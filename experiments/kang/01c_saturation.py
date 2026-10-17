#!/usr/bin/env python
"""Is annotation worth confounded by saturation? The third grid.

    python 01c_saturation.py --scales 0.25 0.5 1 2 4

`01b_encoder_ablation.py` rejected `init="fan_in"` because annotation worth
falls from +14.52 to +8.07 nats under it. Fan-in also takes the pinned fraction
from 71% to 43%, and those two facts are perfectly collinear in that design:
fan-in is its only low-saturation arm. So the rejection has two readings. Either
the annotation matters less under fan-in, or a less saturated model is simply
easier for a shuffled map to fit, in which case annotation worth is partly a
restatement of saturation and the selection rule is confounded by the cost it is
meant to weigh against.

This script separates them with a saturation knob that leaves the annotation
untouched. Scaling `dec_latent` and `cov_path` at initialisation scales the
decoder pre-activation `u = U z + Q c` uniformly, which moves how far units sit
into the flat tail of `tanh` without changing which genes belong to which unit.
The mechanism is the one
`tests/unit/test_cov_conditioning_modes.py::TestSaturationDoesNotDestroyTheSign`
already uses to force saturation deliberately.

**Direction.** Only scaling up sticks. A probe at scale 0.25 converged to the
same 71% pinned as scale 1, because training restores whatever pre-activation
magnitude the likelihood wants, so matched saturation is reached by raising
fan-in toward xavier rather than lowering xavier toward fan-in.

**The arms.** The selected architecture at a ladder of pre-activation scales,
each real and permuted. Annotation worth regressed on the pinned fraction then
gives the slope the confound would require. A flat slope means worth does not
track saturation and `01b`'s rejection of fan-in stands on its own. A steep one
means it does, and fan-in has to be re-judged at matched saturation, which
`--fanin-scale` adds as a further pair.

Writes `outputs/saturation/`:

    arms.csv       one row per arm: scale, fit, pinned fraction, target rank
    summary.json   the arms, worth per scale, and the worth-against-pinned fit
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from pyvae import (
    InformedVAE,
    permute_adjacency,
    set_all_seeds,
    train_ivae_modern,
    unit_contrast,
    unit_sign_structure,
)

HERE = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load("kang_arch", "01_architecture.py")
COND = dict(cov_mode="pathway", cov_in_encoder=False, nonneg_decoder=True)


def scale_pre_activation(model, factor: float) -> None:
    """Scale the decoder pre-activation uniformly, annotation untouched.

    `u = dec_latent(z) + cov_path(cov)`, so scaling both terms and the bias
    scales `u` and therefore `tanh(u)`'s saturation. `dec_out` is left alone:
    touching it would change what each unit writes to its genes, which is the
    annotation's half of the model and must not move.
    """
    dec = model.decoder
    with torch.no_grad():
        dec.dec_latent.weight.mul_(factor)
        dec.dec_latent.bias.mul_(factor)
        dec.cov_path.weight.mul_(factor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scales", type=float, nargs="*",
                    default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--fanin-scales", type=float, nargs="*", default=[],
                    help="also run fan-in init at these pre-activation "
                         "scales. Fan-in sits at 43%% pinned against xavier's "
                         "71%%, and scaling down is a no-op at convergence, so "
                         "matched saturation is reached by scaling fan-in UP "
                         "rather than xavier down.")
    ap.add_argument("--real-only", action="store_true",
                    help="probe mode: skip the permuted twins")
    ap.add_argument("--max-cells", type=int, default=20_000)
    ap.add_argument("--out", default="saturation")
    args = ap.parse_args()

    out_dir = HERE / "outputs" / args.out
    if args.seed != 42:
        out_dir = out_dir / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = A.device()

    adata, adj_df, genes, pathway_names, cov, x, data = A.prepare(args.seed)
    perm_adj = permute_adjacency(adj_df, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    rows = cov.index[cov[A.CTRL].to_numpy() > 0.5]
    if len(rows) > args.max_cells:
        rows = pd.Index(rng.choice(rows, args.max_cells, replace=False))
    cond_cols = [c for c in cov.columns if c.startswith("condition_")]
    cov_a = cov.loc[rows].copy()
    cov_a[cond_cols] = 0.0
    cov_b = cov_a.copy()
    cov_a[A.CTRL], cov_b[A.STIM] = 1.0, 1.0
    print(f"contrast on {len(rows):,} control cells\n")

    specs = {f"scale{s:g}": dict(scale=s, init="xavier") for s in args.scales}
    for fs in args.fanin_scales:
        specs[f"fanin_scale{fs:g}"] = dict(scale=fs, init="fan_in")
    modes = ("real",) if args.real_only else ("real", "perm")
    arms = {f"{m}_{k}": dict(permute=(m == "perm"), **v)
            for k, v in specs.items() for m in modes}

    out = []
    summary = {"seed": args.seed, "epochs": args.epochs,
               "units": len(pathway_names), "target": A.TARGET,
               "conditioning_held_fixed": COND, "base": A.BASE,
               "scales": args.scales, "fanin_scales": args.fanin_scales,
               "real_only": args.real_only, "arms": {}}

    for arm, spec in arms.items():
        set_all_seeds(args.seed)
        model = InformedVAE(
            adj=torch.tensor(np.asarray(perm_adj if spec["permute"] else adj_df),
                             dtype=torch.float32),
            latent_dim=len(pathway_names) // 2, seed=args.seed, l2_lambda=1e-5,
            likelihood="nb", n_cov=cov.shape[1],
            **{**A.BASE, **COND, "init": spec["init"]})
        scale_pre_activation(model, spec["scale"])

        t0 = time.time()
        model, hist = train_ivae_modern(
            model, x_train=data["x_train"], x_val=data["x_val"],
            x_counts_train=data["counts_train"],
            x_counts_val=data["counts_val"], cov_train=data["cov_train"],
            cov_val=data["cov_val"], device=dev, epochs=args.epochs,
            patience=args.epochs, batch_size=128)
        best = float(pd.DataFrame(hist)["val"].min())
        mins = (time.time() - t0) / 60

        sign = unit_sign_structure(model)
        res = unit_contrast(model, x.loc[rows], cov_a, cov_b,
                            pathway_names=pathway_names, gene_names=genes)
        cs, units = res["summary"], res["units"]
        if cs["delta_pre_sd_max"] > A.PRE_SD_TOL:
            raise AssertionError(
                f"{arm}: pre-activation shift varies between cells by "
                f"{cs['delta_pre_sd_max']:.2e}")
        pinned = int((units["saturated_frac"] > 0.5).sum())
        row = {
            "arm": arm, "spec": arm.split("_", 1)[1],
            "permute": spec["permute"], "scale": spec["scale"],
            "init": spec["init"],
            "best_val": round(best, 2), "train_min": round(mins, 1),
            "units_pinned": pinned,
            "frac_pinned": round(pinned / len(units), 4),
            "saturated_frac_max": round(cs["saturated_frac_max"], 4),
            "delta_post_abs_median": float(
                np.median(np.abs(units["delta_post"].to_numpy()))),
            "unit_rank_target": A.rank_of(units["member_shift_z"], A.TARGET),
            "unit_rank_parent": A.rank_of(units["member_shift_z"],
                                          A.TARGET_PARENT),
            "dec_out_frac_negative": round(sign["dec_out_frac_negative"], 4),
            "sign_agree_unit_vs_genes": A.sign_agreement(units),
        }
        out.append(row)
        summary["arms"][arm] = row
        print(f"{arm:<22s} val {best:8.1f} ({mins:4.1f} min)  "
              f"pinned {row['frac_pinned']:>5.0%}  "
              f"|dpost| {row['delta_post_abs_median']:.2e}  "
              f"rank {str(row['unit_rank_target']):>5}")
        del model
        torch.cuda.empty_cache()

    arms_df = pd.DataFrame(out)
    arms_df.to_csv(out_dir / "arms.csv", index=False)

    summary["annotation_worth"] = {}
    for k in specs:
        a, b = f"real_{k}", f"perm_{k}"
        if a in summary["arms"] and b in summary["arms"]:
            summary["annotation_worth"][k] = round(
                summary["arms"][b]["best_val"]
                - summary["arms"][a]["best_val"], 2)

    print(f"\n=== does the pre-activation scale move saturation? ===")
    print(f"\n{'arm':<22s} {'scale':>6s} {'pinned':>7s} {'fit':>9s} "
          f"{'worth':>7s} {'rank':>6s}")
    for k in specs:
        r = summary["arms"].get(f"real_{k}")
        if r is None:
            continue
        w = summary["annotation_worth"].get(k)
        print(f"{k:<22s} {r['scale']:>6g} {r['frac_pinned']:>6.0%} "
              f"{r['best_val']:>9.1f} "
              f"{(f'{w:+.2f}' if w is not None else '-'):>7s} "
              f"{str(r['unit_rank_target']):>6}")

    pins = [summary["arms"][f"real_{k}"]["frac_pinned"] for k in specs
            if f"real_{k}" in summary["arms"]]
    spread = max(pins) - min(pins)
    summary["pinned_spread"] = round(spread, 4)
    print(f"\npinned fraction spans {min(pins):.0%} to {max(pins):.0%} "
          f"(spread {spread:.0%})")
    if spread < 0.10:
        print("  -- the knob does not survive training: the ladder collapsed, "
              "so this design cannot separate worth from saturation.")

    # The regression the confound reading requires. Only meaningful over the
    # xavier arms: a fan-in arm differs in more than its saturation.
    xs = [(summary["arms"][f"real_{k}"]["frac_pinned"],
           summary["annotation_worth"][k])
          for k in specs
          if k.startswith("scale") and k in summary["annotation_worth"]]
    if len(xs) >= 3:
        pin, worth = np.array([p for p, _ in xs]), np.array([w for _, w in xs])
        lr = stats.linregress(pin, worth)
        summary["worth_vs_pinned"] = {
            "n": len(xs), "slope_nats_per_unit_pinned": round(lr.slope, 3),
            "slope_nats_per_10pct": round(lr.slope * 0.10, 3),
            "r": round(lr.rvalue, 3), "p": float(lr.pvalue),
            "intercept": round(lr.intercept, 2)}
        print(f"\nannotation worth against pinned fraction, over "
              f"{len(xs)} xavier arms:")
        print(f"  slope {lr.slope * 0.10:+.2f} nats per 10 points of pinned "
              f"fraction, r = {lr.rvalue:+.3f}, p = {lr.pvalue:.3g}")
        summary["fanin_vs_curve"] = {}
        for fs in args.fanin_scales:
            k = f"fanin_scale{fs:g}"
            fan = summary["annotation_worth"].get(k)
            if fan is None:
                continue
            fr = summary["arms"][f"real_{k}"]
            pred = lr.intercept + lr.slope * fr["frac_pinned"]
            summary["fanin_vs_curve"][k] = {
                "scale": fs, "pinned": fr["frac_pinned"], "worth": fan,
                "predicted_by_curve": round(pred, 2),
                "residual": round(fan - pred, 2)}
            print(f"  fan-in scale {fs:<5g} pinned {fr['frac_pinned']:>4.0%}  "
                  f"worth {fan:+.2f}  curve predicts {pred:+.2f}  "
                  f"residual {fan - pred:+.2f}")
        if summary["fanin_vs_curve"]:
            print("\n  A residual near zero means fan-in's low worth is "
                  "explained by its saturation; a large negative one means it "
                  "is not, and 01b's rejection stands on its own.")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwritten to {out_dir}/")


if __name__ == "__main__":
    main()
