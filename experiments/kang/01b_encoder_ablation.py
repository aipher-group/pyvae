#!/usr/bin/env python
"""Do the encoder-side options earn their place? The second grid.

    python 01b_encoder_ablation.py

`01_architecture.py` varies where the covariate enters and whether the decoder
carries a sign, and holds four other options fixed in `BASE`:

    normalize="batch"        standardise each unit's pre-activation per batch
    informed_decoder=True    mask `dec_out` by the annotation
    nonneg_encoder=True      softplus the encoder's masked weights
    standardize_input=True   z-score each gene before the masked product

Those four were selected in an earlier one-at-a-time grid whose metrics are no
longer valid: `pathway_unit_fidelity` measures distance from initialisation (an
untrained non-negative model scores about 0.75 and the score *falls* with
training) and the model-against-naive win it reported is reproduced by a
permuted annotation. This script re-asks the same question with the read-out
the rest of the pipeline uses, so `BASE` is evidence rather than inheritance.

**The arms.** The conditioning scheme is pinned to the selected one
(`cov_mode="pathway"`, `cov_in_encoder=False`, `nonneg_decoder=True`) and one
encoder-side option moves at a time:

    selected          BASE unchanged; must reproduce `real_signed` from 01
    normalize_none    no pre-activation normalisation
    normalize_layer   LayerNorm instead of BatchNorm
    init_fan_in       fan-in initialisation
    free_encoder      nonneg_encoder=False   (sign freedom restored)
    unscaled_input    standardize_input=False (scale freedom restored)
    free_unscaled     both off; completes the 2x2 with `selected` as "both"
    dense_decoder     informed_decoder=False, so no decoder mask at all
    dense_encoder     informed_encoder=False, so no encoder mask at all

The last two move two flags each, because `nonneg_decoder` requires a masked
decoder to constrain and `nonneg_encoder` requires a masked encoder. They are
the two halves of the same question. The permutation control asks whether
*this* map beats a shuffle; these ask whether a map on that side beats none,
and which side carries it.

`dense_decoder` is scored on fit alone: with no mask on `dec_out` there are no
units to rank. `dense_encoder` keeps the decoder mask, so it is scored like
every other arm, and its encoder-side sign statistics are then taken over every
weight rather than over the live ones.

Each arm runs real and permuted, so annotation worth is comparable with 01's.
18 fits, about 4 min each.

    python 01b_encoder_ablation.py                                 # all of them
    python 01b_encoder_ablation.py --arms real_dense_encoder \
        perm_dense_encoder --append                                # just one

`--append` merges the fitted arms into an existing `arms.csv` and `summary.json`
rather than replacing them, so a new arm costs its own two fits instead of the
whole grid. Every aggregate (annotation worth, the 2x2) is recomputed from the
merged table.

Writes `outputs/encoder_ablation/`:

    arms.csv       one row per arm: fit, target rank, saturation, sign structure
    summary.json   the arms, annotation worth per option, and the 2x2
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

from pyvae import (
    InformedVAE,
    permute_adjacency,
    set_all_seeds,
    train_ivae_modern,
    unit_contrast,
    unit_sign_structure,
)

HERE = Path(__file__).resolve().parent


def _load_arch_module():
    """Reuse 01_architecture.py's data prep, target and helpers.

    Imported by path because the module name starts with a digit. Sharing the
    loader is the point: `selected` here must be the same fit as `real_signed`
    there, which requires identical genes, splits, design and contrast cells.
    """
    spec = importlib.util.spec_from_file_location(
        "kang_arch", HERE / "01_architecture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load_arch_module()

#: The conditioning scheme 01 selected, held fixed here.
COND = dict(cov_mode="pathway", cov_in_encoder=False, nonneg_decoder=True)

OPTIONS = {
    "selected": {},
    "normalize_none": dict(normalize="none"),
    "normalize_layer": dict(normalize="layer"),
    "init_fan_in": dict(init="fan_in"),
    "free_encoder": dict(nonneg_encoder=False),
    "unscaled_input": dict(standardize_input=False),
    "free_unscaled": dict(nonneg_encoder=False, standardize_input=False),
    "dense_decoder": dict(informed_decoder=False, nonneg_decoder=False),
    "dense_encoder": dict(informed_encoder=False, nonneg_encoder=False),
}
ARMS = {f"{m}_{o}": dict(permute=(m == "perm"), **kw)
        for o, kw in OPTIONS.items() for m in ("real", "perm")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--max-cells", type=int, default=20_000)
    ap.add_argument("--append", action="store_true",
                    help="merge into an existing arms.csv and summary.json "
                         "instead of replacing them")
    args = ap.parse_args()

    out_dir = HERE / "outputs" / "encoder_ablation"
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

    out = []
    summary = {"seed": args.seed, "epochs": args.epochs,
               "cells": int(adata.n_obs), "genes": len(genes),
               "units": len(pathway_names), "target": A.TARGET,
               "conditioning_held_fixed": COND,
               "base_under_test": A.BASE, "arms": {}}

    for arm in args.arms:
        spec = ARMS[arm]
        moved = {k: v for k, v in spec.items() if k != "permute"}
        flags = {**A.BASE, **COND, **moved}
        adj = perm_adj if spec["permute"] else adj_df

        set_all_seeds(args.seed)
        model = InformedVAE(
            adj=torch.tensor(np.asarray(adj), dtype=torch.float32),
            latent_dim=len(pathway_names) // 2, seed=args.seed, l2_lambda=1e-5,
            likelihood="nb", n_cov=cov.shape[1], **flags)
        t0 = time.time()
        model, hist = train_ivae_modern(
            model, x_train=data["x_train"], x_val=data["x_val"],
            x_counts_train=data["counts_train"],
            x_counts_val=data["counts_val"], cov_train=data["cov_train"],
            cov_val=data["cov_val"], device=dev, epochs=args.epochs,
            patience=args.epochs, batch_size=128)
        best = float(pd.DataFrame(hist)["val"].min())
        mins = (time.time() - t0) / 60

        # `dense_decoder` is scored on fit alone. With no mask on `dec_out`
        # every gene is a member of every unit, so `member_shift_z` has no
        # member set to sum over and the unit read-out is undefined rather
        # than merely different. The question that arm answers -- does any map
        # beat none -- is answered by held-out loss and annotation worth.
        scored = flags["informed_decoder"]
        sign = unit_sign_structure(model) if scored else {}
        unit_cols = {k: None for k in (
            "unit_rank_target", "unit_rank_parent", "top_unit", "units_pinned",
            "frac_pinned", "saturated_frac_max", "delta_post_abs_median",
            "masked_share", "encoder_frac_negative", "dec_out_frac_negative",
            "sign_agree_unit_vs_genes", "units_without_members")}
        if scored:
            res = unit_contrast(model, x.loc[rows], cov_a, cov_b,
                                pathway_names=pathway_names, gene_names=genes)
            cs, units = res["summary"], res["units"]
            if cs["delta_pre_sd_max"] > A.PRE_SD_TOL:
                raise AssertionError(
                    f"{arm}: the pre-activation shift varies between cells by "
                    f"{cs['delta_pre_sd_max']:.2e}, so this was not a clean "
                    "two-level covariate contrast")
            pinned = int((units["saturated_frac"] > 0.5).sum())
            unit_cols = {
                "unit_rank_target": A.rank_of(units["member_shift_z"], A.TARGET),
                "unit_rank_parent": A.rank_of(units["member_shift_z"],
                                              A.TARGET_PARENT),
                "top_unit": units["member_shift_z"].idxmax(),
                "units_pinned": pinned,
                "frac_pinned": round(pinned / len(units), 4),
                "saturated_frac_max": round(cs["saturated_frac_max"], 4),
                "delta_post_abs_median": float(
                    np.median(np.abs(units["delta_post"].to_numpy()))),
                "masked_share": round(cs["masked_share"], 4),
                "encoder_frac_negative": round(sign["encoder_frac_negative"], 4),
                "dec_out_frac_negative": round(sign["dec_out_frac_negative"], 4),
                "sign_agree_unit_vs_genes": A.sign_agreement(units),
                "units_without_members": int((units["n_genes"] == 0).sum()),
            }

        row = {
            "arm": arm, "option": arm.split("_", 1)[1],
            "permute": spec["permute"],
            "moved": ", ".join(f"{k}={v}" for k, v in moved.items()) or "-",
            "normalize": flags["normalize"],
            "init": flags.get("init", "xavier"),
            "nonneg_encoder": flags["nonneg_encoder"],
            "standardize_input": flags["standardize_input"],
            "informed_encoder": flags.get("informed_encoder", True),
            "informed_decoder": flags["informed_decoder"],
            "nonneg_decoder": flags["nonneg_decoder"],
            "best_val": round(best, 2), "train_min": round(mins, 1),
            "scored_on_units": scored,
            **unit_cols,
        }
        out.append(row)
        summary["arms"][arm] = row

        if scored:
            print(f"{arm:<22s} val {best:8.1f} ({mins:4.1f} min)  "
                  f"unit rank {str(row['unit_rank_target']):>5} "
                  f"(parent {str(row['unit_rank_parent']):>5})  "
                  f"pinned {row['frac_pinned']:.0%}  "
                  f"enc-ve {row['encoder_frac_negative']:.3f}  "
                  f"sign {row['sign_agree_unit_vs_genes']:.3f}")
        else:
            print(f"{arm:<22s} val {best:8.1f} ({mins:4.1f} min)  "
                  f"no decoder mask: scored on fit alone")

        del model
        torch.cuda.empty_cache()

    arms = pd.DataFrame(out)
    prior_csv, prior_json = out_dir / "arms.csv", out_dir / "summary.json"
    if args.append and prior_csv.exists():
        # Merge rather than rebuild. The existing rows are left exactly as they
        # were written, because round-tripping them through this process would
        # change their types without changing their values, and the verifier
        # reads both files.
        kept = pd.read_csv(prior_csv)
        kept = kept[~kept["arm"].isin(arms["arm"])]
        if "informed_encoder" not in kept.columns:
            kept = kept.assign(informed_encoder=True)
        order = {a: i for i, a in enumerate(ARMS)}
        arms = (pd.concat([kept, arms], ignore_index=True)
                .sort_values("arm", key=lambda col: col.map(order))
                .reset_index(drop=True))
        if prior_json.exists():
            merged = json.loads(prior_json.read_text()).get("arms", {})
            for row in merged.values():
                row.setdefault("informed_encoder", True)
            merged.update(summary["arms"])
            summary["arms"] = {a: merged[a] for a in ARMS if a in merged}
        print(f"\nappended {len(out)} arm(s) to {len(arms)} in {prior_csv.name}")
    arms.to_csv(prior_csv, index=False)

    summary["annotation_worth"] = {}
    for opt in OPTIONS:
        a, b = f"real_{opt}", f"perm_{opt}"
        if a in summary["arms"] and b in summary["arms"]:
            summary["annotation_worth"][opt] = round(
                summary["arms"][b]["best_val"]
                - summary["arms"][a]["best_val"], 2)

    ref = summary["arms"].get("real_selected", {})
    print(f"\n=== Kang: do the encoder-side options earn their place? ===")
    print(f"\n{'option':<16s} {'fit':>9s} {'vs selected':>12s} "
          f"{'annot worth':>12s} {'unit rank':>10s} {'pinned':>7s} "
          f"{'enc -ve':>8s} {'sign':>6s}")
    for opt in OPTIONS:
        a = f"real_{opt}"
        if a not in summary["arms"]:
            continue
        r = summary["arms"][a]
        cost = (f"{r['best_val'] - ref['best_val']:+.2f}"
                if ref else "-")
        worth = summary["annotation_worth"].get(opt)
        pin = f"{r['frac_pinned']:.0%}" if r["scored_on_units"] else "-"
        enc = (f"{r['encoder_frac_negative']:.3f}"
               if r["scored_on_units"] else "-")
        sg = (f"{r['sign_agree_unit_vs_genes']:.3f}"
              if r["scored_on_units"] else "-")
        print(f"{opt:<16s} {r['best_val']:9.1f} {cost:>12s} "
              f"{(f'{worth:+.2f}' if worth is not None else '-'):>12s} "
              f"{str(r['unit_rank_target']):>10} "
              f"{pin:>7} {enc:>8} {sg:>6}")

    # The 2x2 of sign freedom against scale freedom, which is the question the
    # retired fidelity experiment was built for.
    cells = {"both": "real_selected", "nonneg_only": "real_unscaled_input",
             "standardize_only": "real_free_encoder",
             "neither": "real_free_unscaled"}
    grid = {k: summary["arms"][v] for k, v in cells.items()
            if v in summary["arms"]}
    if len(grid) == 4:
        summary["sign_scale_2x2"] = {
            k: {"best_val": v["best_val"],
                "unit_rank_target": v["unit_rank_target"],
                "encoder_frac_negative": v["encoder_frac_negative"],
                "frac_pinned": v["frac_pinned"],
                "annotation_worth": summary["annotation_worth"].get(
                    cells[k].removeprefix("real_"))}
            for k, v in grid.items()}
        print("\nnonneg_encoder x standardize_input, real arms:")
        for k in ("both", "nonneg_only", "standardize_only", "neither"):
            v = summary["sign_scale_2x2"][k]
            print(f"  {k:<17s} val {v['best_val']:8.1f}  "
                  f"unit rank {str(v['unit_rank_target']):>5}  "
                  f"enc -ve {v['encoder_frac_negative']:.3f}  "
                  f"worth {v['annotation_worth']}")

    # `nonneg_encoder` is the only option whose effect is structural rather than
    # empirical, so it is asserted rather than argued.
    for arm, r in summary["arms"].items():
        want_zero = r["nonneg_encoder"] and r["scored_on_units"]
        if want_zero and r["encoder_frac_negative"] != 0.0:
            raise AssertionError(
                f"{arm}: nonneg_encoder=True but encoder_frac_negative is "
                f"{r['encoder_frac_negative']}")

    prior_json.write_text(json.dumps(summary, indent=2))
    print(f"\nwritten to {out_dir}/")


if __name__ == "__main__":
    main()
