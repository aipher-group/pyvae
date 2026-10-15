#!/usr/bin/env python
"""Does the pathway annotation affect the result? The version with a known answer.

    python 01_architecture.py

This is `experiments/multiple_mieloma/01_architecture.py` on a dataset where the
answer is fixed before the experiment runs. Kang et al. (2018) is PBMCs,
interferon-beta stimulated against control, so
`REACTOME_INTERFERON_ALPHA_BETA_SIGNALING` is what the model should name. That
makes Kang the external check on whether the architecture the myeloma decision
selects wins for the right reason, rather than merely winning.

**The arms** are the same three library flags crossed with a permuted adjacency:

    shipped    cov_mode="gene"                the dense `(n_genes, n_cov)` term
                                              added after the masked layer
    pathway    cov_mode="pathway"             the covariate's effect becomes a
                                              vector in pathway coordinates
    uncond     + cov_in_encoder=False         and the only route, so the
                                              contrast is one weight per pathway
    signed     + nonneg_decoder=True          and a unit rising raises its
                                              genes, so the sign means something

**What this dataset adds that myeloma cannot.** Two scores per arm: where the
named unit sits among the model's own 1,615, and where the same pathway sits in
the gene-level competitive test that `experiments/kang/FINDINGS.md` reports as
the model's headline. The second is kept here and only here, because it is the
evidence that it was never a model read-out: it never sees the model, and it
ranks the right answer first in **every** arm, permuted ones included.

**What this script does not test.** The four options in `BASE` are held fixed
in every arm here, so nothing below is evidence for them.
`01b_encoder_ablation.py` is where they are tested: it pins the conditioning
scheme at the one this script selects and moves one `BASE` option at a time,
real and permuted, plus a `dense_decoder` arm with no decoder mask at all. Both
scripts share this file's data loader, so `01b`'s `selected` arm is the same fit
as this script's `real_signed`.

One claim is structural rather than empirical and is asserted below rather than
scored: zero inverted encoder units, because `nonneg_encoder` reparameterises
every encoder weight as `softplus(theta) * mask` and makes inversion
unrepresentable. Two metrics the earlier grid used to choose `BASE` are retired
for cause: `pathway_unit_fidelity` medians measured distance from
initialisation -- an untrained non-negative model scores 0.75 and the score
*falls* with training -- and the sample-size experiment's model-against-naive
win is reproduced by a permuted annotation. What that grid never had is the
control both scripts now carry: an arm where the annotation itself is shuffled.

Writes `outputs/architecture/`:

    arms.csv       one row per arm: fit, both ranks, and every diagnostic
    units.csv      the per-unit contrast for every arm
    summary.json   the arms, what the annotation is worth, and the identities
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from pyvae import (
    InformedVAE,
    build_model_config,
    differential_expression,
    load_kang,
    pathway_activity,
    permute_adjacency,
    set_all_seeds,
    train_ivae_modern,
    unit_contrast,
    unit_sign_structure,
)

HERE = Path(__file__).resolve().parent
GMT = HERE / "resources" / "c2.cp.reactome.v7.5.1.symbols.gmt"

#: The answer, fixed before the experiment runs.
TARGET = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING"
#: Its Reactome parent. The hierarchy nests, so the parent ranking first with
#: the child just behind it is agreement, not failure.
TARGET_PARENT = "REACTOME_INTERFERON_SIGNALING"
CTRL, STIM = "condition_control", "condition_stimulated"
PRE_SD_TOL = 1e-4

#: The four encoder-side options shared by every arm, so an arm differs from its
#: neighbour only in the flags named in SCHEMES. Deliberately the same as
#: `experiments/multiple_mieloma/myeloma.py:ARCH_SHIPPED`.
BASE = dict(normalize="batch", informed_decoder=True, nonneg_encoder=True,
            standardize_input=True)
SCHEMES = {
    "shipped": dict(cov_mode="gene"),
    "pathway": dict(cov_mode="pathway"),
    "uncond": dict(cov_mode="pathway", cov_in_encoder=False),
    "signed": dict(cov_mode="pathway", cov_in_encoder=False,
                   nonneg_decoder=True),
}
ARMS = {f"{m}_{s}": dict(permute=(m == "perm"), **kw)
        for s, kw in SCHEMES.items() for m in ("real", "perm")}


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare(seed: int, target_sum: float = 1e4):
    """Kang, the Reactome map, and the condition-by-cell-type design."""
    adata = load_kang(data_folder=str(HERE / "data"), normalize=True,
                      n_genes=None, return_path=False, target_sum=target_sum)
    config = build_model_config(genes=adata.to_df().columns,
                                model_kind="ivae_reactome",
                                resources_dir=str(GMT))
    adj_df = config.model_layer[0]
    genes = list(config.input_genes)
    pathway_names = list(adj_df.columns)
    x = adata.to_df()[genes]
    counts = adata.to_df(layer="counts")[genes]
    cov = pd.get_dummies(adata.obs[["condition", "cell_type"]], dtype=np.float32)

    strat = (adata.obs["condition"].astype(str) + "_"
             + adata.obs["cell_type"].astype(str))
    if strat.value_counts().min() < 2:
        strat = adata.obs["condition"].astype(str)
    i_tr, i_va = train_test_split(np.arange(adata.n_obs), test_size=0.2,
                                  stratify=strat, random_state=seed)
    if len(i_tr) % 128 == 1:
        i_va = np.append(i_va, i_tr[-1])
        i_tr = i_tr[:-1]
    data = {"x_train": x.iloc[i_tr], "x_val": x.iloc[i_va],
            "counts_train": counts.iloc[i_tr], "counts_val": counts.iloc[i_va],
            "cov_train": cov.iloc[i_tr], "cov_val": cov.iloc[i_va]}
    print(f"kang  {adata.n_obs:,} cells x {len(genes):,} genes in the map, "
          f"{len(pathway_names):,} units, {cov.shape[1]} covariate columns")
    print(f"target {TARGET.replace('REACTOME_', '')} present: "
          f"{TARGET in pathway_names}")
    return adata, adj_df, genes, pathway_names, cov, x, data


def sign_agreement(units) -> float:
    """Fraction of units whose own direction matches their genes' direction.

    **Over units that have genes.** A unit with no live decoder weight has no
    member direction to agree or disagree with, so including it scores a
    guaranteed mismatch for a question that was never asked. It is not
    hypothetical: 9 of Kang's 1,615 Reactome sets share no gene with that
    dataset's 7,573-gene panel, and counting them held this statistic at 0.9944
    on a fit where every unit that writes anything agrees perfectly. Their count
    is reported beside this number rather than folded into it.

    Under `nonneg_decoder=True` this is 1.0 exactly, because `member_mean_shift`
    is `delta_post` times a positive weight sum. Without it, it sits near 0.5 --
    which is the whole point of the comparison.
    """
    import numpy as np

    has_genes = units["n_genes"].to_numpy() > 0
    if not has_genes.any():
        return float("nan")
    agree = (np.sign(units["delta_post"].to_numpy()[has_genes])
             == np.sign(units["member_mean_shift"].to_numpy()[has_genes]))
    return round(float(agree.mean()), 4)


def rank_of(series, name) -> int | None:
    if name not in series.index:
        return None
    return int(series.sort_values(ascending=False).index.get_loc(name)) + 1


def centred_maxdev(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs((a - a.mean()) - (b - b.mean())).max())


def forward_pass(model, x, cov_a, cov_b, n_genes, *, batch=512):
    """The decoder's own shift, with `mu` held fixed and with it re-encoded."""
    dev = device()
    model = model.to(dev).eval()
    xv = x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)
    fixed = np.zeros(n_genes)
    total = np.zeros(n_genes)
    with torch.no_grad():
        for s in range(0, len(xv), batch):
            xb = torch.as_tensor(xv[s:s + batch], dtype=torch.float32, device=dev)
            ab = torch.as_tensor(cov_a.to_numpy()[s:s + batch], dtype=torch.float32,
                                 device=dev)
            bb = torch.as_tensor(cov_b.to_numpy()[s:s + batch], dtype=torch.float32,
                                 device=dev)
            mu_a = model.encode(xb, ab)[0]
            mu_b = model.encode(xb, bb)[0]

            def lg(px):
                return torch.log(px.clamp_min(1e-30)).double()

            fixed += (lg(model.decoder(mu_a, bb))
                      - lg(model.decoder(mu_a, ab))).sum(0).cpu().numpy()
            total += (lg(model.decoder(mu_b, bb))
                      - lg(model.decoder(mu_a, ab))).sum(0).cpu().numpy()
    return fixed / len(xv), total / len(xv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--max-cells", type=int, default=20_000)
    args = ap.parse_args()

    out_dir = HERE / "outputs" / "architecture"
    if args.seed != 42:
        out_dir = out_dir / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = device()

    adata, adj_df, genes, pathway_names, cov, x, data = prepare(args.seed)
    perm_adj = permute_adjacency(adj_df, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    # The contrast is evaluated on control cells: what the model does to *these*
    # cells when told they were stimulated. Fixed across arms.
    rows = cov.index[cov[CTRL].to_numpy() > 0.5]
    if len(rows) > args.max_cells:
        rows = pd.Index(rng.choice(rows, args.max_cells, replace=False))
    cond_cols = [c for c in cov.columns if c.startswith("condition_")]
    cov_a = cov.loc[rows].copy()
    cov_a[cond_cols] = 0.0
    cov_b = cov_a.copy()
    cov_a[CTRL], cov_b[STIM] = 1.0, 1.0
    print(f"contrast on {len(rows):,} control cells\n")

    out, unit_rows = [], []
    summary = {"seed": args.seed, "epochs": args.epochs,
               "cells": int(adata.n_obs), "genes": len(genes),
               "units": len(pathway_names), "target": TARGET,
               "contrast_cells": int(len(rows)), "arms": {}}

    for arm in args.arms:
        spec = ARMS[arm]
        flags = {k: v for k, v in spec.items() if k != "permute"}
        adj = perm_adj if spec["permute"] else adj_df

        set_all_seeds(args.seed)
        model = InformedVAE(
            adj=torch.tensor(np.asarray(adj), dtype=torch.float32),
            latent_dim=len(pathway_names) // 2, seed=args.seed, l2_lambda=1e-5,
            likelihood="nb", n_cov=cov.shape[1], **BASE, **flags)
        t0 = time.time()
        model, hist = train_ivae_modern(
            model, x_train=data["x_train"], x_val=data["x_val"],
            x_counts_train=data["counts_train"],
            x_counts_val=data["counts_val"], cov_train=data["cov_train"],
            cov_val=data["cov_val"], device=dev, epochs=args.epochs,
            patience=args.epochs, batch_size=128)
        hist = pd.DataFrame(hist)
        best = float(hist["val"].min())
        mins = (time.time() - t0) / 60

        # Folded in from the retired grid: inversion is unrepresentable under
        # `nonneg_encoder`, so it is asserted rather than scored, and saturation
        # is the metric `normalize="batch"` was chosen for.
        sign = unit_sign_structure(model)
        assert sign["encoder_frac_negative"] == 0.0, (
            "nonneg_encoder should make a negative encoder weight "
            "unrepresentable; this is the one claim the old fidelity grid "
            "established that still stands"
        )

        res = unit_contrast(model, x.loc[rows], cov_a, cov_b,
                           pathway_names=pathway_names, gene_names=genes)
        s = res["summary"]
        if s["delta_pre_sd_max"] > PRE_SD_TOL:
            raise AssertionError(
                f"{arm}: the pre-activation shift varies between cells by "
                f"{s['delta_pre_sd_max']:.2e}, so this was not a clean "
                "two-level covariate contrast")
        units = res["units"]

        # The gene-level baseline, exactly as the old grid computed it: posterior
        # DE, then a competitive rank test against the adjacency.
        xv, cv = data["x_val"], data["cov_val"]
        m_ctrl = cv[CTRL].to_numpy() > 0.5
        m_stim = cv[STIM].to_numpy() > 0.5
        de = differential_expression(
            model, x_a=xv.iloc[m_stim], x_b=xv.iloc[m_ctrl],
            cov_a=cv.iloc[m_stim], cov_b=cv.iloc[m_ctrl], n_samples=25,
            n_pairs=2000, gene_names=genes, seed=args.seed)
        pa = pathway_activity(de, adj_df, statistic="lfc_mean", min_genes=10)

        weight_rank = direct_dev = None
        if flags.get("cov_mode") == "pathway" and not flags.get(
                "cov_in_encoder", True):
            direct = pd.Series(
                model.pathway_contrast(cov_a.to_numpy(), cov_b.to_numpy()
                                       ).cpu().numpy(), index=pathway_names)
            oriented = direct * np.sign(
                units["member_mean_shift"].where(lambda v: v != 0, 1.0))
            weight_rank = rank_of(oriented, TARGET)
            direct_dev = float(np.abs(direct.to_numpy()
                                      - units["delta_pre"].to_numpy()).max())

        fixed, total = forward_pass(model, x.loc[rows], cov_a, cov_b, len(genes))
        checks = {"decomposition_vs_reencoded_maxdev": centred_maxdev(
                      total, res["genes"]["total"]),
                  "counterfactual_vs_bypass_maxdev": None,
                  "counterfactual_vs_reencoded_maxdev": None}
        if flags.get("cov_mode", "gene") == "gene":
            checks["counterfactual_vs_bypass_maxdev"] = centred_maxdev(
                fixed, res["genes"]["bypass"])
        elif not flags.get("cov_in_encoder", True):
            checks["counterfactual_vs_reencoded_maxdev"] = centred_maxdev(
                fixed, total)

        row = {
            "arm": arm, "permute": spec["permute"],
            "cov_mode": flags.get("cov_mode", "gene"),
            "cov_in_encoder": flags.get("cov_in_encoder", True),
            "nonneg_decoder": flags.get("nonneg_decoder", False),
            "best_val": round(best, 2), "train_min": round(mins, 1),
            "unit_rank_target": rank_of(units["member_shift_z"], TARGET),
            "unit_rank_parent": rank_of(units["member_shift_z"], TARGET_PARENT),
            "gene_rank_target": rank_of(pa["z_competitive"], TARGET),
            "gene_z_target": round(float(pa.loc[TARGET, "z_competitive"]), 2)
            if TARGET in pa.index else None,
            "top_unit": units["member_shift_z"].idxmax(),
            "weight_rank_target": weight_rank,
            "masked_share": round(s["masked_share"], 4),
            "direct_vs_forward_maxdev": direct_dev,
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in sign.items()},
            "sign_agree_unit_vs_genes": sign_agreement(units),
            "units_without_members": int((units["n_genes"] == 0).sum()),
            **checks,
        }
        out.append(row)
        summary["arms"][arm] = row
        u = units.copy()
        u.insert(0, "arm", arm)
        u.index.name = "unit"
        unit_rows.append(u.reset_index())

        print(f"{arm:<16s} val {best:8.1f} ({mins:4.1f} min)   "
              f"unit rank {str(row['unit_rank_target']):>5} "
              f"(parent {str(row['unit_rank_parent']):>5})   "
              f"gene rank {str(row['gene_rank_target']):>3} "
              f"(z {row['gene_z_target']})   "
              f"sign agree {row['sign_agree_unit_vs_genes']:.3f}")
        print(f"{'':<16s} top unit {row['top_unit'].replace('REACTOME_', '')[:52]}")

        del model
        torch.cuda.empty_cache()

    arms = pd.DataFrame(out)
    arms.to_csv(out_dir / "arms.csv", index=False)
    pd.concat(unit_rows, ignore_index=True).to_csv(out_dir / "units.csv",
                                                   index=False)

    summary["annotation_worth"] = {}
    for scheme in SCHEMES:
        a, b = f"real_{scheme}", f"perm_{scheme}"
        if a in summary["arms"] and b in summary["arms"]:
            summary["annotation_worth"][scheme] = round(
                summary["arms"][b]["best_val"] - summary["arms"][a]["best_val"], 2)

    base_val = summary["arms"].get("real_shipped", {}).get("best_val")
    print(f"\n=== Kang: {TARGET.replace('REACTOME_', '')} ===")
    print(f"\n{'scheme':<10s} {'fit':>9s} {'vs shipped':>11s} {'annot worth':>12s}"
          f"   unit rank (real/perm)   gene rank (real/perm)")
    for scheme in SCHEMES:
        a, b = f"real_{scheme}", f"perm_{scheme}"
        if a not in summary["arms"]:
            continue
        ra, rb = summary["arms"][a], summary["arms"].get(b, {})
        cost = (f"{ra['best_val'] - base_val:+.2f}"
                if base_val is not None else "-")
        worth = summary["annotation_worth"].get(scheme)
        print(f"{scheme:<10s} {ra['best_val']:9.1f} {cost:>11s} "
              f"{(f'{worth:+.2f}' if worth is not None else '-'):>12s}   "
              f"{str(ra['unit_rank_target']):>5} / "
              f"{str(rb.get('unit_rank_target')):<5}      "
              f"{str(ra['gene_rank_target']):>4} / "
              f"{str(rb.get('gene_rank_target')):<4}")

    gene_ranks = sorted(set(arms["gene_rank_target"].dropna().astype(int)))
    print(f"\ngene-level rank of the known answer, over all {len(arms)} arms: "
          f"{gene_ranks}")
    if gene_ranks == [1]:
        print("  -- 1 in every arm, permuted ones included. The gene-level "
              "read-out does not depend on the annotation being correct.")
    summary["gene_rank_values_over_arms"] = gene_ranks

    IDENTITIES = {
        "decomposition_vs_reencoded_maxdev":
            "masked + bypass == the re-encoded shift",
        "counterfactual_vs_bypass_maxdev":
            "predict_counterfactual == the dense term (gene mode)",
        "counterfactual_vs_reencoded_maxdev":
            "predict_counterfactual == the re-encoded shift (encoder route closed)",
        "direct_vs_forward_maxdev": "pathway_contrast == the forward pass",
    }
    print("\nidentities, worst deviation over the rows where each applies:")
    summary["identities"] = {}
    failed = []
    for col, what in IDENTITIES.items():
        vals = arms[col].dropna()
        if vals.empty:
            continue
        worst = float(vals.max())
        summary["identities"][col] = {"rows": int(len(vals)), "max": worst}
        print(f"  {worst:.2e}  over {len(vals):>2d} rows   {what}")
        if worst > 1e-4:
            failed.append((col, worst))
    if failed:
        raise AssertionError(
            "identities failed, so nothing above can be read: "
            + ", ".join(f"{c} at {v:.2e}" for c, v in failed))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwritten to {out_dir}/")


if __name__ == "__main__":
    main()
