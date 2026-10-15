#!/usr/bin/env python
"""Does the pathway annotation affect the result? The experiment that decides it.

    python 01_architecture.py --study Zavidij_2020 --seed 42

**Why this runs before anything else.** `InformedVAE` is named for a
gene->pathway mask, and in the shipped configuration no result depends on it.
The decoder computes

    logits = dec_out(tanh(dec_latent(z))) + cov_decoder(cov)
             \\_____ through the mask _____/   \\__ dense over genes __/

and `predict_counterfactual` encodes under one covariate and decodes that same
`mu` under the other, which makes the first term identical on both sides. So a
covariate contrast is exactly `cov_decoder[:, b] - cov_decoder[:, a]`: one dense
weight difference that never touches the annotation. This script measures that
rather than asserting it, and then measures what it takes to fix it.

**The arms.** Three library flags, crossed with a permuted adjacency:

    shipped    cov_mode="gene"                        the dense term above
    pathway    cov_mode="pathway"                     the covariate's effect
                                                      becomes a vector in
                                                      pathway coordinates
    uncond     + cov_in_encoder=False                 and the only route, so
                                                      the contrast is one
                                                      weight per pathway
    signed     + nonneg_decoder=True                  and a unit rising raises
                                                      its genes, so the sign
                                                      means something

`permute_adjacency` relabels the genes: set sizes, per-gene degree multiset and
the whole pathway-by-pathway overlap matrix survive, and only which genes sit in
which unit changes. So an arm and its permuted twin are matched in capacity,
sparsity and overlap topology and differ in nothing but the biology.

**Two read-outs per arm, and the gap between them is the finding.**

    gene-level   the competitive rank test on Hallmark, which is the baseline
                 this architecture replaces. It never sees the model -- it
                 scores member against non-member genes of a panel -- so it is
                 kept *here and only here*, because "the right answer ranks
                 first even with a shuffled annotation" is the evidence that the
                 baseline was annotation-blind.
    unit-level   where the named Reactome unit sits among all 1,615, by the
                 shift it imposes on its own member genes.

A model that has learned its annotation should agree with itself. The shipped
architecture does not: it puts the interferon unit in the bottom five.

**The target is not ground truth here.** `Zavidij_2020` has no known answer the
way Kang does. `REACTOME_INTERFERON_ALPHA_BETA_SIGNALING` is used as the target
because the gene-level baseline puts interferon first in both cohorts, so it is
"the unit the baseline points at" and the question is whether the model's own
units agree. `experiments/kang/01_architecture.py` runs the same arms where the
answer *is* known in advance.

Writes `outputs/architecture/<study>[/seed<n>]/`:

    arms.csv       one row per arm: fit, both ranks, and every diagnostic
    units.csv      the per-unit contrast for every arm
    summary.json   the arms, plus the forward-pass identities that must hold
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import train_test_split

import myeloma as mm
from pyvae import (
    InformedVAE,
    pathway_activity,
    permute_adjacency,
    set_all_seeds,
    train_ivae_modern,
    unit_contrast,
    unit_sign_structure,
)

#: The unit the gene-level baseline points at. Not ground truth in this cohort:
#: see the module docstring.
TARGET = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING"
#: Its Reactome parent. Reactome nests, so a child missing while its parent
#: ranks first is agreement, not failure.
TARGET_PARENT = "REACTOME_INTERFERON_SIGNALING"
#: `delta_pre` is a linear image of a constant covariate difference, so any
#: cell-to-cell spread beyond float noise means the contrast was not a clean
#: two-level one and nothing else in the row should be read.
PRE_SD_TOL = 1e-4

#: Each arm as the library flags that produce it. `ARCH_SHIPPED` supplies the
#: four encoder-side options every arm shares, so an arm differs from its
#: neighbour in exactly the flags named here.
SCHEMES = {
    "shipped": dict(cov_mode="gene"),
    "pathway": dict(cov_mode="pathway"),
    "uncond": dict(cov_mode="pathway", cov_in_encoder=False),
    "signed": dict(cov_mode="pathway", cov_in_encoder=False,
                   nonneg_decoder=True),
}
ARMS = {f"{m}_{s}": dict(permute=(m == "perm"), **kw)
        for s, kw in SCHEMES.items() for m in ("real", "perm")}


def prepare(study: str, *, seed: int):
    """The same cells, design, split and panels every other script uses."""
    adata = mm.load_tcells(study, analysed_only=True)
    adata.obs[mm.STAGE_BLOCK] = (adata.obs["subset"].astype(str) + "__"
                                 + adata.obs["cohort"].astype(str))
    blocks = [mm.STAGE_BLOCK, "pheno_h"]
    cov = pd.get_dummies(adata.obs[blocks].astype(str), dtype=np.float32)
    assert cov.sum(axis=1).min() == len(blocks), "a block has a missing value"

    adata, adj_df, gene_names, pathway_names = mm.build_pathways(adata)
    cov = cov.loc[adata.obs_names]
    x = adata.to_df().astype(np.float32)
    counts = adata.to_df(layer="counts").astype(np.float32)

    strat = adata.obs["cohort"].astype(str) + "_" + adata.obs["pheno_h"].astype(str)
    if strat.value_counts().min() < 2:
        strat = adata.obs["cohort"].astype(str)
    i_tr, i_va = train_test_split(np.arange(len(adata)), test_size=0.2,
                                  stratify=strat, random_state=seed)
    # A training split that is 1 mod the batch size leaves a trailing batch of
    # one row, which BatchNorm rejects. Deterministic, and one cell in 100,000.
    if len(i_tr) % mm.BATCH_SIZE == 1:
        i_va = np.append(i_va, i_tr[-1])
        i_tr = i_tr[:-1]
    data = {"x_train": x.iloc[i_tr], "x_val": x.iloc[i_va],
            "counts_train": counts.iloc[i_tr], "counts_val": counts.iloc[i_va],
            "cov_train": cov.iloc[i_tr], "cov_val": cov.iloc[i_va]}
    return adata, adj_df, gene_names, pathway_names, cov, x, data


def fit(adj, cov, data, *, flags, seed: int, epochs: int):
    """One arm. Every difference between arms is in `flags`."""
    set_all_seeds(seed)
    model = InformedVAE(
        adj=torch.tensor(np.asarray(adj), dtype=torch.float32),
        latent_dim=mm.latent_dim_for(adj.shape[1]), seed=seed,
        l2_lambda=mm.L2_LAMBDA, likelihood="nb", n_cov=cov.shape[1],
        **mm.ARCH_SHIPPED, **flags)
    model, history = train_ivae_modern(
        model, x_train=data["x_train"], x_val=data["x_val"],
        x_counts_train=data["counts_train"], x_counts_val=data["counts_val"],
        cov_train=data["cov_train"], cov_val=data["cov_val"],
        device=mm.device(), epochs=epochs, patience=epochs,
        batch_size=mm.BATCH_SIZE)
    return model, pd.DataFrame(history)


def two_level_cov(cov, rows, lineage: str, frm: str, to: str):
    """The same cells' design rows with the stage column set to `frm` and `to`.

    Every other block is left as it is, so the pair differs in two columns and
    nothing else. Returns None when either column is absent, which happens when
    that lineage has no cells in that stage.
    """
    a, b = mm.stage_col(lineage, frm), mm.stage_col(lineage, to)
    if a not in cov.columns or b not in cov.columns:
        return None
    stage_cols = [c for c in cov.columns if c.startswith(f"{mm.STAGE_BLOCK}_")]
    base = cov.loc[rows].copy()
    base[stage_cols] = 0.0
    cov_a, cov_b = base.copy(), base.copy()
    cov_a[a], cov_b[b] = 1.0, 1.0
    return cov_a, cov_b


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
    """1-indexed rank of one label, descending. None when it is absent."""
    if name not in series.index:
        return None
    return int(series.sort_values(ascending=False).index.get_loc(name)) + 1


def forward_pass_checks(model, x, cov_a, cov_b, genes, *, n=512):
    """The decoder's own answer, two ways, for the identities below.

    Returns `(fixed, total)`: the mean log-space shift when `mu` is held fixed
    and decoded under both covariates -- which is exactly what
    `predict_counterfactual` does -- and when the cells are re-encoded under the
    target covariate. Softmax contributes one additive per-cell constant, so
    callers centre before comparing.

    Computed over the same cells the decomposition used, or a comparison fails
    on the cell set rather than on the arithmetic.
    """
    dev = mm.device()
    model = model.to(dev).eval()
    xv = x.values if hasattr(x, "values") else np.asarray(x)
    fixed = np.zeros(len(genes))
    total = np.zeros(len(genes))
    with torch.no_grad():
        for s in range(0, len(xv), n):
            xb = torch.as_tensor(xv[s:s + n], dtype=torch.float32, device=dev)
            ab = torch.as_tensor(cov_a.values[s:s + n], dtype=torch.float32, device=dev)
            bb = torch.as_tensor(cov_b.values[s:s + n], dtype=torch.float32, device=dev)
            mu_a = model.encode(xb, ab)[0]
            mu_b = model.encode(xb, bb)[0]

            def lg(px):
                return torch.log(px.clamp_min(1e-30)).double()

            fixed += (lg(model.decoder(mu_a, bb))
                      - lg(model.decoder(mu_a, ab))).sum(0).cpu().numpy()
            total += (lg(model.decoder(mu_b, bb))
                      - lg(model.decoder(mu_a, ab))).sum(0).cpu().numpy()
    return fixed / len(xv), total / len(xv)


def centred_maxdev(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs((a - a.mean()) - (b - b.mean())).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=mm.ANALYSIS_STUDIES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--max-cells", type=int, default=20_000,
                    help="baseline cells per lineage; the pre-activation shift "
                         "is cell-independent and the post-tanh one varies only "
                         "through saturation, so a seeded subsample suffices")
    args = ap.parse_args()

    out_dir = mm.OUT_DIR / "architecture" / args.study
    if args.seed != 42:
        out_dir = out_dir / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    adata, adj_df, gene_names, pathway_names, cov, x, data = prepare(
        args.study, seed=args.seed)
    hall = mm.read_gmt(mm.HALLMARK_GMT, gene_names)
    perm_adj = permute_adjacency(adj_df, seed=args.seed)
    stages = [s for s in mm.STAGE_ORDER
              if s in set(adata.obs["cohort"].astype(str))]
    frm, to = stages[0], stages[-1]
    rng = np.random.default_rng(args.seed)
    print(f"{args.study}: {adata.n_obs:,} cells x {len(gene_names):,} genes, "
          f"{len(pathway_names):,} units, {cov.shape[1]} covariate columns")
    print(f"contrast {frm}->{to}; target {mm.short(TARGET)}; "
          f"present in the mask: {TARGET in pathway_names}\n")

    # The baseline cells, fixed across arms so a difference is the arm: the
    # healthy cells of each lineage, asked what the model does to them when it
    # is told they came from a later stage.
    base = {}
    for lineage in mm.MAJOR_SUBSETS:
        rows = adata.obs.index[(adata.obs["subset"].astype(str) == lineage)
                               & (adata.obs["cohort"].astype(str) == frm)]
        if len(rows) < 50:
            print(f"{lineage}: only {len(rows)} healthy cells -- skipped")
            continue
        if len(rows) > args.max_cells:
            rows = pd.Index(rng.choice(rows, args.max_cells, replace=False))
        base[lineage] = rows

    rows_out, unit_rows = [], []
    summary = {"study": args.study, "seed": args.seed, "epochs": args.epochs,
               "cells": int(adata.n_obs), "units": len(pathway_names),
               "genes": len(gene_names), "programmes": hall.shape[1],
               "contrast": f"{frm}->{to}", "target": TARGET,
               "baseline_cells": {k: int(len(v)) for k, v in base.items()},
               "arms": {}}

    for arm in args.arms:
        spec = ARMS[arm]
        flags = {k: v for k, v in spec.items() if k != "permute"}
        t0 = time.time()
        model, hist = fit(perm_adj if spec["permute"] else adj_df, cov, data,
                          flags=flags, seed=args.seed, epochs=args.epochs)
        best = float(hist["val"].min())
        mins = (time.time() - t0) / 60
        sign = unit_sign_structure(model)
        summary["arms"][arm] = {"best_val": round(best, 2),
                                "train_min": round(mins, 1), **flags,
                                "permute": spec["permute"],
                                **{k: round(v, 4) if isinstance(v, float) else v
                                   for k, v in sign.items()},
                                "lineages": {}}
        print(f"{arm:<16s} val {best:8.1f}  ({mins:4.1f} min)   "
              f"dec_out negative {sign['dec_out_frac_negative']:.3f}")

        for lineage, rows in base.items():
            pair = two_level_cov(cov, rows, lineage, frm, to)
            if pair is None:
                continue
            cov_a, cov_b = pair
            res = unit_contrast(model, x.loc[rows], cov_a, cov_b,
                                pathway_names=pathway_names,
                                gene_names=gene_names)
            s = res["summary"]
            if s["delta_pre_sd_max"] > PRE_SD_TOL:
                raise AssertionError(
                    f"{arm} {lineage}: the pre-activation shift varies between "
                    f"cells by {s['delta_pre_sd_max']:.2e}, so this was not a "
                    "clean two-level covariate contrast")

            # The unit-level read-out: where the target sits among all units.
            units = res["units"]
            unit_rank = rank_of(units["member_shift_z"], TARGET)
            parent_rank = rank_of(units["member_shift_z"], TARGET_PARENT)

            # The gene-level baseline, on the route that arm actually uses: the
            # full conditional shift, which is the dense term in the shipped arm
            # and the masked one everywhere else.
            de = pd.DataFrame(
                {"lfc_mean": res["genes"]["total"].to_numpy() / np.log(2)},
                index=list(gene_names))
            pa = pathway_activity(de, hall, statistic="lfc_mean", min_genes=10)
            ifn = [p for p in pa.index if "INTERFERON" in p.upper()]
            gene_ranks = sorted(rank_of(pa["z_competitive"], p) for p in ifn)

            # The direct read-out, where the architecture provides one.
            weight_rank = direct_dev = None
            if flags.get("cov_mode") == "pathway" and not flags.get(
                    "cov_in_encoder", True):
                direct = pd.Series(
                    model.pathway_contrast(cov_a.to_numpy(), cov_b.to_numpy()
                                           ).cpu().numpy(), index=pathway_names)
                # Signed by the unit's effect on its own genes, which is the
                # identity only when nonneg_decoder is on.
                oriented = direct * np.sign(units["member_mean_shift"]
                                            .where(lambda v: v != 0, 1.0))
                weight_rank = rank_of(oriented, TARGET)
                direct_dev = float(np.abs(
                    direct.to_numpy() - units["delta_pre"].to_numpy()).max())

            # Three identities, each saying something different, and each
            # applying only where it is meaningful.
            fixed, total = forward_pass_checks(model, x.loc[rows], cov_a, cov_b,
                                               gene_names)
            checks = {
                # Everywhere: the split must reproduce what the decoder does.
                "decomposition_vs_reencoded_maxdev": centred_maxdev(
                    total, res["genes"]["total"]),
                "counterfactual_vs_bypass_maxdev": None,
                "counterfactual_vs_reencoded_maxdev": None,
            }
            if flags.get("cov_mode", "gene") == "gene":
                # The claim about the shipped read-out: holding `mu` fixed
                # cancels the masked layer, so `predict_counterfactual`'s shift
                # IS the dense term and the annotation is absent from it.
                checks["counterfactual_vs_bypass_maxdev"] = centred_maxdev(
                    fixed, res["genes"]["bypass"])
            elif not flags.get("cov_in_encoder", True):
                # And what the fix buys: with `mu` independent of the covariate,
                # re-encoding changes nothing, so the counterfactual convention
                # stops mattering -- the two routes have become one.
                checks["counterfactual_vs_reencoded_maxdev"] = centred_maxdev(
                    fixed, total)
            row = {
                "arm": arm, "lineage": lineage, "permute": spec["permute"],
                "cov_mode": flags.get("cov_mode", "gene"),
                "cov_in_encoder": flags.get("cov_in_encoder", True),
                "nonneg_decoder": flags.get("nonneg_decoder", False),
                "best_val": round(best, 2),
                "unit_rank_target": unit_rank,
                "unit_rank_parent": parent_rank,
                "gene_rank_interferon_best": gene_ranks[0] if gene_ranks else None,
                "top_unit": units["member_shift_z"].idxmax(),
                "top_programme": pa["z_competitive"].idxmax(),
                "weight_rank_target": weight_rank,
                "masked_share": round(s["masked_share"], 4),
                "cos_masked_bypass": round(s["cos_masked_bypass"], 4)
                if np.isfinite(s["cos_masked_bypass"]) else None,
                "dec_out_frac_negative": round(
                    sign["dec_out_frac_negative"], 4),
                "sign_agree_unit_vs_genes": sign_agreement(units),
                "units_without_members": int((units["n_genes"] == 0).sum()),
                "direct_vs_forward_maxdev": direct_dev,
                **checks,
            }
            rows_out.append(row)
            summary["arms"][arm]["lineages"][lineage] = {
                k: v for k, v in row.items()
                if k not in ("arm", "lineage", "permute")}

            u = units.copy()
            u.insert(0, "lineage", lineage)
            u.insert(0, "arm", arm)
            u.index.name = "unit"
            unit_rows.append(u.reset_index())

            print(f"    {lineage}  unit rank {str(unit_rank):>5} "
                  f"(parent {str(parent_rank):>5})   gene rank "
                  f"{str(gene_ranks[0] if gene_ranks else None):>3}   "
                  f"masked_share {s['masked_share']:.3f}   "
                  f"sign agree {row['sign_agree_unit_vs_genes']:.3f}")

        del model
        torch.cuda.empty_cache()

    arms = pd.DataFrame(rows_out)
    arms.to_csv(out_dir / "arms.csv", index=False)
    pd.concat(unit_rows, ignore_index=True).to_csv(out_dir / "units.csv",
                                                   index=False)

    # Real against permuted, per scheme: what the annotation is worth.
    summary["annotation_worth"] = {}
    for scheme in SCHEMES:
        a, b = f"real_{scheme}", f"perm_{scheme}"
        if a in summary["arms"] and b in summary["arms"]:
            summary["annotation_worth"][scheme] = round(
                summary["arms"][b]["best_val"] - summary["arms"][a]["best_val"], 2)

    print(f"\n=== {args.study}, seed {args.seed}: {frm}->{to} ===")
    print(f"\n{'scheme':<10s} {'fit':>9s} {'vs shipped':>11s} "
          f"{'annot worth':>12s}  unit rank (real / perm)")
    base_val = summary["arms"].get("real_shipped", {}).get("best_val")
    for scheme in SCHEMES:
        a, b = f"real_{scheme}", f"perm_{scheme}"
        if a not in summary["arms"]:
            continue
        v = summary["arms"][a]["best_val"]
        cost = f"{v - base_val:+.2f}" if base_val is not None else "-"
        worth = summary["annotation_worth"].get(scheme)
        rr = arms[arms["arm"] == a]["unit_rank_target"].tolist()
        pr = arms[arms["arm"] == b]["unit_rank_target"].tolist()
        print(f"{scheme:<10s} {v:9.1f} {cost:>11s} "
              f"{(f'{worth:+.2f}' if worth is not None else '-'):>12s}  "
              f"{rr} / {pr}")

    # Each identity is checked only on the rows where it applies, and the worst
    # deviation is reported so a pass is a measurement rather than a boolean.
    IDENTITIES = {
        "decomposition_vs_reencoded_maxdev":
            "masked + bypass == the re-encoded shift",
        "counterfactual_vs_bypass_maxdev":
            "predict_counterfactual == the dense term (gene mode: the mask is "
            "absent from it)",
        "counterfactual_vs_reencoded_maxdev":
            "predict_counterfactual == the re-encoded shift (pathway mode with "
            "the encoder route closed: the two routes have become one)",
        "direct_vs_forward_maxdev":
            "pathway_contrast == the forward pass",
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
