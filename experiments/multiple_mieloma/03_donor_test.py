#!/usr/bin/env python
"""A donor-level test of the unit read-out, for one study.

    python 03_donor_test.py --study Zavidij_2020

**Why this exists.** `02_biology.ipynb` reports one signed number per unit per
stage contrast. That number is estimated from cells, and every donor sits at
exactly one stage, so it contains no donor-level variance at all: a large
contrast there is not evidence that the effect would appear in new patients.
This script supplies the missing sampling unit -- one score per **donor**, then
a test across donors.

It is the only place in this experiment where a p-value appears. The notebook
reports ranks and signed contrasts and no test, because a test over 1,615
strictly nested Reactome units is not a test of anything. Here the unit of
sampling is the donor and the multiplicity correction is over units, which is
conservative when they nest -- a parent and its child are near-duplicate
hypotheses and BH treats them as two. That makes a surviving q trustworthy and
a non-surviving one uninformative about any single unit.

**The test runs on the model's own estimator, not on a substitute.** That
distinction decides whether this validates the experiment or quietly replaces
it. A model-free pseudobulk test with the right sampling unit would answer the
donor question and leave "why the VAE at all?" unanswered. So the primary test
refits the *same architecture* on the *same cells* with one change:

    stage block:  subset__cohort   ->   subset__donor

Under pathway-space conditioning that makes each donor's score **immediate**:
it is `cov_path.weight[:, donor] - mean(healthy donors')`, 1,615 signed numbers
read straight off the weights. No cells are re-encoded, nothing is sampled, and
`pathway_activity` -- the competitive gene-level test the previous version of
this script leaned on -- does not appear at all. `pheno_h` stays in the design,
so composition is still conditioned out at the cell level, which is the thing
pseudobulk can only approximate.

A model-free pseudobulk score over the same 1,615 units is computed alongside
and reported as an independent cross-check. Agreement between two estimators
that share only the cells is worth more than either alone; disagreement
localises the problem.

**The test is a trend across all four stages**, not `Non` vs `MM`. That is
forced by the data: `Foster_2026` has one MGUS donor and 3 `Non` against 5 `MM`,
where the smallest attainable Mann-Whitney p is 0.036 and no unit can reach
q < 0.05 after correction. A Spearman trend uses every donor. The endpoint test
is reported beside it with its exact n, and `power_floor` records the smallest q
each design could ever produce, so a null result can be told apart from an
impossible one.

Significance comes from permuting **donor** stage labels. That null keeps each
donor's cells, phenotype mixture and sequencing depth intact and asks only
whether the stage labels carry the trend.

Writes `outputs/donor_test/<study>/`:

    donor_scores.csv   one row per donor x lineage x unit x estimator
    donor_test.csv     one row per unit x lineage x estimator
    summary.json       counts, the power floor, and what survives
    model.pt, model_fingerprint.json, training_history.csv
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import train_test_split

import myeloma as mm
from pyvae import InformedVAE, set_all_seeds, train_ivae_modern

#: A donor x lineage needs this many cells before it is scored at all. Below it
#: the per-donor covariate column is estimated from too little to mean anything.
MIN_CELLS_PER_DONOR = 50
#: And this many within a phenotype level before that level enters the
#: pseudobulk mixture.
MIN_CELLS_PER_LEVEL = 10
#: A unit needs this many measured member genes before either estimator scores
#: it, matching the floor the notebook's read-out uses.
MIN_GENES = 10
#: Permutation draws for the trend test. This is a power parameter, not a
#: cosmetic one. The smallest attainable permutation p is `1/(N_PERM + 1)`, and
#: Benjamini-Hochberg multiplies the smallest p by the number of tests, so the
#: best q this design can ever produce is `N_UNITS / (N_PERM + 1)`. At 10,000
#: draws over 1,262 units that floor is **q = 0.126** -- the primary test could
#: not have returned a single significant unit whatever the data said, which is
#: exactly the null-versus-impossible confusion `power_floor` exists to prevent
#: and which it previously caught only for the endpoint test. 100,000 draws put
#: the floor at 0.013. The null is accumulated in blocks rather than stored, so
#: ten times the draws costs seconds and no extra memory.
N_PERM = 100_000
#: Draws per block. Memory is `n_units x PERM_BLOCK` floats.
PERM_BLOCK = 1_000
ALPHA = 0.05
#: A short run must not overwrite a real one. The workflow trains for this many
#: epochs; anything less is a smoke test and is written to its own directory.
#: Without the guard, `--epochs 3` silently replaces a finished run's tables,
#: Snakemake then sees the output present and skips the real job, and the next
#: person to read those files has no way to tell. `02_biology.ipynb` has had
#: this guard since it was written; this script did not.
CANONICAL_EPOCHS = 250


def bh(p):
    """Benjamini-Hochberg, NaN-safe, returned in the input order."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    v = p[ok]
    if not len(v):
        return out
    order = np.argsort(v)
    ranked = np.minimum.accumulate(
        (v[order] * len(v) / np.arange(1, len(v) + 1))[::-1])[::-1]
    q = np.empty_like(v)
    q[order] = np.minimum(ranked, 1.0)
    out[ok] = q
    return out


# --------------------------------------------------------------------------- #
# The model-side estimator: one covariate column per donor
# --------------------------------------------------------------------------- #
def fit_donor_model(adata, adj_df, *, epochs, seed, out_dir, force):
    """Refit with the stage block interacted with donor rather than cohort.

    Identification: every donor sits at exactly one stage, so a donor column is
    estimated from that donor's own cells and is a deviation from the shared
    intercept, not from their stage. The stage effect is recovered afterwards as
    a contrast between donor columns, which is what gives it donor-level spread.

    The architecture is `mm.ARCH`, unchanged -- that is the whole point. A
    donor-level test of the notebook's read-out is only a test of *that*
    read-out if it fits the same model.
    """
    adata.obs["donor_block"] = (adata.obs["subset"].astype(str) + "__"
                                + adata.obs["donor_id"].astype(str))
    blocks = ["donor_block", "pheno_h"]
    cov = pd.get_dummies(adata.obs[blocks].astype(str), dtype=np.float32)
    assert cov.sum(axis=1).min() == len(blocks), "a block still has a missing value"
    n_cov = cov.shape[1]

    x_trans = adata.to_df().astype(np.float32)
    counts_df = adata.to_df(layer="counts").astype(np.float32)
    strat = adata.obs["cohort"].astype(str) + "_" + adata.obs["pheno_h"].astype(str)
    if strat.value_counts().min() < 2:
        strat = adata.obs["cohort"].astype(str)
    idx_tr, idx_va = train_test_split(np.arange(len(adata)), test_size=0.2,
                                      stratify=strat, random_state=seed)
    if len(idx_tr) % mm.BATCH_SIZE == 1:
        idx_va = np.append(idx_va, idx_tr[-1])
        idx_tr = idx_tr[:-1]
    data = {"x_train": x_trans.iloc[idx_tr], "x_val": x_trans.iloc[idx_va],
            "counts_train": counts_df.iloc[idx_tr],
            "counts_val": counts_df.iloc[idx_va],
            "cov_train": cov.iloc[idx_tr], "cov_val": cov.iloc[idx_va]}

    set_all_seeds(seed)
    model = InformedVAE(adj=torch.tensor(adj_df.values, dtype=torch.float32),
                        latent_dim=mm.latent_dim_for(adj_df.shape[1]), seed=seed,
                        l2_lambda=mm.L2_LAMBDA, likelihood="nb", n_cov=n_cov,
                        **mm.ARCH)
    ckpt, finger = out_dir / "model.pt", out_dir / "model_fingerprint.json"
    # The architecture is part of the fingerprint, so a checkpoint from the
    # shipped decoder can never be adopted here and read as though it came from
    # this one.
    design = {"study": str(adata.obs["study_id"].iloc[0]), "epochs": int(epochs),
              "seed": int(seed), "blocks": blocks, "n_cov": int(n_cov),
              "architecture": {k: str(v) for k, v in sorted(mm.ARCH.items())},
              "cells": int(adata.n_obs), "train_cells": int(len(idx_tr)),
              "stage_block": "donor"}
    if (not force and ckpt.exists() and finger.exists()
            and json.loads(finger.read_text()) == design):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model = model.to(mm.device())
        print(f"reused the saved donor-level fit in {out_dir}")
    else:
        t0 = time.time()
        model, history = train_ivae_modern(
            model, x_train=data["x_train"], x_val=data["x_val"],
            x_counts_train=data["counts_train"],
            x_counts_val=data["counts_val"], cov_train=data["cov_train"],
            cov_val=data["cov_val"], device=mm.device(), epochs=int(epochs),
            patience=int(epochs), batch_size=mm.BATCH_SIZE)
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv",
                                     index=False)
        torch.save(model.state_dict(), ckpt)
        finger.write_text(json.dumps(design, indent=2, sort_keys=True))
        print(f"trained the donor-level fit in {(time.time() - t0) / 60:.1f} min")
    return model, list(cov.columns)


def model_donor_scores(model, cov_columns, adata, pathway_names, usable,
                       lineage, stages):
    """Per-donor signed shift for every unit, from the donor-level fit.

    Each donor's score is their own `cov_path` column minus the **mean of the
    healthy donors' columns** in the same lineage. Healthy donors therefore
    scatter around zero by construction, and that scatter is the donor-level
    noise the notebook's read-out cannot show.

    Under `cov_mode="pathway"` this is the entire conditional effect: there is
    no gene-level covariate term and no second route through the encoder, so
    there is nothing to score, sample or re-encode. It is a column difference.
    """
    W = model.decoder.cov_path.weight.detach().cpu().numpy().astype(np.float64)
    index = {c: i for i, c in enumerate(cov_columns)}
    obs = adata.obs
    sel = obs[obs["subset"].astype(str) == lineage]
    sizes = sel.groupby("donor_id", observed=True).size()
    donors = [d for d, n in sizes.items()
              if n >= MIN_CELLS_PER_DONOR
              and f"donor_block_{lineage}__{d}" in index]
    if not donors:
        return None, None
    stage_of = {d: str(sel.loc[sel["donor_id"] == d, "cohort"].iloc[0])
                for d in donors}
    baseline = [d for d in donors if stage_of[d] == stages[0]]
    if len(baseline) < 2:
        return None, None
    ref = np.mean([W[:, index[f"donor_block_{lineage}__{d}"]] for d in baseline],
                  axis=0)

    rows = {d: W[:, index[f"donor_block_{lineage}__{d}"]] - ref for d in donors}
    scores = pd.DataFrame(rows, index=pathway_names).T[usable]
    return scores, pd.Series(stage_of).reindex(scores.index)


# --------------------------------------------------------------------------- #
# The model-free estimator: phenotype-standardised pseudobulk
# --------------------------------------------------------------------------- #
def pseudobulk_donor_scores(adata, gene_names, adj_df, usable, lineage):
    """Per-donor unit score from pseudobulk, composition held fixed.

    A donor's pseudobulk is a mixture of their phenotypes, so a donor with more
    effector cells looks activated without any unit changing. Each donor's
    profile is therefore built per `pheno_h` level and averaged with **the same
    weights for every donor** -- the study's own phenotype distribution in that
    lineage, renormalised over the levels the donor has. That is the pseudobulk
    analogue of conditioning on phenotype.

    Scored over the same Reactome units as the model side, so the two estimators
    differ in how the score is formed and in nothing else.
    """
    sub = adata[(adata.obs["subset"].astype(str) == lineage).to_numpy()]
    counts, obs = sub.to_df(layer="counts"), sub.obs
    global_w = obs["pheno_h"].astype(str).value_counts(normalize=True)

    profiles, stage_of = {}, {}
    for donor, idx in obs.groupby("donor_id", observed=True).groups.items():
        cells = counts.loc[idx]
        if len(cells) < MIN_CELLS_PER_DONOR:
            continue
        levels = obs.loc[idx, "pheno_h"].astype(str)
        parts, weights = [], []
        for level, rows in levels.groupby(levels, observed=True).groups.items():
            if len(rows) < MIN_CELLS_PER_LEVEL or level not in global_w:
                continue
            s = cells.loc[rows].sum(axis=0)
            parts.append(np.log1p(s / max(s.sum(), 1) * 1e4))
            weights.append(float(global_w[level]))
        if not parts:
            continue
        w = np.asarray(weights) / np.sum(weights)
        profiles[donor] = np.average(np.vstack(parts), axis=0, weights=w)
        stage_of[donor] = str(obs.loc[idx, "cohort"].iloc[0])
    if len(profiles) < 6:
        return None, None

    frame = pd.DataFrame(profiles).T
    frame.columns = gene_names
    # z across donors, so a score says "high relative to the other donors",
    # then competitive centring against the whole panel.
    z = (frame - frame.mean(axis=0)) / frame.std(axis=0).replace(0, np.nan)
    z = z.loc[:, z.notna().all(axis=0)]
    member = adj_df.reindex(z.columns).fillna(0.0).to_numpy() > 0
    values, baseline = z.to_numpy(), z.to_numpy().mean(axis=1)
    out = {p: values[:, member[:, j]].mean(axis=1) - baseline
           for j, p in enumerate(adj_df.columns)
           if p in usable and member[:, j].sum() >= MIN_GENES}
    if not out:
        return None, None
    scores = pd.DataFrame(out, index=z.index)
    return scores, pd.Series(stage_of).reindex(scores.index)


# --------------------------------------------------------------------------- #
# Tests across donors
# --------------------------------------------------------------------------- #
def trend_test(scores, stage_index, rng):
    """Spearman against stage order, with a donor-label permutation p.

    Ranks are computed once and only the stage vector is permuted, which is
    exactly equivalent to re-ranking each time and makes 100,000 draws cheap.

    The null is accumulated in blocks of `PERM_BLOCK` and never materialised in
    full: one `(n_donors, PERM_BLOCK)` matrix of permuted stage vectors, one
    matrix multiply, one comparison, one running count. Storing all 100,000
    draws for 1,262 units would be a gigabyte for no reason.
    """
    y = np.asarray(stage_index, dtype=float)
    values = scores.to_numpy()
    rx = np.apply_along_axis(stats.rankdata, 0, values)
    # A unit that is constant across donors has no rank variation and would
    # divide by zero. It cannot carry a trend, so it is scored as exactly no
    # trend rather than dropped, which keeps every output frame the same width.
    sd = rx.std(axis=0)
    rx = (rx - rx.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    ry = stats.rankdata(y)
    ry = (ry - ry.mean()) / ry.std()
    n = len(ry)
    observed = rx.T @ ry / n
    target = np.abs(observed)[:, None]

    at_least_as_extreme = np.zeros(values.shape[1], dtype=np.int64)
    drawn = 0
    while drawn < N_PERM:
        block = min(PERM_BLOCK, N_PERM - drawn)
        perms = np.empty((n, block))
        for j in range(block):
            perms[:, j] = rng.permutation(ry)
        null = rx.T @ perms / n                      # (n_units, block)
        at_least_as_extreme += (np.abs(null) >= target).sum(axis=1)
        drawn += block
    p = (at_least_as_extreme + 1) / (N_PERM + 1)
    return (pd.Series(observed, index=scores.columns),
            pd.Series(p, index=scores.columns))


def endpoint_test(scores, stage, first, last):
    """Mann-Whitney between the first and last stage, over donors."""
    a = (stage.to_numpy() == first)
    b = (stage.to_numpy() == last)
    if a.sum() < 2 or b.sum() < 2:
        nan = pd.Series(np.nan, index=scores.columns)
        return nan, nan, int(a.sum()), int(b.sum())
    # One apply per field: `Series.apply` returning a namedtuple is unpacked
    # into a DataFrame, so `.statistic` on the result is not available.
    stat = scores.apply(
        lambda c: stats.mannwhitneyu(c[b], c[a],
                                     alternative="two-sided").statistic)
    pval = scores.apply(
        lambda c: stats.mannwhitneyu(c[b], c[a],
                                     alternative="two-sided").pvalue)
    return (stat / (a.sum() * b.sum()), pval, int(a.sum()), int(b.sum()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", required=True, choices=mm.ANALYSIS_STUDIES)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-retrain", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    suffix = "" if args.epochs >= CANONICAL_EPOCHS else f"__epochs{args.epochs}"
    out_dir = mm.OUT_DIR / "donor_test" / f"{args.study}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if suffix:
        print(f"!! {args.epochs} epochs is below the canonical "
              f"{CANONICAL_EPOCHS}: writing to {out_dir} and leaving the real "
              "run alone\n")

    mm.require_inputs()
    adata = mm.load_tcells(args.study, analysed_only=True)
    adata, adj_df, gene_names, pathway_names = mm.build_pathways(adata)
    # The units both estimators can score: enough measured members for the
    # pseudobulk mean to mean anything, which is the same floor the notebook's
    # read-out applies.
    usable = [p for p in pathway_names
              if int((adj_df[p].to_numpy() > 0).sum()) >= MIN_GENES]
    stages = [s for s in mm.STAGE_ORDER
              if s in set(adata.obs["cohort"].astype(str))]
    rank_of = {s: i for i, s in enumerate(stages)}
    print(f"study              {args.study}")
    print(f"cells              {adata.n_obs:,}")
    print(f"stages             {stages}")
    print(f"units              {len(usable):,} of {len(pathway_names):,} with "
          f">= {MIN_GENES} measured members")
    print(f"architecture       {mm.ARCH}\n")

    model, cov_columns = fit_donor_model(
        adata, adj_df, epochs=args.epochs, seed=args.seed, out_dir=out_dir,
        force=args.force_retrain)
    print(f"donor-level design {len(cov_columns)} columns\n")

    score_rows, test_rows, counts, floor = [], [], {}, {}
    for lineage in mm.MAJOR_SUBSETS:
        estimators = {
            "model": model_donor_scores(model, cov_columns, adata,
                                        pathway_names, usable, lineage, stages),
            "pseudobulk": pseudobulk_donor_scores(adata, gene_names, adj_df,
                                                  set(usable), lineage),
        }
        for name, (scores, stage) in estimators.items():
            if scores is None or len(scores) < 6:
                print(f"{lineage} {name}: too few usable donors -- skipped")
                continue
            per_stage = stage.value_counts().reindex(stages).fillna(0).astype(int)
            counts[lineage] = {s: int(per_stage[s]) for s in stages}
            rho, p_trend = trend_test(scores, stage.map(rank_of).to_numpy(), rng)
            auc, p_end, n_a, n_b = endpoint_test(scores, stage, stages[0],
                                                 stages[-1])
            test_rows.append(pd.DataFrame({
                "unit": scores.columns, "lineage": lineage,
                "estimator": name, "n_donors": len(scores),
                "rho_trend": rho.to_numpy(), "p_trend": p_trend.to_numpy(),
                "q_trend": bh(p_trend.to_numpy()),
                "auc_endpoint": auc.reindex(scores.columns).to_numpy(),
                "p_endpoint": p_end.reindex(scores.columns).to_numpy(),
                "q_endpoint": bh(p_end.reindex(scores.columns).to_numpy()),
                f"n_{stages[0]}": n_a, f"n_{stages[-1]}": n_b}))
            long = scores.stack().rename("score").reset_index()
            long.columns = ["donor_id", "unit", "score"]
            long["lineage"], long["estimator"] = lineage, name
            long["stage"] = long["donor_id"].map(stage)
            score_rows.append(long)
            print(f"{lineage} {name:11s} {len(scores)} donors  {dict(per_stage)}")

            if name == "model":
                # The smallest q this design could ever produce. A null result
                # and an impossible one look identical without it.
                best_end = (2.0 / math.comb(n_a + n_b, min(n_a, n_b))
                            if min(n_a, n_b) >= 2 else float("nan"))
                # BH multiplies the smallest p by the number of tests, so
                # the best q either test can produce is that product. Both are
                # recorded: a test whose floor is above ALPHA returned an
                # impossible result, not a negative one.
                n_tested = int(scores.shape[1])
                best_trend = 1.0 / (N_PERM + 1)
                floor[lineage] = {
                    "donors": int(len(scores)),
                    "units_tested": n_tested,
                    "smallest_p_trend": round(best_trend, 7),
                    "best_attainable_q_trend": round(best_trend * n_tested, 4),
                    "trend_can_reach_q05": bool(best_trend * n_tested < ALPHA),
                    "smallest_p_endpoint": (None if math.isnan(best_end)
                                            else round(best_end, 5)),
                    "best_attainable_q_endpoint": (
                        None if math.isnan(best_end)
                        else round(best_end * n_tested, 4)),
                    "endpoint_can_reach_q05": bool(
                        not math.isnan(best_end)
                        and best_end * n_tested < ALPHA),
                }

    tests = pd.concat(test_rows, ignore_index=True)
    tests.to_csv(out_dir / "donor_test.csv", index=False)
    pd.concat(score_rows, ignore_index=True).to_csv(
        out_dir / "donor_scores.csv", index=False)

    def rising(estimator):
        t = tests[(tests["estimator"] == estimator) & (tests["q_trend"] < ALPHA)
                  & (tests["rho_trend"] > 0)]
        n = t.groupby("unit")["lineage"].nunique()
        return sorted(n[n == len(counts)].index)

    both_model, both_pb = rising("model"), rising("pseudobulk")
    print(f"\nrising donor-level trend, q < {ALPHA}, BOTH lineages")
    for label, got in (("model", both_model), ("pseudobulk", both_pb)):
        print(f"  {label:11s} {len(got)}: {[mm.short(p) for p in got][:8]}")
    agree = sorted(set(both_model) & set(both_pb))
    print(f"  both estimators agree on {len(agree)}: "
          f"{[mm.short(p) for p in agree][:8]}")
    print("\nReactome nests, so these lists are families rather than "
          "independent hits:\n  read the names, not the count.")
    blocked = [l for l, f in floor.items() if not f["trend_can_reach_q05"]]
    if blocked:
        print(f"\n!! the trend test CANNOT reach q < {ALPHA} in {blocked}: "
              "its best attainable q is\n   "
              + ", ".join(f"{l} {floor[l]['best_attainable_q_trend']:.3f}"
                          for l in blocked)
              + f".\n   An empty result above is an impossible one, not a "
                "negative one. Raise N_PERM.")

    summary = {
        "study": args.study, "stages": stages,
        "units_tested": int(len(usable)),
        "mask_units": int(len(pathway_names)),
        "epochs": int(args.epochs), "seed": int(args.seed),
        "architecture": {k: str(v) for k, v in sorted(mm.ARCH.items())},
        "min_cells_per_donor": MIN_CELLS_PER_DONOR,
        "min_cells_per_level": MIN_CELLS_PER_LEVEL,
        "min_genes_per_unit": MIN_GENES,
        "permutations": N_PERM, "alpha": ALPHA,
        "donor_design_columns": len(cov_columns),
        "donors_per_stage": counts, "power_floor": floor,
        "trend_both_lineages_model": [mm.short(p) for p in both_model],
        "trend_both_lineages_pseudobulk": [mm.short(p) for p in both_pb],
        "trend_both_lineages_both_estimators": [mm.short(p) for p in agree],
        "nesting_note": ("BH is over strictly nested Reactome units, which is "
                         "conservative: a parent and its child are near-"
                         "duplicate hypotheses and are corrected as two. No "
                         "count over units is a result."),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwritten to {out_dir}/")


if __name__ == "__main__":
    main()
