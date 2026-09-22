#!/usr/bin/env python
"""03_sample_size_experiment.py — does the model beat the naive baseline?

Sweeps sample size and asks: at what point does the model-based pathway
ranking stop beating a two-line log-ratio-of-group-means baseline?

Motivation
----------
``pathway_activity`` never sees the model — it works from a DE table plus an
adjacency. So the naive "compute per-gene mean(stim) - mean(ctrl), feed to
pathway_activity" pipeline is model-free. At full sample size the model has
to be shown to beat this before any claim about the architecture holds. If
the model can't beat naive even at 20,000 cells, the whole story is
unsupported. And if it beats naive at 20,000 but loses at 2,000, we know
where the architecture actually helps.

Two modes
---------
--mode frozen
    Reuse a trained checkpoint. Evaluate both rankings on progressively
    smaller VAL subsets. Cheap (~15 minutes total). BIASED TOWARD THE MODEL
    because the model was trained on the full dataset. So: a model win here
    is *inconclusive*; a model loss is *strong* — if the model can't beat
    naive even given the training advantage, the argument's dead.

--mode retrain
    For each sample size, retrain the model from scratch on a subsampled
    training set. Val stays fixed (a common evaluation basis). Fair
    comparison. Expensive — one training run per sample size, so budget
    accordingly.

Metric
------
External criterion (Carlos's spec): where do the canonical interferon
pathways rank? Reported for both methods at each sample size:

    interferon_ab_rank         REACTOME_INTERFERON_ALPHA_BETA_SIGNALING
                               (the one Section 5 got right in the notebook)
    interferon_parent_rank     REACTOME_INTERFERON_SIGNALING
                               (parent of the above)
    interferon_induction_rank  REACTOME_DDX58_IFIH1_MEDIATED_INDUCTION_
                               OF_INTERFERON_ALPHA_BETA
                               (the one that ranked 1,614/1,615 in the notebook
                                — should improve if the fix works)
    top_20_hit_rate            fraction of the top-20 that are interferon-related

Lower rank number = better (rank 1 = top of the list). "Top-20 hit rate"
counts pathways whose name contains INTERFERON or ISG.

Outputs
-------
experiments/kang/outputs/sample_size_experiment/<timestamp>_<smoke|full>/
    config.json
    per_run/
        <sample_frac>_<method>/
            de.csv           per-gene DE table used
            pa.csv           per-pathway activity ranking
            metrics.json     the interferon-rank numbers
    summary.csv              one row per (sample_frac, method), all metrics

Usage
-----
Smoke  : pixi run python experiments/kang/03_sample_size_experiment.py --smoke
Full frozen (~15 min):
    pixi run python experiments/kang/03_sample_size_experiment.py \\
        --mode frozen --checkpoint-path experiments/kang/outputs/architecture_ablation/\\
20260921_162559_full/baseline/checkpoint.pt

Full retrain (~15 hours):
    pixi run python experiments/kang/03_sample_size_experiment.py --mode retrain
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
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
    set_all_seeds,
    train_ivae_modern,
)


# ---------------------------------------------------------------------------
# The canonical interferon pathways used as the external criterion.
# ---------------------------------------------------------------------------

INTERFERON_AB          = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING"
INTERFERON_PARENT      = "REACTOME_INTERFERON_SIGNALING"
INTERFERON_INDUCTION   = "REACTOME_DDX58_IFIH1_MEDIATED_INDUCTION_OF_INTERFERON_ALPHA_BETA"

CANONICAL_PATHWAYS = [INTERFERON_AB, INTERFERON_PARENT, INTERFERON_INDUCTION]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample-size sweep: model vs naive on Kang.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--smoke", action="store_true",
                   help="Fast smoke: 500 cells, 5 epochs (retrain), small DE draws.")

    p.add_argument("--mode", choices=["frozen", "retrain"], default="frozen",
                   help="frozen: reuse checkpoint. retrain: train per sample size.")
    p.add_argument("--checkpoint-path", default=None,
                   help="Required for --mode frozen. Path to a trained checkpoint.pt.")

    # Data paths (same defaults as 02/04)
    p.add_argument("--data-folder", default="experiments/kang/data")
    p.add_argument("--resources-dir",
                   default="experiments/kang/resources/c2.cp.reactome.v7.5.1.symbols.gmt")
    p.add_argument("--output-root",
                   default="experiments/kang/outputs/sample_size_experiment")

    # Data prep
    p.add_argument("--n-genes", type=int, default=5000)
    p.add_argument("--target-sum", type=float, default=1e4)
    p.add_argument("--sample-fractions", nargs="+", type=float,
                   default=[1.0, 0.5, 0.25, 0.1, 0.05],
                   help="Fractions of the training set to use in retrain mode, "
                        "or of val to use in frozen mode.")

    # Model architecture (for retrain mode; ignored in frozen)
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--l2-lambda", type=float, default=1e-5)
    p.add_argument("--init", default="xavier", choices=["xavier", "fan_in"])
    p.add_argument("--normalize", default="none", choices=["none", "batch", "layer"])
    p.add_argument("--informed-decoder", action="store_true")
    p.add_argument("--nonneg-encoder", action="store_true")
    p.add_argument("--standardize-input", action="store_true")

    # Training (retrain mode)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--device", default="cpu")

    p.add_argument("--seed", type=int, default=42)

    # DE knobs
    p.add_argument("--n-de-samples", type=int, default=25)
    p.add_argument("--n-de-pairs", type=int, default=2000)

    args = p.parse_args()

    if args.smoke:
        args.sample_fractions = [1.0, 0.5, 0.25]
        args.epochs = min(args.epochs, 5)
        args.warmup_epochs = min(args.warmup_epochs, 2)
        args.patience = min(args.patience, 5)
        args.n_de_samples = min(args.n_de_samples, 3)
        args.n_de_pairs = min(args.n_de_pairs, 200)

    if args.mode == "frozen" and args.checkpoint_path is None:
        p.error("--mode frozen requires --checkpoint-path")

    return args


# ---------------------------------------------------------------------------
# Data loading — same setup as 02/04 (whole dataset; subsampling happens later)
# ---------------------------------------------------------------------------

def load_data(args, subsample_cells: int | None = None) -> dict:
    """Load full Kang + adjacency + covariates + 80/20 split. No subsampling here."""
    print(f"[data] load_kang(target_sum={args.target_sum}, n_genes={args.n_genes})")
    adata = load_kang(
        data_folder=args.data_folder,
        normalize=True,
        n_genes=args.n_genes,
        return_path=False,
        target_sum=args.target_sum,
    )

    if subsample_cells is not None and subsample_cells < adata.n_obs:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(adata.n_obs, size=subsample_cells, replace=False)
        adata = adata[idx].copy()
        print(f"[data] subsampled to {subsample_cells} cells")

    print(f"[data] adata: {adata.n_obs} cells x {adata.n_vars} genes")

    x_trans = adata.to_df()
    cfg = build_model_config(
        genes=x_trans.columns,
        model_kind="ivae_reactome",
        resources_dir=args.resources_dir,
    )
    adj_df = cfg.model_layer[0]
    gene_names = list(cfg.input_genes)
    pathway_names = list(adj_df.columns)
    print(f"[data] adj: {len(gene_names)} genes x {len(pathway_names)} pathways")

    x_trans = x_trans[gene_names]
    counts_df = adata.to_df(layer="counts")[gene_names]

    cov = pd.get_dummies(adata.obs[["condition", "cell_type"]], dtype=np.float32)
    cond_ctrl_col = "condition_control"
    cond_stim_col = "condition_stimulated"
    for col in (cond_ctrl_col, cond_stim_col):
        if col not in cov.columns:
            raise ValueError(f"Expected {col!r} in cov; got {list(cov.columns)}")

    strat_full = (
        adata.obs["condition"].astype(str) + "_" + adata.obs["cell_type"].astype(str)
    )
    if strat_full.value_counts().min() >= 2:
        strat = strat_full
    else:
        strat = adata.obs["condition"].astype(str)
    idx_tr, idx_va = train_test_split(
        np.arange(len(adata)), test_size=0.2, stratify=strat,
        random_state=args.seed,
    )
    print(f"[data] train={len(idx_tr)}  val={len(idx_va)}")

    return {
        "x_train":      x_trans.iloc[idx_tr].astype(np.float32),
        "x_val":        x_trans.iloc[idx_va].astype(np.float32),
        "counts_train": counts_df.iloc[idx_tr].astype(np.float32),
        "counts_val":   counts_df.iloc[idx_va].astype(np.float32),
        "cov_train":    cov.iloc[idx_tr],
        "cov_val":      cov.iloc[idx_va],
        "adj_df":        adj_df,
        "gene_names":    gene_names,
        "pathway_names": pathway_names,
        "cond_ctrl_col": cond_ctrl_col,
        "cond_stim_col": cond_stim_col,
        "n_cov":         cov.shape[1],
    }


# ---------------------------------------------------------------------------
# The two ranking methods
# ---------------------------------------------------------------------------

def rank_via_naive(x, adj_df, cov, cond_ctrl_col, cond_stim_col, gene_names):
    """Naive per-gene mean(stim) - mean(ctrl) in log1p space, then pathway_activity.

    Returns (de_df, pa_df).
    """
    ctrl_mask = cov[cond_ctrl_col].values > 0.5
    stim_mask = cov[cond_stim_col].values > 0.5
    if ctrl_mask.sum() == 0 or stim_mask.sum() == 0:
        return None, None

    x_vals = x.values if isinstance(x, pd.DataFrame) else np.asarray(x)
    lfc = x_vals[stim_mask].mean(axis=0) - x_vals[ctrl_mask].mean(axis=0)
    de_df = pd.DataFrame({
        "lfc_mean": lfc,
        "proba_de": np.abs(lfc),  # placeholder; pathway_activity uses lfc_mean
    }, index=gene_names)
    pa_df = pathway_activity(de_df, adj_df, statistic="lfc_mean")
    return de_df, pa_df


def rank_via_model(model, x, cov, adj_df, gene_names, cond_ctrl_col, cond_stim_col, args):
    """Model-based DE via differential_expression, then pathway_activity."""
    ctrl_mask = cov[cond_ctrl_col].values > 0.5
    stim_mask = cov[cond_stim_col].values > 0.5
    if ctrl_mask.sum() == 0 or stim_mask.sum() == 0:
        return None, None

    de_df = differential_expression(
        model,
        x_a=x.iloc[stim_mask] if isinstance(x, pd.DataFrame) else x[stim_mask],
        x_b=x.iloc[ctrl_mask] if isinstance(x, pd.DataFrame) else x[ctrl_mask],
        cov_a=cov.iloc[stim_mask],
        cov_b=cov.iloc[ctrl_mask],
        n_samples=args.n_de_samples,
        n_pairs=args.n_de_pairs,
        gene_names=gene_names,
        seed=args.seed,
    )
    pa_df = pathway_activity(de_df, adj_df, statistic="lfc_mean")
    return de_df, pa_df


def rank_of_pathway(pa_df: pd.DataFrame, pathway: str) -> float:
    """1-indexed rank of a pathway in the (already sorted) pa_df, or NaN if missing."""
    if pa_df is None or pathway not in pa_df.index:
        return float("nan")
    return float(list(pa_df.index).index(pathway) + 1)


def top_20_interferon_hit_rate(pa_df: pd.DataFrame) -> float:
    """Fraction of the top-20 pathways whose name contains 'INTERFERON' or 'ISG'."""
    if pa_df is None or len(pa_df) == 0:
        return float("nan")
    top20 = pa_df.head(20).index.tolist()
    if not top20:
        return float("nan")
    hits = sum(1 for p in top20 if ("INTERFERON" in p.upper()) or ("ISG" in p.upper()))
    return hits / len(top20)


def summarise_ranking(pa_df: pd.DataFrame) -> dict:
    return {
        "interferon_ab_rank":        rank_of_pathway(pa_df, INTERFERON_AB),
        "interferon_parent_rank":    rank_of_pathway(pa_df, INTERFERON_PARENT),
        "interferon_induction_rank": rank_of_pathway(pa_df, INTERFERON_INDUCTION),
        "top_20_hit_rate":           top_20_interferon_hit_rate(pa_df),
    }


# ---------------------------------------------------------------------------
# Model training (retrain mode only)
# ---------------------------------------------------------------------------

def build_model_from_args(data: dict, args) -> InformedVAE:
    n_pathways = data["adj_df"].shape[1]
    latent_dim = args.latent_dim if args.latent_dim is not None else n_pathways // 2
    adj_tensor = torch.tensor(data["adj_df"].values, dtype=torch.float32)
    return InformedVAE(
        adj=adj_tensor,
        latent_dim=latent_dim,
        seed=args.seed,
        l2_lambda=args.l2_lambda,
        likelihood="nb",
        n_cov=data["n_cov"],
        init=args.init,
        normalize=args.normalize,
        informed_decoder=args.informed_decoder,
        nonneg_encoder=args.nonneg_encoder,
        standardize_input=args.standardize_input,
    )


def train_from_args(model: InformedVAE, data: dict, args):
    return train_ivae_modern(
        model,
        x_train=data["x_train"],
        x_val=data["x_val"],
        x_counts_train=data["counts_train"],
        x_counts_val=data["counts_val"],
        cov_train=data["cov_train"],
        cov_val=data["cov_val"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
    )


# ---------------------------------------------------------------------------
# Frozen-mode pipeline
# ---------------------------------------------------------------------------

def run_frozen(args, out_root: Path) -> list[dict]:
    """Load full data + one checkpoint; subsample VAL at each fraction."""
    print(f"[frozen] loading checkpoint from {args.checkpoint_path}")
    data = load_data(args)  # full data
    model = build_model_from_args(data, args)
    state = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True)
    model.load_state_dict(state)
    model.to(args.device)

    results = []
    per_run_dir = out_root / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    for frac in args.sample_fractions:
        n_val = int(len(data["x_val"]) * frac)
        if n_val < 4:
            print(f"[frozen] frac={frac} would give n_val={n_val}, skipping (too small)")
            continue
        val_idx = np.sort(rng.choice(len(data["x_val"]), size=n_val, replace=False))

        x_v   = data["x_val"].iloc[val_idx]
        cov_v = data["cov_val"].iloc[val_idx]

        for method in ("naive", "model"):
            tag = f"{frac:g}_{method}"
            print(f"\n[frozen] frac={frac}  method={method}  n_val={n_val}")
            run_dir = per_run_dir / tag
            run_dir.mkdir(exist_ok=True)

            if method == "naive":
                de_df, pa_df = rank_via_naive(
                    x_v, data["adj_df"], cov_v,
                    data["cond_ctrl_col"], data["cond_stim_col"], data["gene_names"],
                )
            else:
                de_df, pa_df = rank_via_model(
                    model, x_v, cov_v, data["adj_df"], data["gene_names"],
                    data["cond_ctrl_col"], data["cond_stim_col"], args,
                )

            if pa_df is not None:
                de_df.to_csv(run_dir / "de.csv")
                pa_df.to_csv(run_dir / "pa.csv")

            metrics = summarise_ranking(pa_df)
            metrics.update({"sample_fraction": frac, "method": method, "n_val": n_val})
            with (run_dir / "metrics.json").open("w") as f:
                json.dump(metrics, f, indent=2, default=str)
            print(f"  interferon_ab_rank        = {metrics['interferon_ab_rank']}")
            print(f"  interferon_induction_rank = {metrics['interferon_induction_rank']}")
            print(f"  top_20_hit_rate           = {metrics['top_20_hit_rate']}")
            results.append(metrics)

    return results


# ---------------------------------------------------------------------------
# Retrain-mode pipeline
# ---------------------------------------------------------------------------

def run_retrain(args, out_root: Path) -> list[dict]:
    """For each fraction, subsample TRAIN and retrain from scratch. Val stays fixed."""
    print("[retrain] loading full data (val will stay fixed across fractions)")
    data = load_data(args)
    full_train_size = len(data["x_train"])

    results = []
    per_run_dir = out_root / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    for frac in args.sample_fractions:
        n_train = int(full_train_size * frac)
        if n_train < 32:
            print(f"[retrain] frac={frac} would give n_train={n_train}, skipping")
            continue

        print(f"\n[retrain] frac={frac}  n_train={n_train}")
        train_idx = np.sort(rng.choice(full_train_size, size=n_train, replace=False))

        subset = {
            **data,
            "x_train":      data["x_train"].iloc[train_idx],
            "counts_train": data["counts_train"].iloc[train_idx],
            "cov_train":    data["cov_train"].iloc[train_idx],
        }

        set_all_seeds(args.seed)
        model = build_model_from_args(subset, args)
        t0 = time.time()
        model, _history = train_from_args(model, subset, args)
        train_time = time.time() - t0
        print(f"  [train] {train_time:.1f}s on {n_train} cells")

        for method in ("naive", "model"):
            tag = f"{frac:g}_{method}"
            run_dir = per_run_dir / tag
            run_dir.mkdir(exist_ok=True)

            if method == "naive":
                de_df, pa_df = rank_via_naive(
                    data["x_val"], data["adj_df"], data["cov_val"],
                    data["cond_ctrl_col"], data["cond_stim_col"], data["gene_names"],
                )
            else:
                de_df, pa_df = rank_via_model(
                    model, data["x_val"], data["cov_val"], data["adj_df"],
                    data["gene_names"], data["cond_ctrl_col"], data["cond_stim_col"],
                    args,
                )

            if pa_df is not None:
                de_df.to_csv(run_dir / "de.csv")
                pa_df.to_csv(run_dir / "pa.csv")

            metrics = summarise_ranking(pa_df)
            metrics.update({
                "sample_fraction": frac, "method": method,
                "n_train": n_train, "train_time_s": train_time,
            })
            with (run_dir / "metrics.json").open("w") as f:
                json.dump(metrics, f, indent=2, default=str)
            print(f"  {method:5s}  interferon_ab_rank={metrics['interferon_ab_rank']} "
                  f" induction_rank={metrics['interferon_induction_rank']} "
                  f" top20={metrics['top_20_hit_rate']}")
            results.append(metrics)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "smoke" if args.smoke else "full"
    out_root = Path(args.output_root) / f"{ts}_{tag}_{args.mode}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[main] output root: {out_root}")
    print(f"[main] mode: {args.mode}  fractions: {args.sample_fractions}")

    args_dict = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    with (out_root / "config.json").open("w") as f:
        json.dump(args_dict, f, indent=2, default=str)

    if args.mode == "frozen":
        results = run_frozen(args, out_root)
    else:
        results = run_retrain(args, out_root)

    if not results:
        print("[main] no sample sizes produced results; nothing to summarise")
        return

    summary = pd.DataFrame(results)
    summary_path = out_root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n{'=' * 66}")
    print(f"[summary] {summary_path}")
    print(f"{'=' * 66}")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.float_format", "{:.4f}".format,
    ):
        # Sort by fraction desc so 100% is at top of the print
        print(summary.sort_values(["sample_fraction", "method"], ascending=[False, True]).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[main] interrupted by user", file=sys.stderr)
        sys.exit(130)
