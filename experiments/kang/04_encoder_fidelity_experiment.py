#!/usr/bin/env python
"""04_encoder_fidelity_experiment.py — 2x2 cross of nonneg_encoder x standardize_input.

Four configurations testing Carlos's central hypothesis: the encoder's sign
freedom and scale freedom produce the "one loud gene wins" failure mode.
Separates the sign-freedom fix (nonneg_encoder) from the scale-freedom fix
(standardize_input) from the combination.

    1. baseline           nonneg_encoder=False, standardize_input=False
    2. nonneg_only        nonneg_encoder=True,  standardize_input=False
    3. standardize_only   nonneg_encoder=False, standardize_input=True
    4. both               nonneg_encoder=True,  standardize_input=True

Note that config #1 (baseline) reproduces the same architecture as
02_architecture_ablation.py's baseline. Running both scripts gives two
independent measurements of the same setup — useful for gauging seed
sensitivity.

All 4 configs train with covariate conditioning (condition + cell_type
one-hot). Same production hyperparameters as 02_architecture_ablation.py.

Metrics per config
------------------
- best_val_loss                : minimum validation ELBO.
- fidelity_median_abs_corr     : median |corr| from pathway_unit_fidelity.
                                 Higher = pathway units track their own gene
                                 sets on average.
- fidelity_fraction_below_02   : fraction of units with |corr| < 0.2. These
                                 units aren't tracking their pathway at all.
- fidelity_fraction_inverted   : fraction of units with corr < 0. These units
                                 have learned an upside-down representation.
- unit_vs_naive_auc_diff_mean  : mean over pathways of (unit_auc - naive_auc).
                                 Per pathway, compute AUC of (per-cell unit
                                 activation) as classifier for control vs
                                 stimulated. Compare to AUC of the naive
                                 baseline (per-cell mean of z-scored member
                                 genes). Positive = model unit adds signal
                                 beyond the naive average.
- unit_vs_naive_auc_frac_wins  : fraction of pathways where unit_auc > naive_auc.
- competitive_z_interferon     : signed z-score for interferon-a/b pathway
                                 from decoder-side pathway_activity. Same
                                 metric as 02_architecture_ablation.py for
                                 cross-script comparison.

Skip counterfactual correlation here (already reported by
02_architecture_ablation.py under the same training setup — this script
tests a different hypothesis and doesn't add value by re-computing that).

Outputs
-------
experiments/kang/outputs/encoder_fidelity_experiment/<timestamp>_<smoke|full>/
    config.json               the exact CLI args used
    <config_name>/
        metrics.json          all metrics for this config
        fidelity.csv          per-pathway pathway_unit_fidelity output
        auc_diff.csv          per-pathway unit_auc, naive_auc, diff
        history.csv           per-epoch training curves
        checkpoint.pt         if --save-checkpoints
    summary.csv               one row per config, all summary metrics

Usage
-----
Smoke  : pixi run python experiments/kang/04_encoder_fidelity_experiment.py --smoke
Full   : pixi run python experiments/kang/04_encoder_fidelity_experiment.py
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
import scipy.stats
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from pyvae import (
    InformedVAE,
    build_model_config,
    differential_expression,
    load_kang,
    pathway_activity,
    pathway_unit_fidelity,
    set_all_seeds,
    train_ivae_modern,
)


# ---------------------------------------------------------------------------
# Configurations — 2x2 cross
# ---------------------------------------------------------------------------

CONFIGS = [
    {"name": "baseline",         "kwargs": {"nonneg_encoder": False, "standardize_input": False}},
    {"name": "nonneg_only",      "kwargs": {"nonneg_encoder": True,  "standardize_input": False}},
    {"name": "standardize_only", "kwargs": {"nonneg_encoder": False, "standardize_input": True}},
    {"name": "both",             "kwargs": {"nonneg_encoder": True,  "standardize_input": True}},
]

INTERFERON_PATHWAY = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Encoder fidelity 2x2 cross on Kang.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--smoke", action="store_true",
                   help="Fast smoke test: 500 cells, 5 epochs, small DE draws.")

    # Data paths — same defaults as 02_architecture_ablation.py.
    p.add_argument("--data-folder", default="experiments/kang/data")
    p.add_argument("--resources-dir",
                   default="experiments/kang/resources/c2.cp.reactome.v7.5.1.symbols.gmt",
                   help="Path to the Reactome GMT file (matches the notebook).")
    p.add_argument("--output-root",
                   default="experiments/kang/outputs/encoder_fidelity_experiment")

    # Data prep — same defaults as 02.
    p.add_argument("--n-cells", type=int, default=None)
    p.add_argument("--n-genes", type=int, default=5000)
    p.add_argument("--target-sum", type=float, default=1e4)

    # Model
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--l2-lambda", type=float, default=1e-5)

    # Training — notebook-matching defaults (same as 02).
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--device", default="cpu")

    p.add_argument("--seed", type=int, default=42)

    # Metric-computation knobs.
    p.add_argument("--fidelity-min-genes", type=int, default=10,
                   help="Skip pathways with < this many genes in pathway_unit_fidelity.")
    p.add_argument("--n-de-samples", type=int, default=25)
    p.add_argument("--n-de-pairs", type=int, default=2000)

    p.add_argument("--save-checkpoints", action="store_true")

    args = p.parse_args()

    if args.smoke:
        if args.n_cells is None:
            args.n_cells = 500
        args.epochs = min(args.epochs, 5)
        args.warmup_epochs = min(args.warmup_epochs, 2)
        args.patience = min(args.patience, 5)
        args.n_de_samples = min(args.n_de_samples, 3)
        args.n_de_pairs = min(args.n_de_pairs, 200)
        args.fidelity_min_genes = min(args.fidelity_min_genes, 3)

    return args


# ---------------------------------------------------------------------------
# Data loading — identical to 02_architecture_ablation.py
# ---------------------------------------------------------------------------

def load_data(args) -> dict:
    """Load Kang + Reactome adjacency + covariates; split into train/val."""
    print(f"[data] load_kang(target_sum={args.target_sum}, n_genes={args.n_genes})")
    adata = load_kang(
        data_folder=args.data_folder,
        normalize=True,
        n_genes=args.n_genes,
        return_path=False,
        target_sum=args.target_sum,
    )

    if args.n_cells is not None and args.n_cells < adata.n_obs:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(adata.n_obs, size=args.n_cells, replace=False)
        adata = adata[idx].copy()
        print(f"[data] subsampled to {args.n_cells} cells")

    print(f"[data] adata: {adata.n_obs} cells x {adata.n_vars} genes")

    x_trans = adata.to_df()
    config = build_model_config(
        genes=x_trans.columns,
        model_kind="ivae_reactome",
        resources_dir=args.resources_dir,
    )
    adj_df = config.model_layer[0]
    gene_names = list(config.input_genes)
    pathway_names = list(adj_df.columns)
    print(f"[data] adj: {len(gene_names)} genes x {len(pathway_names)} pathways")

    if INTERFERON_PATHWAY not in pathway_names:
        print(f"[warn] {INTERFERON_PATHWAY} not in adj_df; competitive_z will be NaN")

    x_trans = x_trans[gene_names]
    counts_df = adata.to_df(layer="counts")[gene_names]

    cov = pd.get_dummies(
        adata.obs[["condition", "cell_type"]],
        dtype=np.float32,
    )
    print(f"[data] cov: {cov.shape[1]} columns")

    cond_ctrl_col = "condition_control"
    cond_stim_col = "condition_stimulated"
    for col in (cond_ctrl_col, cond_stim_col):
        if col not in cov.columns:
            raise ValueError(f"Expected covariate column {col!r} in cov; got {list(cov.columns)}")

    strat_full = (
        adata.obs["condition"].astype(str)
        + "_"
        + adata.obs["cell_type"].astype(str)
    )
    min_stratum = strat_full.value_counts().min()
    if min_stratum >= 2:
        strat = strat_full
        strat_label = "condition x cell_type"
    else:
        strat = adata.obs["condition"].astype(str)
        strat_label = f"condition only (smallest cell_type stratum has {min_stratum})"
    idx_tr, idx_va = train_test_split(
        np.arange(len(adata)),
        test_size=0.2,
        stratify=strat,
        random_state=args.seed,
    )
    print(f"[data] stratified by {strat_label}")
    print(f"[data] train={len(idx_tr)}  val={len(idx_va)}")

    return {
        "x_train": x_trans.iloc[idx_tr].astype(np.float32),
        "x_val":   x_trans.iloc[idx_va].astype(np.float32),
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
# Build & train — identical shape to 02
# ---------------------------------------------------------------------------

def build_model(config: dict, data: dict, args) -> InformedVAE:
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
        **config["kwargs"],
    )


def train_one(model: InformedVAE, data: dict, args) -> tuple[InformedVAE, dict]:
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
# Metrics
# ---------------------------------------------------------------------------

def _run_pathway_unit_fidelity(model, data, args) -> pd.DataFrame:
    """Compute per-pathway fidelity on val cells."""
    return pathway_unit_fidelity(
        model,
        x=data["x_val"].values,
        adj=data["adj_df"],
        cov=data["cov_val"].values,
        min_genes=args.fidelity_min_genes,
    )


def _summarise_fidelity(fid: pd.DataFrame) -> dict:
    """Extract the three fidelity summary numbers Carlos asked for."""
    abs_corr = fid["abs_corr"].dropna()
    if len(abs_corr) == 0:
        return {
            "fidelity_median_abs_corr": float("nan"),
            "fidelity_fraction_below_02": float("nan"),
            "fidelity_fraction_inverted": float("nan"),
        }
    # Fraction inverted: over units that were tested (corr not NaN).
    sign = fid["sign"].dropna()
    return {
        "fidelity_median_abs_corr":   float(abs_corr.median()),
        "fidelity_fraction_below_02": float((abs_corr < 0.2).mean()),
        "fidelity_fraction_inverted": float((sign < 0).mean()) if len(sign) else float("nan"),
    }


def _compute_unit_vs_naive_auc(model, data) -> pd.DataFrame:
    """For each pathway, compute unit_auc and naive_auc for condition classification.

    unit_auc  : AUC of h[:, j] as a classifier for control vs stimulated on val.
    naive_auc : AUC of the mean z-scored expression of pathway j's member genes.

    Returns a DataFrame with one row per pathway, columns unit_auc, naive_auc, diff.
    Pathways where AUC can't be computed (only one class in val, or zero-variance
    predictor) get NaN.
    """
    device = next(model.parameters()).device
    x_val_np = data["x_val"].values
    x_val_t = torch.tensor(x_val_np, dtype=torch.float32, device=device)
    cov_val_t = torch.tensor(data["cov_val"].values, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        _, _, h = model.encode(x_val_t, cov_val_t)
    h_np = h.cpu().numpy()

    y = data["cov_val"][data["cond_stim_col"]].values > 0.5  # 1 = stimulated

    # Only compute AUC if both classes are present.
    if y.sum() == 0 or y.sum() == len(y):
        n_pathways = h_np.shape[1]
        return pd.DataFrame({
            "unit_auc": [float("nan")] * n_pathways,
            "naive_auc": [float("nan")] * n_pathways,
            "diff": [float("nan")] * n_pathways,
        }, index=data["pathway_names"])

    # Z-score gene expression per gene across val cells (for the naive baseline).
    x_mean = x_val_np.mean(axis=0, keepdims=True)
    x_std = x_val_np.std(axis=0, keepdims=True)
    x_z = (x_val_np - x_mean) / (x_std + 1e-8)

    adj_arr = data["adj_df"].values.astype(bool)
    n_pathways = h_np.shape[1]
    unit_auc = np.full(n_pathways, np.nan, dtype=np.float64)
    naive_auc = np.full(n_pathways, np.nan, dtype=np.float64)

    for j in range(n_pathways):
        h_j = h_np[:, j]
        if h_j.std() > 1e-8:
            try:
                unit_auc[j] = float(roc_auc_score(y, h_j))
            except ValueError:
                pass
        member_mask = adj_arr[:, j]
        if member_mask.sum() > 0:
            naive_j = x_z[:, member_mask].mean(axis=1)
            if naive_j.std() > 1e-8:
                try:
                    naive_auc[j] = float(roc_auc_score(y, naive_j))
                except ValueError:
                    pass

    return pd.DataFrame(
        {"unit_auc": unit_auc, "naive_auc": naive_auc, "diff": unit_auc - naive_auc},
        index=data["pathway_names"],
    )


def _summarise_auc_diff(auc_df: pd.DataFrame) -> dict:
    """Extract summary numbers from the per-pathway AUC comparison."""
    diff = auc_df["diff"].dropna()
    if len(diff) == 0:
        return {
            "unit_vs_naive_auc_diff_mean": float("nan"),
            "unit_vs_naive_auc_frac_wins": float("nan"),
        }
    return {
        "unit_vs_naive_auc_diff_mean": float(diff.mean()),
        "unit_vs_naive_auc_frac_wins": float((diff > 0).mean()),
    }


def _compute_competitive_z_interferon(model, data, args) -> float:
    """Signed z for INTERFERON_ALPHA_BETA_SIGNALING via decoder-side pathway_activity.

    Same construction as 02_architecture_ablation.py so results are directly
    comparable across scripts.
    """
    ctrl_mask = data["cov_val"][data["cond_ctrl_col"]].values > 0.5
    stim_mask = data["cov_val"][data["cond_stim_col"]].values > 0.5
    if ctrl_mask.sum() == 0 or stim_mask.sum() == 0:
        return float("nan")

    de = differential_expression(
        model,
        x_a=data["x_val"].iloc[stim_mask],
        x_b=data["x_val"].iloc[ctrl_mask],
        cov_a=data["cov_val"].iloc[stim_mask],
        cov_b=data["cov_val"].iloc[ctrl_mask],
        n_samples=args.n_de_samples,
        n_pairs=args.n_de_pairs,
        gene_names=data["gene_names"],
        seed=args.seed,
    )
    pa = pathway_activity(de, data["adj_df"], statistic="lfc_mean")

    if INTERFERON_PATHWAY not in pa.index:
        return float("nan")
    p = pa.loc[INTERFERON_PATHWAY, "pvalue"]
    effect = pa.loc[INTERFERON_PATHWAY, "effect"]
    if not np.isfinite(p) or not np.isfinite(effect) or effect == 0:
        return float("nan")
    p_clip = max(p, 1e-300)
    z = scipy.stats.norm.ppf(1 - p_clip / 2)
    return float(np.sign(effect) * z)


def compute_metrics(model, history, data, args, cfg_dir: Path) -> dict:
    """Run all metrics for this config; save per-pathway tables to cfg_dir."""
    print("  [metric] pathway_unit_fidelity...")
    fid = _run_pathway_unit_fidelity(model, data, args)
    fid.to_csv(cfg_dir / "fidelity.csv")
    fid_summary = _summarise_fidelity(fid)

    print("  [metric] unit_vs_naive_auc...")
    auc_df = _compute_unit_vs_naive_auc(model, data)
    auc_df.to_csv(cfg_dir / "auc_diff.csv")
    auc_summary = _summarise_auc_diff(auc_df)

    print("  [metric] competitive_z_interferon...")
    z_ifn = _compute_competitive_z_interferon(model, data, args)

    best_val = float(min(history["val"])) if history["val"] else float("nan")

    metrics = {
        "best_val_loss": best_val,
        **fid_summary,
        **auc_summary,
        "competitive_z_interferon": z_ifn,
    }
    for k, v in metrics.items():
        print(f"  [metric] {k:32s} = {v}")
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "smoke" if args.smoke else "full"
    out_root = Path(args.output_root) / f"{ts}_{tag}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[main] output root: {out_root}")
    print(f"[main] mode:        {tag}")
    print(f"[main] epochs={args.epochs} batch_size={args.batch_size} device={args.device}")

    args_dict = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    with (out_root / "config.json").open("w") as f:
        json.dump(args_dict, f, indent=2, default=str)

    data = load_data(args)

    results = []
    for config in CONFIGS:
        name = config["name"]
        print(f"\n{'=' * 66}")
        print(f"[config] {name}  kwargs={config['kwargs']}")
        print(f"{'=' * 66}")
        cfg_dir = out_root / name
        cfg_dir.mkdir(exist_ok=True)

        set_all_seeds(args.seed)
        model = build_model(config, data, args)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  [build] {n_params:,} parameters")

        t0 = time.time()
        model, history = train_one(model, data, args)
        train_time = time.time() - t0
        n_epochs = len(history["train"])
        print(f"  [train] ran {n_epochs} epochs in {train_time:.1f}s "
              f"({train_time / max(n_epochs, 1):.1f}s / epoch)")

        pd.DataFrame(history).to_csv(cfg_dir / "history.csv", index=False)
        if args.save_checkpoints:
            torch.save(model.state_dict(), cfg_dir / "checkpoint.pt")

        metrics = compute_metrics(model, history, data, args, cfg_dir)
        metrics["n_epochs_run"] = n_epochs
        metrics["train_time_s"] = train_time
        with (cfg_dir / "metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)
        results.append({"config": name, **metrics})

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
        print(summary.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[main] interrupted by user", file=sys.stderr)
        sys.exit(130)
