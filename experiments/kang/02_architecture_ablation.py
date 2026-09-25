#!/usr/bin/env python
"""02_architecture_ablation.py — one-at-a-time A/B of encoder-constraint options.

Runs 5 configurations of InformedVAE on Kang, changing exactly one thing per
row from the notebook baseline:

    1. baseline          — all defaults, matches current notebook architecture
    2. init_fan_in       — init="fan_in"
    3. normalize_batch   — normalize="batch"
    4. normalize_layer   — normalize="layer"
    5. informed_decoder  — informed_decoder=True

The 2x2 cross of nonneg_encoder x standardize_input is a SEPARATE experiment
(04_encoder_fidelity_experiment.py). Not in this script.

All 5 configs train the NB InformedVAE with covariate conditioning
(condition + cell_type one-hot). This closes limitation 1 in the notebook
Takeaways (which is the whole point of Phase 1d) and lets us report
counterfactual correlation as a comparable metric across the sweep.

Metrics reported per config
---------------------------
- best_val_loss              : minimum validation ELBO (beta=1.0) across epochs.
- tanh_saturation            : fraction of pathway-layer activations with
                               |h| > 0.99 on the val set. Higher = more units
                               stuck at the tanh rails and unable to discriminate.
- competitive_z_interferon   : signed z-score for the interferon-alpha/beta
                               pathway from pathway_activity on the decoder-side
                               DE table. Larger absolute value = stronger
                               separation from the panel.
- sign_agreement             : fraction of pathways where the encoder-side
                               bayes_factor_da sign matches the decoder-side
                               pathway_activity sign.
- counterfactual_correlation : Pearson correlation, across genes, between
                               the predicted control -> stimulated shift
                               (from predict_counterfactual) and the real
                               shift (mean_stim - mean_ctrl in log1p space).

Outputs
-------
experiments/kang/outputs/architecture_ablation/<timestamp>_<smoke|full>/
    config.json                          the exact CLI args used
    <config_name>/
        metrics.json                     the five metrics for this config
        history.csv                     per-epoch training curves
        checkpoint.pt                    if --save-checkpoints
    summary.csv                          one row per config, all metrics

Usage
-----
Smoke  : pixi run python experiments/kang/02_architecture_ablation.py --smoke
Full   : pixi run python experiments/kang/02_architecture_ablation.py

Production defaults match the notebook's train_ivae_modern call:
epochs=100, batch_size=128, patience=20, lr=1e-3, weight_decay=1e-6,
warmup_epochs=10, max_grad_norm=1.0. Override any of them via CLI.
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
from sklearn.model_selection import train_test_split

from pyvae import (
    InformedVAE,
    bayes_factor_da,
    build_model_config,
    differential_expression,
    load_kang,
    pathway_activity,
    set_all_seeds,
    swap_condition,
    train_ivae_modern,
)


# ---------------------------------------------------------------------------
# Configurations — one-at-a-time from baseline
# ---------------------------------------------------------------------------

CONFIGS = [
    {"name": "baseline",         "kwargs": {}},
    {"name": "init_fan_in",      "kwargs": {"init": "fan_in"}},
    {"name": "normalize_batch",  "kwargs": {"normalize": "batch"}},
    {"name": "normalize_layer",  "kwargs": {"normalize": "layer"}},
    {"name": "informed_decoder", "kwargs": {"informed_decoder": True}},
]

INTERFERON_PATHWAY = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Architecture ablation on Kang.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Modes
    p.add_argument("--smoke", action="store_true",
                   help="Fast smoke test: 500 cells, 5 epochs, small DE draws.")

    # Data paths
    p.add_argument("--data-folder", default="experiments/kang/data",
                   help="Kang h5ad cache dir.")
    p.add_argument("--resources-dir", default="experiments/kang/resources/c2.cp.reactome.v7.5.1.symbols.gmt",
                   help="Path to the Reactome GMT file (matches the notebook).")
    p.add_argument("--output-root", default="experiments/kang/outputs/architecture_ablation",
                   help="Where per-run subdirectories are created.")

    # Data prep
    p.add_argument("--n-cells", type=int, default=None,
                   help="Subsample to N cells; None uses the full ~24k dataset.")
    p.add_argument("--n-genes", type=int, default=5000,
                   help="Top HVGs kept before pathway alignment. "
                        "After sync_gexp_adj typically ~2400 survive.")
    p.add_argument("--target-sum", type=float, default=1e4,
                   help="load_kang target_sum (matches Phase 1e default).")

    # Model
    p.add_argument("--latent-dim", type=int, default=None,
                   help="Bottleneck dim; None = n_pathways // 2 (InformedVAE default).")
    p.add_argument("--l2-lambda", type=float, default=1e-5)

    # Training (notebook-matching defaults)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--device", default="cpu")

    # Reproducibility
    p.add_argument("--seed", type=int, default=42)

    # Metric-computation knobs (kept small for smoke, larger for prod)
    p.add_argument("--n-de-samples", type=int, default=25,
                   help="differential_expression n_samples")
    p.add_argument("--n-de-pairs", type=int, default=2000,
                   help="differential_expression n_pairs")
    p.add_argument("--n-cf-cells", type=int, default=1000,
                   help="Max control cells used for counterfactual correlation.")

    # Outputs
    p.add_argument("--save-checkpoints", action="store_true")

    args = p.parse_args()

    # Smoke overrides
    if args.smoke:
        if args.n_cells is None:
            args.n_cells = 500
        args.epochs = min(args.epochs, 5)
        args.warmup_epochs = min(args.warmup_epochs, 2)
        args.patience = min(args.patience, 5)
        args.n_de_samples = min(args.n_de_samples, 3)
        args.n_de_pairs = min(args.n_de_pairs, 200)
        args.n_cf_cells = min(args.n_cf_cells, 100)

    return args


# ---------------------------------------------------------------------------
# Data loading (shared across configs)
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

    # ---- Reactome adjacency via build_model_config (Phase 1c-refactored path) ----
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

    # Filter to synced genes
    x_trans = x_trans[gene_names]
    counts_df = adata.to_df(layer="counts")[gene_names]

    # ---- Covariate matrix: condition + cell_type one-hot ----
    cov = pd.get_dummies(
        adata.obs[["condition", "cell_type"]],
        dtype=np.float32,
    )
    print(f"[data] cov: {cov.shape[1]} columns  {list(cov.columns)}")

    cond_ctrl_col = "condition_control"
    cond_stim_col = "condition_stimulated"
    for col in (cond_ctrl_col, cond_stim_col):
        if col not in cov.columns:
            raise ValueError(f"Expected covariate column {col!r} in cov; got {list(cov.columns)}")

    # ---- Stratified train/val split (80/20 on condition x cell_type) ----
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
        # Some (condition, cell_type) combos have <2 cells (typical for smoke
        # or small subsamples). Fall back to condition-only stratification.
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
# Build & train one config
# ---------------------------------------------------------------------------

def build_model(config: dict, data: dict, args) -> InformedVAE:
    """Construct an NB InformedVAE for this config."""
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
    """Train with train_ivae_modern under the matched notebook hyperparameters."""
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
# Metrics — the five per-config numbers
# ---------------------------------------------------------------------------

def _split_val_by_condition(data: dict):
    """Return (ctrl_mask, stim_mask) as numpy boolean masks over val set."""
    ctrl_mask = data["cov_val"][data["cond_ctrl_col"]].values > 0.5
    stim_mask = data["cov_val"][data["cond_stim_col"]].values > 0.5
    return ctrl_mask, stim_mask


def metric_tanh_saturation(model: InformedVAE, data: dict, args) -> float:
    """Fraction of pathway-layer activations with |h| > 0.99 on val cells."""
    device = next(model.parameters()).device
    x_val_t = torch.tensor(data["x_val"].values, dtype=torch.float32, device=device)
    cov_val_t = torch.tensor(data["cov_val"].values, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        _, _, h = model.encode(x_val_t, cov_val_t)
    return float((h.abs() > 0.99).float().mean().cpu().item())


def _run_encoder_bf(model, data, ctrl_mask, stim_mask, args) -> pd.DataFrame:
    """Encoder-side bayes_factor_da, stimulated vs control."""
    device = next(model.parameters()).device
    x_stim = torch.tensor(
        data["x_val"].iloc[stim_mask].values, dtype=torch.float32, device=device
    )
    x_ctrl = torch.tensor(
        data["x_val"].iloc[ctrl_mask].values, dtype=torch.float32, device=device
    )
    cov_stim = torch.tensor(
        data["cov_val"].iloc[stim_mask].values, dtype=torch.float32, device=device
    )
    cov_ctrl = torch.tensor(
        data["cov_val"].iloc[ctrl_mask].values, dtype=torch.float32, device=device
    )
    return bayes_factor_da(
        model,
        x_a=x_stim, x_b=x_ctrl,
        cov_a=cov_stim, cov_b=cov_ctrl,
        pathway_names=data["pathway_names"],
        seed=args.seed,
    )


def _run_decoder_de_pa(model, data, ctrl_mask, stim_mask, args):
    """Decoder-side differential_expression + pathway_activity, stim vs ctrl."""
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
    return de, pa


def _extract_competitive_z(pa: pd.DataFrame) -> float:
    """Signed z for INTERFERON_ALPHA_BETA_SIGNALING from pathway_activity output."""
    if pa is None or INTERFERON_PATHWAY not in pa.index:
        return float("nan")
    p = pa.loc[INTERFERON_PATHWAY, "pvalue"]
    effect = pa.loc[INTERFERON_PATHWAY, "effect"]
    if not np.isfinite(p) or not np.isfinite(effect) or effect == 0:
        return float("nan")
    # Clamp p to avoid inf when pvalue is exactly 0.
    p_clip = max(p, 1e-300)
    z = scipy.stats.norm.ppf(1 - p_clip / 2)
    return float(np.sign(effect) * z)


def _compute_sign_agreement(
    bf_encoder: pd.DataFrame, pa: pd.DataFrame, data: dict
) -> float:
    """Fraction of pathways where encoder BF sign matches decoder pa sign."""
    if bf_encoder is None or pa is None:
        return float("nan")
    encoder = bf_encoder.reindex(data["pathway_names"])["bf"]
    decoder_sign = pa.reindex(data["pathway_names"])["sign"]
    encoder_sign = np.sign(encoder.values)
    decoder_sign = decoder_sign.values

    # Both must be signed (not NaN, not 0) for a comparison to be meaningful.
    signed = (
        np.isfinite(encoder_sign) & np.isfinite(decoder_sign)
        & (encoder_sign != 0) & (decoder_sign != 0)
    )
    if signed.sum() == 0:
        return float("nan")
    return float((encoder_sign[signed] == decoder_sign[signed]).mean())


def metric_counterfactual_correlation(model, data, args) -> float:
    """Pearson corr between predicted ctrl->stim shift and real shift, across genes.

    For a random subset of control val cells, uses predict_counterfactual to
    predict their expression under the stimulated covariate. The predicted
    mean log1p shift (across cells) is compared to the real shift
    (mean_stim - mean_ctrl in log1p space, on val cells).
    """
    ctrl_mask, stim_mask = _split_val_by_condition(data)
    if ctrl_mask.sum() == 0 or stim_mask.sum() == 0:
        return float("nan")

    x_ctrl = data["x_val"].iloc[ctrl_mask]
    x_stim = data["x_val"].iloc[stim_mask]
    counts_ctrl = data["counts_val"].iloc[ctrl_mask]
    cov_ctrl = data["cov_val"].iloc[ctrl_mask]

    if len(x_ctrl) > args.n_cf_cells:
        rng = np.random.default_rng(args.seed)
        cf_idx = rng.choice(len(x_ctrl), size=args.n_cf_cells, replace=False)
        x_ctrl = x_ctrl.iloc[cf_idx]
        counts_ctrl = counts_ctrl.iloc[cf_idx]
        cov_ctrl = cov_ctrl.iloc[cf_idx]

    device = next(model.parameters()).device
    x_ctrl_t = torch.tensor(x_ctrl.values, dtype=torch.float32, device=device)
    cov_ctrl_t = torch.tensor(cov_ctrl.values, dtype=torch.float32, device=device)
    lib_ctrl_t = torch.tensor(
        counts_ctrl.values.sum(axis=1, keepdims=True),
        dtype=torch.float32,
        device=device,
    )

    # Swap condition_control -> condition_stimulated in the covariate frame.
    cov_to = swap_condition(cov_ctrl, from_label="control", to_label="stimulated")
    cov_to_t = torch.tensor(cov_to.values, dtype=torch.float32, device=device)

    predicted_stim_counts = model.predict_counterfactual(
        x_ctrl_t, lib_ctrl_t, cov_ctrl_t, cov_to_t
    )

    # Predicted shift (per gene): mean over cells of log1p(pred_stim) - x_ctrl.
    predicted_shift = (
        torch.log1p(predicted_stim_counts).mean(dim=0) - x_ctrl_t.mean(dim=0)
    ).detach().cpu().numpy()

    # Real shift (per gene): mean over val cells: mean_stim - mean_ctrl in log1p.
    # x_val is already log1p-normalised (via load_kang normalize=True), so this
    # is a log1p-space mean difference — same units as predicted_shift.
    real_shift = x_stim.values.mean(axis=0) - x_ctrl.values.mean(axis=0)

    if predicted_shift.shape != real_shift.shape:
        return float("nan")

    ps = predicted_shift - predicted_shift.mean()
    rs = real_shift - real_shift.mean()
    denom = np.sqrt((ps ** 2).sum() * (rs ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((ps * rs).sum() / denom)


def compute_metrics(model, history, data, args) -> dict:
    """Return dict with all five metrics + best_val_loss."""
    ctrl_mask, stim_mask = _split_val_by_condition(data)
    has_both = ctrl_mask.sum() > 0 and stim_mask.sum() > 0

    # Encoder-side BF and decoder-side DE+PA are shared across two metrics,
    # so compute once here.
    bf_encoder = None
    pa = None
    if has_both:
        print("  [metric] encoder-side bayes_factor_da...")
        bf_encoder = _run_encoder_bf(model, data, ctrl_mask, stim_mask, args)
        print("  [metric] decoder-side differential_expression + pathway_activity...")
        _, pa = _run_decoder_de_pa(model, data, ctrl_mask, stim_mask, args)

    print("  [metric] tanh_saturation...")
    tanh_sat = metric_tanh_saturation(model, data, args)

    print("  [metric] competitive_z_interferon...")
    z_ifn = _extract_competitive_z(pa)

    print("  [metric] sign_agreement...")
    sign_agree = _compute_sign_agreement(bf_encoder, pa, data)

    print("  [metric] counterfactual_correlation...")
    cf_corr = metric_counterfactual_correlation(model, data, args)

    best_val = float(min(history["val"])) if history["val"] else float("nan")

    metrics = {
        "best_val_loss":              best_val,
        "tanh_saturation":            tanh_sat,
        "competitive_z_interferon":   z_ifn,
        "sign_agreement":             sign_agree,
        "counterfactual_correlation": cf_corr,
    }
    for k, v in metrics.items():
        print(f"  [metric] {k:30s} = {v}")
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

    # Persist args for later reproduction.
    args_dict = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    with (out_root / "config.json").open("w") as f:
        json.dump(args_dict, f, indent=2, default=str)

    # Shared data (loaded once, used by every config).
    data = load_data(args)

    results = []
    for config in CONFIGS:
        name = config["name"]
        print(f"\n{'=' * 66}")
        print(f"[config] {name}  kwargs={config['kwargs']}")
        print(f"{'=' * 66}")
        cfg_dir = out_root / name
        cfg_dir.mkdir(exist_ok=True)

        # Fresh seed per config so RNG state is comparable at model init.
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

        # Save history + optional checkpoint.
        pd.DataFrame(history).to_csv(cfg_dir / "history.csv", index=False)
        if args.save_checkpoints:
            torch.save(model.state_dict(), cfg_dir / "checkpoint.pt")

        # Metrics.
        metrics = compute_metrics(model, history, data, args)
        metrics["n_epochs_run"] = n_epochs
        metrics["train_time_s"] = train_time
        with (cfg_dir / "metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)

        results.append({"config": name, **metrics})

    # Summary CSV.
    summary = pd.DataFrame(results)
    summary_path = out_root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n{'=' * 66}")
    print(f"[summary] {summary_path}")
    print(f"{'=' * 66}")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", "{:.4f}".format,
    ):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[main] interrupted by user", file=sys.stderr)
        sys.exit(130)
