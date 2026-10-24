"""
biological_questions.py — exploratory figures answering Carlos's suggestion:
"explore MM stages beyond cross-cohort agreement."

v3: adds `low_signal` class and tightens `peak_dominance` so noise pathways
    don't flood the real timing categories.
    See classify_timing() docstring for the rule.

Run from the repo root:

    python experiments/multiple_mieloma/biological_questions.py --fig fig5

Outputs land in experiments/multiple_mieloma/outputs/biological_questions/.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
BIOLOGY_DIR = REPO_ROOT / "experiments" / "multiple_mieloma" / "outputs" / "biology"
OUT_DIR = REPO_ROOT / "experiments" / "multiple_mieloma" / "outputs" / "biological_questions"

COHORTS = {
    "Zavidij_2020": BIOLOGY_DIR / "Zavidij_2020" / "unit_contrasts.csv",
    "Foster_2026": BIOLOGY_DIR / "Foster_2026" / "unit_contrasts.csv",
}

CONTRAST_ORDER = ["Non->MGUS", "Non->SMM", "Non->MM"]
STAGE_LABELS = ["Non", "MGUS", "SMM", "MM"]
STAGE_X = {"Non->MGUS": 1, "Non->SMM": 2, "Non->MM": 3}

LINEAGES = ["CD4", "CD8"]

# Order controls legend order; also sets which classes get columns in the summary.
CLASS_COLORS = {
    "early_peak":    "#9467BD",   # purple      — argmax before MM, peak >> endpoint
    "early_rising":  "#D62728",   # red         — argmax at MM, substantial at MGUS
    "progressive":   "#1F77B4",   # blue        — argmax at MM, moderate at MGUS
    "late_emerging": "#2CA02C",   # green       — argmax at MM, flat until MM
    "non_monotonic": "#808080",   # grey        — sign flip outside null band
    "low_signal":    "#D8D8D8",   # pale grey   — no meaningful activity anywhere
}

CLASS_LABEL = {
    "early_peak":    "Early-peak",
    "early_rising":  "Early-rising",
    "progressive":   "Progressive",
    "late_emerging": "Late-emerging",
    "non_monotonic": "Non-monotonic",
    "low_signal":    "Low-signal",
}

INTERFERON_PATTERN = re.compile(r"interferon.*alpha.*beta", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_timing(
    z_mgus: float,
    z_smm: float,
    z_mm: float,
    early_thr: float = 0.5,
    late_thr: float = 0.3,
    null_eps: float = 0.15,
    peak_dominance: float = 1.2,
    min_signal_eps: float = 0.3,
) -> str:
    """Classify a unit's three-point (z_MGUS, z_SMM, z_MM) trajectory.

    Robustness tricks
    -----------------
    * Low-signal floor (`min_signal_eps`): if max(|z|) < min_signal_eps, the
      unit has no meaningful activity at any stage → labeled `low_signal`.
      Keeps noise pathways out of the real timing categories.
    * Null-band tolerance (`null_eps`): values with |z| < null_eps are treated
      as neutral for the sign-flip check — tiny negatives next to a positive
      signal don't demote the trajectory to non_monotonic.
    * argmax-based primary split: `argmax(|z|)` determines whether the signal
      peaks at an early stage (MGUS/SMM) or at the endpoint (MM).

    Classes
    -------
    * low_signal    : max(|z|) < min_signal_eps. No signal to classify.
    * early_peak    : argmax(|z|) at MGUS or SMM, AND |peak| >= |z_MM| * peak_dominance (1.2).
                      → signal reaches max activity before MM, substantially above endpoint.
    * early_rising  : argmax at MM (or early with |peak|/|z_MM| < peak_dominance),
                      |z_MGUS|/|z_MM| >= early_thr (0.5).
                      → signal already a substantial fraction of endpoint at MGUS.
    * progressive   : argmax at MM, early_thr > |z_MGUS|/|z_MM| >= late_thr (0.3).
                      → builds smoothly across stages.
    * late_emerging : argmax at MM, |z_MGUS|/|z_MM| < late_thr AND
                      |z_SMM|/|z_MM| < late_thr.
                      → signal only appears at MM.
    * non_monotonic : sign flip among values outside the null band, OR z_mm == 0,
                      OR missing values.
    """
    if not (np.isfinite(z_mgus) and np.isfinite(z_smm) and np.isfinite(z_mm)):
        return "non_monotonic"
    if abs(z_mm) == 0.0:
        return "non_monotonic"

    vals = np.array([z_mgus, z_smm, z_mm])
    abs_vals = np.abs(vals)

    # Low-signal floor: no stage has meaningful activity
    if abs_vals.max() < min_signal_eps:
        return "low_signal"

    # Sign-flip check with null-band tolerance
    outside = vals[abs_vals >= null_eps]
    if len(outside) >= 2:
        signs_out = np.sign(outside)
        if not np.all(signs_out == signs_out[0]):
            return "non_monotonic"

    # argmax of magnitude
    peak_idx = int(np.argmax(abs_vals))  # 0=MGUS, 1=SMM, 2=MM
    peak_mag = float(abs_vals[peak_idx])
    mm_mag = abs(z_mm)

    # Early peak: argmax before MM and peak substantially larger than MM
    if peak_idx in (0, 1) and peak_mag >= mm_mag * peak_dominance:
        return "early_peak"

    # Otherwise, classify by how much of the MM endpoint is already there at MGUS
    r_mgus = abs(z_mgus) / mm_mag
    r_smm = abs(z_smm) / mm_mag

    if r_mgus >= early_thr:
        return "early_rising"
    if r_mgus < late_thr and r_smm < late_thr:
        return "late_emerging"
    return "progressive"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_cohort_trajectories(csv_path: Path) -> pd.DataFrame:
    """Pivot a cohort's unit_contrasts.csv to one row per (lineage, unit) with
    columns z_mgus, z_smm, z_mm.
    """
    df = pd.read_csv(csv_path)
    df = df[df["lineage"].isin(LINEAGES)].copy()
    pivot = df.pivot_table(
        index=["lineage", "unit"],
        columns="contrast",
        values="member_shift_z",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.rename(
        columns={
            "Non->MGUS": "z_mgus",
            "Non->SMM":  "z_smm",
            "Non->MM":   "z_mm",
        }
    )
    for col in ["z_mgus", "z_smm", "z_mm"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    return pivot[["lineage", "unit", "z_mgus", "z_smm", "z_mm"]]


def _find_interferon_unit(unit_names: pd.Series) -> str | None:
    """Case-insensitive search for 'interferon alpha beta' in unit names."""
    matches = [u for u in unit_names if INTERFERON_PATTERN.search(u.replace("_", " "))]
    if not matches:
        return None
    # Prefer the shortest match (most specific, excludes compound names)
    return min(matches, key=len)


# ---------------------------------------------------------------------------
# fig5 — pathway timing
# ---------------------------------------------------------------------------
def fig5_pathway_timing(
    out_dir: Path = OUT_DIR,
    top_k: int = 10,
    early_thr: float = 0.5,
    late_thr: float = 0.3,
    null_eps: float = 0.15,
    peak_dominance: float = 1.2,
    min_signal_eps: float = 0.3,
) -> dict:
    """Four-panel figure: Zavidij × Foster × CD4 × CD8 pathway trajectories.

    Top-k units per panel (ranked by |z_mm|) are drawn as 3-point ladders
    (Non=0 pinned, then MGUS/SMM/MM points colored by timing classification).
    Interferon α/β is highlighted in bold with a label.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and classify every unit in every cohort × lineage
    records = []
    for cohort_name, csv_path in COHORTS.items():
        pivot = _load_cohort_trajectories(csv_path)
        pivot["cohort"] = cohort_name
        pivot["class"] = [
            classify_timing(
                r.z_mgus, r.z_smm, r.z_mm,
                early_thr=early_thr, late_thr=late_thr,
                null_eps=null_eps, peak_dominance=peak_dominance,
                min_signal_eps=min_signal_eps,
            )
            for r in pivot.itertuples(index=False)
        ]
        pivot["abs_z_mm"] = pivot["z_mm"].abs()
        records.append(pivot)

    classification_df = pd.concat(records, ignore_index=True)
    classification_csv = out_dir / "unit_timing_classification.csv"
    classification_df.to_csv(classification_csv, index=False)

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=False)
    cohort_order = ["Zavidij_2020", "Foster_2026"]

    for i, cohort_name in enumerate(cohort_order):
        highlight_unit = _find_interferon_unit(
            classification_df[classification_df["cohort"] == cohort_name]["unit"]
        )
        for j, lineage in enumerate(LINEAGES):
            ax = axes[i, j]
            panel = (
                classification_df[
                    (classification_df["cohort"] == cohort_name)
                    & (classification_df["lineage"] == lineage)
                ]
                .dropna(subset=["z_mm"])
                .sort_values("abs_z_mm", ascending=False)
                .head(top_k)
                .reset_index(drop=True)
            )

            # If interferon α/β isn't in top_k but exists, add it to the panel
            if highlight_unit is not None and highlight_unit not in panel["unit"].values:
                extra = classification_df[
                    (classification_df["cohort"] == cohort_name)
                    & (classification_df["lineage"] == lineage)
                    & (classification_df["unit"] == highlight_unit)
                ]
                if len(extra) > 0:
                    panel = pd.concat([panel, extra], ignore_index=True)

            for _, row in panel.iterrows():
                is_highlight = row["unit"] == highlight_unit
                color = CLASS_COLORS[row["class"]]
                lw = 3.0 if is_highlight else 1.6
                alpha = 1.0 if is_highlight else 0.70
                z_order = 10 if is_highlight else 2
                xs = [0, 1, 2, 3]
                ys = [0.0, row["z_mgus"], row["z_smm"], row["z_mm"]]
                ax.plot(
                    xs, ys,
                    color=color, linewidth=lw, alpha=alpha,
                    marker="o", markersize=6 if is_highlight else 4,
                    zorder=z_order,
                )
                if is_highlight:
                    ax.annotate(
                        "IFN α/β",
                        xy=(xs[-1], ys[-1]),
                        xytext=(6, 0),
                        textcoords="offset points",
                        fontsize=10,
                        fontweight="bold",
                        color=color,
                        verticalalignment="center",
                    )

            ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(STAGE_LABELS)
            ax.set_xlabel("Stage" if i == 1 else "")
            ax.set_ylabel("member_shift_z (vs Non)" if j == 0 else "")
            ax.set_title(f"{cohort_name} — {lineage}", fontsize=11)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.set_xlim(-0.15, 3.55)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=color, lw=2.4, label=CLASS_LABEL[cls])
        for cls, color in CLASS_COLORS.items()
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=6,
        frameon=False,
        fontsize=10,
    )

    fig.suptitle(
        "fig5 — Pathway timing across the MM precursor sequence\n"
        f"Top-{top_k} units per panel by |z_mm|. Non pinned at 0. "
        "Foster MGUS = 1 donor (interpret with caution).",
        fontsize=12,
        y=1.00,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    pdf_path = out_dir / "fig5_pathway_timing.pdf"
    png_path = out_dir / "fig5_pathway_timing.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=200)
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    # Summary table — class counts per cohort × lineage
    summary = (
        classification_df.groupby(["cohort", "lineage", "class"])
        .size()
        .reset_index(name="n_units")
        .pivot_table(
            index=["cohort", "lineage"],
            columns="class",
            values="n_units",
            fill_value=0,
        )
        .reset_index()
    )
    for cls in CLASS_COLORS:
        if cls not in summary.columns:
            summary[cls] = 0
    summary = summary[["cohort", "lineage"] + list(CLASS_COLORS)]
    summary["total"] = summary[list(CLASS_COLORS)].sum(axis=1)
    summary_csv = out_dir / "fig5_timing_summary.csv"
    summary.to_csv(summary_csv, index=False)

    # Where did interferon α/β land?
    ifn_rows = classification_df[
        classification_df["unit"].str.contains("INTERFERON_ALPHA", case=False, na=False)
    ].sort_values(["cohort", "lineage", "unit"])
    ifn_csv = out_dir / "fig5_interferon_rows.csv"
    ifn_rows.to_csv(ifn_csv, index=False)

    print(f"✓ fig5_pathway_timing — wrote:")
    print(f"    {pdf_path}")
    print(f"    {png_path}")
    print(f"    {classification_csv}")
    print(f"    {summary_csv}")
    print(f"    {ifn_csv}")
    print()
    print("Thresholds used:")
    print(f"    early_thr={early_thr}  late_thr={late_thr}  "
          f"null_eps={null_eps}  peak_dominance={peak_dominance}  "
          f"min_signal_eps={min_signal_eps}")
    print()
    print("Class distribution (one row per cohort × lineage):")
    print(summary.to_string(index=False))
    print()
    print("Interferon α/β rows (where it lands):")
    if len(ifn_rows) > 0:
        print(ifn_rows.to_string(index=False))
    else:
        print("    (none found — check unit naming)")

    return {
        "classification": classification_df,
        "summary": summary,
        "interferon_rows": ifn_rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Biological questions figures.")
    parser.add_argument(
        "--fig",
        choices=["fig5", "all"],
        default="fig5",
        help="Which figure to generate (default: fig5).",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--early-thr", type=float, default=0.5,
                        help="Ratio |z_MGUS|/|z_MM| above which a unit is early_rising.")
    parser.add_argument("--late-thr", type=float, default=0.3,
                        help="Ratio below which both |z_MGUS|/|z_MM| and |z_SMM|/|z_MM| "
                             "must fall for late_emerging.")
    parser.add_argument("--null-eps", type=float, default=0.15,
                        help="Values with |z| < null_eps are treated as sign-neutral.")
    parser.add_argument("--peak-dominance", type=float, default=1.2,
                        help="For early_peak class: |peak at MGUS or SMM| must be "
                             ">= peak_dominance * |z_MM|.")
    parser.add_argument("--min-signal-eps", type=float, default=0.3,
                        help="If max(|z|) < min_signal_eps, unit is classified as "
                             "low_signal (no meaningful activity anywhere).")
    args = parser.parse_args()

    if args.fig in ("fig5", "all"):
        fig5_pathway_timing(
            top_k=args.top_k,
            early_thr=args.early_thr,
            late_thr=args.late_thr,
            null_eps=args.null_eps,
            peak_dominance=args.peak_dominance,
            min_signal_eps=args.min_signal_eps,
        )


if __name__ == "__main__":
    main()
