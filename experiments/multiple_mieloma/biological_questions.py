"""
biological_questions.py 
"explore MM stages beyond cross-cohort agreement."

v3: adds `low_signal` class and tightens `peak_dominance` so noise pathways
    don't flood the real timing categories. See classify_timing() docstring.
v4: adds fig6 — donor-level heterogeneity (PCA + MM-shift score).
v5: adds fig7 — driver-gene attribution for defended units, from
    table2_drivers.csv. First-pass only; richer top-k version requires
    re-running the biology notebook with model-weight access.
v6: fig7 Panel B rebuilt as 8-dot scatter (defended units only — table2
    carries 4 per cohort), with per-dot labels and sides-agree highlight.
v7: fig7 Panel B replaced with horizontal grouped bar chart (encoder dark /
    decoder light bars per unit), eliminating label overlap.
v8: adds fig8 — CD4 vs CD8 lineage divergence (Q#5), classifying each
    unit's endpoint behavior as shared / lineage-specific / opposing.
v8.1: fig8 defended-tier labels get per-cohort × per-unit placement with
    thin leader lines so clustered labels don't overlap.

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


#Paths (relative to repo root)

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

#Order controls legend order; also sets which classes get columns in the summary.
CLASS_COLORS = {
    "early_peak":    "#9467BD",   # purple       argmax before MM, peak >> endpoint
    "early_rising":  "#D62728",   # red          argmax at MM, substantial at MGUS
    "progressive":   "#1F77B4",   # blue         argmax at MM, moderate at MGUS
    "late_emerging": "#2CA02C",   # green        argmax at MM, flat until MM
    "non_monotonic": "#808080",   # grey         sign flip outside null band
    "low_signal":    "#D8D8D8",   # pale grey    no meaningful activity anywhere
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


#Classification

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

    #Low-signal floor: no stage has meaningful activity
    if abs_vals.max() < min_signal_eps:
        return "low_signal"

    #Sign-flip check with null-band tolerance
    outside = vals[abs_vals >= null_eps]
    if len(outside) >= 2:
        signs_out = np.sign(outside)
        if not np.all(signs_out == signs_out[0]):
            return "non_monotonic"

    #argmax of magnitude
    peak_idx = int(np.argmax(abs_vals))  # 0=MGUS, 1=SMM, 2=MM
    peak_mag = float(abs_vals[peak_idx])
    mm_mag = abs(z_mm)

    #Early peak: argmax before MM and peak substantially larger than MM
    if peak_idx in (0, 1) and peak_mag >= mm_mag * peak_dominance:
        return "early_peak"

    #Otherwise, classify by how much of the MM endpoint is already there at MGUS
    r_mgus = abs(z_mgus) / mm_mag
    r_smm = abs(z_smm) / mm_mag

    if r_mgus >= early_thr:
        return "early_rising"
    if r_mgus < late_thr and r_smm < late_thr:
        return "late_emerging"
    return "progressive"


#Data loading

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
    #Prefer the shortest match (most specific, excludes compound names)
    return min(matches, key=len)


#fig5 — pathway timing

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

    #Load and classify every unit in every cohort × lineage
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

    #Figure
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

            #If interferon α/β isn't in top_k but exists, add it to the panel
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

    #Legend
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

    #Summary table — class counts per cohort × lineage
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

    #Where did interferon α/β land
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


#fig6 — donor-level heterogeneity


DONOR_SCORES_PATHS = {
    "Zavidij_2020": REPO_ROOT / "experiments" / "multiple_mieloma" / "outputs"
                    / "donor_test" / "Zavidij_2020" / "donor_scores.csv",
    "Foster_2026":  REPO_ROOT / "experiments" / "multiple_mieloma" / "outputs"
                    / "donor_test" / "Foster_2026" / "donor_scores.csv",
}

STAGE_ORDER_FULL = ["Non", "MGUS", "SMM", "MM"]
STAGE_X_INT = {"Non": 0, "MGUS": 1, "SMM": 2, "MM": 3}
STAGE_COLORS = {
    "Non":  "#2CA02C",   #green
    "MGUS": "#FF7F0E",   #orange
    "SMM":  "#D62728",   #red
    "MM":   "#17202A",   #near-black
}


def _compute_mm_shift(
    coords: np.ndarray,
    stages: np.ndarray,
    non_stage: str = "Non",
    mm_stage: str = "MM",
) -> np.ndarray:
    """Project each donor's PC coordinates onto the Non → MM centroid axis.

    Returns a 1-D array (one score per donor) where:
        Non centroid → 0
        MM centroid  → 1
    Values between show "fractional shift" of a donor's pathway profile from
    the Non centroid toward the MM centroid. NaN if either centroid is missing.
    """
    non_mask = stages == non_stage
    mm_mask = stages == mm_stage
    if non_mask.sum() == 0 or mm_mask.sum() == 0:
        return np.full(len(coords), np.nan)
    non_centroid = coords[non_mask].mean(axis=0)
    mm_centroid = coords[mm_mask].mean(axis=0)
    axis = mm_centroid - non_centroid
    axis_norm_sq = float(np.dot(axis, axis))
    if axis_norm_sq == 0:
        return np.full(len(coords), np.nan)
    return (coords - non_centroid) @ axis / axis_norm_sq


def fig6_donor_heterogeneity(
    out_dir: Path = OUT_DIR,
    estimator: str = "model",
    n_components: int = 3,
) -> dict:
    """Donor-level heterogeneity within MM precursor stages (Q#2).

    Builds two figures from donor_scores.csv (per-donor pathway activity):
      * fig6a_donor_pca   — 2x2 PCA scatter of donors per cohort × lineage.
      * fig6b_mm_shift    — 2x2 strip plot of per-donor MM-shift score.

    Writes CSVs:
      * donor_shift_scores.csv — per-donor shift score with PC1/PC2 and stage.
      * fig6_shift_summary.csv — mean±std of shift score per cohort×lineage×stage.
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "fig6 needs scikit-learn. Install with: pip install scikit-learn"
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_order = ["Zavidij_2020", "Foster_2026"]

    all_shift_records: list[dict] = []
    pca_data: dict = {}

    for cohort_name in cohort_order:
        csv_path = DONOR_SCORES_PATHS[cohort_name]
        df = pd.read_csv(csv_path)
        df = df[df["estimator"] == estimator]

        for lineage in LINEAGES:
            sub = df[df["lineage"] == lineage]
            if len(sub) == 0:
                continue

            #Donor × unit pivot
            M = sub.pivot_table(
                index="donor_id", columns="unit", values="score", aggfunc="first"
            )
            donors = M.index.values
            donor_stage_map = sub.groupby("donor_id")["stage"].first()
            stages = donor_stage_map.loc[donors].values

            #Drop zero-variance columns, then standardize
            X = M.fillna(0.0).values
            col_std = X.std(axis=0)
            X = X[:, col_std > 0]
            if X.shape[1] == 0:
                continue
            Xs = StandardScaler().fit_transform(X)

            #PCA — cap n_components by matrix shape
            k = min(n_components, Xs.shape[0] - 1, Xs.shape[1])
            pca = PCA(n_components=k)
            coords = pca.fit_transform(Xs)

            #MM shift score
            shift = _compute_mm_shift(coords, stages)

            #Record per-donor
            for idx, (donor, stage_val) in enumerate(zip(donors, stages)):
                all_shift_records.append({
                    "cohort": cohort_name,
                    "lineage": lineage,
                    "donor_id": donor,
                    "stage": stage_val,
                    "mm_shift": shift[idx],
                    "pc1": coords[idx, 0],
                    "pc2": coords[idx, 1] if coords.shape[1] > 1 else np.nan,
                    "pc3": coords[idx, 2] if coords.shape[1] > 2 else np.nan,
                })

            pca_data[(cohort_name, lineage)] = {
                "coords": coords,
                "donors": donors,
                "stages": stages,
                "var_ratio": pca.explained_variance_ratio_,
            }

    shift_df = pd.DataFrame(all_shift_records)
    shift_csv = out_dir / "donor_shift_scores.csv"
    shift_df.to_csv(shift_csv, index=False)

    #fig6a — PCA grid
    fig_a, axes_a = plt.subplots(2, 2, figsize=(13, 10))
    for i, cohort_name in enumerate(cohort_order):
        for j, lineage in enumerate(LINEAGES):
            ax = axes_a[i, j]
            key = (cohort_name, lineage)
            if key not in pca_data:
                ax.set_title(f"{cohort_name} — {lineage} (no data)")
                ax.axis("off")
                continue
            data = pca_data[key]
            coords = data["coords"]
            stages = data["stages"]
            var_ratio = data["var_ratio"]

            for stage in STAGE_ORDER_FULL:
                mask = stages == stage
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=STAGE_COLORS[stage], s=120, alpha=0.80,
                    label=f"{stage} (n={int(mask.sum())})",
                    edgecolor="white", linewidth=1.0,
                    zorder=3,
                )

            ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5, zorder=1)
            ax.axvline(0, color="gray", linewidth=0.4, alpha=0.5, zorder=1)
            ax.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}%)")
            ax.set_ylabel(
                f"PC2 ({var_ratio[1] * 100:.1f}%)" if len(var_ratio) > 1 else "PC2 (n/a)"
            )
            ax.set_title(f"{cohort_name} — {lineage}", fontsize=11)
            ax.legend(fontsize=9, loc="best", frameon=False)
            ax.grid(True, alpha=0.25)

    fig_a.suptitle(
        f"fig6a — Donor pathway-profile heterogeneity (PCA)\n"
        f"estimator = {estimator}. One dot = one donor.",
        fontsize=12, y=1.00,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_a.savefig(out_dir / "fig6a_donor_pca.pdf", bbox_inches="tight", dpi=200)
    fig_a.savefig(out_dir / "fig6a_donor_pca.png", bbox_inches="tight", dpi=200)
    plt.close(fig_a)

    #fig6b — MM-shift strip plot 
    fig_b, axes_b = plt.subplots(2, 2, figsize=(13, 10))
    rng = np.random.RandomState(42)
    for i, cohort_name in enumerate(cohort_order):
        for j, lineage in enumerate(LINEAGES):
            ax = axes_b[i, j]
            sub = shift_df[
                (shift_df["cohort"] == cohort_name)
                & (shift_df["lineage"] == lineage)
            ]
            if len(sub) == 0:
                ax.set_title(f"{cohort_name} — {lineage} (no data)")
                ax.axis("off")
                continue

            for stage in STAGE_ORDER_FULL:
                stage_sub = sub[sub["stage"] == stage]
                if len(stage_sub) == 0:
                    continue
                x = STAGE_X_INT[stage] + rng.uniform(-0.18, 0.18, size=len(stage_sub))
                ax.scatter(
                    x, stage_sub["mm_shift"],
                    c=STAGE_COLORS[stage], s=100, alpha=0.80,
                    edgecolor="white", linewidth=0.8,
                    label=f"{stage} (n={len(stage_sub)})",
                    zorder=3,
                )

            ax.axhline(0, color="gray", linewidth=0.6, linestyle=":", alpha=0.7)
            ax.axhline(1, color="gray", linewidth=0.6, linestyle=":", alpha=0.7)
            ax.axhline(0.5, color="red", linewidth=0.4, linestyle="--", alpha=0.4)
            ax.text(3.3, 0.0, "Non", color="gray", fontsize=9, va="center")
            ax.text(3.3, 1.0, "MM",  color="gray", fontsize=9, va="center")
            ax.text(3.3, 0.5, "half-shift", color="red", fontsize=8, va="center", alpha=0.7)

            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(STAGE_ORDER_FULL)
            ax.set_xlabel("Stage" if i == 1 else "")
            ax.set_ylabel(
                "MM-shift score" if j == 0 else ""
            )
            ax.set_title(f"{cohort_name} — {lineage}", fontsize=11)
            ax.set_xlim(-0.4, 3.9)
            ax.grid(True, alpha=0.25)

    fig_b.suptitle(
        "fig6b — Donor MM-shift score across stages\n"
        "Projection onto Non → MM centroid axis (PC space). "
        "0 = Non centroid, 1 = MM centroid.",
        fontsize=12, y=1.00,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_b.savefig(out_dir / "fig6b_mm_shift.pdf", bbox_inches="tight", dpi=200)
    fig_b.savefig(out_dir / "fig6b_mm_shift.png", bbox_inches="tight", dpi=200)
    plt.close(fig_b)

    #Summary table
    summ = (
        shift_df.groupby(["cohort", "lineage", "stage"])["mm_shift"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summ_csv = out_dir / "fig6_shift_summary.csv"
    summ.to_csv(summ_csv, index=False)

    #Flag MM-like precursor donors (MGUS/SMM with shift > 0.5)
    anomalies = shift_df[
        (shift_df["stage"].isin(["MGUS", "SMM"]))
        & (shift_df["mm_shift"] > 0.5)
    ].sort_values(["cohort", "lineage", "mm_shift"], ascending=[True, True, False])

    print("✓ fig6_donor_heterogeneity — wrote:")
    print(f"    {out_dir / 'fig6a_donor_pca.pdf'}")
    print(f"    {out_dir / 'fig6a_donor_pca.png'}")
    print(f"    {out_dir / 'fig6b_mm_shift.pdf'}")
    print(f"    {out_dir / 'fig6b_mm_shift.png'}")
    print(f"    {shift_csv}")
    print(f"    {summ_csv}")
    print()
    print(f"Settings: estimator={estimator}, n_components={n_components}")
    print()
    print("Shift-score summary (mean ± std per stage):")
    print(summ.to_string(index=False))
    print()
    print("MM-like precursor donors (MGUS/SMM with mm_shift > 0.5):")
    if len(anomalies) > 0:
        print(anomalies.to_string(index=False))
    else:
        print("    (none — no precursor donors crossed the half-shift threshold)")

    return {
        "shift_scores": shift_df,
        "pca_data": pca_data,
        "summary": summ,
        "anomalies": anomalies,
    }


# fig7 — driver-gene attribution for defended units 

BIOLOGY_TABLE2_PATHS = {
    "Zavidij_2020": REPO_ROOT / "experiments" / "multiple_mieloma" / "outputs"
                    / "biology" / "Zavidij_2020" / "table2_drivers.csv",
    "Foster_2026":  REPO_ROOT / "experiments" / "multiple_mieloma" / "outputs"
                    / "biology" / "Foster_2026" / "table2_drivers.csv",
}


#Short display names for defended-tier units
UNIT_SHORT_NAMES = {
    "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING": "IFN α/β",
    "REACTOME_MITOCHONDRIAL_TRANSLATION": "Mito. translation",
    "REACTOME_OAS_ANTIVIRAL_RESPONSE": "OAS antiviral",
    "REACTOME_CROSS_PRESENTATION_OF_SOLUBLE_EXOGENOUS_ANTIGENS_ENDOSOMES":
        "Cross-presentation",
    "REACTOME_CELL_SURFACE_INTERACTIONS_AT_THE_VASCULAR_WALL":
        "Vascular wall",
    "REACTOME_SLC_MEDIATED_TRANSMEMBRANE_TRANSPORT": "SLC transport",
    "REACTOME_FCERI_MEDIATED_MAPK_ACTIVATION": "FCERI MAPK",
}


def _short_unit(unit: str) -> str:
    """Return a short display name for a Reactome unit; fall back to a trim."""
    if unit in UNIT_SHORT_NAMES:
        return UNIT_SHORT_NAMES[unit]
    return unit.replace("REACTOME_", "").replace("_", " ").title()[:28]


def fig7_driver_attribution(
    out_dir: Path = OUT_DIR,
    highlight_unit: str = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING",
) -> dict:
    """fig7 — gene-level attribution for defended Reactome units.

    Uses table2_drivers.csv (top-1 driver per encoder/decoder side per unit).

    Panel A: Bar chart of highlight_unit's four top-driver observations
             (Zavidij-encoder, Zavidij-decoder, Foster-encoder, Foster-decoder).
             IFITM1 highlighted whenever it appears, since it's the cross-cohort
             driver.
    Panel B: Attribution concentration landscape — scatter of all units'
             encoder_top_share vs decoder_top_share, per cohort. The highlight
             unit is circled per cohort.

    Writes a cross-cohort driver concordance CSV alongside.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {name: pd.read_csv(p) for name, p in BIOLOGY_TABLE2_PATHS.items()}

    ifn_rows: dict = {}
    for cohort_name, df in tables.items():
        row = df[df["unit"] == highlight_unit]
        if len(row) > 0:
            ifn_rows[cohort_name] = row.iloc[0].to_dict()

    if not ifn_rows:
        raise RuntimeError(f"{highlight_unit} not found in any cohort table")

    cohort_order = ["Zavidij_2020", "Foster_2026"]
    cohort_dot_color = {
        "Zavidij_2020": "#4C72B0",  # blue
        "Foster_2026":  "#DD8452",  # orange
    }

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(17, 8),
        gridspec_kw={"width_ratios": [1.0, 1.5]},
    )

    #Panel A: bar chart of defended-unit drivers
    labels, shares, genes, bar_colors = [], [], [], []
    for cohort_name in cohort_order:
        if cohort_name not in ifn_rows:
            continue
        r = ifn_rows[cohort_name]
        for side in ("encoder", "decoder"):
            labels.append(f"{cohort_name.replace('_', ' ')}\n{side}")
            shares.append(r[f"{side}_top_share"])
            gene = r[f"{side}_top"]
            genes.append(gene)
            bar_colors.append("#D62728" if gene == "IFITM1" else "#808080")

    x_pos = np.arange(len(labels))
    bars = ax_a.bar(x_pos, shares, color=bar_colors,
                    edgecolor="black", linewidth=1.0)

    for bar, gene, share in zip(bars, genes, shares):
        bx = bar.get_x() + bar.get_width() / 2
        by = bar.get_height()
        ax_a.text(bx, by + 0.015, gene, ha="center", fontsize=11,
                  fontweight="bold",
                  color="#D62728" if gene == "IFITM1" else "black")
        ax_a.text(bx, by / 2, f"{share * 100:.1f}%", ha="center",
                  fontsize=10, color="white", fontweight="bold")

    even_share = next(iter(ifn_rows.values())).get("even_share")
    if even_share is not None:
        ax_a.axhline(even_share, color="gray", linewidth=0.8,
                     linestyle="--", alpha=0.6,
                     label=f"even-share baseline (1 / n_genes = {even_share:.3f})")

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(labels, fontsize=9)
    ax_a.set_ylabel("Top-driver share of unit attribution")
    unit_short = highlight_unit.replace("REACTOME_", "").replace("_", " ").title()
    ax_a.set_title(
        f"{unit_short}\nIFITM1 (red) is a top driver in both cohorts",
        fontsize=11,
    )
    ax_a.legend(fontsize=9, loc="upper left", frameon=False)
    ax_a.grid(True, axis="y", alpha=0.3)
    ax_a.set_ylim(0, max(shares) * 1.18 if shares else 1.0)

    #Panel B: horizontal grouped bar chart 
    #Collect all 8 unit-cohort combinations, sorted within cohort by endpoint z
    unit_rows: list[dict] = []
    for cohort_name in cohort_order:
        df = (
            tables[cohort_name]
            .sort_values("member_shift_z_endpoint", ascending=False)
        )
        for _, r in df.iterrows():
            unit_rows.append({
                "unit_short": _short_unit(r["unit"]),
                "cohort": cohort_name,
                "cohort_short": cohort_name.split("_")[0],
                "encoder_share": float(r["encoder_top_share"]),
                "encoder_top": r["encoder_top"],
                "decoder_share": float(r["decoder_top_share"]),
                "decoder_top": r["decoder_top"],
                "is_ifn": r["unit"] == highlight_unit,
                "sides_agree": bool(r["sides_agree"]),
            })

    n_rows = len(unit_rows)
    bar_h = 0.38

    for i, u in enumerate(unit_rows):
        y_center = n_rows - 1 - i  #top-down natural reading
        cc = cohort_dot_color[u["cohort"]]
        edge_c = "red" if u["is_ifn"] else "none"
        edge_lw = 1.8 if u["is_ifn"] else 0

        #Encoder (dark) — above center
        ax_b.barh(
            y_center + bar_h / 2, u["encoder_share"], bar_h,
            color=cc, alpha=0.95, edgecolor=edge_c, linewidth=edge_lw,
            zorder=3,
        )
        #Decoder (light) — below center
        ax_b.barh(
            y_center - bar_h / 2, u["decoder_share"], bar_h,
            color=cc, alpha=0.45, edgecolor=edge_c, linewidth=edge_lw,
            zorder=3,
        )

        #Gene labels at bar tips (encoder, then decoder)
        for yoff, side, gene, share in [
            (bar_h / 2, "enc", u["encoder_top"], u["encoder_share"]),
            (-bar_h / 2, "dec", u["decoder_top"], u["decoder_share"]),
        ]:
            is_ifitm1 = gene == "IFITM1"
            is_sides_agree_gene = (
                u["sides_agree"] and gene == u["encoder_top"]
            )
            text_color = (
                "#D62728" if is_ifitm1
                else ("#000000" if is_sides_agree_gene else "#333333")
            )
            fw = "bold" if (is_ifitm1 or is_sides_agree_gene) else "normal"
            ax_b.text(
                share + 0.012, y_center + yoff,
                f"{side}: {gene}  ({share * 100:.1f}%)",
                fontsize=8.5, va="center",
                color=text_color, fontweight=fw,
            )

    #Y-axis labels — unit (cohort)
    ax_b.set_yticks(list(range(n_rows - 1, -1, -1)))
    ax_b.set_yticklabels(
        [f"{u['unit_short']}  [{u['cohort_short']}]" for u in unit_rows],
        fontsize=9,
    )
    ax_b.set_ylim(-0.6, n_rows - 0.4)

    #Legend
    from matplotlib.lines import Line2D as _L2D
    from matplotlib.patches import Patch as _Patch
    legend_handles = [
        _Patch(facecolor=cohort_dot_color["Zavidij_2020"], alpha=0.95,
               label="Zavidij 2020"),
        _Patch(facecolor=cohort_dot_color["Foster_2026"], alpha=0.95,
               label="Foster 2026"),
        _Patch(facecolor="gray", alpha=0.95, label="encoder (top bar)"),
        _Patch(facecolor="gray", alpha=0.45, label="decoder (bottom bar)"),
        _L2D([0], [0], marker="s", color="w",
             markerfacecolor="w", markeredgecolor="red", markeredgewidth=1.8,
             markersize=11, label="IFN α/β (shared defended unit)"),
    ]
    ax_b.legend(
        handles=legend_handles, fontsize=8, loc="upper right",
        frameon=False, bbox_to_anchor=(1.0, 1.0),
    )

    ax_b.set_xlabel("Top-driver share of unit attribution")
    ax_b.set_title(
        "All 8 defended-tier driver observations\n"
        "IFITM1 (red) is the only gene shared across cohorts; "
        "PSMB9 is the only sides-agree unit",
        fontsize=11,
    )
    ax_b.set_xlim(0, 1.18)
    ax_b.grid(True, axis="x", alpha=0.3, zorder=1)
    ax_b.axvline(1 / 59, color="gray", linewidth=0.6, linestyle="--",
                 alpha=0.5, zorder=1)

    fig.suptitle(
        f"fig7 — Gene-level attribution of defended Reactome units",
        fontsize=12, y=1.00,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    pdf_path = out_dir / "fig7_driver_attribution.pdf"
    png_path = out_dir / "fig7_driver_attribution.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=200)
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    #Cross-cohort driver concordance
    merged = tables["Zavidij_2020"].merge(
        tables["Foster_2026"], on="unit", suffixes=("_zav", "_fos")
    )
    enc_concordant = merged["encoder_top_zav"] == merged["encoder_top_fos"]
    dec_concordant = merged["decoder_top_zav"] == merged["decoder_top_fos"]
    concordance = pd.DataFrame({
        "unit": merged["unit"],
        "encoder_top_zav": merged["encoder_top_zav"],
        "encoder_top_fos": merged["encoder_top_fos"],
        "encoder_concordant": enc_concordant,
        "decoder_top_zav": merged["decoder_top_zav"],
        "decoder_top_fos": merged["decoder_top_fos"],
        "decoder_concordant": dec_concordant,
        "any_side_shared_gene": (
            (merged["encoder_top_zav"] == merged["encoder_top_fos"])
            | (merged["encoder_top_zav"] == merged["decoder_top_fos"])
            | (merged["decoder_top_zav"] == merged["encoder_top_fos"])
            | (merged["decoder_top_zav"] == merged["decoder_top_fos"])
        ),
    })
    concordance_csv = out_dir / "fig7_cross_cohort_driver_concordance.csv"
    concordance.to_csv(concordance_csv, index=False)

    #Defended-tier set analysis (which units are shared vs cohort-specific)
    zav_units = set(tables["Zavidij_2020"]["unit"])
    fos_units = set(tables["Foster_2026"]["unit"])
    shared_units = zav_units & fos_units
    zav_only = zav_units - fos_units
    fos_only = fos_units - zav_units

    #Any gene appearing more than once across all 8 driver observations
    all_driver_genes = []
    for cohort_name, df in tables.items():
        for _, r in df.iterrows():
            all_driver_genes.append(
                (cohort_name, r["unit"], "encoder", r["encoder_top"])
            )
            all_driver_genes.append(
                (cohort_name, r["unit"], "decoder", r["decoder_top"])
            )
    gene_df = pd.DataFrame(
        all_driver_genes, columns=["cohort", "unit", "side", "gene"]
    )
    gene_freq = gene_df["gene"].value_counts()
    repeat_genes = gene_freq[gene_freq > 1]

    print("✓ fig7_driver_attribution — wrote:")
    print(f"    {pdf_path}")
    print(f"    {png_path}")
    print(f"    {concordance_csv}")
    print()
    print(f"Highlighted unit: {highlight_unit}")
    for cohort_name in cohort_order:
        if cohort_name not in ifn_rows:
            continue
        r = ifn_rows[cohort_name]
        print(f"  {cohort_name}:")
        print(f"    encoder_top = {r['encoder_top']} "
              f"({r['encoder_top_share'] * 100:.2f}%)")
        print(f"    decoder_top = {r['decoder_top']} "
              f"({r['decoder_top_share'] * 100:.2f}%)")
        print(f"    sides_agree = {r['sides_agree']}, "
              f"endpoint z = {r['member_shift_z_endpoint']:.2f}")
    print()

    print("Defended-tier set analysis:")
    print(f"    Zavidij: {len(zav_units)} units     Foster: {len(fos_units)} units")
    print(f"    Shared across cohorts: {len(shared_units)}")
    for u in sorted(shared_units):
        print(f"        - {u}")
    print(f"    Zavidij-only: {len(zav_only)}")
    for u in sorted(zav_only):
        print(f"        - {u}")
    print(f"    Foster-only:  {len(fos_only)}")
    for u in sorted(fos_only):
        print(f"        - {u}")
    print()
    print("Genes appearing as top-driver more than once across all 8 "
          "defended driver observations:")
    if len(repeat_genes) > 0:
        for gene, n in repeat_genes.items():
            rows = gene_df[gene_df["gene"] == gene]
            print(f"    {gene} (×{n}):")
            for _, g in rows.iterrows():
                print(f"        {g['cohort']}  {_short_unit(g['unit']):<24} "
                      f"{g['side']}")
    else:
        print("    (none — every defended driver observation is a different gene)")

    return {
        "ifn_rows": ifn_rows,
        "concordance": concordance,
        "shared_units": shared_units,
        "zav_only": zav_only,
        "fos_only": fos_only,
        "repeat_genes": repeat_genes,
        "tables": tables,
    }


# fig8 — CD4 vs CD8 lineage divergence

LINEAGE_CLASS_COLORS = {
    "shared_concordant": "#1F77B4",   # blue        both lineages respond
    "cd4_specific":      "#E377C2",   # pink        CD4 only
    "cd8_specific":      "#9467BD",   # purple      CD8 only
    "lineage_skewed":    "#FFBB33",   # amber       both respond, one dominant
    "opposing":          "#D62728",   # red         opposite signs, both significant
    "low_signal":        "#D8D8D8",   # pale grey   neither lineage significant
}

LINEAGE_CLASS_LABEL = {
    "shared_concordant": "Shared",
    "cd4_specific":      "CD4-specific",
    "cd8_specific":      "CD8-specific",
    "lineage_skewed":    "Lineage-skewed",
    "opposing":          "Opposing",
    "low_signal":        "Low-signal",
}

#Defended-tier units per cohort (from fig7 / table2_drivers)
DEFENDED_UNITS = {
    "Zavidij_2020": [
        "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING",
        "REACTOME_MITOCHONDRIAL_TRANSLATION",
        "REACTOME_OAS_ANTIVIRAL_RESPONSE",
        "REACTOME_CROSS_PRESENTATION_OF_SOLUBLE_EXOGENOUS_ANTIGENS_ENDOSOMES",
    ],
    "Foster_2026": [
        "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING",
        "REACTOME_CELL_SURFACE_INTERACTIONS_AT_THE_VASCULAR_WALL",
        "REACTOME_SLC_MEDIATED_TRANSMEMBRANE_TRANSPORT",
        "REACTOME_FCERI_MEDIATED_MAPK_ACTIVATION",
    ],
}


#Per-cohort × per-unit label placement for fig8.
#Offset is in POINTS from the dot; ha/va controls text anchor.
FIG8_LABEL_PLACEMENT = {
    "Zavidij_2020": {
        "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING":
            {"offset": (0, 22), "ha": "center", "va": "bottom"},
        "REACTOME_MITOCHONDRIAL_TRANSLATION":
            {"offset": (26, 10), "ha": "left", "va": "center"},
        "REACTOME_OAS_ANTIVIRAL_RESPONSE":
            {"offset": (28, -18), "ha": "left", "va": "top"},
        "REACTOME_CROSS_PRESENTATION_OF_SOLUBLE_EXOGENOUS_ANTIGENS_ENDOSOMES":
            {"offset": (-20, 24), "ha": "right", "va": "bottom"},
    },
    "Foster_2026": {
        "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING":
            {"offset": (10, 24), "ha": "left", "va": "bottom"},
        "REACTOME_CELL_SURFACE_INTERACTIONS_AT_THE_VASCULAR_WALL":
            {"offset": (28, -16), "ha": "left", "va": "top"},
        "REACTOME_FCERI_MEDIATED_MAPK_ACTIVATION":
            {"offset": (-22, -24), "ha": "right", "va": "top"},
        "REACTOME_SLC_MEDIATED_TRANSMEMBRANE_TRANSPORT":
            {"offset": (26, 0), "ha": "left", "va": "center"},
    },
}


def classify_lineage(
    z_cd4: float,
    z_cd8: float,
    min_signal_eps: float = 0.3,
    ratio_shared: float = 2.0,
) -> str:
    """Classify a unit's CD4/CD8 endpoint-contrast behavior.

    Classes
    -------
    * low_signal        : max(|z|) < min_signal_eps (filtered out)
    * cd4_specific      : |z_CD4| >= eps, |z_CD8| < eps
    * cd8_specific      : |z_CD8| >= eps, |z_CD4| < eps
    * opposing          : both >= eps, opposite signs
    * shared_concordant : both >= eps, same sign, magnitude ratio <= ratio_shared
    * lineage_skewed    : both >= eps, same sign, magnitude ratio > ratio_shared
    """
    if not (np.isfinite(z_cd4) and np.isfinite(z_cd8)):
        return "low_signal"

    a_cd4 = abs(z_cd4)
    a_cd8 = abs(z_cd8)

    if max(a_cd4, a_cd8) < min_signal_eps:
        return "low_signal"

    cd4_sig = a_cd4 >= min_signal_eps
    cd8_sig = a_cd8 >= min_signal_eps

    if cd4_sig and not cd8_sig:
        return "cd4_specific"
    if cd8_sig and not cd4_sig:
        return "cd8_specific"

    #Both above threshold
    if np.sign(z_cd4) != np.sign(z_cd8):
        return "opposing"

    ratio = max(a_cd4, a_cd8) / max(min(a_cd4, a_cd8), 1e-9)
    if ratio <= ratio_shared:
        return "shared_concordant"
    return "lineage_skewed"


def _load_cohort_endpoint_by_lineage(csv_path: Path) -> pd.DataFrame:
    """Load unit_contrasts.csv and pivot to one row per unit with CD4/CD8 endpoint z.

    Returns columns: unit, z_cd4, z_cd8
    """
    df = pd.read_csv(csv_path)
    df = df[(df["contrast"] == "Non->MM") & (df["lineage"].isin(["CD4", "CD8"]))]
    pivot = df.pivot_table(
        index="unit", columns="lineage", values="member_shift_z", aggfunc="first"
    ).reset_index()
    pivot = pivot.rename(columns={"CD4": "z_cd4", "CD8": "z_cd8"})
    for c in ["z_cd4", "z_cd8"]:
        if c not in pivot.columns:
            pivot[c] = np.nan
    return pivot[["unit", "z_cd4", "z_cd8"]]


def fig8_lineage_divergence(
    out_dir: Path = OUT_DIR,
    min_signal_eps: float = 0.3,
    ratio_shared: float = 2.0,
) -> dict:
    """fig8 — CD4 vs CD8 endpoint divergence per Reactome unit.

    Two scatter panels (Zavidij, Foster). Each dot = one unit, colored by
    its lineage classification. IFN α/β highlighted with a red circle;
    other defended-tier units with a black circle.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_order = ["Zavidij_2020", "Foster_2026"]

    cohort_paths = {
        "Zavidij_2020": BIOLOGY_DIR / "Zavidij_2020" / "unit_contrasts.csv",
        "Foster_2026":  BIOLOGY_DIR / "Foster_2026" / "unit_contrasts.csv",
    }

    all_records: list[dict] = []
    for cohort_name in cohort_order:
        df = _load_cohort_endpoint_by_lineage(cohort_paths[cohort_name])
        for _, r in df.iterrows():
            cls = classify_lineage(
                r["z_cd4"], r["z_cd8"],
                min_signal_eps=min_signal_eps, ratio_shared=ratio_shared,
            )
            all_records.append({
                "cohort": cohort_name,
                "unit": r["unit"],
                "z_cd4": r["z_cd4"],
                "z_cd8": r["z_cd8"],
                "lineage_class": cls,
                "is_ifn": "INTERFERON_ALPHA_BETA_SIGNALING" in r["unit"],
                "is_defended": r["unit"] in DEFENDED_UNITS[cohort_name],
            })

    class_df = pd.DataFrame(all_records)
    class_csv = out_dir / "lineage_classification.csv"
    class_df.to_csv(class_csv, index=False)

    #Figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

    for j, cohort_name in enumerate(cohort_order):
        ax = axes[j]
        sub = class_df[class_df["cohort"] == cohort_name]

        #Reference lines: diagonal, signal floors (±min_signal_eps)
        lim = max(sub["z_cd4"].abs().max(), sub["z_cd8"].abs().max()) * 1.08
        lim = max(lim, 1.0)
        ax.plot([-lim, lim], [-lim, lim], color="gray", linewidth=0.6,
                linestyle=":", alpha=0.6, zorder=1)
        ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5, zorder=1)
        ax.axvline(0, color="gray", linewidth=0.4, alpha=0.5, zorder=1)
        for v in (-min_signal_eps, min_signal_eps):
            ax.axhline(v, color="lightgray", linewidth=0.5,
                       linestyle="--", alpha=0.6, zorder=1)
            ax.axvline(v, color="lightgray", linewidth=0.5,
                       linestyle="--", alpha=0.6, zorder=1)

        #Plot each class
        for cls, color in LINEAGE_CLASS_COLORS.items():
            cls_sub = sub[sub["lineage_class"] == cls]
            if len(cls_sub) == 0:
                continue
            #Skip low_signal in main view (they just clutter) — show as pale background
            alpha = 0.20 if cls == "low_signal" else 0.75
            size = 10 if cls == "low_signal" else 30
            ax.scatter(
                cls_sub["z_cd4"], cls_sub["z_cd8"],
                c=color, s=size, alpha=alpha,
                edgecolor="none",
                label=f"{LINEAGE_CLASS_LABEL[cls]} (n={len(cls_sub)})",
                zorder=3 if cls != "low_signal" else 2,
            )

        #Highlight defended-tier units
        defended_sub = sub[sub["is_defended"]]
        placements = FIG8_LABEL_PLACEMENT.get(cohort_name, {})
        for _, r in defended_sub.iterrows():
            is_ifn = r["is_ifn"]
            ring_color = "red" if is_ifn else "black"
            ring_lw = 2.2 if is_ifn else 1.3
            ax.scatter(
                [r["z_cd4"]], [r["z_cd8"]],
                s=180, facecolor="none",
                edgecolor=ring_color, linewidth=ring_lw,
                zorder=10,
            )
            #Label with per-unit placement + thin leader line
            label = _short_unit(r["unit"])
            p = placements.get(
                r["unit"],
                {"offset": (10, 10), "ha": "left", "va": "bottom"},
            )
            ax.annotate(
                label,
                xy=(r["z_cd4"], r["z_cd8"]),
                xytext=p["offset"], textcoords="offset points",
                fontsize=8.5, fontweight="bold" if is_ifn else "normal",
                color=ring_color,
                ha=p["ha"], va=p["va"],
                arrowprops=dict(
                    arrowstyle="-",
                    color=ring_color,
                    lw=0.6,
                    alpha=0.55,
                    shrinkA=0, shrinkB=8,
                ),
                zorder=11,
            )

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("CD4 member_shift_z (Non → MM)")
        ax.set_ylabel("CD8 member_shift_z (Non → MM)")
        ax.set_title(f"{cohort_name}", fontsize=11)
        ax.legend(fontsize=8, loc="best", frameon=False)
        ax.grid(True, alpha=0.25, zorder=0)

    fig.suptitle(
        "fig8 — CD4 vs CD8 lineage divergence per Reactome unit\n"
        f"min_signal_eps = {min_signal_eps}, ratio_shared = {ratio_shared}x. "
        "Red ring: IFN α/β. Black ring: other defended-tier units.",
        fontsize=12, y=1.00,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    pdf_path = out_dir / "fig8_lineage_divergence.pdf"
    png_path = out_dir / "fig8_lineage_divergence.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=200)
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    #Summary table
    summary = (
        class_df.groupby(["cohort", "lineage_class"])
        .size().reset_index(name="n_units")
        .pivot_table(
            index="cohort", columns="lineage_class", values="n_units",
            fill_value=0,
        )
        .reset_index()
    )
    for cls in LINEAGE_CLASS_COLORS:
        if cls not in summary.columns:
            summary[cls] = 0
    summary = summary[["cohort"] + list(LINEAGE_CLASS_COLORS)]
    summary["total"] = summary[list(LINEAGE_CLASS_COLORS)].sum(axis=1)
    summary_csv = out_dir / "fig8_lineage_summary.csv"
    summary.to_csv(summary_csv, index=False)

    #Defended-tier lineage classifications
    defended_detail = class_df[class_df["is_defended"]].sort_values(
        ["cohort", "unit"]
    )[["cohort", "unit", "z_cd4", "z_cd8", "lineage_class"]]

    print(f"✓ fig8_lineage_divergence — wrote:")
    print(f"    {pdf_path}")
    print(f"    {png_path}")
    print(f"    {class_csv}")
    print(f"    {summary_csv}")
    print()
    print(f"Thresholds: min_signal_eps={min_signal_eps}, "
          f"ratio_shared={ratio_shared}x")
    print()
    print("Class distribution:")
    print(summary.to_string(index=False))
    print()
    print("Defended-tier units' lineage classification:")
    print(defended_detail.to_string(index=False))

    return {
        "classifications": class_df,
        "summary": summary,
        "defended_detail": defended_detail,
    }


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="Biological questions figures.")
    parser.add_argument(
        "--fig",
        choices=["fig5", "fig6", "fig7", "fig8", "all"],
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
    parser.add_argument("--estimator", type=str, default="model",
                        help="(fig6) estimator column value to filter donor_scores.csv "
                             "(default: model).")
    parser.add_argument("--n-components", type=int, default=3,
                        help="(fig6) number of PCA components to keep (default: 3).")
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
    if args.fig in ("fig6", "all"):
        fig6_donor_heterogeneity(
            estimator=args.estimator,
            n_components=args.n_components,
        )
    if args.fig in ("fig7", "all"):
        fig7_driver_attribution()
    if args.fig in ("fig8", "all"):
        fig8_lineage_divergence(
            min_signal_eps=args.min_signal_eps,
        )


if __name__ == "__main__":
    main()
