#!/usr/bin/env python
"""Thesis-side extra figures, assembled from the pipeline's existing outputs.

    python thesis_figures.py              # all figures
    python thesis_figures.py --only 1     # just the first figure

These figures sit alongside three paper figures (`04_figures.py`) but
answer different questions, aimed at a thesis reader rather than a paper
reviewer. 

    fig1_reactome_hierarchy    the Reactome nesting of the defended finding,
                               as a nested-box diagram, plus a panoramic view
                               of the top-20 units grouped by their family
    fig2_pseudobulk_vs_pyvae   the two estimators' per-donor trend statistics
                               on one scatter per cohort x lineage
    fig3_donor_scores_top10    per-donor dot plots across stages for the
                               top-10 endpoint units (union, donor-scored)
    fig4_donor_pvalue_qq       calibration of the donor-level trend test
                               against a uniform null

Pure assembly, same as 04_figures.py:
- reads CSVs and JSONs that the Snakemake grid wrote under `outputs/`
- does not import pyvae, torch, anndata or myeloma.py
- writes to `outputs/thesis_figures/`

Palette and short() match Carlos's conventions in myeloma.py so the thesis
figures sit next to the paper figures without a style clash.

    cohort colours   Zavidij_2020 = blue  (#2a78d6)
                     Foster_2026  = orange (#eb6834)
    accent           C_INK  = #0b0b0b
                     C_GREY = #8a8a86
                     C_FAINT = #dcdcd8
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Constants and paths

C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"
C_GREY, C_FAINT, C_INK = "#8a8a86", "#dcdcd8", "#0b0b0b"
STUDY_COLOURS = {"Zavidij_2020": C_BLUE, "Foster_2026": C_ORANGE}
STUDIES = ["Zavidij_2020", "Foster_2026"]
MAJOR_LINEAGES = ["CD4", "CD8"]
ALPHA = 0.05

HERE = Path(__file__).resolve().parent
OUT_BASE = HERE / "outputs"
OUT_DIR = OUT_BASE / "thesis_figures"
GMT_PATH = HERE / "resources" / "c2.cp.reactome.v7.5.1.symbols.gmt"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.5,
        "font.size": 9, "legend.frameon": False,
    })


def short(name: str) -> str:
    for p in ("REACTOME_", "HALLMARK_"):
        name = name.removeprefix(p)
    return name.replace("_", " ").lower()


# Shorter display forms for a few verbose Reactome names that otherwise
# overflow their panels. Only used for display; the raw name is preserved
# in every file the pipeline writes.
PRETTY = {
    "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING":
        "interferon α/β signaling",
    "REACTOME_INTERFERON_GAMMA_SIGNALING":
        "interferon γ signaling",
    "REACTOME_DDX58_IFIH1_MEDIATED_INDUCTION_OF_INTERFERON_ALPHA_BETA":
        "DDX58 / IFIH1-mediated IFN α/β induction",
    "REACTOME_INTERFERON_SIGNALING":
        "interferon signaling",
    "REACTOME_CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM":
        "cytokine signaling in immune system",
    "REACTOME_DISEASES_OF_SIGNAL_TRANSDUCTION_BY_GROWTH_FACTOR_RECEPTORS_AND_SECOND_MESSENGERS":
        "diseases of signal transduction (GF receptors)",
    "REACTOME_INTRACELLULAR_SIGNALING_BY_SECOND_MESSENGERS":
        "intracellular signaling (second messengers)",
    "REACTOME_SIGNALING_BY_RHO_GTPASES_MIRO_GTPASES_AND_RHOBTB3":
        "signaling by Rho GTPases / Miro / RHOBTB3",
    "REACTOME_FACTORS_INVOLVED_IN_MEGAKARYOCYTE_DEVELOPMENT_AND_PLATELET_PRODUCTION":
        "megakaryocyte development & platelet production",
    "REACTOME_CROSS_PRESENTATION_OF_SOLUBLE_EXOGENOUS_ANTIGENS_ENDOSOMES":
        "cross-presentation of soluble antigens (endosomes)",
    "REACTOME_ANTIGEN_PROCESSING_UBIQUITINATION_PROTEASOME_DEGRADATION":
        "antigen processing: ubiquitination + proteasome",
    "REACTOME_HSP90_CHAPERONE_CYCLE_FOR_STEROID_HORMONE_RECEPTORS_SHR_IN_THE_PRESENCE_OF_LIGAND":
        "HSP90 chaperone cycle for SHR (with ligand)",
    "REACTOME_TRANSCRIPTIONAL_REGULATION_OF_WHITE_ADIPOCYTE_DIFFERENTIATION":
        "white adipocyte differentiation (transcription)",
    "REACTOME_CASPASE_ACTIVATION_VIA_DEATH_RECEPTORS_IN_THE_PRESENCE_OF_LIGAND":
        "caspase activation via death receptors",
    "REACTOME_ASSOCIATION_OF_TRIC_CCT_WITH_TARGET_PROTEINS_DURING_BIOSYNTHESIS":
        "TRiC/CCT association during protein synthesis",
    "REACTOME_MATURATION_OF_SARS_COV_1_NUCLEOPROTEIN":
        "SARS-CoV-1 nucleoprotein maturation",
    "REACTOME_PROCESSING_OF_CAPPED_INTRON_CONTAINING_PRE_MRNA":
        "pre-mRNA processing (capped, intron-containing)",
}


def pretty(name: str) -> str:
    """Reader-friendly display name. Falls back to short() if no override."""
    return PRETTY.get(name, short(name))



# Shared loaders

def load_gmt(path: Path) -> dict[str, set[str]]:
    """Parse the Reactome GMT into {pathway_name: set(gene_symbols)}."""
    sets = {}
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            sets[parts[0]] = set(parts[2:])
    return sets


def load_endpoint_ranks(studies: list[str]) -> dict[str, dict[str, int]]:
    """Per-study: {unit -> rank by endpoint mean over major lineages}."""
    ranks = {}
    for study in studies:
        contrasts = pd.read_csv(OUT_BASE / "biology" / study / "unit_contrasts.csv")
        summary = json.loads(
            (OUT_BASE / "biology" / study / "summary.json").read_text())
        endpoint = summary["endpoint_contrast"]
        wide = (contrasts[contrasts["contrast"] == endpoint]
                .pivot(index="unit", columns="lineage",
                       values="member_shift_z"))
        lins = [l for l in MAJOR_LINEAGES if l in wide.columns]
        endpoint_mean = wide[lins].mean(axis=1)
        ranks[study] = endpoint_mean.rank(ascending=False).astype(int).to_dict()
    return ranks


# fig1: Reactome hierarchy — focused + panoramic

def _parent_of(unit: str, sets: dict[str, set[str]]) -> str | None:
    """Smallest set in the collection that strictly contains this unit.

    "Strictly" = unit's genes are a proper subset. Returns None if nothing
    larger contains it (i.e. this unit is at the top of its lineage within
    the C2:CP:REACTOME collection).
    """
    g = sets[unit]
    parents = [(name, s) for name, s in sets.items()
               if len(s) > len(g) and g.issubset(s)]
    if not parents:
        return None
    return min(parents, key=lambda ns: len(ns[1]))[0]


def _root_family(unit: str, sets: dict[str, set[str]]) -> str:
    """Walk up until no larger containing set exists; return that set's name."""
    cur = unit
    seen = {cur}
    while True:
        p = _parent_of(cur, sets)
        if p is None or p in seen:
            return cur
        cur = p
        seen.add(cur)


def _draw_nested_box(ax, x: float, y: float, w: float, h: float, *,
                     label: str, member_count: int, ranks: dict[str, int],
                     face: str, edge: str, lw: float = 1.0,
                     label_fontsize: float = 8.5, pad: float = 0.03,
                     stats_anchor: str = "inside_bottom",
                     wrap_chars: int = 32,
                     short_stats: bool = False) -> None:
    """One Reactome set drawn as a labelled rectangle, with per-study ranks.

    stats_anchor:
      "inside_bottom"      — stats stacked on two lines at the bottom INSIDE
                             the box; good for leaves with room to spare.
      "same_row_right"     — stats on the SAME row as the label, right-
                             aligned inside the top of the box; good for
                             parent boxes with a nested child below.
    """
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=0.02",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=2)
    ax.add_patch(rect)
    display = pretty(label)
    name_lines = textwrap.wrap(display, wrap_chars)[:3]
    # Label at top-left of the box.
    ax.text(x + 0.015, y + h - 0.015, "\n".join(name_lines),
            fontsize=label_fontsize, fontweight="bold", color=C_INK,
            ha="left", va="top", zorder=3)
    # Per-cohort ranks in one short string.
    # Short form uses first-letter abbreviation ("Z #1484 · F #260")
    # to fit inside narrow leaf boxes; full form ("Zavidij #581 · Foster #513")
    # is used in the roomier parent boxes.
    def _fmt(study: str) -> str:
        key = study.split("_")[0][0] if short_stats else study.split("_")[0]
        return f"{key} #{ranks.get(study, 'n/a')}"
    rank_bits = "  ·  ".join(_fmt(s) for s in STUDIES)
    if stats_anchor == "inside_bottom":
        # Two lines, bottom-left, so the longest cohort rank fits inside
        # even the narrow leaf boxes.
        ax.text(x + 0.015, y + 0.015,
                f"{member_count} genes\n{rank_bits}",
                fontsize=label_fontsize - 1.6, color=C_INK,
                ha="left", va="bottom", zorder=3, linespacing=1.15)
    else:  # stacked_top — stats on the row immediately below the label
        # Count how many lines the label took so stats sits below it, not
        # on top of its last line.
        n_label_lines = len(name_lines)
        line_dy = 0.028   # rough display spacing per label line in data coords
        ax.text(x + 0.015, y + h - 0.015 - n_label_lines * line_dy,
                f"{member_count} genes  ·  {rank_bits}",
                fontsize=label_fontsize - 1.5, color=C_GREY,
                ha="left", va="top", zorder=3)


def draw_focused_interferon(ax, sets: dict[str, set[str]],
                            ranks: dict[str, dict[str, int]]) -> None:
    """The interferon family drawn as physically nested boxes.

    cytokine signaling in immune system   (root family)
    └── interferon signaling              (middle)
         ├── interferon alpha beta signaling   (defended)
         ├── interferon gamma signaling
         └── ddx58 ifih1 mediated ...
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # Pathway identifiers in the GMT.
    cytokine = "REACTOME_CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM"
    ifn      = "REACTOME_INTERFERON_SIGNALING"
    ifn_ab   = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING"
    ifn_g    = "REACTOME_INTERFERON_GAMMA_SIGNALING"
    ifn_ddx  = "REACTOME_DDX58_IFIH1_MEDIATED_INDUCTION_OF_INTERFERON_ALPHA_BETA"

    # Rank lookup helper that handles missing units.
    def ranks_for(name):
        return {s: ranks[s].get(name, None) for s in STUDIES}

    # Root: cytokine signaling in immune system (big outer frame)
    _draw_nested_box(ax, x=0.02, y=0.06, w=0.96, h=0.84,
                     label=cytokine, member_count=len(sets[cytokine]),
                     ranks=ranks_for(cytokine),
                     face="#f7f5f0", edge=C_GREY, lw=1.2,
                     stats_anchor="stacked_top", wrap_chars=42,
                     label_fontsize=9)

    # Mid: interferon signaling (nested in the root; children below)
    _draw_nested_box(ax, x=0.06, y=0.14, w=0.88, h=0.64,
                     label=ifn, member_count=len(sets[ifn]),
                     ranks=ranks_for(ifn),
                     face="#eceae2", edge=C_GREY, lw=1.0,
                     stats_anchor="stacked_top", wrap_chars=42,
                     label_fontsize=8.5)

    # Three leaves of interferon signaling, laid side by side.
    leaf_y, leaf_h = 0.20, 0.42
    leaf_w = 0.275
    gap = 0.018
    xs = [0.08, 0.08 + leaf_w + gap, 0.08 + 2 * (leaf_w + gap)]
    for xi, name, is_defended in zip(xs, [ifn_ab, ifn_g, ifn_ddx],
                                      [True, False, False]):
        _draw_nested_box(
            ax, x=xi, y=leaf_y, w=leaf_w, h=leaf_h,
            label=name, member_count=len(sets[name]),
            ranks=ranks_for(name),
            face=C_ORANGE if is_defended else "white",
            edge=C_INK if is_defended else C_GREY,
            lw=1.6 if is_defended else 0.8,
            label_fontsize=7.4, wrap_chars=18,
            stats_anchor="inside_bottom", short_stats=True)
    # One-line legend telling the reader what Z/F mean in the leaves.
    ax.text(0.5, 0.10, "Z = Zavidij_2020   ·   F = Foster_2026",
            fontsize=6.6, color=C_GREY, ha="center", va="center",
            style="italic")

    ax.set_title(
        "a — the defended finding, in its Reactome context\n"
        "nested boxes = strict gene-set containment; the orange leaf is "
        "the defended finding (rank 1 in both cohorts)",
        fontsize=9.2, loc="left", pad=6)


def draw_panoramic_top20(ax, sets: dict[str, set[str]],
                         ranks: dict[str, dict[str, int]]) -> None:
    """Top-20 units from each cohort grouped by their root Reactome family.

    Shows how a count of "20 units" collapses into a smaller number of
    biological families, which is the §3.6.1 point visualised.
    """
    # Union of each cohort's top 20.
    top = set()
    for study in STUDIES:
        top |= set(sorted(ranks[study],
                          key=lambda u: ranks[study][u])[:20])
    top = sorted(top)
    # Keep only units present in the GMT (the biology outputs rank the
    # pyvae units; essentially all are Reactome, but defensive).
    top = [u for u in top if u in sets]

    # Group by root family.
    family_of = {u: _root_family(u, sets) for u in top}
    families = {}
    for u in top:
        families.setdefault(family_of[u], []).append(u)
    # Order families by how many top units they contain (then by family size).
    family_order = sorted(families.keys(),
                          key=lambda f: (-len(families[f]), -len(sets[f])))

    # One row per unit, grouped visually by family. x = cohort rank (log).
    # When a family contains exactly one unit AND that unit IS the family
    # itself (a top-level Reactome set in the collection), collapse to one
    # row — otherwise the same name appears twice stacked on top of itself.
    rows = []
    for fam in family_order:
        members = sorted(families[fam],
                         key=lambda u: min(ranks[s].get(u, 10**6)
                                           for s in STUDIES))
        if len(members) == 1 and members[0] == fam:
            rows.append(("standalone", fam, None))
            continue
        rows.append(("family", fam, len(members)))
        for u in members:
            rows.append(("unit", u, None))

    def _trunc(s: str, w: int = 60) -> str:
        return s if len(s) <= w else s[: w - 1] + "…"

    # Labels sit in the LEFT MARGIN (outside the plot area) using the
    # blended transform y-in-data, x-in-axes-fraction. The dots stay inside
    # the plot area so the two never collide. clip_on=False lets the label
    # text extend into the figure's left margin.
    yax = ax.get_yaxis_transform()
    yy = np.arange(len(rows))[::-1]

    for i, (kind, name, extra) in enumerate(rows):
        y = yy[i]
        if kind == "family":
            fam_name = _trunc(pretty(name), 46)
            # Family header: subtle horizontal rule the full width of the
            # plot, plus a bold label in the left margin.
            ax.axhline(y, color=C_GREY, lw=0.5, alpha=0.5, zorder=1)
            ax.text(-0.015, y, f"{fam_name}  "
                              f"({extra} unit{'s' if extra != 1 else ''})",
                    transform=yax, fontsize=7.6, fontweight="bold",
                    color=C_INK, ha="right", va="center", clip_on=False)
        elif kind == "standalone":
            # Single-unit "self-family" row — bold label, dots on same line.
            ax.axhline(y, color=C_GREY, lw=0.5, alpha=0.5, zorder=1)
            for study in STUDIES:
                r = ranks[study].get(name)
                if r is None:
                    continue
                ax.scatter(r, y, s=22,
                           c=STUDY_COLOURS[study],
                           edgecolors=C_INK, linewidths=0.3, zorder=4)
            ax.text(-0.015, y, _trunc(pretty(name), 44) + "  (standalone)",
                    transform=yax, fontsize=7.0, fontweight="bold",
                    color=C_INK, ha="right", va="center", clip_on=False)
        else:
            # Regular unit — dots in plot, plain label in margin, indented
            # so it reads as a child of the family header above.
            for study in STUDIES:
                r = ranks[study].get(name)
                if r is None:
                    continue
                ax.scatter(r, y, s=18,
                           c=STUDY_COLOURS[study],
                           edgecolors=C_INK, linewidths=0.3, zorder=4)
            ax.text(-0.015, y, _trunc(pretty(name), 48),
                    transform=yax, fontsize=6.6, color=C_INK,
                    ha="right", va="center", clip_on=False)

    ax.set_xscale("log")
    ax.set_xlim(1, max(len(ranks[s]) for s in STUDIES) * 1.2)
    ax.set_ylim(-1, len(rows))
    ax.set_yticks([])
    ax.set_xlabel("rank among all units (log, 1 = best)", fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True, which="both", alpha=0.25, lw=0.4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    # Cohort legend placed just below the plot title, inside the plot area
    # on the right where no family headers reach.
    handles = [Line2D([], [], marker="o", ls="", color=STUDY_COLOURS[s],
                      markeredgecolor=C_INK, markersize=5,
                      label=s.split("_")[0]) for s in STUDIES]
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              bbox_to_anchor=(1.0, 1.02), framealpha=0.0)

    ax.set_title(
        "b — top-20 units from each cohort, grouped by Reactome family\n"
        "many units collapse into a few families — the thesis's "
        "'families, not rows' rule in one view",
        fontsize=9, loc="left", pad=6)


def fig1_reactome_hierarchy() -> None:
    sets = load_gmt(GMT_PATH)
    ranks = load_endpoint_ranks(STUDIES)

    # Two panels:
    #   a — nested boxes for the interferon family (centred square axis)
    #   b — top-20 of each cohort grouped by Reactome family (wide axis
    #       with a generous left margin for the unit labels)
    fig = plt.figure(figsize=(9.0, 13.5))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[0.95, 1.65], hspace=0.28,
        left=0.04, right=0.97, top=0.95, bottom=0.03)

    draw_focused_interferon(fig.add_subplot(gs[0]), sets, ranks)
    ax_b = fig.add_subplot(gs[1])
    draw_panoramic_top20(ax_b, sets, ranks)
    # Shift panel b's left edge inward so the ~40%-wide label margin
    # (populated by family headers and unit names via get_yaxis_transform)
    # sits inside the figure, not off its left edge.
    pos = ax_b.get_position()
    ax_b.set_position([0.42, pos.y0, 0.55, pos.height])

    fig.suptitle(
        "Reactome nesting of the defended finding",
        fontsize=11.5, fontweight="bold", y=0.985)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fig1_reactome_hierarchy"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  {out.name}.pdf / .png")


# fig2: pseudobulk vs pyvae donor-level trend — four panels (cohort x lineage)

INTERFERON_UNIT = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING"


def _load_donor_test(study: str) -> pd.DataFrame:
    """Donor-level test table for one cohort."""
    return pd.read_csv(OUT_BASE / "donor_test" / study / "donor_test.csv")


def _paired_rho(df: pd.DataFrame, lineage: str) -> pd.DataFrame:
    """Pivot donor_test.csv into (unit, rho_model, rho_pseudobulk,
    q_model, q_pseudobulk) for one lineage, inner-joined on unit."""
    sub = df[df["lineage"] == lineage]
    model = sub[sub["estimator"] == "model"].set_index("unit")
    pseudo = sub[sub["estimator"] == "pseudobulk"].set_index("unit")
    both = model.index.intersection(pseudo.index)
    out = pd.DataFrame({
        "rho_model":    model.loc[both, "rho_trend"],
        "rho_pseudo":   pseudo.loc[both, "rho_trend"],
        "q_model":      model.loc[both, "q_trend"],
        "q_pseudo":     pseudo.loc[both, "q_trend"],
    })
    return out


def _draw_scatter_panel(ax, df: pd.DataFrame, title_prefix: str,
                        xlabel: str, ylabel: str, alpha: float = ALPHA
                        ) -> dict:
    """One scatter panel: pyvae ρ (x) vs pseudobulk ρ (y), coloured by
    cross-estimator significance class."""
    x = df["rho_model"].to_numpy()
    y = df["rho_pseudo"].to_numpy()
    sig_m = df["q_model"].to_numpy() < alpha
    sig_p = df["q_pseudo"].to_numpy() < alpha

    classes = {
        "neither":  ~sig_m & ~sig_p,
        "pyvae":    sig_m  & ~sig_p,
        "pseudo":   ~sig_m & sig_p,
        "both":     sig_m  & sig_p,
    }

    # Draw neither first (bottom), then the signified classes on top so
    # they're visible against the cloud.
    draw_specs = [
        ("neither", mm_c := C_FAINT, 4,  0.0, None),
        ("pseudo",  C_GREY,          12, 0.6, None),
        ("pyvae",   C_BLUE,          16, 0.9, None),
        ("both",    C_INK,           22, 0.9, "both"),
    ]
    for key, colour, size, edge_lw, _ in draw_specs:
        mask = classes[key]
        if not mask.any():
            continue
        ax.scatter(x[mask], y[mask], s=size, c=colour,
                   edgecolors=C_INK if edge_lw > 0 else "none",
                   linewidths=edge_lw, zorder=3 if key == "neither" else 4)

    # Reference lines: zero and y = x.
    ax.axhline(0, color=C_GREY, lw=0.5, zorder=1)
    ax.axvline(0, color=C_GREY, lw=0.5, zorder=1)
    lo = min(float(np.nanmin(x)), float(np.nanmin(y))) - 0.03
    hi = max(float(np.nanmax(x)), float(np.nanmax(y))) + 0.03
    ax.plot([lo, hi], [lo, hi], color=mm_c, lw=0.8, ls="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    # Mark the thesis's defended unit explicitly.
    if INTERFERON_UNIT in df.index:
        xi, yi = float(df.loc[INTERFERON_UNIT, "rho_model"]), \
                 float(df.loc[INTERFERON_UNIT, "rho_pseudo"])
        ax.scatter([xi], [yi], s=70, facecolors=C_ORANGE,
                   edgecolors=C_INK, linewidths=0.8, zorder=5)
        ax.annotate("interferon α/β", (xi, yi),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=7.4, fontweight="bold", color=C_INK,
                    ha="left", va="bottom", zorder=6)

    # Spearman ρ over all paired units.
    from scipy import stats as _stats
    rho = float(_stats.spearmanr(x, y, nan_policy="omit").statistic)

    # Per-class counts and the panel-wide ρ are returned so the caller
    # can bake them into the panel title (keeps title and counts on
    # distinct rows above the plot — the only way to avoid both covering
    # data dots AND colliding with each other).
    counts = {k: int(v.sum()) for k, v in classes.items()}

    # Two-line panel title: panel letter + cohort × lineage on line 1,
    # per-cohort ρ + per-class counts on line 2. matplotlib owns the
    # vertical spacing so the two lines never collide.
    counts_line = (f"ρ = {rho:+.2f}  ·  n = {len(df):,}  ·  "
                   f"both: {counts['both']}  ·  "
                   f"pyvae-only: {counts['pyvae']}  ·  "
                   f"pseudo-only: {counts['pseudo']}")
    ax.set_title(f"{title_prefix}\n{counts_line}",
                 fontsize=8.4, loc="left", pad=4, linespacing=1.4)
    ax.set_xlabel(xlabel, fontsize=7.4)
    ax.set_ylabel(ylabel, fontsize=7.4)
    ax.tick_params(labelsize=6.4)
    ax.grid(True, alpha=0.2, lw=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return counts | {"rho": rho}


def fig2_pseudobulk_vs_pyvae() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.5), constrained_layout=True)
    for i, study in enumerate(STUDIES):
        df = _load_donor_test(study)
        for j, lineage in enumerate(MAJOR_LINEAGES):
            paired = _paired_rho(df, lineage)
            _draw_scatter_panel(
                axes[i, j], paired,
                title_prefix=f"{chr(ord('a') + i*2 + j)} — "
                             f"{study.split('_')[0]}  ·  {lineage}",
                xlabel="pyvae donor-trend ρ",
                ylabel="pseudobulk donor-trend ρ")

    # Shared legend at the top of the figure.
    handles = [
        Line2D([], [], marker="o", ls="", color=C_FAINT, markersize=4,
               label="not significant in either"),
        Line2D([], [], marker="o", ls="", color=C_GREY, markeredgecolor=C_INK,
               markeredgewidth=0.6, markersize=5, label="pseudobulk only"),
        Line2D([], [], marker="o", ls="", color=C_BLUE, markeredgecolor=C_INK,
               markeredgewidth=0.6, markersize=6, label="pyvae only"),
        Line2D([], [], marker="o", ls="", color=C_INK,
               markeredgewidth=0.6, markersize=7, label="both"),
        Line2D([], [], marker="o", ls="", color=C_ORANGE,
               markeredgecolor=C_INK, markeredgewidth=0.7, markersize=8,
               label="defended finding (interferon α/β)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=7.4,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(
        "pyvae vs pseudobulk donor-level trend "
        "(ρ per unit, BH q < 0.05)",
        fontsize=11.5, fontweight="bold", y=1.075)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fig2_pseudobulk_vs_pyvae"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}.pdf / .png")

# fig3: per-donor score distributions for the top 10 units

STAGE_ORDER = ["Non", "MGUS", "SMM", "MM"]


def _top10_union(ranks: dict[str, dict[str, int]],
                 scored_units: set[str] | None = None) -> list[str]:
    """Union of each cohort's top-10 endpoint units, filtered to those
    actually scored at the donor level (≥10 measured members per cohort,
    the donor test's inclusion rule). Order: top in Zavidij first, then
    any new Foster additions."""
    seen, out = set(), []
    for study in STUDIES:
        picked = 0
        for u in sorted(ranks[study], key=lambda u: ranks[study][u]):
            if scored_units is not None and u not in scored_units:
                continue
            if u in seen:
                picked += 1
                if picked >= 10:
                    break
                continue
            out.append(u)
            seen.add(u)
            picked += 1
            if picked >= 10:
                break
    return out


def _trend_qs(study: str) -> pd.DataFrame:
    """Per-(unit, lineage) donor-level trend q, model estimator only."""
    df = _load_donor_test(study)
    df = df[df["estimator"] == "model"]
    return df.set_index(["unit", "lineage"])[["q_trend", "rho_trend"]]


def _draw_donor_cell(ax, scores_cell: pd.DataFrame, trends: pd.DataFrame,
                     unit: str, show_xlabels: bool, show_ylabel: bool,
                     unit_label: str) -> None:
    """One panel: per-donor dots across stages, CD4 and CD8 side-by-side."""
    sub = scores_cell[(scores_cell["unit"] == unit) &
                      (scores_cell["estimator"] == "model")]
    if sub.empty:
        ax.text(0.5, 0.5, "unit not scored\nfor this cohort",
                transform=ax.transAxes, fontsize=7, color=C_GREY,
                ha="center", va="center", style="italic")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    rng = np.random.default_rng(0)
    stage_x = {s: i for i, s in enumerate(STAGE_ORDER)}
    offsets = {"CD4": -0.17, "CD8": +0.17}

    for lineage, dx in offsets.items():
        colour = {"CD4": C_BLUE, "CD8": C_ORANGE}[lineage]
        for stage in STAGE_ORDER:
            cell = sub[(sub["lineage"] == lineage) & (sub["stage"] == stage)]
            if cell.empty:
                continue
            n = len(cell)
            # Jittered x positions; keep jitter small so CD4/CD8 don't cross.
            xs = (stage_x[stage] + dx
                  + rng.normal(0, 0.035, n).clip(-0.09, 0.09))
            ys = cell["score"].to_numpy()
            ax.scatter(xs, ys, s=16, c=colour, alpha=0.85,
                       edgecolors="none", zorder=3)
            # Median line (short horizontal segment) per group.
            med = float(np.median(ys))
            ax.plot([stage_x[stage] + dx - 0.11, stage_x[stage] + dx + 0.11],
                    [med, med], color=C_INK, lw=1.0, zorder=4)

        # Significance mark at the TOP of the panel if trend q<0.05 for this
        # (unit, lineage). Positioned in axes-fraction (y=0.92) and in data
        # x-coord (just inside the leftmost stage), so it clearly belongs
        # to THIS panel and never bleeds into the neighbouring panel.
        key = (unit, lineage)
        if key in trends.index and float(trends.loc[key, "q_trend"]) < ALPHA:
            x_pos = -0.42 if lineage == "CD4" else -0.28
            ax.scatter(x_pos, 0.92, s=48, facecolors="none",
                       edgecolors=colour, linewidths=1.5, zorder=6,
                       clip_on=False, transform=ax.get_xaxis_transform())

    ax.axhline(0, color=C_GREY, lw=0.5, zorder=1)
    ax.set_xticks(list(stage_x.values()))
    if show_xlabels:
        ax.set_xticklabels(STAGE_ORDER, fontsize=7)
    else:
        ax.set_xticklabels([])
    ax.set_xlim(-0.55, len(STAGE_ORDER) - 0.45)
    ax.tick_params(labelsize=6.2, length=0)
    if show_ylabel:
        ax.set_ylabel(unit_label, fontsize=7.2, rotation=0, ha="right",
                      va="center", labelpad=12, linespacing=1.05)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.2, lw=0.4)
    ax.grid(axis="x", visible=False)


def fig3_donor_violin_top10() -> None:
    ranks = load_endpoint_ranks(STUDIES)

    # Load donor scores once; use the SET of scored units to pre-filter.
    scores = {s: pd.read_csv(OUT_BASE / "donor_test" / s / "donor_scores.csv")
              for s in STUDIES}
    scored = {u for s in STUDIES for u in scores[s]["unit"].unique()}

    units = _top10_union(ranks, scored_units=scored)
    trends = {s: _trend_qs(s) for s in STUDIES}

    # Grid: one row per unit, one column per cohort. Tight vertical pitch
    # so the figure doesn't become a scroll.
    n_rows = len(units)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(9.5, 0.75 * n_rows + 0.9),
        sharex=True, sharey="row",
        gridspec_kw=dict(hspace=0.35, wspace=0.08,
                         left=0.18, right=0.98, top=0.93, bottom=0.05))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for r, unit in enumerate(units):
        unit_label = textwrap.fill(short(unit), 22)
        for c, study in enumerate(STUDIES):
            _draw_donor_cell(
                axes[r, c], scores[study], trends[study], unit,
                show_xlabels=(r == n_rows - 1),
                show_ylabel=(c == 0),
                unit_label=unit_label)
            if r == 0:
                axes[r, c].set_title(
                    f"{study.split('_')[0]}  —  per-donor pyvae score",
                    fontsize=9, pad=4, loc="center")

    # Shared legend ABOVE the top panel row.
    handles = [
        Line2D([], [], marker="o", ls="", color=C_BLUE, markersize=5,
               label="CD4 donor"),
        Line2D([], [], marker="o", ls="", color=C_ORANGE, markersize=5,
               label="CD8 donor"),
        Line2D([], [], color=C_INK, lw=1.5, label="per-group median"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor=C_INK, markeredgewidth=1.3, markersize=8,
               label="donor-level trend q < 0.05 (per lineage)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=7.6,
               bbox_to_anchor=(0.5, 0.985), frameon=False)
    fig.suptitle(
        "per-donor pyvae scores across stages "
        "(top-10 endpoint units per cohort, union, donor-scored)",
        fontsize=11, fontweight="bold", y=1.00)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fig3_donor_scores_top10"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}.pdf / .png")

# fig4: donor-level p-value QQ against the uniform null

def _draw_qq_panel(ax, p_vals: np.ndarray, title_prefix: str,
                   show_xlabel: bool, show_ylabel: bool) -> None:
    """QQ of observed p_trend against the uniform(0, 1) null, plus the
    p-value histogram as a small inset."""
    p = np.sort(p_vals[~np.isnan(p_vals)])
    n = len(p)
    expected = (np.arange(1, n + 1) - 0.5) / n

    # Pointwise 95% band from Beta(k, n-k+1) quantiles — standard for QQ.
    from scipy import stats as _stats
    k = np.arange(1, n + 1)
    lo = _stats.beta.ppf(0.025, k, n - k + 1)
    hi = _stats.beta.ppf(0.975, k, n - k + 1)

    ax.fill_between(expected, lo, hi, color=C_FAINT, alpha=0.55, zorder=1,
                    label="95% pointwise band")
    ax.plot([0, 1], [0, 1], color=C_GREY, lw=0.7, ls="--", zorder=2,
            label="perfect calibration")
    ax.scatter(expected, p, s=5, c=C_BLUE, alpha=0.75, edgecolors="none",
               zorder=3)

    # Inflation factor λ: ratio of observed median to expected median (0.5).
    # λ > 1 = inflated, λ < 1 = conservative, λ ≈ 1 = well-sized.
    lam = float(np.median(p) / 0.5)
    n_sig_005 = int((p < 0.05).sum())
    n_exp_005 = int(round(0.05 * n))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(
        f"{title_prefix}\n"
        f"n = {n:,} units  ·  λ = {lam:.2f}  ·  "
        f"p<0.05: {n_sig_005} obs vs {n_exp_005} expected",
        fontsize=8.4, loc="left", pad=4, linespacing=1.4)
    if show_xlabel:
        ax.set_xlabel("expected p (uniform null)", fontsize=7.4)
    if show_ylabel:
        ax.set_ylabel("observed p_trend", fontsize=7.4)
    ax.tick_params(labelsize=6.4)
    ax.grid(True, alpha=0.2, lw=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig4_donor_pvalue_qq() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.8), constrained_layout=True)
    for i, study in enumerate(STUDIES):
        df = _load_donor_test(study)
        df = df[df["estimator"] == "model"]
        for j, lineage in enumerate(MAJOR_LINEAGES):
            p = df[df["lineage"] == lineage]["p_trend"].to_numpy()
            _draw_qq_panel(
                axes[i, j], p,
                title_prefix=f"{chr(ord('a') + i*2 + j)} — "
                             f"{study.split('_')[0]}  ·  {lineage}",
                show_xlabel=(i == 1),
                show_ylabel=(j == 0))

    # Shared legend (one dot family + one line) at the top.
    handles = [
        Line2D([], [], color=C_GREY, lw=1.0, ls="--",
               label="perfect calibration (y = x)"),
        mpatches.Patch(facecolor=C_FAINT, alpha=0.55,
                       label="95% pointwise band under the uniform null"),
        Line2D([], [], marker="o", ls="", color=C_BLUE, markersize=5,
               label="one unit (sorted p_trend)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=7.4,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(
        "donor-level trend-test calibration (QQ against uniform)",
        fontsize=11.5, fontweight="bold", y=1.07)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fig4_donor_pvalue_qq"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}.pdf / .png")



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=int, choices=[1, 2, 3, 4], default=None,
                    help="draw only this figure (1-4); default: all")
    args = ap.parse_args()
    apply_style()

    builders = {1: fig1_reactome_hierarchy,
                2: fig2_pseudobulk_vs_pyvae,
                3: fig3_donor_violin_top10,
                4: fig4_donor_pvalue_qq}
    picks = [args.only] if args.only else list(builders)
    for i in picks:
        print(f"fig{i}:")
        try:
            builders[i]()
        except NotImplementedError as e:
            print(f"  (skipped: {e})")


if __name__ == "__main__":
    main()
