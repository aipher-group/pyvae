#!/usr/bin/env python
"""The three paper figures, assembled from the tables the pipeline wrote.

    python 04_figures.py

Pure assembly. It reads the CSVs that `01_architecture.py`, `02_biology.ipynb`
and `03_donor_test.py` already wrote and draws from them -- no model, no counts
matrix, no GPU, a few seconds. It is a script rather than cells in the notebook
for one structural reason: the notebook is **one study per run** and the
comparison page needs both.

Three pages, each as `.pdf` for the paper and `.png` for looking at:

    fig1_<study A>.pdf      one study, four panels
    fig2_<study B>.pdf      the other study, identical layout and colour scale
    fig3_comparison.pdf     what replicates and what does not

Plus `fig_panel_data.csv`, which holds every number that appears in the three
figures. `verify_findings.py` reads that, so the prose is checked against the
figures rather than against a re-derivation that could drift from them.

Conventions the figures depend on, each easy to get wrong:

**One colour scale across both studies, on a within-study scale.** Figures 1
and 2 are meant to be laid side by side, so they share one `TwoSlopeNorm`. But
`member_shift_z` is **not on a common absolute scale across cohorts**: it
carries `delta_post`, which depends on how far each fit pushed its units into
the flat tail of `tanh`, and `Foster_2026`'s endpoint values run about seven
times larger than `Zavidij_2020`'s. Sharing a raw scale therefore renders the
smaller cohort a blank panel -- which is exactly what an earlier version of
this script did.

So each cohort's values are divided by its own robust scale (the 99th
percentile of |`member_shift_z`| over all its units, lineages and contrasts)
before colouring, and the shared norm is in those units. A colour then means
"this far out for its own cohort", which is the comparison these two pages can
honestly support. The raw values, and each cohort's divisor, are both in
`fig_panel_data.csv`.

**The ladder, never the steps.** Every heat-map column is a contrast against the
healthy baseline, so a row reads left to right as a trajectory.

**No significance mark on any unit panel.** There is no test over units -- 1,615
strictly nested Reactome units cannot carry one -- so colour is an effect size
and nothing on those panels encodes a p-value. The only `q` in these figures is
donor-level, where the sampling unit is the donor, and it is drawn as a mark on
the donor panels alone.

**Row counts are not results.** Reactome nests, so neighbouring rows are often
parent and child of one family. Every panel caption that lists units says so.

Palette: the three categorical slots pass an all-pairs colour-vision check
against a light print surface (worst adjacent pair deltaE 9.2 deutan, 27.6
normal). The aqua slot sits at 2.74:1 contrast, below the 3:1 floor, which
obligates relief rather than disqualifying it -- so every aqua mark is either
directly labelled or inside a table. Do not reuse aqua for an unlabelled mark.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from scipy import stats

import myeloma as mm

ALPHA = 0.05
#: Page geometry: a portrait page at a two-column journal's full text width,
#: 183 x 244 mm. Every page is saved at exactly this size -- `bbox_inches` is
#: deliberately NOT "tight", because cropping to content gives each page a
#: different width and breaks the one property figures 1 and 2 exist to have:
#: that they can be laid on top of each other.
PAGE = (7.2, 9.6)
#: Rows in the ladder panel. Chosen once, applied to both studies, so the two
#: pages show the same units and can be read against each other.
N_ROWS_UP, N_ROWS_DOWN = 18, 7
#: Units drawn on the donor panel, by the study's own endpoint ranking.
N_DONOR_UNITS = 6


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load(study: str) -> dict:
    """Every table one study contributes, or a clear error naming the rule."""
    bio = mm.OUT_DIR / "biology" / study
    donor = mm.OUT_DIR / "donor_test" / study
    arch = mm.OUT_DIR / "architecture" / study
    need = {
        "summary": (bio / "summary.json", "units"),
        "contrasts": (bio / "unit_contrasts.csv", "units"),
        "agreement": (bio / "table3_lineage_agreement.csv", "units"),
        "design": (bio / "table1_design.csv", "units"),
        "donor_test": (donor / "donor_test.csv", "donor_test"),
        "donor_scores": (donor / "donor_scores.csv", "donor_test"),
        "donor_summary": (donor / "summary.json", "donor_test"),
        "arch_summary": (arch / "summary.json", "architecture"),
    }
    out = {"study": study}
    for key, (path, rule) in need.items():
        if not path.exists():
            raise SystemExit(
                f"missing {path}\nrun `snakemake {rule}` for {study} first")
        out[key] = (json.loads(path.read_text()) if path.suffix == ".json"
                    else pd.read_csv(path))
    out["agreement"] = out["agreement"].set_index("contrast")
    out["design"] = out["design"].set_index("stage")
    out["ladder"] = list(out["summary"]["ladder_contrasts"])
    out["endpoint"] = out["summary"]["endpoint_contrast"]
    return out


def wide(d: dict, contrast: str) -> pd.DataFrame:
    """units x lineage of `member_shift_z` for one contrast."""
    c = d["contrasts"]
    return (c[c["contrast"] == contrast]
            .pivot(index="unit", columns="lineage", values="member_shift_z"))


def major(d: dict) -> list[str]:
    cols = wide(d, d["endpoint"]).columns
    return [l for l in mm.MAJOR_SUBSETS if l in cols]


def endpoint_mean(d: dict) -> pd.Series:
    """Each unit's endpoint contrast, averaged over the major lineages."""
    return wide(d, d["endpoint"])[major(d)].mean(axis=1)


# --------------------------------------------------------------------------- #
# Shared drawing helpers
# --------------------------------------------------------------------------- #
def tag(ax, letter: str, dx: float = -0.16, dy: float = 1.16) -> None:
    """A panel letter, clear of the title.

    `dy` is above every two-line panel title this page uses. An earlier version
    sat at 1.045 and the letter landed inside the title text of panels b and c.
    """
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def strip_spines(ax, keep=()) -> None:
    for side in ("top", "right", "left", "bottom"):
        if side not in keep:
            ax.spines[side].set_visible(False)


def wrap(name: str, width: int = 44) -> str:
    """Up to two lines, for a panel with room for them."""
    return "\n".join(textwrap.wrap(mm.short(name), width)[:2])


def clip(name: str, width: int = 42) -> str:
    """One line, truncated. For panels with more rows than vertical room."""
    short = mm.short(name)
    return short if len(short) <= width else short[:width - 1] + "\u2026"


def study_scale(d: dict) -> float:
    """The cohort's own robust magnitude, used to put both pages on one scale.

    The 99th percentile of |member_shift_z| over everything that cohort
    produced. Robust rather than the maximum, because one saturated outlier
    would otherwise set the scale for 1,615 units.
    """
    v = np.abs(d["contrasts"]["member_shift_z"].to_numpy())
    return float(np.nanpercentile(v, 99)) or 1e-9


def ladder_heatmap(ax, d, order, *, norm, cmap, fontsize=5.4):
    """One row per unit, one column per lineage x ladder contrast.

    Values are divided by the cohort's own scale; see the module docstring.
    """
    lin = major(d)
    frames = {f"{l}|{c}": wide(d, c)[l].reindex(order) for l in lin
              for c in d["ladder"] if l in wide(d, c).columns}
    M = pd.DataFrame(frames) / study_scale(d)
    ax.imshow(M.to_numpy(), aspect="auto", cmap=cmap, norm=norm)
    n_per = len(d["ladder"])
    for k in range(1, len(lin)):
        ax.axvline(k * n_per - 0.5, color="white", lw=2.6)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels([c.split("|")[1].split("->")[1] for c in M.columns],
                       fontsize=6.4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([wrap(p, 46) for p in order], fontsize=fontsize)
    for k, l in enumerate(lin):
        ax.annotate(l, ((k + 0.5) * n_per - 0.5, -0.95), fontsize=7.6,
                    fontweight="bold", ha="center", annotation_clip=False)
    ax.tick_params(length=0)
    ax.grid(False)
    return M


def donor_strip(ax, d, units, *, colours, stages):
    """Per-donor model score against stage, for a few units."""
    sc = d["donor_scores"]
    sc = sc[sc["estimator"] == "model"]
    tests = d["donor_test"]
    tests = tests[tests["estimator"] == "model"].set_index(["unit", "lineage"])
    rank_of = {s: i for i, s in enumerate(stages)}
    rng = np.random.default_rng(0)
    rows = []
    for i, unit in enumerate(units):
        for off, (lin, colour) in enumerate(colours.items()):
            sel = sc[(sc["unit"] == unit) & (sc["lineage"] == lin)]
            if sel.empty:
                continue
            x = sel["stage"].map(rank_of).to_numpy(dtype=float)
            x = x + (off - 0.5) * 0.3 + rng.normal(0, 0.035, len(x))
            y = sel["score"].to_numpy()
            # Scaled per unit so six units with different magnitudes share one
            # axis; the trend and its q are what this panel reports, not the
            # raw value, which is in donor_scores.csv.
            span = np.abs(y).max() or 1.0
            ax.scatter(x, i + 0.34 * y / span, s=9, c=colour, linewidths=0,
                       alpha=0.85, zorder=3)
            key = (unit, lin)
            if key in tests.index:
                row = tests.loc[key]
                if float(row["q_trend"]) < ALPHA:
                    # At a fixed position beside the row label, not at the
                    # mean of the points. The trend is a property of the whole
                    # row; drawing the ring inside the cloud put it over
                    # whichever stage happened to hold the most donors, which
                    # reads as "significant at SMM", and rings on adjacent
                    # rows overlapped each other's points.
                    ax.scatter(-0.42, i + (off - 0.5) * 0.26, s=30,
                               facecolors="none", edgecolors=colour,
                               linewidths=1.4, zorder=5, clip_on=False)
                rows.append({"unit": unit, "lineage": lin,
                             "rho_trend": float(row["rho_trend"]),
                             "q_trend": float(row["q_trend"]),
                             "n_donors": int(row["n_donors"])})
        ax.axhline(i, color=mm.C_FAINT, lw=0.6, zorder=1)
    ax.set_yticks(range(len(units)))
    ax.set_yticklabels([wrap(p, 36) for p in units], fontsize=5.8)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=7)
    ax.set_xlim(-0.55, len(stages) - 0.4)
    ax.set_ylim(-0.7, len(units) - 0.3)
    ax.tick_params(length=0)
    ax.grid(False)
    strip_spines(ax, keep=("bottom",))
    return rows


# --------------------------------------------------------------------------- #
# Page 1 and 2: one study each
# --------------------------------------------------------------------------- #
def per_study_page(d: dict, *, norm, order, out: Path) -> list[dict]:
    study = d["study"]
    lin_colours = {l: mm.LINEAGE_COLOURS[l] for l in major(d)}
    cmap = mm.diverging()
    stages = list(d["summary"]["stages"])
    panel = []

    fig = plt.figure(figsize=PAGE)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[3.3, 1.0, 1.25],
                  hspace=0.60, wspace=0.46,
                  left=0.30, right=0.95, top=0.94, bottom=0.05)

    # (a) the ladder
    ax = fig.add_subplot(gs[0, :])
    M = ladder_heatmap(ax, d, order, norm=norm, cmap=cmap)
    tag(ax, "a", dx=-0.40, dy=1.07)
    ax.set_title(f"{study}: the units the stage sequence moves\n"
                 f"top {N_ROWS_UP} and bottom {N_ROWS_DOWN} by the endpoint "
                 "mean over both lineages; Reactome nests, so read families",
                 fontsize=8, pad=16)
    cb = fig.colorbar(ax.images[0], ax=ax, shrink=0.55, pad=0.015)
    cb.set_label("member_shift_z / cohort scale", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    scale = study_scale(d)
    panel.append({"figure": study, "panel": "a", "unit": "", "lineage": "",
                  "contrast": "", "quantity": "cohort_scale",
                  "value": scale})
    for col in M.columns:
        lin, contrast = col.split("|")
        for unit, value in M[col].items():
            panel.append({"figure": study, "panel": "a", "unit": unit,
                          "lineage": lin, "contrast": contrast,
                          "quantity": "member_shift_z",
                          "value": float(value) * scale})

    # (b) lineage agreement at the endpoint
    ax = fig.add_subplot(gs[1, 0])
    w = wide(d, d["endpoint"])
    lin = major(d)
    if len(lin) == 2:
        x, y = w[lin[0]].to_numpy(), w[lin[1]].to_numpy()
        ax.scatter(x, y, s=3, c=mm.C_FAINT, linewidths=0)
        lo, hi = np.nanpercentile(np.r_[x, y], [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], color=mm.C_ORANGE, lw=1.0, ls="--")
        ax.axhline(0, color=mm.C_GREY, lw=0.6)
        ax.axvline(0, color=mm.C_GREY, lw=0.6)
        row = d["agreement"].loc[d["endpoint"]]
        ax.set_xlabel(f"{lin[0]} member_shift_z", fontsize=7)
        ax.set_ylabel(f"{lin[1]} member_shift_z", fontsize=7)
        ax.set_title(f"lineage agreement, {d['endpoint'].replace('->', '→')}\n"
                     f"rho {row['rho']:+.2f} [{row['ci_lo']:+.2f}, "
                     f"{row['ci_hi']:+.2f}], limiting n = "
                     f"{int(row['limiting cells']):,} cells", fontsize=7.4)
        panel.append({"figure": study, "panel": "b", "unit": "",
                      "lineage": "|".join(lin), "contrast": d["endpoint"],
                      "quantity": "rho_lineage", "value": float(row["rho"])})
    ax.tick_params(labelsize=6)
    strip_spines(ax, keep=("left", "bottom"))
    tag(ax, "b")

    # (c) what the annotation is worth, from the decision script
    ax = fig.add_subplot(gs[1, 1])
    arms = d["arch_summary"]["arms"]
    schemes = ["shipped", "pathway", "uncond", "signed"]
    have = [s for s in schemes if f"real_{s}" in arms and f"perm_{s}" in arms]
    worth = [arms[f"perm_{s}"]["best_val"] - arms[f"real_{s}"]["best_val"]
             for s in have]
    bars = ax.bar(range(len(have)), worth, color=[
        mm.C_ORANGE if s == "signed" else mm.C_GREY for s in have], width=0.66)
    ax.bar_label(bars, [f"{v:+.1f}" for v in worth], fontsize=6, padding=1.5)
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels(have, fontsize=6.6, rotation=20, ha="right")
    ax.set_ylabel("permuted − real, nats", fontsize=7)
    ax.set_title("what the annotation is worth\n"
                 "a shuffled map costs the real one nothing if the\n"
                 "mask is not in the contrast path", fontsize=7.4)
    ax.tick_params(labelsize=6)
    ax.grid(axis="x", visible=False)
    strip_spines(ax, keep=("left", "bottom"))
    tag(ax, "c", dx=-0.22)
    for s, v in zip(have, worth):
        panel.append({"figure": study, "panel": "c", "unit": "", "lineage": "",
                      "contrast": s, "quantity": "annotation_worth_nats",
                      "value": float(v)})

    # (d) the donor-level test
    ax = fig.add_subplot(gs[2, :])
    # Only units the donor test could actually score. It drops units with
    # fewer than 10 measured members, so the study's leading unit is not
    # guaranteed to be there -- and an unfiltered pick draws an empty row with
    # a label on it, which reads as "no donor signal" rather than "not tested".
    scored = set(d["donor_scores"]["unit"])
    ranked = endpoint_mean(d).sort_values(ascending=False)
    units = [u for u in ranked.index if u in scored][:N_DONOR_UNITS]
    missing = [u for u in ranked.index[:N_DONOR_UNITS] if u not in scored]
    if missing:
        print(f"    {len(missing)} leading unit(s) not donor-scored "
              f"(<10 measured members): {[mm.short(u) for u in missing]}")
    rows = donor_strip(ax, d, units, colours=lin_colours, stages=stages)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l,
                          markersize=4) for l, c in lin_colours.items()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0),
              fontsize=6.4)
    floor = d["donor_summary"]["power_floor"]
    n_don = ", ".join(f"{l} n={v['donors']}" for l, v in floor.items())
    ax.set_title(f"the same units with the donor as the sampling unit "
                 f"({n_don})\nper-donor score against stage, scaled per row; "
                 f"ring at the left edge = trend q < {ALPHA} over "
                 f"{d['donor_summary']['units_tested']:,} units",
                 fontsize=7.4, pad=8)
    tag(ax, "d", dx=-0.40, dy=1.10)
    for r in rows:
        panel.append({"figure": study, "panel": "d", "unit": r["unit"],
                      "lineage": r["lineage"], "contrast": "donor_trend",
                      "quantity": "rho_trend", "value": r["rho_trend"]})
        panel.append({"figure": study, "panel": "d", "unit": r["unit"],
                      "lineage": r["lineage"], "contrast": "donor_trend",
                      "quantity": "q_trend", "value": r["q_trend"]})

    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  {out.name}.pdf / .png")
    return panel


# --------------------------------------------------------------------------- #
# Page 3: the comparison
# --------------------------------------------------------------------------- #
def comparison_page(a: dict, b: dict, *, out: Path) -> list[dict]:
    panel = []
    ea, eb = endpoint_mean(a), endpoint_mean(b)
    shared = ea.index.intersection(eb.index)
    fig = plt.figure(figsize=PAGE)
    # Panel b carries twenty long Reactome names down its left edge, so it
    # takes a full row and has its horizontal extent set explicitly afterwards:
    # a shared GridSpec `left` wide enough for those labels would waste a third
    # of every other panel, and leaving it at 0.12 ran them over panel a.
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.35, 1.2],
                  hspace=0.62, wspace=0.40,
                  left=0.13, right=0.96, top=0.90, bottom=0.05)

    # (a) endpoint against endpoint, every shared unit
    ax = fig.add_subplot(gs[0, 0])
    x, y = ea[shared].to_numpy(), eb[shared].to_numpy()
    rho = float(stats.spearmanr(x, y).statistic)
    ax.scatter(x, y, s=3, c=mm.C_FAINT, linewidths=0)
    ax.axhline(0, color=mm.C_GREY, lw=0.6)
    ax.axvline(0, color=mm.C_GREY, lw=0.6)
    ax.set_xlabel(f"{a['study']}", fontsize=7)
    ax.set_ylabel(f"{b['study']}", fontsize=7)
    ax.set_title(f"endpoint member_shift_z, {len(shared):,} shared units\n"
                 f"rho {rho:+.2f}", fontsize=7.4)
    ax.tick_params(labelsize=6)
    strip_spines(ax, keep=("left", "bottom"))
    tag(ax, "a")
    panel.append({"figure": "comparison", "panel": "a", "unit": "",
                  "lineage": "", "contrast": "endpoint",
                  "quantity": "rho_cross_study", "value": rho})

    # (b) each study's leaders, and where they sit in the other
    ax = fig.add_subplot(gs[1, :])
    _box = ax.get_position()
    ax.set_position([0.34, _box.y0, 0.60, _box.height])
    ra = ea.rank(ascending=False)
    rb = eb.rank(ascending=False)
    top_a = list(ea.sort_values(ascending=False).index[:10])
    top_b = list(eb.sort_values(ascending=False).index[:10])
    labels, here, there, colours = [], [], [], []
    for u in top_a:
        labels.append(clip(u))
        here.append(float(ra[u]))
        there.append(float(rb[u]) if u in rb.index else np.nan)
        colours.append(mm.STUDY_COLOURS[a["study"]])
    for u in top_b:
        labels.append(clip(u))
        here.append(float(rb[u]))
        there.append(float(ra[u]) if u in ra.index else np.nan)
        colours.append(mm.STUDY_COLOURS[b["study"]])
    yy = np.arange(len(labels))[::-1]
    ax.hlines(yy, here, there, color=mm.C_FAINT, lw=1.0, zorder=1)
    ax.scatter(here, yy, s=16, c=colours, zorder=3, label="its own study")
    ax.scatter(there, yy, s=16, facecolors="none", edgecolors=colours,
               linewidths=1.0, zorder=3, label="the other study")
    ax.set_xscale("log")
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=5.2)
    ax.axhline(len(top_b) - 0.5, color=mm.C_GREY, lw=0.8, ls=":")
    ax.set_xlabel("rank among all units (log)", fontsize=7)
    ax.set_title(f"each cohort's ten leading units, and where the other puts "
                 f"them\n(upper block: {a['study']}; lower: {b['study']})",
                 fontsize=7.4)
    ax.legend(fontsize=5.6, loc="lower right")
    ax.tick_params(labelsize=6, length=0)
    ax.grid(axis="y", visible=False)
    strip_spines(ax, keep=("bottom",))
    tag(ax, "b", dx=-0.30, dy=1.09)
    for u, h, t in zip(top_a + top_b, here, there):
        panel.append({"figure": "comparison", "panel": "b", "unit": u,
                      "lineage": "", "contrast": "endpoint_rank",
                      "quantity": "rank_own", "value": h})
        panel.append({"figure": "comparison", "panel": "b", "unit": u,
                      "lineage": "", "contrast": "endpoint_rank",
                      "quantity": "rank_other", "value": t})

    # (c) the donor-level trend, study against study
    ax = fig.add_subplot(gs[0, 1])
    def trend(d):
        t = d["donor_test"]
        t = t[t["estimator"] == "model"]
        return t.groupby("unit")["rho_trend"].mean()
    ta, tb = trend(a), trend(b)
    both = ta.index.intersection(tb.index)
    rho_d = (float(stats.spearmanr(ta[both], tb[both]).statistic)
             if len(both) > 2 else float("nan"))
    ax.scatter(ta[both], tb[both], s=3, c=mm.C_FAINT, linewidths=0)
    ax.axhline(0, color=mm.C_GREY, lw=0.6)
    ax.axvline(0, color=mm.C_GREY, lw=0.6)
    ax.set_xlabel(f"{a['study']} donor trend rho", fontsize=7)
    ax.set_ylabel(f"{b['study']} donor trend rho", fontsize=7)
    ax.set_title(f"donor-level trend, {len(both):,} shared units\n"
                 f"rho {rho_d:+.2f}", fontsize=7.4)
    ax.tick_params(labelsize=6)
    strip_spines(ax, keep=("left", "bottom"))
    tag(ax, "c")
    panel.append({"figure": "comparison", "panel": "c", "unit": "",
                  "lineage": "", "contrast": "donor_trend",
                  "quantity": "rho_cross_study", "value": rho_d})

    # (d) what the architecture is worth, both studies side by side
    ax = fig.add_subplot(gs[2, 0])
    schemes = ["shipped", "pathway", "uncond", "signed"]
    width = 0.38
    for k, d in enumerate((a, b)):
        arms = d["arch_summary"]["arms"]
        have = [s for s in schemes if f"real_{s}" in arms]
        ranks = [np.mean([v["unit_rank_target"]
                          for v in arms[f"real_{s}"]["lineages"].values()])
                 for s in have]
        ax.bar(np.arange(len(have)) + (k - 0.5) * width, ranks, width=width,
               color=mm.STUDY_COLOURS[d["study"]], label=d["study"])
    ax.set_yscale("log")
    ax.set_xticks(range(len(schemes)))
    ax.set_xticklabels(schemes, fontsize=6.6, rotation=20, ha="right")
    ax.set_ylabel("unit rank of the interferon target (log)", fontsize=7)
    ax.set_title("where each architecture puts the unit the\ngene-level "
                 "baseline points at; lower is better", fontsize=7.4)
    ax.legend(fontsize=5.8)
    ax.tick_params(labelsize=6)
    ax.grid(axis="x", visible=False)
    strip_spines(ax, keep=("left", "bottom"))
    tag(ax, "d")

    # (e) the table: what each study says, in words
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    lines = [["", a["study"], b["study"]]]
    for label, key in [("donors", "donors"), ("cells", "cells")]:
        lines.append([label, f"{a['summary'][key]:,}", f"{b['summary'][key]:,}"])
    lines.append(["validation loss",
                  f"{a['summary']['best_val_loss']:,.1f}",
                  f"{b['summary']['best_val_loss']:,.1f}"])
    lines.append(["units saturated > half",
                  f"{a['summary'].get('units_saturated_gt_half', '?')}",
                  f"{b['summary'].get('units_saturated_gt_half', '?')}"])
    lines.append(["donor trend, both lineages",
                  str(len(a["donor_summary"]["trend_both_lineages_model"])),
                  str(len(b["donor_summary"]["trend_both_lineages_model"]))])
    lines.append(["and the pseudobulk agrees on",
                  str(len(a["donor_summary"]
                          ["trend_both_lineages_both_estimators"])),
                  str(len(b["donor_summary"]
                          ["trend_both_lineages_both_estimators"]))])
    table = ax.table(cellText=lines[1:], colLabels=lines[0],
                     cellLoc="left", loc="upper center",
                     colWidths=[0.52, 0.24, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(5.6)
    table.scale(1.0, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(mm.C_FAINT)
        if r == 0:
            cell.set_text_props(fontweight="bold")
    ax.set_title("the two cohorts side by side\ncounts over nested units are "
                 "descriptive, not results", fontsize=7.4, pad=10)
    tag(ax, "e", dx=-0.02, dy=1.02)

    fig.suptitle("What replicates between the two cohorts, read over the "
                 "model's own units", fontsize=9.6, y=0.975)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  {out.name}.pdf / .png")
    return panel


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--studies", nargs=2, default=mm.ANALYSIS_STUDIES)
    args = ap.parse_args()
    mm.apply_style()

    out_dir = mm.OUT_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    a, b = (load(s) for s in args.studies)

    # One row order and one colour scale for both pages, from the union of the
    # two studies' endpoint means. Two studies drawn on two scales would make
    # the comparison these pages exist for a lie.
    # Standardised before averaging, for the same reason the colour is: an
    # unscaled mean is whichever cohort has the larger numbers.
    union = pd.concat([(endpoint_mean(a) / study_scale(a)).rename("a"),
                       (endpoint_mean(b) / study_scale(b)).rename("b")], axis=1)
    ranking = union.mean(axis=1).sort_values(ascending=False)
    order = list(ranking.index[:N_ROWS_UP]) + list(ranking.index[-N_ROWS_DOWN:])
    # In cohort-scale units, so the norm is the same object for both pages and
    # neither is washed out by the other's magnitude.
    span = max(float(np.nanpercentile(
        np.abs(d["contrasts"]["member_shift_z"].to_numpy()) / study_scale(d),
        99.9)) for d in (a, b)) or 1e-9
    norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
    print(f"cohort scales: {a['study']} {study_scale(a):.4f}, "
          f"{b['study']} {study_scale(b):.4f}; shared span {span:.2f} "
          f"cohort units over {len(order)} rows")

    panel = []
    panel += per_study_page(a, norm=norm, order=order,
                            out=out_dir / f"fig1_{a['study']}")
    panel += per_study_page(b, norm=norm, order=order,
                            out=out_dir / f"fig2_{b['study']}")
    panel += comparison_page(a, b, out=out_dir / "fig3_comparison")

    frame = pd.DataFrame(panel)
    frame.to_csv(out_dir / "fig_panel_data.csv", index=False)
    print(f"\n{len(frame):,} plotted numbers -> {out_dir}/fig_panel_data.csv")


if __name__ == "__main__":
    main()
