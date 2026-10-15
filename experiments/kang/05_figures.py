#!/usr/bin/env python
"""Kang's four figures, assembled from the tables the read-out already wrote.

    python 05_figures.py

Pure assembly. It reads what `02_kang_units.ipynb` wrote and draws from it: no
model, no counts matrix, no GPU, about ten seconds. Everything it needs is in
four files, all of them small enough to keep beside the paper:

    outputs/units/unit_contrasts.csv   one row per unit, the ranked read-out
    outputs/units/ig_target.csv        integrated gradients inside the target
    outputs/units/target_cells.csv     one row per cell, the target unit
    outputs/units/summary.json         the run's own scalars

Four pages, as `.png` at 300 dpi:

    fig1_ranked_units.png      the units the condition moves, both ends
    fig2_target_activation.png what the encoder reads per cell
    fig3_ig_target.png         which genes the unit reads the pathway through
    fig4_umap.png              a picture, carrying no claim

**Why this is a script and not cells in the notebook.** The notebook fits a
model, which needs a GPU and five minutes. Figure work is iterative and belongs
on a laptop. Splitting them means the aesthetics can be changed without a card,
and that editing a figure cannot invalidate a fit: `02_kang_units.ipynb` is a
declared input of a GPU rule, so touching it makes Snakemake delete the
read-out and refit, while this file is an input of a CPU rule that writes
nothing but pictures.

It also means there is one place where each figure is drawn. The notebook used
to both compute and plot, so a figure edited here and a figure edited there
would diverge silently.

Conventions, each easy to get wrong:

**Pre-activation, never activation.** `tanh(pre)` is pinned at 1.000 for most
cells in this fit, so a plot of the activation is a flat line at the top of the
axis that hides the thing worth seeing. Figures 2 and 4 both show `pre`, and
figure 2 marks where `tanh` passes 0.99.

**Every cell is held at `control` in the covariate.** The gap between the two
colours in figure 2 is therefore not the contrast. It is what the encoder reads
off each cell's own expression, since `cov_in_encoder=False` leaves `z`
depending on the transcriptome alone. The covariate's own shift is one number,
in the title, added equally to every point. That the two routes agree in
direction is the check worth taking from the panel.

**Figure 4 centres colour on the median, not on zero.** The unit sits far from
zero in every cell, so a scale centred at zero paints every panel one colour,
which is what an earlier version of this figure did.

**No significance mark anywhere.** Reactome nests strictly, so no valid test
exists over these 1,615 units. Colour and bar length are effect sizes.

**The known answer is always in the panel.** Figure 1 draws both ends of the
ranking and splices the target and its parent in wherever they fall, because a
figure that shows the top fifteen and quietly omits the target when the target
did badly is a figure that can only ever look good.

Palette: the same two categorical slots `experiments/multiple_mieloma` uses, so
one hue means the same thing across both experiments. Measured rather than
judged: worst pair OKLab deltaE 24.7 protan, 33.6 normal vision, both above the
3:1 contrast floor against a light surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent

#: The answer, fixed before the experiment ran, and its Reactome parent.
#: Reactome nests, so the parent ranking alongside the child is agreement.
TARGET = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING"
PARENT = "REACTOME_INTERFERON_SIGNALING"

C_BLUE, C_ORANGE, C_GREY, C_FAINT = "#2a78d6", "#eb6834", "#8a8a86", "#dcdcd8"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_orange", [C_BLUE, "#e8e8e6", C_ORANGE])

STYLE = {
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.frameon": False,
}

#: Figure 1 shows this many units from each end of the ranking.
N_UP, N_DOWN = 15, 8
#: Figure 3 shows this many genes, after dropping the zero-attribution padding.
N_GENES_IG = 18


# --------------------------------------------------------------------------- #
def load(out_dir: Path) -> dict:
    """Every table the figures need, or a clear error naming the rule."""
    need = {
        "units": out_dir / "unit_contrasts.csv",
        "ig": out_dir / "ig_target.csv",
        "cells": out_dir / "target_cells.csv",
        "summary": out_dir / "summary.json",
    }
    d = {}
    for key, path in need.items():
        if not path.exists():
            raise SystemExit(
                f"missing {path}\nrun `snakemake units` first (it needs a GPU; "
                "this script does not)")
        d[key] = (json.loads(path.read_text()) if path.suffix == ".json"
                  else pd.read_csv(path))
    d["units"] = d["units"].set_index("unit")
    d["ig"] = d["ig"].set_index("gene")
    d["ranked"] = d["units"].sort_values("member_shift_z", ascending=False)
    # The cell-type order is the one figure 2 draws left to right, and figure 4
    # labels its third panel with. Derived, so the two panels cannot disagree.
    d["order"] = (d["cells"].groupby("cell_type")["pre"].mean()
                  .sort_values(ascending=False).index.tolist())
    if d["summary"].get("smoke"):
        print("[warn] this read-out was a smoke run: 4 epochs, 2,000 genes. "
              "Any figure from it should be thrown away.")
    return d


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path.name}")


# --------------------------------------------------------------------------- #
def fig1_ranked_units(d: dict, figs: Path) -> None:
    """Both ends of the unit ranking, with the known answer spliced in."""
    ranked, units = d["ranked"], d["units"]
    rank_target = int(d["summary"]["unit_rank_target"])

    ends = pd.concat([ranked.head(N_UP), ranked.tail(N_DOWN)])
    extra = [p for p in (TARGET, PARENT)
             if p in ranked.index and p not in ends.index]
    top = (pd.concat([ends, ranked.loc[extra]])
           .sort_values("member_shift_z", ascending=False) if extra else ends)
    labels = [s.replace("REACTOME_", "").replace("_", " ")[:52]
              for s in top.index]
    colours = [C_ORANGE if v > 0 else C_BLUE for v in top["member_shift_z"]]

    fig, ax = plt.subplots(figsize=(7.4, 0.29 * len(top) + 1.2))
    y = np.arange(len(top))[::-1]
    ax.barh(y, top["member_shift_z"].to_numpy(), color=colours, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    for yy, name in zip(y, top.index):
        if name in (TARGET, PARENT):
            ax.annotate("◀ known answer" if name == TARGET else "◀ its parent",
                        (top.loc[name, "member_shift_z"], yy), fontsize=7.5,
                        va="center",
                        ha="left" if top.loc[name, "member_shift_z"] > 0
                        else "right", color="#0b0b0b", fontweight="bold")
    ax.axvline(0, color=C_GREY, lw=0.8)
    ax.set_xlabel("member_shift_z   (control → stimulated)")
    ax.tick_params(length=0)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Kang: the units the condition moves, {len(units):,} ranked, "
                 f"target at {rank_target}", fontsize=10)
    save(fig, figs / "fig1_ranked_units.png")


def fig2_target_activation(d: dict, figs: Path) -> None:
    """The target unit's pre-activation, by condition and cell type."""
    cells, order = d["cells"], d["order"]
    seed = int(d["summary"]["seed"])
    sat_line = float(np.arctanh(0.99))

    fig, ax = plt.subplots(figsize=(1.05 * len(order) + 2.2, 3.8))
    rng = np.random.default_rng(seed)
    for i, ct in enumerate(order):
        for off, (cond, colour) in enumerate([("control", C_BLUE),
                                              ("stimulated", C_ORANGE)]):
            v = cells.loc[(cells["cell_type"] == ct)
                          & (cells["condition"] == cond), "pre"].to_numpy()
            if not len(v):
                continue
            xpos = i + (off - 0.5) * 0.34
            ax.scatter(xpos + rng.normal(0, 0.045, len(v)), v, s=1.6, c=colour,
                       alpha=0.35, linewidths=0)
            ax.plot([xpos - 0.13, xpos + 0.13], [np.median(v)] * 2,
                    color="#0b0b0b", lw=1.6, solid_capstyle="butt", zorder=5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.axhline(sat_line, color="#0b0b0b", lw=1.0, ls="--")
    ax.annotate("tanh saturates above here", (len(order) - 0.5, sat_line),
                fontsize=7.5, va="bottom", ha="right", color="#52514e")
    ax.set_ylabel("unit pre-activation")
    ax.grid(axis="x", visible=False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l,
                          markersize=5)
               for c, l in [(C_BLUE, "control"), (C_ORANGE, "stimulated")]]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title(f"{TARGET.replace('REACTOME_', '').replace('_', ' ')}: what "
                 "the encoder reads per cell, every cell held at control\n(bar "
                 "= median; the covariate adds "
                 f"{d['units'].loc[TARGET, 'delta_pre']:+.3f} on top of all of "
                 "them)", fontsize=9.5)
    save(fig, figs / "fig2_target_activation.png")


def fig3_ig_target(d: dict, figs: Path) -> None:
    """Integrated gradients inside the target unit, top genes."""
    # Genes with exactly zero attribution are padding, not findings: the mask
    # puts every live encoder weight on a member, so most of the panel
    # contributes nothing and sorting ties alphabetically fills it with noise.
    ig = d["ig"]
    t = ig[ig["attribution"] > 0].head(N_GENES_IG).iloc[::-1]
    # The share a member would carry if all of them contributed equally. The
    # unit's measured member count, which is what `n_genes` counts, not the
    # size of the Reactome set before the panel was intersected with it.
    even = 1.0 / float(d["units"].loc[TARGET, "n_genes"])

    fig, ax = plt.subplots(figsize=(6.6, 0.28 * len(t) + 1.1))
    ax.barh(range(len(t)), t["share"].to_numpy(),
            color=[C_ORANGE if m else C_FAINT for m in t["is_member"]],
            edgecolor=C_GREY, linewidth=0.4, height=0.7)
    ax.axvline(even, color=C_GREY, ls="--", lw=1)
    ax.annotate("equal share", (even, len(t) - 0.2), fontsize=7,
                color="#52514e", ha="center", va="bottom")
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t.index, fontsize=8)
    ax.set_xlabel("share of total |attribution|")
    ax.tick_params(length=0)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{TARGET.replace('REACTOME_', '').replace('_', ' ')}: "
                 "encoder sensitivity (filled = annotated member)", fontsize=10)
    save(fig, figs / "fig3_ig_target.png")


def fig4_umap(d: dict, figs: Path) -> None:
    """The scanpy frame, coloured by the same quantity as figure 2."""
    cells, order = d["cells"], d["order"]
    if not {"umap1", "umap2"} <= set(cells.columns):
        raise SystemExit(
            f"{TARGET} cells table has no UMAP coordinates\nthe read-out that "
            "wrote it predates them: re-run `snakemake units`")
    pre = cells["pre"].to_numpy()
    u = cells[["umap1", "umap2"]].to_numpy()

    centre = float(np.median(pre))
    span = float(np.nanquantile(np.abs(pre - centre), 0.99)) or 1e-9
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    for ax, cond in zip(axes[:2], ["control", "stimulated"]):
        m = (cells["condition"] == cond).to_numpy()
        ax.scatter(u[~m, 0], u[~m, 1], s=1.2, c=C_FAINT, linewidths=0)
        sct = ax.scatter(u[m, 0], u[m, 1], s=2.2, c=pre[m], cmap=DIVERGING,
                         vmin=centre - span, vmax=centre + span, linewidths=0)
        ax.set_title(f"{cond}   {int(m.sum()):,} cells", fontsize=10)
    fig.colorbar(sct, ax=axes[1], shrink=0.8, label="unit pre-activation")
    for ct in order:
        m = (cells["cell_type"] == ct).to_numpy()
        axes[2].scatter(u[m, 0], u[m, 1], s=1.2, c=C_FAINT, linewidths=0)
        axes[2].annotate(ct, (np.median(u[m, 0]), np.median(u[m, 1])),
                         fontsize=7.5, ha="center", fontweight="bold",
                         color="#0b0b0b",
                         bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                   ec="none", alpha=0.8))
    axes[2].set_title("cell type", fontsize=10)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(f"{TARGET.replace('REACTOME_', '').replace('_', ' ')}: the "
                 "encoder's read per cell, standard scanpy frame", y=1.02)
    save(fig, figs / "fig4_umap.png")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs" / "units",
                    help="where the read-out wrote its tables")
    ap.add_argument("--figs", type=Path, default=None,
                    help="where to write the figures (default: <out-dir>/figs)")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    figs = (args.figs or out_dir / "figs").resolve()
    figs.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(STYLE)

    d = load(out_dir)
    print(f"{len(d['units']):,} units, {len(d['cells']):,} cells, "
          f"{len(d['order'])} cell types, target at "
          f"{d['summary']['unit_rank_target']}\nwriting to {figs}")
    fig1_ranked_units(d, figs)
    fig2_target_activation(d, figs)
    fig3_ig_target(d, figs)
    fig4_umap(d, figs)


if __name__ == "__main__":
    main()
