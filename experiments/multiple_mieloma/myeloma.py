"""Shared loading, contrast and plotting helpers for the myeloma experiment.

All four workflow steps import this module -- `01_architecture.py`,
`02_biology.ipynb`, `03_donor_test.py` and `04_figures.py` -- so the filter
chain, the pathway map, the phenotype harmonisation, the **architecture** and
the palette are defined once. Before it existed the same twenty lines were
pasted into four files and had already drifted: two of them disagreed about
whether doublets were dropped.

The experiment analyses `ANALYSIS_STUDIES` — the two datasets that span all four
disease stages — one at a time, **never pooled**. The reason is in FINDINGS.md
under "Decisions, dataset by dataset": 69% of the atlas's non-cancer donors sit
in studies that contain no patients, so a pooled disease contrast is
substantially a contrast between studies and `study_id` cannot be adjusted for
without becoming collinear with the thing being estimated.

**Everything in here is used by the current workflow.** That is enforced by
hand rather than by a tool, so when something loses its last caller, delete it
rather than leaving it -- git history is the archive. The earlier pipeline's
retired helpers (a measured depth axis, a model-free pseudobulk path, a handful
of one-line conveniences) were removed with the `sandbox/` tree they served.

Nothing here trains a model or reads `outputs/`. It is the layer below both.

Paths are resolved against this file's own directory, so an import works from
anywhere and a notebook run under papermill with a different cwd still finds the
data.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from pyvae import build_model_config, pathway_activity

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESOURCES_DIR = HERE / "resources"
OUT_DIR = HERE / "outputs"

TCELL_H5AD = DATA_DIR / "Tcell.h5ad"
DONOR_META = DATA_DIR / "metadata-donor.csv"
#: Defines the gene panel and the encoder mask. Hierarchical, which is fine for
#: a mask and fatal for a count -- so it is never the thing tested.
GMT_PATH = RESOURCES_DIR / "c2.cp.reactome.v7.5.1.symbols.gmt"
#: Not a read-out any more. `01_architecture.py` scores the *baseline* on these
#: 50 curated programmes -- the gene-level competitive test the architecture
#: replaces -- because "the right answer ranks first even with a permuted
#: annotation" is the evidence that the baseline was annotation-blind. No result
#: is reported on this collection. See `resources/PROVENANCE.md`.
HALLMARK_GMT = RESOURCES_DIR / "h.all.v7.5.1.symbols.gmt"

#: Disease stages in order. Not an alphabetical sort — it is the clinical
#: progression, and several read-outs take the first and last element.
STAGE_ORDER = ["Non", "MGUS", "SMM", "MM"]

#: The two studies that span all four disease stages. Each is analysed as an
#: independent dataset and the two are never pooled: in this atlas the non-cancer
#: donors come mostly from studies that contribute no patients, so a pooled fit
#: makes "Non versus MM" and "these studies versus those" nearly the same
#: comparison. Fitting one study at a time removes the study covariate entirely
#: rather than merely shrinking it.
#:
#: Bailur_2019 (no SMM) and Liu_2021 (no non-cancer donors) are deliberately not
#: here. Neither can carry a four-stage contrast, and Bailur's control group is
#: anomalous in a way no recorded metadata field indexes — see FINDINGS.md.
#: The atlas labels its own cohort `This_study`, which means nothing outside the
#: source file. It is renamed on load to the `<FirstAuthor>_<Year>` convention
#: the other ten studies already use -- the atlas is Foster et al. 2026,
#: *Nat Commun* 17:2400 -- so no figure, table or output path carries the label.
#: Applied after the donor-metadata join, which keys on the original ids, and it
#: rewrites the matching `donor_id` prefix so the two never disagree.
STUDY_RENAME = {"This_study": "Foster_2026"}

ANALYSIS_STUDIES = ["Zavidij_2020", "Foster_2026"]


#: T-cell lineages in the atlas, largest first. All five are fitted: the point
#: of not subsetting up front is that the model sees the whole T-cell manifold,
#: and the embedding then has real lineage structure to overlay activations onto.
#: The three small ones are kept and flagged as under-powered rather than dropped.
SUBSETS = ["CD4", "CD8", "MAIT_gdT", "ISG", "Prolif."]

#: The lineages with enough cells for a per-lineage stage contrast to be worth
#: reading. The others get a column in the model and a warning in the read-out.
MAJOR_SUBSETS = ["CD4", "CD8"]


#: How `pheno` is harmonised to a level set that is *identical in both analysed
#: studies*, so a phenotype block can be conditioned on in each of them.
#:
#: The atlas already gives both studies one shared vocabulary -- Foster et al.
#: 2026 (Nat Commun 17:2400) integrate all 11 studies with scVI and then cluster
#: and phenotype **once**, so Zavidij's cells carry the atlas's labels, not
#: Zavidij's own. Only two levels stop that vocabulary being usable as a
#: covariate, and both are merged here for reasons in the data, not taste:
#:
#: 1. `Teff`, `IFN-stin.` and `Proliferating` are lineage-agnostic by
#:    construction. The paper defines Teff by effector molecules (TNF, IFNG)
#:    rather than by lineage, and the atlas's own `subset` column refuses to
#:    place Teff or IFN-stin. in CD4 or CD8 -- it files both under `ISG`. The
#:    paper resolves Teff's lineage by TCR clone sharing ("CD8+ Teff clones"),
#:    which cannot be reproduced here: Zavidij has no scTCR data at all
#:    (`has_tcr` is False for all 9,863 cells, 0 clone_ids), so a clone-based
#:    split would exist in one study and not the other.
#:    In `Foster_2026` `Teff` is also 95.5% MM, which makes it a near-perfect MM
#:    indicator, and it is the shallowest cluster in the study (median 764 genes
#:    against 914-1,721 elsewhere). Pooling the three removes both problems.
#: 2. `CD8+Tex` is absent from Zavidij (0 cells) and is 27 cells in `Foster_2026`,
#:    16 of them in donors the authors excluded. It is folded into `CD8+Tte`,
#:    the terminal state it is adjacent to. The atlas's `CD8.Tex_outlier` flag
#:    marks one donor, in Zheng_2021 -- not in either analysed study.
PHENO_MERGE = {
    "Teff": "lineage-agnostic",
    "IFN-stin.": "lineage-agnostic",
    "Proliferating": "lineage-agnostic",
    "CD8+Tex": "CD8+Tte",
}


#: Name of the cohort-by-lineage interaction block, and how its levels are spelled.
#: `__` because no subset or cohort label contains it.
STAGE_BLOCK = "stage"

#: Donor-metadata columns worth carrying into `.obs`. `noncancer_sampling` is
#: here because it looks like a confounder and is not one — see 01's section 1.
DONOR_COLS = ["cohort", "sex", "age", "noncancer_sampling", "risk_ISS",
              "risk_SMM_mayo", "analysed"]

#: The architecture, in one place because `02_biology.ipynb` and
#: `03_donor_test.py` must fit the *same* model for their read-outs to be
#: comparable, and because `01_architecture.py` has to be able to name the
#: baseline it beats.
#: The architecture as shipped: the covariate's effect is a dense
#: `(n_genes, n_cov)` term added *after* the masked layer, so it bypasses the
#: annotation entirely. Kept only so `01_architecture.py` has a baseline arm;
#: no result is read from it. See that script and `FINDINGS.md` R1.
ARCH_SHIPPED = dict(normalize="batch", informed_decoder=True,
                    nonneg_encoder=True, standardize_input=True)

#: The architecture every result uses, selected by `01_architecture.py`.
#:
#:   cov_mode="pathway"      the covariate's effect is a vector in pathway
#:                           coordinates, reaching a gene only through the mask
#:   cov_in_encoder=False    and it is the *whole* conditional effect, because
#:                           the covariate cannot also reach z
#:   nonneg_decoder=True     and a unit rising raises its genes, so a pathway's
#:                           sign means what it says
#:
#: Together these make the stage contrast one signed number per pathway, read
#: straight off `InformedVAE.pathway_contrast`. They cost validation loss where
#: the dense term is well estimated (+2.8 nats on `Zavidij_2020`, +9.7 on
#: `Foster_2026`); the justification is that the real annotation then beats a
#: size-matched shuffle by +5.9 and +10.7 nats against +0.6 and +1.3 before.
ARCH = dict(**ARCH_SHIPPED, cov_mode="pathway", cov_in_encoder=False,
            nonneg_decoder=True)
L2_LAMBDA = 1e-5
#: A trailing batch of one row crashes BatchNorm, which is why callers check
#: `n_train % BATCH_SIZE == 1` and move one cell to validation.
BATCH_SIZE = 128


def latent_dim_for(n_pathways: int) -> int:
    """Half the pathway units, which is the Kang experiment's rule."""
    return int(n_pathways) // 2


MPL_STYLE = {
    # 300 dpi on save, set once here rather than passed at every savefig call.
    # figure.dpi stays low so inline rendering in Jupyter is not sluggish.
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
}


def apply_style() -> None:
    """Make every figure in this experiment look like the others."""
    import matplotlib as mpl
    mpl.rcParams.update(MPL_STYLE)


# Categorical slots, checked all-pairs for colour-vision deficiency before use,
# and a diverging ramp built from the same two poles with a NEUTRAL midpoint.
# Two rules these exist to enforce. A hue is never generated: a fifth series
# folds into "other" rather than inventing a colour. And the diverging ramp is
# used for one quantity at one scale per figure -- reusing it at a second scale
# is the standard way to make a heatmap lie.
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_GREY, C_FAINT, C_INK = "#8a8a86", "#dcdcd8", "#0b0b0b"
#: The two studies always take the same two colours, in this order, in every
#: figure. Colour follows the dataset and never its rank.
STUDY_COLOURS = {"Zavidij_2020": C_BLUE, "Foster_2026": C_ORANGE}
#: The two lineages, likewise fixed.
LINEAGE_COLOURS = {"CD4": C_BLUE, "CD8": C_ORANGE}


def diverging():
    """Blue -> near-white -> orange, for a signed quantity centred on zero."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "blue_grey_orange", [C_BLUE, "#e8e8e6", C_ORANGE])


def short(pathway: str) -> str:
    """Strip the collection prefix and make a set name readable.

    `REACTOME_INTERFERON_SIGNALING` -> `interferon signaling`
    `HALLMARK_INTERFERON_ALPHA_RESPONSE` -> `interferon alpha response`
    """
    for prefix in ("REACTOME_", "HALLMARK_"):
        pathway = pathway.removeprefix(prefix)
    return pathway.replace("_", " ").lower()


def device() -> str:
    """Fastest available torch device: cuda, then Apple MPS, then cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def require_inputs() -> None:
    """Fail early and legibly when the ungettable inputs are missing."""
    missing = [p for p in (TCELL_H5AD, DONOR_META, GMT_PATH, HALLMARK_GMT)
               if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing inputs:\n  "
            + "\n  ".join(str(p) for p in missing)
            + "\n\nNeither can be fetched by a script — Zenodo serves the atlas "
              "behind a browser check. Run ./00_setup.sh for the download URLs "
              "and the expected paths; resources/PROVENANCE.md records which "
              "MSigDB collection does which job.")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_tcells(study: str | None = None, *, subset: str | None = None,
                analysed_only: bool = False,
                verbose: bool = False) -> ad.AnnData:
    """Bone-marrow T cells with donor metadata joined.

    This is the one definition of the cell set. Three filters, in this order:
    bone marrow, non-doublet, and a donor whose cohort is known. `study`
    restricts to one `study_id`, which is how the per-study fits get their data,
    and it takes the renamed id -- `Foster_2026`, not the atlas's `This_study`.

    `subset` additionally restricts to one lineage. It is **not** used by the
    fits: an earlier version of this experiment subset to CD8 before training,
    and that was the wrong order. Fitting the whole T-cell compartment gives the
    model 2 to 4 times the cells, gives the embedding real lineage structure to
    overlay onto, and lets the stage contrast be read per lineage from one fit
    rather than compared across separately-trained ones. The argument stays
    because the audit uses it to slice, and because reproducing the old CD8-only
    fit is then one keyword away.

    `scr_doublet` is a categorical of the *strings* "False" and "True" in this
    atlas rather than of bools, hence the string comparison — a plain truth test
    silently keeps every doublet.

    `analysed_only` honours the atlas's own donor-level `analysed` flag, which
    `dataDescription.xlsx` defines as "whether donor was included in analysis in
    analysis besides clustering/phenotyping". It is off by default so that
    existing outputs keep reproducing, and `02_biology.ipynb` turns it on. It
    only ever removes cells from `Foster_2026` -- 3 donors and 18,226 cells, 12.7%
    of that study; all 30 Zavidij donors are flagged True.

    A `pheno_h` column is always added: `pheno` with `PHENO_MERGE` applied, which
    is the phenotype level set shared by both analysed studies.
    """
    require_inputs()
    adata = ad.read_h5ad(TCELL_H5AD)
    donor_meta = pd.read_csv(DONOR_META, index_col="donor_id")
    cols = [c for c in DONOR_COLS if c in donor_meta.columns]
    adata.obs = adata.obs.merge(donor_meta[cols], left_on="donor_id",
                                right_index=True, how="left",
                                validate="many_to_one")
    for old, new in STUDY_RENAME.items():
        adata.obs["study_id"] = (adata.obs["study_id"].astype(str)
                                 .replace({old: new}).astype("category"))
        adata.obs["donor_id"] = (adata.obs["donor_id"].astype(str)
                                 .str.replace(f"{old}.", f"{new}.", regex=False))

    steps = [
        ("start", None),
        ("tissue == BM", adata.obs["tissue"] == "BM"),
        ("non-doublet", adata.obs["scr_doublet"].astype(str) == "False"),
        ("cohort known", adata.obs["cohort"].notna()),
    ]
    if subset is not None:
        steps.append((f"subset == {subset}", adata.obs["subset"] == subset))
    if analysed_only:
        steps.append(("analysed by authors",
                      adata.obs["analysed"].astype(str) == "True"))
    keep = np.ones(adata.n_obs, dtype=bool)
    rows = []
    for label, mask in steps:
        if mask is not None:
            keep &= np.asarray(mask)
        rows.append((label, int(keep.sum()),
                     adata.obs.loc[keep, "donor_id"].nunique()))
    adata = adata[keep].copy()

    adata.obs["pheno_h"] = (adata.obs["pheno"].astype(str)
                            .replace(PHENO_MERGE).astype("category"))

    if study is not None:
        sel = adata.obs["study_id"].astype(str) == str(study)
        if not sel.any():
            raise ValueError(
                f"study {study!r} has no bone-marrow CD8 cells after filtering; "
                f"available: {sorted(adata.obs['study_id'].astype(str).unique())}")
        adata = adata[np.asarray(sel)].copy()
        rows.append((f"study == {study}", adata.n_obs,
                     adata.obs["donor_id"].nunique()))

    if verbose:
        print("Filter attrition:")
        for label, n, d in rows:
            print(f"  [{label:<24s}] {n:>8,} cells   {d:>3d} donors")
        print(f"\nFinal: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
        print(f"       {adata.obs['donor_id'].nunique()} donors x "
              f"{adata.obs['cohort'].nunique()} cohorts x "
              f"{adata.obs['pheno'].nunique()} phenotypes x "
              f"{adata.obs['subset'].nunique()} lineages")
        print("\ncells per lineage:")
        counts = adata.obs["subset"].value_counts()
        for name in [s for s in SUBSETS if s in counts.index]:
            print(f"  {name:<10} {counts[name]:>8,}")
    return adata


def build_pathways(adata: ad.AnnData):
    """Reactome adjacency for this panel, and `adata` restricted to its genes.

    `build_model_config` takes `var_names` rather than a frame on purpose: the
    full 116k x 34k matrix does not need to be materialised to decide which
    genes are in a pathway, and on a laptop it cannot be.

    Returns `(adata, adj_df, gene_names, pathway_names)` with `adata` subset and
    column-ordered to `gene_names`, so `adata.X` and `adj_df` rows always agree.
    """
    config = build_model_config(genes=adata.var_names,
                                model_kind="ivae_reactome",
                                resources_dir=str(GMT_PATH))
    adj_df = config.model_layer[0]
    gene_names = list(config.input_genes)
    return (adata[:, gene_names].copy(), adj_df, gene_names,
            list(adj_df.columns))


def adjacency_only():
    """The Reactome adjacency for this panel, without materialising any counts.

    `build_model_config` decides pathway membership from gene names alone, so a
    backed read of `var_names` is enough. That lets
    `03_compare_two_studies.ipynb` ask which genes two pathways share without
    loading 281,119 x 34,849 cells, and without re-deriving a gene list that
    could drift from the one the fits used.

    Returns `(adj_df, gene_names)`.
    """
    var_names = ad.read_h5ad(TCELL_H5AD, backed="r").var_names
    config = build_model_config(genes=var_names, model_kind="ivae_reactome",
                                resources_dir=str(GMT_PATH))
    return config.model_layer[0], list(config.input_genes)


# --------------------------------------------------------------------------- #
# Model-free read-outs
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Covariates
# --------------------------------------------------------------------------- #


def stage_col(subset: str, cohort: str) -> str:
    """Column name of one cohort-by-lineage cell in the covariate frame."""
    return f"{STAGE_BLOCK}_{subset}__{cohort}"


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def standard_umap(counts: pd.DataFrame, obs: pd.DataFrame, *, seed: int = 42,
                  n_hvg: int = 2000, n_pcs: int = 50, n_neighbors: int = 15):
    """The textbook scanpy embedding, starting from raw counts.

    `normalize_total` to 10k, `log1p`, highly variable genes, scale, PCA,
    neighbours, UMAP — scanpy's own tutorial recipe, and the one
    `experiments/kang/` used.

    It is the right frame for **one** experiment and the wrong one for a pooled
    atlas: run across all ten studies here it returns a map of who sequenced the
    cell, with the studies separating and the phenotypes not. That is why an
    earlier version of this experiment embedded on the atlas's integrated
    `X_scVI` latent instead. Fitting one study at a time removes the reason, so
    this is primary again and `X_scVI` becomes the comparison.

    Normalising here rather than reusing the atlas's `.X` keeps the notebook
    independent of somebody else's preprocessing: the only input is the counts
    layer.

    Returns an AnnData carrying `X_pca`, `X_umap` and the neighbour graph.
    """
    import scanpy as sc

    emb = ad.AnnData(X=np.asarray(counts.values, dtype=np.float32),
                     obs=obs.copy(),
                     var=pd.DataFrame(index=pd.Index(counts.columns, name="gene")))
    sc.pp.normalize_total(emb, target_sum=1e4)
    sc.pp.log1p(emb)
    sc.pp.highly_variable_genes(emb, n_top_genes=min(n_hvg, emb.n_vars - 1),
                                flavor="seurat")
    emb = emb[:, emb.var["highly_variable"].values].copy()
    sc.pp.scale(emb, max_value=10)
    sc.tl.pca(emb, n_comps=min(n_pcs, emb.n_obs - 1, emb.n_vars - 1),
              random_state=seed)
    sc.pp.neighbors(emb, n_neighbors=n_neighbors, random_state=seed)
    sc.tl.umap(emb, random_state=seed)
    return emb


#: The cumulative contrasts: every later stage against the healthy baseline.
LADDER_NAMES = [f"{STAGE_ORDER[0]}->{s}" for s in STAGE_ORDER[1:]]


def read_gmt(path, gene_names, *, min_genes: int = 10):
    """Gene-set adjacency over `gene_names`, from a GMT file.

    `build_model_config` is the route for the collection that becomes the encoder
    mask; this is the route for a collection that is only ever *tested*, which
    needs no model config and must be aligned to the gene panel the fit actually
    used. Sets are restricted to measured genes before the `min_genes` floor is
    applied, so the floor means "enough members we can see" and not "enough
    members annotated".

    Returns a 0/1 DataFrame indexed by `gene_names`, columns the set names.
    """
    position = {g: i for i, g in enumerate(gene_names)}
    names, columns = [], []
    for line in Path(path).read_text().splitlines():
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 3:
            continue
        hits = [position[g] for g in fields[2:] if g in position]
        if len(hits) < min_genes:
            continue
        column = np.zeros(len(gene_names), dtype=np.float32)
        column[hits] = 1.0
        names.append(fields[0])
        columns.append(column)
    if not names:
        raise ValueError(f"{path} shares no set of >= {min_genes} genes with the panel")
    return pd.DataFrame(np.array(columns).T, index=list(gene_names), columns=names)


def stage_contrast_weights(W_cov, cov_columns, subset: str, frm: str, to: str):
    """Per-gene logit shift for one lineage's `frm -> to` stage contrast.

    One weight difference, identical for every cell of that lineage, holding
    phenotype, sex and depth bin fixed by construction. Returns None when either
    column is absent, which happens when that lineage has no cells in that stage.
    """
    idx = {c: i for i, c in enumerate(cov_columns)}
    a, b = stage_col(subset, frm), stage_col(subset, to)
    if a not in idx or b not in idx:
        return None
    return W_cov[:, idx[b]] - W_cov[:, idx[a]]


def pooled_stage_contrast(W_cov, cov_columns, cell_counts, frm: str, to: str):
    """The all-lineage stage contrast, pooled from the per-lineage ones.

    A cell-count-weighted mean over whichever lineages have both stages, with the
    weight taken as the smaller of the two cell counts so a lineage that is
    well-sampled at one stage and nearly absent at the other cannot dominate.

    This is what a plain, non-interacted `cohort` block would have estimated, so
    reporting both answers "is the stage effect shared across lineages, or is it
    one lineage showing through a pooled average".

    `cell_counts` maps (subset, cohort) -> cells. Returns `(delta, weights)`.
    """
    deltas, weights = [], {}
    for s in SUBSETS:
        d = stage_contrast_weights(W_cov, cov_columns, s, frm, to)
        if d is None:
            continue
        w = float(min(cell_counts.get((s, frm), 0), cell_counts.get((s, to), 0)))
        if w <= 0:
            continue
        deltas.append(d * w)
        weights[s] = w
    if not deltas:
        return None, {}
    total = sum(weights.values())
    return np.sum(deltas, axis=0) / total, {k: v / total for k, v in weights.items()}
