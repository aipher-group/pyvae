from dataclasses import dataclass, field
from itertools import chain, repeat  # noqa: F401
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd


@dataclass
class InformedModelConfig:
    """
    Container for the biological configuration of an iVAE model.

    Fields
    ------
    model_kind : str
        Identifier for the model variant.  Supported values:
        "ivae_reactome_hierarchical_d2" and "ivae_reactome".
    frac : float
        Sampling fraction (kept for API compatibility; not used in the
        PyTorch implementation).
    n_encoding_layers : int
        Number of constrained encoder layers (usually 2).
    adj_name : list of str
        Human-readable name for each adjacency layer, e.g. ["d2_pathways"].
    adj_activ : list of str
        Activation function for each layer, e.g. ["tanh"].
    input_genes : Any
        Index of genes retained after intersecting the expression data with
        the adjacency matrix (output of sync_gexp_adj).
    layer_entity_names : list
        Column names (pathway names) for each adjacency DataFrame.
    model_layer : list of pd.DataFrame
        The actual binary adjacency DataFrames, one per encoding layer.
    """

    model_kind: str = ""
    frac: float = 0.0
    n_encoding_layers: int = 1
    adj_name: List[str] = field(default_factory=list)
    adj_activ: List[str] = field(default_factory=list)
    input_genes: Any = None
    layer_entity_names: List[Any] = field(default_factory=list)
    model_layer: List[Any] = field(default_factory=list)


def get_reactome_adj(pth: Optional[Path] = None) -> pd.DataFrame:
    """
    Parse a Reactome GMT file into a binary gene x pathway indicator matrix.

    GMT format (one pathway per line, tab-separated):
        PATHWAY_NAME  <tab>  DESCRIPTION  <tab>  GENE1  <tab>  GENE2  ...

    Read the file line by line, unpack each line into a pathway name and its
    gene list, then pivot the result into a wide binary DataFrame:
    index = gene symbols, columns = pathway names, values in {0, 1}.

    Parameters
    ----------
    pth : Path or str
        Path to a Reactome GMT file.  Required.

    Returns
    -------
    pd.DataFrame, shape (n_genes, n_pathways)
        Binary indicator matrix.

    Raises
    ------
    ValueError
        If pth is None.
    """
    if pth is None:
        raise ValueError(
            "pth is required for pyvae - pass the GMT file path explicitly."
        )

    pathways = {}
    with Path(pth).open("r") as f:
        for line in f:
            name, _, *genes = line.strip().split("\t")
            pathways[name] = genes

    records = chain.from_iterable(
        zip(repeat(name), genes) for name, genes in pathways.items()
    )
    df_long = pd.DataFrame(records, columns=["geneset", "genesymbol"])

    reactome = (
        df_long.drop_duplicates()
        .assign(belongs_to=1)
        .pivot(columns="geneset", index="genesymbol", values="belongs_to")
        .fillna(0)
    )
    return reactome


def get_reactome_hierarchical_adj(pth: Any) -> pd.DataFrame:
    """
    Load a pre-built hierarchical Reactome indicator matrix.

    The file is a TSV with rows = gene symbols (first column as index),
    columns = pathway names, values in {0, 1}.

    Depth-2 (D2) means the pathway hierarchy was truncated at level 2,
    giving a manageable number of pathways (~1 000) while preserving
    biological resolution.

    Parameters
    ----------
    pth : str or Path
        Path to the TSV file (e.g. "resources/Matrix_1_genes_D2.tsv").

    Returns
    -------
    pd.DataFrame, shape (n_genes, n_pathways)
        Binary indicator matrix.
    """
    # return pd.read_csv(pth, sep="\t", index_col=0)
    return pd.read_csv(
        pth, sep="\t", index_col=0, keep_default_na=False, na_values=[""]
    )


def sync_gexp_adj(genes, adj: pd.DataFrame):
    """
    Intersect a gene list with the adjacency matrix index.

    Ensures that the gene expression matrix and the adjacency matrix share
    exactly the same genes before building the model.  Genes present in one
    but not the other are silently dropped.

    Parameters
    ----------
    genes : array-like of str
        Gene names from the expression dataset (e.g. adata.var_names).
    adj : pd.DataFrame
        Binary adjacency matrix with gene symbols as its index.

    Returns
    -------
    gene_list : pd.Index
        Intersection of genes and adj.index.
    adj_filtered : pd.DataFrame
        adj restricted to gene_list rows (all columns kept).
    """

    gene_list = adj.index.intersection(genes)
    adj_filtered = adj.loc[gene_list, :]
    return gene_list, adj_filtered


def build_model_config(
    genes,
    model_kind: str,
    frac: Optional[float] = None,
    resources_dir: Optional[Any] = None,
) -> InformedModelConfig:
    """
    Build an InformedModelConfig for the given model kind.

    Supported model kinds
    ---------------------
    "ivae_reactome_hierarchical_d2"
        Uses the depth-2 Reactome hierarchy stored in
        resources_dir / "Matrix_1_genes_D2.tsv".
        Call get_reactome_hierarchical_adj then sync_gexp_adj.

    "ivae_reactome"
        Uses the full Reactome GMT file at resources_dir.
        Call get_reactome_adj then sync_gexp_adj.

    Both branches return an InformedModelConfig with n_encoding_layers=1.

    Parameters
    ----------
    genes : array-like of str
        Gene names from the expression dataset.
    model_kind : str
        One of "ivae_reactome_hierarchical_d2" or "ivae_reactome".
    frac : float, optional
        Sampling fraction (stored in config, not used in PyTorch version).
    resources_dir : str or Path, optional
        For "ivae_reactome_hierarchical_d2": directory containing
        Matrix_1_genes_D2.tsv.
        For "ivae_reactome": path to the GMT file itself.

    Returns
    -------
    InformedModelConfig

    Raises
    ------
    ValueError
        If resources_dir is None for a model kind that requires it.
    NotImplementedError
        If model_kind is not one of the supported values.
    """
    if model_kind == "ivae_reactome_hierarchical_d2":
        if resources_dir is None:
            raise ValueError(
                "resources_dir must be provided for ivae_reactome_hierarchical_d2 "
                "(directory containing Matrix_1_genes_D2.tsv)."
            )
        adj_path = Path(resources_dir) / "Matrix_1_genes_D2.tsv"
        adj = get_reactome_hierarchical_adj(adj_path)

        input_genes, adj = sync_gexp_adj(genes, adj)

        return InformedModelConfig(
            model_kind=model_kind,
            frac=frac if frac is not None else 0.0,
            n_encoding_layers=1,
            adj_name=["d2_pathways"],
            adj_activ=["tanh"],
            input_genes=input_genes,
            layer_entity_names=[adj.columns],
            model_layer=[adj],
        )

    if model_kind == "ivae_reactome":
        if resources_dir is None:
            raise ValueError(
                "resources_dir must be the path to the Reactome GMT file for ivae_reactome."
            )
        adj = get_reactome_adj(resources_dir)

        input_genes, adj = sync_gexp_adj(genes, adj)

        return InformedModelConfig(
            model_kind=model_kind,
            frac=frac if frac is not None else 0.0,
            n_encoding_layers=1,
            adj_name=["pathways"],
            adj_activ=["tanh"],
            input_genes=input_genes,
            layer_entity_names=[adj.columns],
            model_layer=[adj],
        )

    raise NotImplementedError(
        f"model_kind '{model_kind}' is not supported. "
        "Choose from: ivae_reactome_hierarchical_d2, ivae_reactome."
    )


def swap_condition(
    cov: pd.DataFrame,
    from_label: str,
    to_label: str,
    prefix: str = "condition",
) -> pd.DataFrame:
    """Return a copy of ``cov`` with the one-hot condition columns flipped.

    Zeroes the ``{prefix}_{from_label}`` column and sets
    ``{prefix}_{to_label}`` to 1, for every row. All other columns (e.g.
    ``cell_type_*``) are left unchanged. Both column names must already exist
    in ``cov.columns``.

    ``prefix`` names the obs column the one-hots came from, so any categorical
    covariate can be swapped, not only one called ``condition``. The other
    covariate blocks are untouched by construction, which is what makes the
    swap a contrast at fixed batch: flip ``cohort`` and every ``study_id_*``
    column stays where it was.

    Used to build the ``cov_to`` argument of
    ``InformedVAE.predict_counterfactual``: encode with the cell's real
    covariate, decode with the swapped covariate, and read out the predicted
    counts under the counterfactual condition.

    Parameters
    ----------
    cov : covariate DataFrame produced by ``pd.get_dummies(obs[[...]])``, with
        one-hot columns including at least ``{prefix}_{from_label}`` and
        ``{prefix}_{to_label}``.
    from_label : the condition label the cells were actually observed under
        (e.g. ``"control"``).
    to_label : the counterfactual condition label to decode under (e.g.
        ``"stimulated"``).
    prefix : the one-hot column prefix, i.e. the name of the obs column
        ``pd.get_dummies`` was given. Defaults to ``"condition"``; pass
        ``"cohort"``, ``"treatment"`` or whatever the covariate is called.

    Returns
    -------
    cov_to : a copy of ``cov`` with the two condition columns flipped. Other
        columns (cell type, etc.) are unchanged.
    """
    from_col = f"{prefix}_{from_label}"
    to_col = f"{prefix}_{to_label}"
    missing = [c for c in (from_col, to_col) if c not in cov.columns]
    if missing:
        raise KeyError(
            f"columns not found in cov: {missing}; expected both "
            f"{from_col!r} and {to_col!r}. Make sure pd.get_dummies was called "
            f"on the FULL obs (or use pd.Categorical with an explicit categories "
            f"list) so both condition columns exist before splitting into "
            f"subsets."
        )
    cov_to = cov.copy()
    cov_to[from_col] = 0
    cov_to[to_col] = 1
    return cov_to


def permute_adjacency(adj, seed: int = 42):
    """Shuffle an adjacency's gene rows: the null for "does the annotation matter".

    Returns a copy of ``adj`` with its rows in a seeded random order. Because a
    gene's whole membership row travels with it, this is a **relabelling of the
    genes** rather than a reshuffling of the structure, and that makes the null
    tighter than it first looks. Preserved exactly: unit count, every single
    pathway's size, the multiset of per-gene membership counts, and the entire
    pathway-by-pathway overlap matrix -- the partition over gene slots is
    untouched and only the slot-to-identity mapping moves. Destroyed: which
    *named* genes share a unit, and therefore any biology a unit could learn.

    So a model fitted on the permuted map is matched to the real one in capacity,
    sparsity and overlap topology, and differs in nothing but which genes sit
    where. That is what makes real-against-permuted a test of the annotation's
    *content* rather than of sparsity as a regulariser.

    This is the control that separates the two. Without it, a masked layer
    beating a dense one shows only that sparsity helps. With it, measured on
    three datasets: with the covariate routed through the pathway layer the real
    map beats a permuted one by 5.9 to 13.9 nats of validation loss, while in the
    shipped gene-level conditioning the same comparison is 0.6 to 2.2 nats and the
    read-out is reproduced at rho 0.97.

    Parameters
    ----------
    adj : pandas.DataFrame or numpy.ndarray or torch.Tensor
        Gene-by-pathway adjacency, shape ``(n_genes, n_pathways)``. A DataFrame
        keeps its index and columns, so the permuted map is a drop-in
        replacement: unit *names* are unchanged and only their membership moves.
        That is deliberate -- the point is to ask whether the name still predicts
        anything.
    seed : int, default 42
        Row order is a function of this alone.

    Returns
    -------
    Same type as ``adj``, with rows permuted.

    Examples
    --------
    >>> perm = permute_adjacency(adj_df, seed=42)
    >>> (perm.sum(axis=0) == adj_df.sum(axis=0)).all()   # set sizes preserved
    True
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    try:
        import pandas as pd

        is_frame = isinstance(adj, pd.DataFrame)
    except ImportError:  # pragma: no cover - pandas is a hard dependency
        is_frame = False

    if is_frame:
        order = rng.permutation(adj.shape[0])
        out = adj.iloc[order].copy()
        out.index = adj.index
        return out

    try:
        import torch

        if isinstance(adj, torch.Tensor):
            order = torch.as_tensor(rng.permutation(adj.shape[0]),
                                    device=adj.device)
            return adj[order].clone()
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass

    arr = np.asarray(adj)
    return arr[rng.permutation(arr.shape[0])].copy()
