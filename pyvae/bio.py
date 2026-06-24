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
