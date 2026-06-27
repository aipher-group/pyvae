from pathlib import Path
from urllib.request import urlretrieve

import scanpy as sc


def load_kang(data_folder=".", normalize=True, n_genes=None, return_path=False):
    """
    Load the Kang et al. 2018 PBMC IFN-beta stimulation dataset.

    This dataset contains ~25 000 peripheral blood mononuclear cells (PBMCs)
    profiled under two conditions: control (unstimulated) and stimulated
    (IFN-beta treated).

    The file is downloaded automatically from Figshare on the first call
    and cached locally.  Processing steps in order:

    1. Create the destination directory and define local file paths.
    2. Download from Figshare if the file is missing or empty.
       Wrap the read in error handling: if the file is corrupt, delete it,
       re-download, and try once more.
    3. Fix the condition labels: rename "ctrl" -> "control" and
       "stim" -> "stimulated" in obs["label"], then rename that column
       to "condition".
    4. Preserve the raw counts in adata.layers["counts"] before any
       transformation.
    5. If normalize is True: apply library-size normalisation followed
       by log1p.
    6. If n_genes is set: select the top n highly-variable genes using
       scanpy's dispersion-based seurat flavour on the (log-normalised)
       expression matrix, then subset adata. Best used with normalize=True.
    7. Write the processed object to disk and return it (or its path).

    Parameters
    ----------
    data_folder : str or Path
        Directory where files are stored / downloaded.
    normalize : bool
        Apply library-size normalisation and log1p.
    n_genes : int or None
        Keep only the top n highly-variable genes.
    return_path : bool
        Return the saved .h5ad path instead of the AnnData object.

    Returns
    -------
    adata : AnnData  (or Path if return_path is True)
    """

    KANG_URL = "https://figshare.com/ndownloader/files/34464122"

    data_folder = Path(data_folder)
    data_folder.mkdir(parents=True, exist_ok=True)

    source_path = data_folder / "kang_counts_25k.h5ad"
    out_path = data_folder / "kang_processed.h5ad"

    if not source_path.exists() or source_path.stat().st_size == 0:
        urlretrieve(KANG_URL, source_path)

    try:
        adata = sc.read_h5ad(source_path)
    except Exception:
        source_path.unlink(missing_ok=True)
        urlretrieve(KANG_URL, source_path)
        adata = sc.read_h5ad(source_path)

    adata.obs["label"] = adata.obs["label"].replace(
        {"ctrl": "control", "stim": "stimulated"}
    )
    adata.obs = adata.obs.rename(columns={"label": "condition"})

    adata.layers["counts"] = adata.X.copy()

    if normalize:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    if n_genes is not None:
        # Dispersion-based `seurat` flavour (the scanpy default) on the
        # log-normalised matrix. Same intent as `seurat_v3` (keep the top
        # n_genes most variable genes) but pure numpy/scipy — no scikit-misc.
        sc.pp.highly_variable_genes(adata, n_top_genes=n_genes)
        adata = adata[:, adata.var["highly_variable"]].copy()

    adata.write_h5ad(out_path)

    if return_path:
        return out_path

    return adata
