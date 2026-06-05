"""
Dataset loading for pyvae.
"""

from pathlib import Path
from urllib.request import urlretrieve  # noqa: F401

import scanpy as sc  # noqa: F401


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
       the seurat_v3 flavour on the raw count layer, then subset adata.
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
    # source_path: path to the raw downloaded file
    # out_path: path where the processed file is saved
    raise NotImplementedError
