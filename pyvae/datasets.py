"""Loaders for the datasets used in the pyvae examples and notebooks.

Phase 1e housekeeping in ``load_kang``:

- ``target_sum`` is now an explicit parameter with default ``1e4``. Previously
  scanpy's default (``target_sum=None``) scaled each cell to the dataset
  median library size, which shifts between runs whenever HVG selection
  changes — meaning two runs with different arg sets produced losses that
  could not be compared. A fixed target restores comparability.
- ``kang_processed.h5ad`` is now written only when ``return_path=True``.
  Previously the write happened unconditionally and the file was never read
  back, so a stale copy from an earlier arg set could sit on disk looking
  authoritative.
- Condition-label renaming now uses ``label.cat.rename_categories(...)``
  when the column is a pandas ``CategoricalDtype``. ``Series.replace`` on
  a categorical is deprecated and will stop renaming in a future pandas.
  A non-categorical column still uses ``.replace``, so the helper is safe
  regardless of what scanpy hands us.
- ``KANG_URL`` promoted to module level and pointed at
  ``https://api.figshare.com/v2/file/download/34464122``. The previous
  Figshare download URL routes through a WAF that sometimes returns an
  HTML browser-check page instead of the h5ad file.
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import scanpy as sc


# Module-level constant. Promoted from inside ``load_kang`` so tests can assert
# on its value directly and so downstream tools can look it up without
# constructing the loader.
KANG_URL = "https://api.figshare.com/v2/file/download/34464122"


def _rename_condition_labels(obs: pd.DataFrame) -> pd.DataFrame:
    """Rename ctrl/stim to control/stimulated, then rename the column to 'condition'.

    Handles both categorical and object-dtype label columns. For categoricals,
    uses ``.cat.rename_categories`` (the currently-supported API); for
    non-categorical dtypes, falls back to ``.replace``. Returns a new
    DataFrame; the input is not mutated.

    Parameters
    ----------
    obs : pd.DataFrame
        The AnnData ``obs`` frame, expected to contain a ``"label"`` column
        with values in ``{"ctrl", "stim"}``.

    Returns
    -------
    pd.DataFrame
        A copy of ``obs`` with ``"label"`` renamed to ``"condition"`` and
        values renamed to ``"control"`` / ``"stimulated"``.
    """
    obs = obs.copy()
    mapping = {"ctrl": "control", "stim": "stimulated"}
    label = obs["label"]
    if isinstance(label.dtype, pd.CategoricalDtype):
        obs["label"] = label.cat.rename_categories(mapping)
    else:
        obs["label"] = label.replace(mapping)
    obs = obs.rename(columns={"label": "condition"})
    return obs


def load_kang(
    data_folder=".",
    normalize=True,
    n_genes=None,
    return_path=False,
    target_sum: float = 1e4,
):
    """Load the Kang et al. 2018 PBMC IFN-beta stimulation dataset.

    This dataset contains ~25 000 peripheral blood mononuclear cells (PBMCs)
    profiled under two conditions: control (unstimulated) and stimulated
    (IFN-beta treated).

    The file is downloaded automatically from Figshare on the first call
    and cached locally. Processing steps in order:

    1. Create the destination directory and define local file paths.
    2. Download from Figshare if the file is missing or empty. Wrap the read
       in error handling: if the file is corrupt, delete it, re-download,
       and try once more.
    3. Fix the condition labels: rename ``"ctrl" -> "control"`` and
       ``"stim" -> "stimulated"`` in ``obs["label"]``, then rename that
       column to ``"condition"``. Uses ``.cat.rename_categories`` for
       categorical columns (the currently-supported API for renaming
       categoricals) and ``.replace`` for object-dtype columns.
    4. Preserve the raw counts in ``adata.layers["counts"]`` before any
       transformation.
    5. If ``normalize`` is True: apply library-size normalisation to a fixed
       ``target_sum`` per cell (default ``1e4``), followed by log1p. A fixed
       target keeps losses comparable across runs; scanpy's default
       (``target_sum=None``) instead scales to the dataset median, which
       shifts whenever the cell or gene selection changes.
    6. If ``n_genes`` is set: select the top n highly-variable genes using
       scanpy's dispersion-based seurat flavour on the (log-normalised)
       expression matrix, then subset adata. Best used with ``normalize=True``.
    7. If ``return_path`` is True, write the processed object to disk and
       return its path; otherwise return the AnnData object directly. The
       write is gated on the flag so a stale ``kang_processed.h5ad`` from
       an earlier arg set can't sit on disk looking authoritative.

    Parameters
    ----------
    data_folder : str or Path
        Directory where files are stored / downloaded.
    normalize : bool, default True
        Apply library-size normalisation and log1p.
    n_genes : int or None, default None
        Keep only the top n highly-variable genes.
    return_path : bool, default False
        When True, write ``kang_processed.h5ad`` to disk and return its
        path. When False, return the in-memory AnnData without writing.
    target_sum : float, default 1e4
        Target library size per cell for ``sc.pp.normalize_total``. Ignored
        when ``normalize=False``.

    Returns
    -------
    AnnData or Path
        The processed AnnData object, or the path to the written .h5ad file
        when ``return_path=True``.
    """
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

    adata.obs = _rename_condition_labels(adata.obs)

    adata.layers["counts"] = adata.X.copy()

    if normalize:
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)

    if n_genes is not None:
        # Dispersion-based `seurat` flavour (the scanpy default) on the
        # log-normalised matrix. Same intent as `seurat_v3` (keep the top
        # n_genes most variable genes) but pure numpy/scipy — no scikit-misc.
        sc.pp.highly_variable_genes(adata, n_top_genes=n_genes)
        adata = adata[:, adata.var["highly_variable"]].copy()

    if return_path:
        adata.write_h5ad(out_path)
        return out_path

    return adata
