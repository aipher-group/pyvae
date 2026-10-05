"""Tests for the Phase 1e changes to datasets.py.

Covers:

- ``KANG_URL`` is the API-endpoint URL, not the WAF-fronted one.
- ``target_sum`` parameter defaults to 1e4 and is threaded through to
  ``sc.pp.normalize_total``.
- ``kang_processed.h5ad`` is written only when ``return_path=True``.
- ``_rename_condition_labels`` handles both categorical and object-dtype
  label columns, and renames the column to 'condition'.

Design decisions:

- The download and read steps are stubbed via monkeypatch so the tests
  don't touch the network or the ~200MB h5ad file. A tiny synthetic AnnData
  stands in for the real one.
- The rename helper is tested in isolation because it's the piece most
  likely to break under pandas API changes (categorical .replace deprecation)
  — a targeted unit test catches that regression cheaply.
- Every test that writes to disk uses ``pytest``'s ``tmp_path`` fixture,
  so we never pollute the real ``experiments/kang/`` directory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import pyvae.datasets as ds
from pyvae.datasets import KANG_URL, _rename_condition_labels, load_kang


# ---------- KANG_URL: module-level constant, correct value ----------


class TestKangUrl:
    def test_kang_url_is_module_level_constant(self):
        """The URL is accessible without constructing the loader."""
        assert KANG_URL == ds.KANG_URL

    def test_kang_url_uses_api_endpoint(self):
        """The API URL is more reliable than the ndownloader WAF-fronted one."""
        assert KANG_URL == "https://api.figshare.com/v2/file/download/34464122"


# ---------- _rename_condition_labels: categorical + object dtypes ----------


class TestRenameConditionLabels:
    def test_object_dtype_labels_are_renamed(self):
        obs = pd.DataFrame({"label": ["ctrl", "stim", "ctrl"]})
        out = _rename_condition_labels(obs)
        assert list(out["condition"]) == ["control", "stimulated", "control"]

    def test_categorical_labels_are_renamed_via_cat_api(self):
        """Categorical dtype must go through cat.rename_categories."""
        obs = pd.DataFrame({
            "label": pd.Categorical(["ctrl", "stim", "ctrl", "stim"])
        })
        out = _rename_condition_labels(obs)
        assert list(out["condition"]) == ["control", "stimulated", "control", "stimulated"]

    def test_categorical_output_preserves_dtype(self):
        """After renaming, a categorical stays categorical."""
        obs = pd.DataFrame({
            "label": pd.Categorical(["ctrl", "stim", "ctrl"])
        })
        out = _rename_condition_labels(obs)
        assert isinstance(out["condition"].dtype, pd.CategoricalDtype)

    def test_column_renamed_from_label_to_condition(self):
        obs = pd.DataFrame({"label": ["ctrl", "stim"]})
        out = _rename_condition_labels(obs)
        assert "condition" in out.columns
        assert "label" not in out.columns

    def test_original_dataframe_is_not_mutated(self):
        """Helper returns a new frame; caller's copy stays intact."""
        obs = pd.DataFrame({"label": ["ctrl", "stim"]})
        _ = _rename_condition_labels(obs)
        assert list(obs["label"]) == ["ctrl", "stim"]
        assert "label" in obs.columns

    def test_no_deprecation_warning_on_categorical(self):
        """The categorical path should not trigger the .replace deprecation."""
        obs = pd.DataFrame({
            "label": pd.Categorical(["ctrl", "stim", "ctrl"])
        })
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            warnings.simplefilter("error", FutureWarning)
            _ = _rename_condition_labels(obs)


# ---------- load_kang: target_sum default and threading ----------


def _make_fake_adata(n_cells: int = 12, n_genes: int = 8) -> AnnData:
    """Tiny synthetic AnnData that mimics the raw Kang download shape.

    - X: integer-valued expression matrix (counts, since normalize hasn't run yet)
    - obs['label']: categorical with 'ctrl' / 'stim' values
    """
    rng = np.random.default_rng(0)
    X = rng.integers(0, 20, size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame({
        "label": pd.Categorical(["ctrl"] * (n_cells // 2) + ["stim"] * (n_cells // 2)),
    })
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    return AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def patch_download_and_read(monkeypatch, tmp_path):
    """Stub out urlretrieve and sc.read_h5ad so tests don't touch the network."""
    def fake_urlretrieve(url, path):
        # Write a placeholder byte so the size-check passes.
        with open(path, "wb") as f:
            f.write(b"x")

    def fake_read_h5ad(_path):
        return _make_fake_adata()

    monkeypatch.setattr(ds, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(ds.sc, "read_h5ad", fake_read_h5ad)
    return tmp_path


class TestLoadKangTargetSum:
    def test_target_sum_defaults_to_1e4(self):
        """The signature default is 1e4 — the fixed value that keeps runs comparable."""
        import inspect
        sig = inspect.signature(load_kang)
        assert sig.parameters["target_sum"].default == 1e4

    def test_target_sum_is_passed_to_normalize_total(self, patch_download_and_read, monkeypatch):
        """A non-default target_sum reaches sc.pp.normalize_total."""
        seen = {}

        def fake_normalize_total(adata, target_sum=None):
            seen["target_sum"] = target_sum

        monkeypatch.setattr(ds.sc.pp, "normalize_total", fake_normalize_total)
        # log1p still runs; that's OK — it's a no-op on our fake data.
        load_kang(data_folder=patch_download_and_read, target_sum=5000.0)
        assert seen["target_sum"] == 5000.0

    def test_default_target_sum_reaches_normalize_total(self, patch_download_and_read, monkeypatch):
        """The default 1e4 reaches sc.pp.normalize_total when no override is passed."""
        seen = {}

        def fake_normalize_total(adata, target_sum=None):
            seen["target_sum"] = target_sum

        monkeypatch.setattr(ds.sc.pp, "normalize_total", fake_normalize_total)
        load_kang(data_folder=patch_download_and_read)
        assert seen["target_sum"] == 1e4


# ---------- load_kang: return_path gating on the write ----------


class TestReturnPathGating:
    def test_return_path_false_does_not_write_processed_file(
        self, patch_download_and_read
    ):
        """Default behaviour: no kang_processed.h5ad on disk after loading."""
        tmp_path = patch_download_and_read
        adata = load_kang(data_folder=tmp_path, normalize=False)
        assert not isinstance(adata, type(tmp_path))  # returned the AnnData, not a Path
        assert not (tmp_path / "kang_processed.h5ad").exists(), (
            "return_path=False must not write kang_processed.h5ad — otherwise a "
            "stale copy from an earlier arg set could sit on disk looking authoritative."
        )

    def test_return_path_true_writes_and_returns_path(self, patch_download_and_read):
        tmp_path = patch_download_and_read
        result = load_kang(data_folder=tmp_path, normalize=False, return_path=True)
        assert isinstance(result, type(tmp_path))
        assert result == tmp_path / "kang_processed.h5ad"
        assert result.exists()


# ---------- load_kang: end-to-end sanity with stubbed download ----------


class TestLoadKangEndToEnd:
    def test_returns_anndata_by_default(self, patch_download_and_read):
        adata = load_kang(data_folder=patch_download_and_read, normalize=False)
        assert isinstance(adata, AnnData)

    def test_condition_column_is_renamed(self, patch_download_and_read):
        adata = load_kang(data_folder=patch_download_and_read, normalize=False)
        assert "condition" in adata.obs.columns
        assert "label" not in adata.obs.columns
        assert set(adata.obs["condition"].unique()) == {"control", "stimulated"}

    def test_raw_counts_preserved_in_layer(self, patch_download_and_read):
        """adata.layers['counts'] holds the pre-normalisation matrix."""
        adata = load_kang(data_folder=patch_download_and_read, normalize=False)
        assert "counts" in adata.layers
        # X and layers['counts'] are the same when normalize=False.
        np.testing.assert_array_equal(np.asarray(adata.X), np.asarray(adata.layers["counts"]))
