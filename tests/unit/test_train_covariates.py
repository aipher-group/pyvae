"""Tests for the Phase 1d changes to train.py.

Covers:

- ``_check_cov`` — all four rejection modes (no-cov / stray-cov / wrong-width /
  wrong-cell-count) and the accepting case.
- ``train_ivae`` refusing a conditional model.
- ``train_ivae_modern`` end-to-end with covariates.
- Every matrix argument accepts torch / numpy / pandas via ``as_float_tensor``.
- ``history["lr"]`` reports the rate the epoch actually trained at, not the
  one queued for the next.

Fixtures deliberately kept tiny so the whole file runs in a couple of seconds
on CPU. Real training tests belong in the ablation experiment scripts, not
the unit-test suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pyvae.models import InformedVAE
from pyvae.train import _check_cov, train_ivae, train_ivae_modern


# ---------- Fixtures ----------


def make_adj(n_genes: int = 20, n_pathways: int = 3) -> torch.Tensor:
    """Same adjacency shape as elsewhere in the tests."""
    adj = torch.zeros(n_genes, n_pathways)
    adj[0:10, 0] = 1
    adj[5:15, 1] = 1
    adj[10:20, 2] = 1
    return adj


def make_synthetic_kang(n_cells: int = 32, n_genes: int = 20, seed: int = 0):
    """Small NB-friendly dataset with a two-level covariate."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 20, size=(n_cells, n_genes)).astype(np.float32)
    x = np.log1p(counts).astype(np.float32)
    # Split cells 50/50 into two conditions for the one-hot covariate.
    half = n_cells // 2
    cov = np.zeros((n_cells, 2), dtype=np.float32)
    cov[:half, 0] = 1
    cov[half:, 1] = 1
    return x, counts, cov


# ---------- _check_cov ----------


class TestCheckCov:
    def test_conditional_model_without_cov_raises(self):
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, likelihood="nb", n_cov=2)
        x = torch.randn(16, 20)
        with pytest.raises(ValueError, match="cov_train is None"):
            _check_cov(model, cov=None, x=x, name="cov_train")

    def test_unconditional_model_with_cov_raises(self):
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, likelihood="nb", n_cov=0)
        x = torch.randn(16, 20)
        cov = torch.zeros(16, 2)
        with pytest.raises(ValueError, match="silently discard"):
            _check_cov(model, cov=cov, x=x, name="cov_train")

    def test_cov_wrong_width_raises(self):
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, likelihood="nb", n_cov=2)
        x = torch.randn(16, 20)
        cov = torch.zeros(16, 5)  # width=5, expected 2
        with pytest.raises(ValueError, match="shape"):
            _check_cov(model, cov=cov, x=x, name="cov_train")

    def test_cov_wrong_cell_count_raises(self):
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, likelihood="nb", n_cov=2)
        x = torch.randn(16, 20)
        cov = torch.zeros(8, 2)  # cells=8, expected 16
        with pytest.raises(ValueError, match="shape"):
            _check_cov(model, cov=cov, x=x, name="cov_train")

    def test_valid_cov_returns_silently(self):
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, likelihood="nb", n_cov=2)
        x = torch.randn(16, 20)
        cov = torch.zeros(16, 2)
        result = _check_cov(model, cov=cov, x=x, name="cov_train")
        assert result is None

    def test_unconditional_model_with_none_cov_returns_silently(self):
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, likelihood="nb", n_cov=0)
        x = torch.randn(16, 20)
        result = _check_cov(model, cov=None, x=x, name="cov_train")
        assert result is None

    def test_error_message_names_the_argument(self):
        """Argument name reaches the error message so callers can locate it."""
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, likelihood="nb", n_cov=2)
        x = torch.randn(16, 20)
        with pytest.raises(ValueError, match="cov_val"):
            _check_cov(model, cov=None, x=x, name="cov_val")


# ---------- train_ivae refuses conditional models ----------


class TestTrainIvaeRefusesConditionalModel:
    def test_conditional_model_raises_immediately(self):
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, likelihood="nb", n_cov=2)
        x, counts, _ = make_synthetic_kang()
        with pytest.raises(ValueError, match="train_ivae_modern"):
            train_ivae(
                model,
                x_train=x,
                x_val=x,
                x_counts_train=counts,
                x_counts_val=counts,
                epochs=1,
            )


# ---------- train_ivae_modern with covariates ----------


class TestTrainIvaeModernWithCovariates:
    def test_conditional_run_end_to_end(self):
        """A conditional model trains for a few epochs and produces a valid history."""
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, likelihood="nb", n_cov=2)
        x, counts, cov = make_synthetic_kang()
        # 80/20 split for train/val.
        n_train = 24
        model, history = train_ivae_modern(
            model,
            x_train=x[:n_train],
            x_val=x[n_train:],
            x_counts_train=counts[:n_train],
            x_counts_val=counts[n_train:],
            cov_train=cov[:n_train],
            cov_val=cov[n_train:],
            epochs=3,
            batch_size=8,
            warmup_epochs=2,
        )
        assert len(history["train"]) == 3
        assert len(history["val"]) == 3
        assert len(history["lr"]) == 3
        for v in history["val"]:
            assert torch.isfinite(torch.tensor(v))

    def test_unconditional_model_without_cov_still_works(self):
        """Backward compat: a n_cov=0 model trains as before, no cov args."""
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, likelihood="nb", n_cov=0)
        x, counts, _ = make_synthetic_kang()
        model, history = train_ivae_modern(
            model,
            x_train=x[:24],
            x_val=x[24:],
            x_counts_train=counts[:24],
            x_counts_val=counts[24:],
            epochs=2,
            batch_size=8,
            warmup_epochs=1,
        )
        assert len(history["train"]) == 2

    def test_conditional_model_without_cov_raises_before_training(self):
        """A conditional model without covariates should fail up front, not silently."""
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, likelihood="nb", n_cov=2)
        x, counts, _ = make_synthetic_kang()
        with pytest.raises(ValueError, match="cov_train is None"):
            train_ivae_modern(
                model,
                x_train=x[:24],
                x_val=x[24:],
                x_counts_train=counts[:24],
                x_counts_val=counts[24:],
                # cov_train and cov_val deliberately omitted
                epochs=1,
            )

    def test_unconditional_model_with_cov_raises_before_training(self):
        """An unconditional model handed a covariate should refuse up front."""
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, likelihood="nb", n_cov=0)
        x, counts, cov = make_synthetic_kang()
        with pytest.raises(ValueError, match="silently discard"):
            train_ivae_modern(
                model,
                x_train=x[:24],
                x_val=x[24:],
                x_counts_train=counts[:24],
                x_counts_val=counts[24:],
                cov_train=cov[:24],
                cov_val=cov[24:],
                epochs=1,
            )

    def test_cov_shape_mismatch_raises_before_training(self):
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, likelihood="nb", n_cov=2)
        x, counts, _ = make_synthetic_kang()
        bad_cov = np.zeros((10, 2), dtype=np.float32)  # only 10 rows, need 24
        with pytest.raises(ValueError, match="shape"):
            train_ivae_modern(
                model,
                x_train=x[:24],
                x_val=x[24:],
                x_counts_train=counts[:24],
                x_counts_val=counts[24:],
                cov_train=bad_cov,
                cov_val=bad_cov,
                epochs=1,
            )


# ---------- Every matrix argument accepts torch / numpy / pandas ----------


class TestInputCoercion:
    @pytest.mark.parametrize("input_kind", ["numpy", "torch", "pandas"])
    def test_train_ivae_modern_accepts_all_input_types(self, input_kind):
        """The trainer should accept any of torch.Tensor, numpy.ndarray,
        or pandas.DataFrame — Carlos's `as_float_tensor` fix in Phase 1c."""
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, likelihood="nb", n_cov=0)
        x, counts, _ = make_synthetic_kang()

        if input_kind == "numpy":
            x_tr, x_va = x[:24], x[24:]
            c_tr, c_va = counts[:24], counts[24:]
        elif input_kind == "torch":
            x_tr = torch.from_numpy(x[:24])
            x_va = torch.from_numpy(x[24:])
            c_tr = torch.from_numpy(counts[:24])
            c_va = torch.from_numpy(counts[24:])
        elif input_kind == "pandas":
            x_tr = pd.DataFrame(x[:24])
            x_va = pd.DataFrame(x[24:])
            c_tr = pd.DataFrame(counts[:24])
            c_va = pd.DataFrame(counts[24:])

        model, history = train_ivae_modern(
            model,
            x_train=x_tr,
            x_val=x_va,
            x_counts_train=c_tr,
            x_counts_val=c_va,
            epochs=2,
            batch_size=8,
            warmup_epochs=1,
        )
        assert len(history["train"]) == 2

    def test_train_ivae_accepts_numpy_input(self):
        """Regression: train_ivae used .values, which broke on numpy inputs."""
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, likelihood="nb", n_cov=0)
        x, counts, _ = make_synthetic_kang()
        # Pass numpy arrays; without as_float_tensor this would crash on .values.
        model, history = train_ivae(
            model,
            x_train=x[:24],
            x_val=x[24:],
            x_counts_train=counts[:24],
            x_counts_val=counts[24:],
            epochs=2,
            batch_size=8,
        )
        assert len(history["train"]) == 2


# ---------- history["lr"] fix ----------


class TestHistoryLrRecording:
    def test_history_lr_length_matches_epochs(self):
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, likelihood="nb", n_cov=0)
        x, counts, _ = make_synthetic_kang()
        _, history = train_ivae_modern(
            model,
            x_train=x[:24],
            x_val=x[24:],
            x_counts_train=counts[:24],
            x_counts_val=counts[24:],
            epochs=4,
            batch_size=8,
            warmup_epochs=1,
        )
        assert len(history["lr"]) == 4

    def test_first_epoch_lr_equals_initial_lr(self):
        """LR logged at epoch 0 is the initial LR, not one cosine step further.

        This is the specific bug Phase 1d fixes: previously the LR was recorded
        AFTER scheduler.step(), so it reported the rate queued for epoch 1.
        Now it's recorded before, so epoch 0's logged LR equals the initial LR.
        """
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, likelihood="nb", n_cov=0)
        x, counts, _ = make_synthetic_kang()
        initial_lr = 1e-3
        _, history = train_ivae_modern(
            model,
            x_train=x[:24],
            x_val=x[24:],
            x_counts_train=counts[:24],
            x_counts_val=counts[24:],
            epochs=4,
            batch_size=8,
            warmup_epochs=1,
            lr=initial_lr,
        )
        assert history["lr"][0] == pytest.approx(initial_lr)

    def test_lr_decreases_across_epochs(self):
        """Cosine annealing: LR should monotonically decrease over epochs."""
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, likelihood="nb", n_cov=0)
        x, counts, _ = make_synthetic_kang()
        _, history = train_ivae_modern(
            model,
            x_train=x[:24],
            x_val=x[24:],
            x_counts_train=counts[:24],
            x_counts_val=counts[24:],
            epochs=5,
            batch_size=8,
            warmup_epochs=1,
        )
        lrs = history["lr"]
        for i in range(1, len(lrs)):
            assert lrs[i] < lrs[i - 1], (
                f"Cosine annealing should give monotonically decreasing LR; "
                f"got history['lr'] = {lrs}"
            )
