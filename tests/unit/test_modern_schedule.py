"""Unit and integration tests for the modernized training loop (#12).

Covers:
  * kl_beta_schedule as a pure function (acceptance values plus
    the linear-ramp midpoint).
  * train_ivae_modern trained end-to-end on synthetic Poisson counts:
    reconstruction component of the validation loss must decrease across
    epochs (not the full loss; during warmup the validation ELBO curve
    can dip and rise as KL takes hold, so we track the recon term alone).
  * Cosine LR schedule actually runs (final LR strictly lower than initial).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pyvae.models import InformedVAE
from pyvae.train import kl_beta_schedule, train_ivae_modern


# kl_beta_schedule unit tests

def test_kl_beta_schedule_start_of_warmup_is_zero():
    """beta(0, 10) == 0.0 — acceptance value."""
    assert kl_beta_schedule(0, 10) == 0.0


def test_kl_beta_schedule_end_of_warmup_is_one():
    """beta(10, 10) == 1.0 — acceptance value."""
    assert kl_beta_schedule(10, 10) == 1.0


def test_kl_beta_schedule_past_warmup_is_capped_at_one():
    """beta(20, 10) == 1.0 — acceptance value (stays capped at 1.0)."""
    assert kl_beta_schedule(20, 10) == 1.0


def test_kl_beta_schedule_warmup_disabled_returns_one():
    """beta(0, 0) == 1.0 — acceptance value (no division by zero)."""
    assert kl_beta_schedule(0, 0) == 1.0


def test_kl_beta_schedule_midpoint_is_linear():
    """beta(5, 10) == 0.5 — confirms the ramp is genuinely linear, not stepped."""
    assert kl_beta_schedule(5, 10) == 0.5


# train_ivae_modern integration test

def _synthetic_counts(n_cells: int, n_genes: int, seed: int) -> pd.DataFrame:
    """Draw non-trivial Poisson counts as a DataFrame."""
    rng = np.random.default_rng(seed)
    rates = rng.uniform(0.5, 10.0, size=n_genes)
    counts = rng.poisson(rates, size=(n_cells, n_genes)).astype(np.float32)
    return pd.DataFrame(counts)


def test_modern_training_recon_decreases_with_warmup():
    """Reconstruction term of val loss decreases across epochs under KL warmup.

    Per pitfall: the full validation loss can be non-monotonic during
    warmup because it is evaluated at beta=1.0 while the model was trained at
    beta<1.0; the *reconstruction component* (unaffected by beta) is the
    correct convergence indicator here.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    n_genes = 20
    n_pathways = 8
    adj = torch.randint(0, 2, size=(n_genes, n_pathways), dtype=torch.float32)

    x_counts_train = _synthetic_counts(n_cells=64, n_genes=n_genes, seed=1)
    x_counts_val = _synthetic_counts(n_cells=16, n_genes=n_genes, seed=2)
    x_train = np.log1p(x_counts_train)
    x_val = np.log1p(x_counts_val)

    model = InformedVAE(adj=adj, latent_dim=4, seed=42, likelihood="nb")

    _, history = train_ivae_modern(
        model,
        x_train,
        x_val,
        x_counts_train=x_counts_train,
        x_counts_val=x_counts_val,
        epochs=10,
        batch_size=16,
        patience=100,
        lr=1e-3,
        weight_decay=1e-6,
        warmup_epochs=3,
        max_grad_norm=5.0,
        device="cpu",
    )

    # History has all five expected keys, each with 10 entries.
    for key in ("train", "val", "recon", "beta", "lr"):
        assert key in history, f"history missing {key!r}"
        assert len(history[key]) == 10, f"history[{key!r}] wrong length"

    assert all(np.isfinite(v) for v in history["recon"]), "non-finite recon"

    # Reconstruction term of val loss strictly decreases start -> end.
    assert history["recon"][-1] < history["recon"][0], (
        f"NB reconstruction did not decrease under modern schedule: "
        f"start={history['recon'][0]:.4f}, end={history['recon'][-1]:.4f}"
    )


def test_modern_training_lr_is_lower_at_end():
    """Cosine schedule actually ran: end-of-training LR is strictly below start LR."""
    torch.manual_seed(0)
    np.random.seed(0)

    n_genes = 20
    n_pathways = 8
    adj = torch.randint(0, 2, size=(n_genes, n_pathways), dtype=torch.float32)

    x_counts_train = _synthetic_counts(n_cells=32, n_genes=n_genes, seed=3)
    x_counts_val = _synthetic_counts(n_cells=8, n_genes=n_genes, seed=4)
    x_train = np.log1p(x_counts_train)
    x_val = np.log1p(x_counts_val)

    model = InformedVAE(adj=adj, latent_dim=4, seed=42, likelihood="nb")
    lr_init = 1e-3

    _, history = train_ivae_modern(
        model,
        x_train,
        x_val,
        x_counts_train=x_counts_train,
        x_counts_val=x_counts_val,
        epochs=10,
        batch_size=16,
        patience=100,
        lr=lr_init,
        weight_decay=1e-6,
        warmup_epochs=3,
        max_grad_norm=5.0,
        device="cpu",
    )

    assert history["lr"][-1] < history["lr"][0], (
        f"LR at end ({history['lr'][-1]}) is not strictly less than "
        f"LR at start ({history['lr'][0]}); cosine annealing did not run."
    )


def test_modern_training_rejects_gaussian_model():
    """train_ivae_modern raises if called on a Gaussian model."""
    adj = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    model = InformedVAE(adj=adj, latent_dim=1, seed=42, likelihood="gaussian")

    x_train = pd.DataFrame(np.random.randn(4, 2).astype(np.float32))
    x_val = pd.DataFrame(np.random.randn(2, 2).astype(np.float32))

    with pytest.raises(ValueError, match="train_ivae_modern requires"):
        train_ivae_modern(
            model, x_train, x_val,
            x_counts_train=x_train, x_counts_val=x_val,
            epochs=1, batch_size=2, device="cpu",
        )