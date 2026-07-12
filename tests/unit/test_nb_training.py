"""Integration test: NB path trains, validation loss decreases"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from pyvae.models import InformedVAE
from pyvae.train import train_ivae


def _synthetic_counts(n_cells: int, n_genes: int, seed: int) -> pd.DataFrame:
    """Draw non-trivial random NB-like counts as a DataFrame.

    Uses a simple Poisson draw with per-gene rates on the order of 1-10, which
    gives realistic sparsity and dynamic range for a small test.
    """
    rng = np.random.default_rng(seed)
    rates = rng.uniform(0.5, 10.0, size=n_genes)
    counts = rng.poisson(rates, size=(n_cells, n_genes)).astype(np.float32)
    return pd.DataFrame(counts)


def test_nb_training_decreases_validation_loss():
    torch.manual_seed(0)
    np.random.seed(0)

    n_genes = 20
    n_pathways = 8
    adj = torch.randint(0, 2, size=(n_genes, n_pathways), dtype=torch.float32)

    x_counts_train = _synthetic_counts(n_cells=64, n_genes=n_genes, seed=1)
    x_counts_val = _synthetic_counts(n_cells=16, n_genes=n_genes, seed=2)
    # Encoder input: log1p of counts, as the standard preprocessing.
    x_train = np.log1p(x_counts_train)
    x_val = np.log1p(x_counts_val)

    model = InformedVAE(
        adj=adj, latent_dim=4, seed=42, likelihood="nb",
    )

    _, history = train_ivae(
        model,
        x_train,
        x_val,
        x_counts_train=x_counts_train,
        x_counts_val=x_counts_val,
        epochs=10,
        batch_size=16,
        patience=100,
        lr=1e-3,
        device="cpu",
    )

    assert len(history["val"]) == 10
    assert all(np.isfinite(v) for v in history["val"])
    # Validation loss must be lower at the end than at the start.
    assert history["val"][-1] < history["val"][0], (
        f"NB validation loss did not decrease: "
        f"start={history['val'][0]:.4f}, end={history['val'][-1]:.4f}"
    )