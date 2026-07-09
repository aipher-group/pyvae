# The refactor into shared components must not change behavior. Pin it by
# training a tiny model for a few epochs with a fixed seed and asserting the
# loss matches a value recorded BEFORE the refactor.
#
# Workflow:
#   1. Before refactoring, run a short fixed-seed training and copy the final
#      loss into EXPECTED_FINAL_LOSS below.
#   2. Do the refactor.
#   3. Remove the skip and run this test; it must match.


import numpy as np
import pandas as pd
import torch

from pyvae.models import InformedVAE
from pyvae.train import train_ivae


# Pin the value produced by _run_baseline_training below on the pre refactor
# code. Any behavior change during the refactor (weight init order, layer
# ordering, activation swap) shifts this number and fails the test.
EXPECTED_FINAL_LOSS = 23.0879344940
TOLERANCE = 1e-5


def _run_baseline_training() -> float:
    """Deterministic fixed-seed training run whose final val loss is
    the behavior-preservation ground truth for the component refactor (#5)."""
    torch.manual_seed(0)
    np.random.seed(0)

    adj = torch.randint(0, 2, size=(20, 8), dtype=torch.float32)
    x_train = pd.DataFrame(np.random.randn(24, 20).astype(np.float32))
    x_val = pd.DataFrame(np.random.randn(8, 20).astype(np.float32))

    model = InformedVAE(adj=adj, latent_dim=4, seed=42)

    _, history = train_ivae(
        model,
        x_train,
        x_val,
        epochs=3,
        batch_size=8,
        patience=100,
        lr=1e-3,
        device="cpu",
    )
    return history["val"][-1]


def test_refactor_preserves_loss():
    """Fixed seed + fixed data must give identical loss after the component refactor."""
    actual = _run_baseline_training()
    diff = abs(actual - EXPECTED_FINAL_LOSS)
    assert diff < TOLERANCE, (
        f"Refactor changed behavior. "
        f"Expected {EXPECTED_FINAL_LOSS}, got {actual:.10f}, "
        f"diff = {diff:.2e}"
    )
