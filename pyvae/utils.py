"""
Utilities for reproducibility.
"""

import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """
    Set random seeds for full reproducibility across all numerical libraries.

    Call this once at the start of your script/notebook before constructing
    the model or loading data.  Seeded libraries:

    - random   (Python standard library)
    - numpy    (NumPy RNG)
    - torch    (CPU operations)
    - torch.cuda (GPU operations, if a CUDA device is available)

    Parameters
    ----------
    seed : int
        Seed value to use.  42 is a common default.
    """
    # TODO: seed random, np.random, torch, and torch.cuda (if available)
    raise NotImplementedError
