"""
Training loop for InformedVAE with early stopping.
"""

from __future__ import annotations

import torch  # noqa: F401
from torch.utils.data import DataLoader, TensorDataset  # noqa: F401

from pyvae.models import InformedVAE


def train_ivae(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
    model: InformedVAE,
    x_train,
    x_val,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 100,
    lr: float = 1e-5,
    device: str = "cpu",
) -> tuple[InformedVAE, dict]:
    """
    Train InformedVAE with Adam and early stopping.

    Implement a standard mini-batch training loop:

    - Move the model to the target device.
    - Use the Adam optimiser with the given learning rate.  Set eps=1e-7
      (lower than the default) for better numerical stability with small lr.
    - Convert the DataFrames to float32 tensors and create a shuffled
      DataLoader for the training set.
    - For each epoch, run a train phase (model.train()) over all mini-batches
      followed by a validation phase (model.eval(), no gradient computation).
    - After each backward pass, clip gradients before the optimiser step.
      The reconstruction loss is scaled by n_genes, so raw gradients can be
      large at the start of training.
    - Implement early stopping: keep track of the best validation loss seen
      so far and a patience counter.  Stop early if the counter reaches the
      patience limit.
    - Append the per-epoch train and validation losses to history.

    Parameters
    ----------
    model : InformedVAE
    x_train : pd.DataFrame, dtype compatible with float32
        Training split - rows are cells, columns are genes.
    x_val : pd.DataFrame, dtype compatible with float32
        Validation split.
    epochs : int
        Maximum number of full passes over the training data.
    batch_size : int
        Number of cells per mini-batch.
    patience : int
        Epochs with no improvement on val_loss before early stopping fires.
    lr : float
        Learning rate for Adam.
    device : str
        "cpu" or "cuda".

    Returns
    -------
    model : InformedVAE
        Trained model, still on device.
    history : dict
        {"train": [...], "val": [...]} - one entry per completed epoch.
    """
    from tqdm.auto import tqdm  # noqa: PLC0415

    # optimizer: Adam optimiser for model.parameters()
    # loader: DataLoader wrapping the training tensor
    # history: dict with "train" and "val" loss lists
    raise NotImplementedError
