from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from pyvae.models import InformedVAE


def kl_beta_schedule(epoch: int, warmup_epochs: int) -> float:
    """Linear KL warmup: 0 at epoch=0, ramps to 1.0 at ``epoch >= warmup_epochs``.

    When ``warmup_epochs == 0`` the warmup is disabled and this always returns
    1.0 (equivalent to standard ELBO training from epoch 0). Guarding this case
    also avoids the division by zero that a naive ``epoch / warmup_epochs``
    would trigger.

    Parameters
    ----------
    epoch : current epoch index (0-based).
    warmup_epochs : number of epochs over which to ramp beta from 0 to 1.

    Returns
    -------
    beta : float in [0.0, 1.0].
    """
    if warmup_epochs <= 0:
        return 1.0
    return min(epoch / warmup_epochs, 1.0)


def train_ivae(
    model: InformedVAE,
    x_train,
    x_val,
    x_counts_train=None,
    x_counts_val=None,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 100,
    lr: float = 1e-5,
    device: str = "cpu",
) -> tuple[InformedVAE, dict]:
    is_nb = model.likelihood_kind == "nb"
    if is_nb and (x_counts_train is None or x_counts_val is None):
        raise ValueError("NB likelihood requires both x_counts_train and x_counts_val")

    model.to(device)

    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32)

    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())

    if is_nb:
        x_counts_train_tensor = torch.tensor(x_counts_train.values, dtype=torch.float32)
        x_counts_val_tensor = torch.tensor(x_counts_val.values, dtype=torch.float32)
        train_dataset = TensorDataset(x_train_tensor, x_counts_train_tensor)
        val_dataset = TensorDataset(x_val_tensor, x_counts_val_tensor)
    else:
        train_dataset = TensorDataset(x_train_tensor)
        val_dataset = TensorDataset(x_val_tensor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    history = {"train": [], "val": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    from tqdm.auto import tqdm

    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if is_nb:
                x_batch, counts_batch = batch
                x_batch = x_batch.to(device)
                counts_batch = counts_batch.to(device)
                library = counts_batch.sum(1, keepdim=True)
            else:
                (x_batch,) = batch
                x_batch = x_batch.to(device)

            optimizer.zero_grad()
            recon, mu, log_var, h = model(x_batch)
            if is_nb:
                loss = model.loss(
                    x_batch,
                    recon,
                    mu,
                    log_var,
                    h,
                    counts=counts_batch,
                    library=library,
                )
            else:
                loss = model.loss(x_batch, recon, mu, log_var, h)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_train_loss / n_batches

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if is_nb:
                    x_batch, counts_batch = batch
                    x_batch = x_batch.to(device)
                    counts_batch = counts_batch.to(device)
                    library = counts_batch.sum(1, keepdim=True)
                    recon, mu, log_var, h = model(x_batch)
                    val_loss_sum += model.loss(
                        x_batch,
                        recon,
                        mu,
                        log_var,
                        h,
                        counts=counts_batch,
                        library=library,
                    ).item() * len(x_batch)
                else:
                    (x_batch,) = batch
                    x_batch = x_batch.to(device)
                    recon, mu, log_var, h = model(x_batch)
                    val_loss_sum += model.loss(
                        x_batch, recon, mu, log_var, h
                    ).item() * len(x_batch)
        val_loss = val_loss_sum / len(x_val_tensor)

        history["train"].append(avg_train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
