from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from pyvae.components import nb_reconstruction_loss
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


def train_ivae_modern(
    model: InformedVAE,
    x_train,
    x_val,
    x_counts_train,
    x_counts_val,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    warmup_epochs: int = 10,
    max_grad_norm: float = 5.0,
    device: str = "cpu",
) -> tuple[InformedVAE, dict]:
    """Train an NB-likelihood InformedVAE with KL warmup, AdamW, cosine LR
    annealing, and gradient-norm clipping.

    Unlike ``train_ivae``, this trainer assumes ``model.likelihood_kind == "nb"``
    and that count/library data is always supplied (no Gaussian branch).

    Returns
    -------
    model : the trained model, with best-validation-loss weights restored.
    history : dict with per-epoch lists for "train", "val", "recon", "beta",
        and "lr". "val" is the ELBO evaluated at beta=1.0 every epoch (fair
        comparison across epochs); "recon" is the NB reconstruction term
        alone (unaffected by beta warmup, so its trajectory reflects the
        model's actual fit to the data).
    """
    if model.likelihood_kind != "nb":
        raise ValueError(
            f"train_ivae_modern requires model.likelihood_kind == 'nb', "
            f"got {model.likelihood_kind!r}"
        )

    model.to(device)

    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32)
    x_counts_train_tensor = torch.tensor(x_counts_train.values, dtype=torch.float32)
    x_counts_val_tensor = torch.tensor(x_counts_val.values, dtype=torch.float32)

    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())

    train_dataset = TensorDataset(x_train_tensor, x_counts_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, x_counts_val_tensor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train": [], "val": [], "recon": [], "beta": [], "lr": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    from tqdm.auto import tqdm

    for epoch in tqdm(range(epochs)):
        beta_epoch = kl_beta_schedule(epoch, warmup_epochs)

        model.train()
        epoch_train_loss = 0.0
        n_batches = 0
        for x_batch, counts_batch in train_loader:
            x_batch = x_batch.to(device)
            counts_batch = counts_batch.to(device)
            library = counts_batch.sum(1, keepdim=True)

            optimizer.zero_grad()
            recon, mu, log_var, h = model(x_batch)
            loss = model.loss(
                x_batch,
                recon,
                mu,
                log_var,
                h,
                counts=counts_batch,
                library=library,
                beta=beta_epoch,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            epoch_train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_train_loss = epoch_train_loss / n_batches

        model.eval()
        val_loss_sum = 0.0
        val_recon_sum = 0.0
        with torch.no_grad():
            for x_batch, counts_batch in val_loader:
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
                    beta=1.0,
                ).item() * len(x_batch)

                recon_term = nb_reconstruction_loss(
                    counts_batch, recon, library, model.decoder.theta
                )
                val_recon_sum += recon_term.item() * len(x_batch)

        val_loss = val_loss_sum / len(x_val_tensor)
        val_recon = val_recon_sum / len(x_val_tensor)

        history["train"].append(avg_train_loss)
        history["val"].append(val_loss)
        history["recon"].append(val_recon)
        history["beta"].append(beta_epoch)
        history["lr"].append(optimizer.param_groups[0]["lr"])

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
