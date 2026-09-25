"""Training loops for InformedVAE.

Phase 1d additions:

- Covariate threading through ``train_ivae_modern`` via new ``cov_train`` and
  ``cov_val`` parameters. When present, they are batched alongside x and counts
  and passed to the model forward. Closes limitation #1 in the notebook
  Takeaways (the counterfactual demo was deferred because the modern trainer
  didn't wire covariates through).
- ``_check_cov`` helper that catches the three quiet failure modes: a
  conditional model with no covariate, an unconditional model handed a
  covariate, and a shape that matches neither ``n_cov`` nor the cell count.
  A silently ignored covariate produces a run that looks healthy but has
  learned nothing about the label — worse than an explicit error.
- ``train_ivae`` now refuses a model with ``n_cov > 0``, since the classic
  loop passes no covariate to the encoder — a conditional model there
  would train silently without conditioning.
- Every matrix argument in both trainers is routed through
  ``as_float_tensor``, replacing the previous ``.values``-only path that
  silently restricted the trainers to pandas inputs.
- ``history["lr"]`` in the modern trainer is now recorded before
  ``scheduler.step()``, so it reports the rate the epoch actually trained
  at rather than the one queued for the next epoch.
"""
from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from pyvae.components import as_float_tensor, nb_reconstruction_loss
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


def _check_cov(model: InformedVAE, cov, x, name: str) -> None:
    """Validate a covariate matrix against a model's ``n_cov`` and the cell count.

    Catches the three quiet failure modes:

    1. ``model.n_cov > 0`` but ``cov is None`` — the encoder would fall back
       through its own ``ValueError``, but the message is opaque at the
       training-loop level. Raise here with the argument name.
    2. ``model.n_cov == 0`` but ``cov is not None`` — the encoder would
       silently discard the covariate. A run that looks healthy but has
       learned nothing about the label is worse than an explicit refusal.
    3. ``cov.shape`` doesn't match ``(x.shape[0], model.n_cov)`` — a
       transposed or truncated covariate would broadcast into shapes that
       don't crash but produce garbage.

    Parameters
    ----------
    model : InformedVAE
        The model whose ``n_cov`` attribute defines the expected width.
    cov : array-like or None
        The covariate matrix to validate. Must already be a torch.Tensor
        by the time this is called (pass the output of ``as_float_tensor``).
    x : torch.Tensor
        The corresponding cell matrix; ``cov.shape[0]`` must match ``x.shape[0]``.
    name : str
        Argument name to include in the error message (e.g. ``"cov_train"``).

    Raises
    ------
    ValueError
        With a message that names both the argument and the failure mode.

    Returns
    -------
    None. Returns silently when the covariate is valid.
    """
    if model.n_cov > 0 and cov is None:
        raise ValueError(
            f"{name} is None but the model was built with n_cov={model.n_cov}. "
            f"Pass a covariate matrix of shape ({x.shape[0]}, {model.n_cov})."
        )
    if model.n_cov == 0 and cov is not None:
        raise ValueError(
            f"{name} was provided but the model has n_cov=0. "
            f"An unconditional model would silently discard the covariate. "
            f"Either drop {name} or rebuild the model with n_cov > 0."
        )
    if cov is None:
        return
    expected = (x.shape[0], model.n_cov)
    if tuple(cov.shape) != expected:
        raise ValueError(
            f"{name} has shape {tuple(cov.shape)}; expected {expected} "
            f"({x.shape[0]} cells x {model.n_cov} covariates)."
        )


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
    if model.n_cov > 0:
        raise ValueError(
            f"train_ivae does not thread covariates through the encoder; "
            f"got a model with n_cov={model.n_cov}. Use train_ivae_modern "
            f"with cov_train and cov_val instead."
        )

    is_nb = model.likelihood_kind == "nb"
    if is_nb and (x_counts_train is None or x_counts_val is None):
        raise ValueError("NB likelihood requires both x_counts_train and x_counts_val")

    model.to(device)

    x_train_tensor = as_float_tensor(x_train, name="x_train")
    x_val_tensor = as_float_tensor(x_val, name="x_val")

    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())

    if is_nb:
        x_counts_train_tensor = as_float_tensor(x_counts_train, name="x_counts_train")
        x_counts_val_tensor = as_float_tensor(x_counts_val, name="x_counts_val")
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
    cov_train=None,
    cov_val=None,
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

    Parameters
    ----------
    model : InformedVAE
        Must have ``likelihood_kind == "nb"``. If ``model.n_cov > 0``, both
        ``cov_train`` and ``cov_val`` must be supplied.
    x_train, x_val : array-like
        Log-normalized expression matrices, shape (n_cells, n_genes).
    x_counts_train, x_counts_val : array-like
        Raw count matrices, shape (n_cells, n_genes).
    cov_train, cov_val : array-like or None, default None
        One-hot covariate matrices, shape (n_cells, n_cov). Required when
        the model was built with ``n_cov > 0``; forbidden otherwise. See
        ``_check_cov``.
    (other hyperparameters unchanged)

    Returns
    -------
    model : the trained model, with best-validation-loss weights restored.
    history : dict with per-epoch lists for "train", "val", "recon", "beta",
        and "lr". "val" is the ELBO evaluated at beta=1.0 every epoch (fair
        comparison across epochs); "recon" is the NB reconstruction term
        alone (unaffected by beta warmup, so its trajectory reflects the
        model's actual fit to the data). "lr" is recorded BEFORE
        ``scheduler.step()`` so it reports the rate this epoch trained at,
        not the one queued for the next.
    """
    if model.likelihood_kind != "nb":
        raise ValueError(
            f"train_ivae_modern requires model.likelihood_kind == 'nb', "
            f"got {model.likelihood_kind!r}"
        )

    model.to(device)

    x_train_tensor = as_float_tensor(x_train, name="x_train")
    x_val_tensor = as_float_tensor(x_val, name="x_val")
    x_counts_train_tensor = as_float_tensor(x_counts_train, name="x_counts_train")
    x_counts_val_tensor = as_float_tensor(x_counts_val, name="x_counts_val")

    # Covariate handling. Validate up front so a shape bug fails on the first
    # tensor conversion rather than at some unclear point inside the epoch.
    if cov_train is not None:
        cov_train_tensor = as_float_tensor(cov_train, name="cov_train")
    else:
        cov_train_tensor = None
    if cov_val is not None:
        cov_val_tensor = as_float_tensor(cov_val, name="cov_val")
    else:
        cov_val_tensor = None
    _check_cov(model, cov_train_tensor, x_train_tensor, "cov_train")
    _check_cov(model, cov_val_tensor, x_val_tensor, "cov_val")

    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())

    # Build datasets. When covariates are present, they are the third tensor
    # in the tuple; the loop below unpacks accordingly.
    if cov_train_tensor is not None:
        train_dataset = TensorDataset(x_train_tensor, x_counts_train_tensor, cov_train_tensor)
        val_dataset = TensorDataset(x_val_tensor, x_counts_val_tensor, cov_val_tensor)
    else:
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

        # Record LR BEFORE the epoch runs — this is the rate every batch
        # in this epoch will use. If we recorded it after scheduler.step(),
        # we'd log the rate queued for the NEXT epoch instead.
        current_lr = optimizer.param_groups[0]["lr"]

        model.train()
        epoch_train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            if cov_train_tensor is not None:
                x_batch, counts_batch, cov_batch = batch
                cov_batch = cov_batch.to(device)
            else:
                x_batch, counts_batch = batch
                cov_batch = None
            x_batch = x_batch.to(device)
            counts_batch = counts_batch.to(device)
            library = counts_batch.sum(1, keepdim=True)

            optimizer.zero_grad()
            recon, mu, log_var, h = model(x_batch, cov_batch)
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
            for batch in val_loader:
                if cov_val_tensor is not None:
                    x_batch, counts_batch, cov_batch = batch
                    cov_batch = cov_batch.to(device)
                else:
                    x_batch, counts_batch = batch
                    cov_batch = None
                x_batch = x_batch.to(device)
                counts_batch = counts_batch.to(device)
                library = counts_batch.sum(1, keepdim=True)

                recon, mu, log_var, h = model(x_batch, cov_batch)
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
        history["lr"].append(current_lr)

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
