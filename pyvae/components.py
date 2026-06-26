"""
Shared building blocks for the informed VAE.

Splits the model into reusable pieces so the classic model and later variants
are assembled from the same parts instead of duplicating code:

    Encoder            genes -> pathway activations h -> (mu, log_var)
    reparameterise     (mu, log_var) -> sampled latent z
    DenseDecoder       z -> reconstructed genes
    GaussianLikelihood reconstruction loss object (swappable later for NB/ZINB)

Scaffold only: signatures, docstrings and expected returns are given; bodies
raise NotImplementedError. This is a behavior-preserving refactor, so each
piece must reproduce what pyvae/models.py::InformedVAE does today.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pyvae.layers import InformedLinear


def reparameterise(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Sample z ~ N(mu, sigma^2) with the reparameterization trick.

    sigma = exp(0.5 * log_var); z = mu + sigma * eps, eps ~ N(0, I).
    Move the body from InformedVAE.reparameterise unchanged.

    Parameters
    ----------
    mu : (batch, latent) posterior mean.
    log_var : (batch, latent) posterior log-variance (clamped upstream).

    Returns
    -------
    z : (batch, latent) sampled latent, differentiable wrt mu and log_var.
    """
    raise NotImplementedError


def gaussian_kl(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL( N(mu, sigma^2) || N(0, I) ), summed over latent dims, mean over batch.

    Must equal the KL term in InformedVAE.loss:
        -0.5 * (1 + log_var - mu**2 - exp(log_var)).sum(dim=1).mean()

    Parameters
    ----------
    mu : (batch, latent) posterior mean.
    log_var : (batch, latent) posterior log-variance.

    Returns
    -------
    kl : scalar tensor (0-dim).
    """
    raise NotImplementedError


class Encoder(nn.Module):
    """Informed encoder: genes -> pathway activations h -> (mu, log_var).

    Wraps the masked first layer (InformedLinear, tanh) and the two linear
    heads. Reproduces InformedVAE.encode, including the clamp of log_var to
    [-3, 3].

    Parameters
    ----------
    adj : (n_genes, n_pathways) binary mask tensor.
    latent_dim : size of the latent space.
    """

    def __init__(self, adj: torch.Tensor, latent_dim: int):
        super().__init__()
        # self.informed = InformedLinear(adj, activation="tanh")
        # self.fc_mean = nn.Linear(n_pathways, latent_dim)
        # self.fc_log_var = nn.Linear(n_pathways, latent_dim)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a batch of expression.

        Parameters
        ----------
        x : (batch, n_genes) log1p-normalized expression.

        Returns
        -------
        mu : (batch, latent) posterior mean.
        log_var : (batch, latent) posterior log-variance, clamped to [-3, 3].
        h : (batch, n_pathways) pathway activations (used by the L2 term and
            for interpretation).
        """
        raise NotImplementedError


class DenseDecoder(nn.Module):
    """Dense decoder: latent -> pathways (tanh) -> genes.

    Reproduces InformedVAE.decode (dec_latent + dec_out).

    Parameters
    ----------
    latent_dim : size of the latent space.
    n_pathways : width of the hidden pathway layer.
    n_genes : number of output genes.
    """

    def __init__(self, latent_dim: int, n_pathways: int, n_genes: int):
        super().__init__()
        # self.dec_latent = nn.Linear(latent_dim, n_pathways)
        # self.dec_out = nn.Linear(n_pathways, n_genes)
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent samples back to gene space.

        Parameters
        ----------
        z : (batch, latent) latent samples.

        Returns
        -------
        recon : (batch, n_genes) reconstructed expression.
        """
        raise NotImplementedError


class GaussianLikelihood(nn.Module):
    """Reconstruction loss object (MSE), so the loss term is swappable later.

    Wraps the reconstruction term in InformedVAE.loss:
        F.mse_loss(recon, x, reduction="none").sum(dim=1).mean()

    Keeping it as an object lets a future count likelihood (NB/ZINB) drop in
    without touching the model assembly.
    """

    def __call__(self, recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction loss for a batch.

        Parameters
        ----------
        recon : (batch, n_genes) reconstructed expression.
        x : (batch, n_genes) target expression.

        Returns
        -------
        loss : scalar tensor (0-dim), summed over genes, mean over batch.
        """
        raise NotImplementedError
