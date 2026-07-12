"""
Shared building blocks for the informed VAE.

Splits the model into reusable pieces so the classic model and later variants
are assembled from the same parts instead of duplicating code:

    Encoder            genes -> pathway activations h -> (mu, log_var)
    reparameterise     (mu, log_var) -> sampled latent z
    DenseDecoder       z -> reconstructed genes
    CountDecoder       z -> px_scale (proportions, softmax over genes)
    GaussianLikelihood reconstruction loss object for the Gaussian path
    gaussian_kl        closed-form KL( N(mu, sigma^2) || N(0, I) )
    nb_log_prob        negative binomial log-PMF, for the count-likelihood path
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    sigma = torch.exp(0.5 * log_var)
    eps = torch.randn_like(sigma)
    return mu + sigma * eps


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
    return -0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1).mean()


def nb_log_prob(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Elementwise log-PMF of the negative binomial distribution.

    Parameterization
    ----------------
    Mean-dispersion form (as in scVI): ``Var(X) = mu + mu**2 / theta``.
    Large ``theta`` -> Poisson-like; small ``theta`` -> heavily overdispersed.

    Formula
    -------
        log p(x | mu, theta) = lgamma(x + theta) - lgamma(theta) - lgamma(x + 1)
                             + theta * (log(theta) - log(theta + mu))
                             + x * (log(mu) - log(theta + mu))

    An ``eps`` is added inside every ``log`` to avoid ``log(0) = -inf``.

    Parameters
    ----------
    x : (batch, n_genes) integer counts (as float tensor).
    mu : (batch, n_genes) predicted mean, must be non-negative.
    theta : (n_genes,) or (batch, n_genes) dispersion, must be positive.
    eps : small constant added inside every log for numerical stability.

    Returns
    -------
    log_prob : same shape as x, elementwise log-PMF.
    """
    log_theta_mu_eps = torch.log(theta + mu + eps)
    return (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
        + theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
    )


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
        n_pathways = adj.shape[1]
        self.informed = InformedLinear(adj, activation="tanh")
        self.fc_mean = nn.Linear(n_pathways, latent_dim)
        self.fc_log_var = nn.Linear(n_pathways, latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        h = self.informed(x)
        mu = self.fc_mean(h)
        log_var = torch.clamp(self.fc_log_var(h), -3, 3)
        return mu, log_var, h


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
        self.dec_latent = nn.Linear(latent_dim, n_pathways)
        self.dec_out = nn.Linear(n_pathways, n_genes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent samples back to gene space.

        Parameters
        ----------
        z : (batch, latent) latent samples.

        Returns
        -------
        recon : (batch, n_genes) reconstructed expression.
        """

        h_prime = torch.tanh(self.dec_latent(z))
        x_hat = self.dec_out(h_prime)
        return x_hat


class CountDecoder(nn.Module):
    """Count decoder: latent -> pathways (tanh) -> gene logits -> softmax proportions.

    Produces ``px_scale``, the per-cell gene expression proportions summing to 1
    across the gene axis. The mean of the negative binomial (``mu = px_scale *
    library``) is computed downstream in ``InformedVAE.loss`` where the per-cell
    library size is available.

    A learnable per-gene log-dispersion ``px_r`` is stored on the module; its
    exponential (always positive) is exposed as ``theta`` for the NB likelihood.

    Parameters
    ----------
    latent_dim : size of the latent space.
    n_pathways : width of the hidden pathway layer.
    n_genes : number of output genes.
    """

    def __init__(self, latent_dim: int, n_pathways: int, n_genes: int):
        super().__init__()
        self.dec_latent = nn.Linear(latent_dim, n_pathways)
        self.dec_out = nn.Linear(n_pathways, n_genes)
        # Per-gene log-dispersion, exponentiated in ``theta`` to enforce positivity.
        self.px_r = nn.Parameter(torch.zeros(n_genes))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent samples to per-cell gene proportions.

        Parameters
        ----------
        z : (batch, latent) latent samples.

        Returns
        -------
        px_scale : (batch, n_genes) softmax proportions, one row per cell,
            summing to 1 across the gene axis.
        """
        h_prime = torch.tanh(self.dec_latent(z))
        logits = self.dec_out(h_prime)
        return torch.softmax(logits, dim=1)

    @property
    def theta(self) -> torch.Tensor:
        """Per-gene NB dispersion, exp(px_r), always positive."""
        return torch.exp(self.px_r)


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
        return F.mse_loss(recon, x, reduction="none").sum(dim=1).mean()
