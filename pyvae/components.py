"""Components — helpers, encoder, decoders, likelihoods.

This module holds:

- Small utility functions: ``reparameterise``, ``gaussian_kl``, ``nb_log_prob``,
  ``nb_reconstruction_loss``, ``as_float_tensor``, ``_make_dec_out``.
- The ``Encoder`` module.
- The two decoder modules: ``DenseDecoder`` (Gaussian likelihood) and
  ``CountDecoder`` (NB likelihood).
- The ``GaussianLikelihood`` module.

Phase 1c additions:

- ``as_float_tensor`` coerces torch / numpy / pandas / scipy-sparse inputs
  to a float32 tensor, raising a TypeError that names the offending argument.
  Fixes the previous silent restriction where the trainers called ``.values``
  on inputs, which quietly assumed pandas.
- ``_make_dec_out`` returns either an ``nn.Linear`` (dense decoder) or an
  ``InformedLinear`` (masked decoder, gated by the ``informed_decoder`` flag
  on ``InformedVAE``). The masked decoder deliberately uses ``mask_bias=False``
  so every gene keeps a free per-gene baseline, even those with no pathway
  annotation.
- ``Encoder`` now threads ``init``, ``normalize``, ``nonneg``, and
  ``standardize_input`` through to its ``InformedLinear``.
- ``DenseDecoder`` and ``CountDecoder`` now accept ``adj`` and ``init`` to
  route through ``_make_dec_out``.

Every new option defaults to the pre-existing behaviour, so the golden
regression test still passes bit-for-bit.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyvae.layers import InformedLinear


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def reparameterise(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
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
    log_theta_mu_eps = torch.log(theta + mu + eps)
    return (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
        + theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
    )


def nb_reconstruction_loss(
    counts: torch.Tensor,
    px_scale: torch.Tensor,
    library: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    mu = px_scale * library
    return -nb_log_prob(counts, mu, theta).sum(dim=1).mean()


def as_float_tensor(data, name: str = "input") -> torch.Tensor:
    """Coerce array-like input to a float32 ``torch.Tensor``.

    Accepts ``torch.Tensor``, ``numpy.ndarray``, ``pandas.DataFrame``,
    ``pandas.Series``, and scipy sparse matrices. Anything else raises
    ``TypeError`` naming the argument so the caller can locate the problem.

    Parameters
    ----------
    data : Any
        The value to coerce.
    name : str, default "input"
        Argument name to include in the error message. Callers should pass
        the parameter name they're validating, e.g. ``name="x"``.

    Returns
    -------
    torch.Tensor
        A float32 tensor. Sparse inputs are densified in memory.

    Raises
    ------
    TypeError
        If ``data`` is not one of the supported types. The message names
        both the argument (``name``) and the type it received.

    Notes
    -----
    The current trainers call ``.values`` on inputs, which works for pandas
    but fails obscurely on a plain tensor (``Tensor.values`` resolves to a
    bound method). This helper is the intended one-stop coercion so callers
    do not need to know the input's provenance.
    """
    # Fast path: torch.Tensor. Always safe to cast to float.
    if isinstance(data, torch.Tensor):
        return data.float()

    # numpy.ndarray. Lazy import so this module works without numpy present.
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data.astype(np.float32, copy=False))
    except ImportError:
        pass

    # pandas.DataFrame / Series.
    try:
        import pandas as pd
        if isinstance(data, (pd.DataFrame, pd.Series)):
            import numpy as np
            return torch.from_numpy(data.values.astype(np.float32, copy=False))
    except ImportError:
        pass

    # scipy sparse.
    try:
        from scipy import sparse
        if sparse.issparse(data):
            import numpy as np
            return torch.from_numpy(data.toarray().astype(np.float32, copy=False))
    except ImportError:
        pass

    raise TypeError(
        f"{name} must be torch.Tensor, numpy.ndarray, pandas.DataFrame/Series, "
        f"or scipy.sparse; got {type(data).__name__}"
    )


def _make_dec_out(
    n_pathways: int,
    n_genes: int,
    adj: torch.Tensor | None = None,
    init: str = "xavier",
) -> nn.Module:
    """Build the decoder's output layer, dense or masked.

    Parameters
    ----------
    n_pathways : int
        Input dimensionality (the pathway-layer width of the decoder).
    n_genes : int
        Output dimensionality (number of genes to predict).
    adj : torch.Tensor or None, default None
        Adjacency of shape ``(n_genes, n_pathways)``. When ``None``, returns
        a dense ``nn.Linear(n_pathways, n_genes)``. When provided, returns an
        ``InformedLinear(adj.T, ...)`` so each pathway's output only affects
        its member genes.
    init : {"xavier", "fan_in"}, default "xavier"
        Weight initialisation for the masked variant. Ignored when ``adj`` is
        None (nn.Linear uses PyTorch's default kaiming_uniform_).

    Returns
    -------
    nn.Module
        Either ``nn.Linear`` or ``InformedLinear`` matching the two cases above.

    Raises
    ------
    ValueError
        If ``adj`` is provided but does not have shape ``(n_genes, n_pathways)``.

    Notes
    -----
    ``mask_bias=False`` on the masked variant is deliberate: the bias here
    is a per-gene baseline expression level and must stay free even for a
    gene with no annotations, which would otherwise sit at zero logit
    forever. Only the weight matrix is masked, not the bias.
    """
    if adj is None:
        return nn.Linear(n_pathways, n_genes)

    if adj.shape != (n_genes, n_pathways):
        raise ValueError(
            f"adj must have shape ({n_genes}, {n_pathways}); got {tuple(adj.shape)}"
        )

    return InformedLinear(
        adj.T,
        activation="linear",
        init=init,
        mask_bias=False,
    )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """Masked linear encoder followed by two dense heads for mu and log_var.

    The masked layer is an :class:`InformedLinear`; Phase 1c added keyword-only
    parameters that let ``InformedVAE`` pass through the four encoder-side
    constraint options (``init``, ``normalize``, ``nonneg``, ``standardize_input``).
    Every option defaults to the pre-existing behaviour.
    """

    def __init__(
        self,
        adj: torch.Tensor,
        latent_dim: int,
        n_cov: int = 0,
        *,
        init: str = "xavier",
        normalize: str = "none",
        nonneg: bool = False,
        standardize_input: bool = False,
    ):
        super().__init__()
        n_pathways = adj.shape[1]
        self.n_cov = n_cov
        self.informed = InformedLinear(
            adj,
            activation="tanh",
            init=init,
            normalize=normalize,
            nonneg=nonneg,
            standardize_input=standardize_input,
        )
        if n_cov > 0:
            self.fc_mean = nn.Linear(n_pathways + n_cov, latent_dim)
            self.fc_log_var = nn.Linear(n_pathways + n_cov, latent_dim)
        else:
            self.fc_mean = nn.Linear(n_pathways, latent_dim)
            self.fc_log_var = nn.Linear(n_pathways, latent_dim)

    def forward(
        self, x: torch.Tensor, cov: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.informed(x)
        if self.n_cov > 0:
            if cov is None:
                raise ValueError(
                    f"Encoder was built with n_cov={self.n_cov} > 0 but forward() "
                    f"got cov=None; pass a (batch, {self.n_cov}) covariate tensor"
                )
            h_cat = torch.cat([h, cov], dim=1)
            mu = self.fc_mean(h_cat)
            log_var = torch.clamp(self.fc_log_var(h_cat), -3, 3)
        else:
            mu = self.fc_mean(h)
            log_var = torch.clamp(self.fc_log_var(h), -3, 3)
        return mu, log_var, h


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------


class DenseDecoder(nn.Module):
    """Decoder for the Gaussian likelihood.

    Phase 1c added the ``adj`` and ``init`` keyword-only parameters. When
    ``adj`` is provided (i.e. ``InformedVAE(informed_decoder=True)`` at the
    model level), the ``dec_out`` layer is masked so each pathway output
    only affects its member genes. Otherwise ``dec_out`` remains a dense
    ``nn.Linear`` and current behaviour is reproduced exactly.
    """

    def __init__(
        self,
        latent_dim: int,
        n_pathways: int,
        n_genes: int,
        n_cov: int = 0,
        *,
        adj: torch.Tensor | None = None,
        init: str = "xavier",
    ):
        super().__init__()
        self.n_cov = n_cov
        self.dec_latent = nn.Linear(latent_dim, n_pathways)
        self.dec_out = _make_dec_out(n_pathways, n_genes, adj, init)
        if n_cov > 0:
            # No bias: the per-gene baseline is provided by dec_out's bias;
            # this layer should only capture the shift due to the label.
            self.cov_decoder = nn.Linear(n_cov, n_genes, bias=False)

    def forward(self, z: torch.Tensor, cov: torch.Tensor | None = None) -> torch.Tensor:
        h_prime = torch.tanh(self.dec_latent(z))
        x_hat = self.dec_out(h_prime)
        if self.n_cov > 0:
            if cov is None:
                raise ValueError(
                    f"DenseDecoder was built with n_cov={self.n_cov} > 0 but "
                    f"forward() got cov=None; pass a (batch, {self.n_cov}) "
                    f"covariate tensor"
                )
            x_hat = x_hat + self.cov_decoder(cov)
        return x_hat


class CountDecoder(nn.Module):
    """Decoder for the negative-binomial likelihood.

    Phase 1c added the ``adj`` and ``init`` keyword-only parameters. When
    ``adj`` is provided, the ``dec_out`` layer that produces logits is masked
    so each pathway output only affects its member genes' logits. Softmax
    is applied on the (batched) logit vector as before; the mask does not
    change the softmax's normalisation.
    """

    def __init__(
        self,
        latent_dim: int,
        n_pathways: int,
        n_genes: int,
        n_cov: int = 0,
        *,
        adj: torch.Tensor | None = None,
        init: str = "xavier",
    ):
        super().__init__()
        self.n_cov = n_cov
        self.dec_latent = nn.Linear(latent_dim, n_pathways)
        self.dec_out = _make_dec_out(n_pathways, n_genes, adj, init)
        if n_cov > 0:
            # No bias: the per-gene baseline is provided by dec_out's bias;
            # this layer should only capture the shift due to the label.
            self.cov_decoder = nn.Linear(n_cov, n_genes, bias=False)
        # Per-gene log-dispersion, exponentiated in ``theta`` to enforce positivity.
        self.px_r = nn.Parameter(torch.zeros(n_genes))

    def forward(self, z: torch.Tensor, cov: torch.Tensor | None = None) -> torch.Tensor:
        h_prime = torch.tanh(self.dec_latent(z))
        logits = self.dec_out(h_prime)
        if self.n_cov > 0:
            if cov is None:
                raise ValueError(
                    f"CountDecoder was built with n_cov={self.n_cov} > 0 but "
                    f"forward() got cov=None; pass a (batch, {self.n_cov}) "
                    f"covariate tensor"
                )
            logits = logits + self.cov_decoder(cov)
        return torch.softmax(logits, dim=1)

    @property
    def theta(self) -> torch.Tensor:
        """Per-gene NB dispersion, exp(px_r), always positive."""
        return torch.exp(self.px_r)


class GaussianLikelihood(nn.Module):
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
