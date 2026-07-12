"""Unit tests for the negative binomial building blocks in components.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import nbinom

from pyvae.components import CountDecoder, nb_log_prob

# nb_log_prob

def test_nb_log_prob_matches_scipy():
    """Formula check against scipy.stats.nbinom.logpmf.

    scipy uses the (n, p) parameterisation: n = theta, p = theta / (theta + mu).
    Our formula uses the (mu, theta) mean-dispersion parameterisation; the two
    are algebraically equivalent, so they must agree to floating-point precision.
    """
    x = torch.tensor([[0.0, 1.0, 2.0, 5.0, 10.0]])
    mu = torch.tensor([[0.5, 1.0, 2.0, 5.0, 10.0]])
    theta = torch.tensor([1.0, 2.0, 5.0, 10.0, 20.0])

    ours = nb_log_prob(x, mu, theta).detach().numpy().flatten()

    scipy_vals = np.array(
        [
            nbinom.logpmf(
                int(x[0, i].item()),
                n=theta[i].item(),
                p=theta[i].item() / (theta[i].item() + mu[0, i].item()),
            )
            for i in range(5)
        ]
    )

    np.testing.assert_allclose(ours, scipy_vals, atol=1e-5)


def test_nb_log_prob_stays_finite_at_zero():
    """log(0) = -inf must not propagate; the eps guard has to keep results finite.

    Tests the common single-cell case: many genes have zero counts and the
    model predicts near-zero mean.
    """
    x = torch.zeros(3, 4)
    mu = torch.zeros(3, 4)
    theta = torch.ones(4)

    lp = nb_log_prob(x, mu, theta)
    assert torch.isfinite(lp).all()


def test_nb_log_prob_broadcasts_theta_across_batch():
    """theta is (n_genes,) but x and mu are (batch, n_genes); broadcasting must work."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mu = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
    theta = torch.tensor([1.0, 2.0, 5.0])   # (n_genes,)

    lp = nb_log_prob(x, mu, theta)
    assert lp.shape == (2, 3)
    assert torch.isfinite(lp).all()

# CountDecoder

def test_count_decoder_output_shape_and_softmax():
    """CountDecoder.forward(z) returns per-cell proportions summing to 1 over genes."""
    dec = CountDecoder(latent_dim=3, n_pathways=4, n_genes=5)
    z = torch.randn(2, 3)

    px_scale = dec(z)

    assert px_scale.shape == (2, 5)
    # Each cell (row) sums to 1 across the gene axis.
    torch.testing.assert_close(
        px_scale.sum(dim=1),
        torch.ones(2),
        rtol=1e-5, atol=1e-5,
    )
    assert (px_scale >= 0).all()
    assert (px_scale <= 1).all()


def test_count_decoder_theta_property_is_positive_and_matches_expr():
    """theta property returns exp(px_r), always strictly positive."""
    dec = CountDecoder(latent_dim=3, n_pathways=4, n_genes=5)

    theta = dec.theta
    assert theta.shape == (5,)
    assert (theta > 0).all()
    # px_r is initialised to zeros, so theta initialises to exp(0) = 1.
    torch.testing.assert_close(theta, torch.ones(5))

    # After a manual perturbation, the property tracks px_r via autograd-safe exp.
    with torch.no_grad():
        dec.px_r.copy_(torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0]))
    theta = dec.theta
    expected = torch.tensor([1.0, np.e, 1.0 / np.e, np.e**2, 1.0 / np.e**2])
    torch.testing.assert_close(theta, expected, rtol=1e-5, atol=1e-5)


def test_informed_vae_nb_builds_and_forward_returns_correct_shapes():
    """InformedVAE(likelihood='nb') constructs a NB-flavoured model whose forward
    returns per-cell proportions, and whose loss accepts counts and library kwargs.
    """
    from pyvae.models import InformedVAE

    adj = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)
    model = InformedVAE(adj=adj, latent_dim=2, seed=42, likelihood="nb")

    # Model reports its likelihood 
    assert model.likelihood_kind == "nb"
    # decoder is the CountDecoder, exposing theta
    assert isinstance(model.decoder, CountDecoder)
    assert model.likelihood is None

    x = torch.randn(3, 4)
    counts = torch.tensor(
        [[10.0, 5.0, 20.0, 15.0],
         [8.0, 12.0, 7.0, 25.0],
         [30.0, 3.0, 40.0, 10.0]],
    )
    library = counts.sum(1, keepdim=True)

    recon, mu, log_var, h = model(x)
    assert recon.shape == (3, 4)
    # NB path returns px_scale, so each cell's recon sums to 1 across genes
    torch.testing.assert_close(recon.sum(dim=1), torch.ones(3), rtol=1e-5, atol=1e-5)
    assert mu.shape == (3, 2)
    assert log_var.shape == (3, 2)
    assert h.shape == (3, 2)

    loss = model.loss(x, recon, mu, log_var, h, counts=counts, library=library)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_informed_vae_nb_loss_rejects_missing_counts_or_library():
    """The NB loss branch raises if counts or library is missing."""
    from pyvae.models import InformedVAE

    adj = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    model = InformedVAE(adj=adj, latent_dim=1, seed=42, likelihood="nb")

    x = torch.randn(2, 2)
    recon, mu, log_var, h = model(x)

    with pytest.raises(ValueError, match="NB likelihood requires"):
        model.loss(x, recon, mu, log_var, h)

    counts = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="NB likelihood requires"):
        model.loss(x, recon, mu, log_var, h, counts=counts)   # no library


def test_informed_vae_rejects_unknown_likelihood():
    """Typos in likelihood argument fail loudly at construction, not later."""
    from pyvae.models import InformedVAE

    adj = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    with pytest.raises(ValueError, match="unknown likelihood"):
        InformedVAE(adj=adj, latent_dim=1, seed=42, likelihood="gausian")   # typo