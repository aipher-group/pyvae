"""Unit and integration tests for covariate conditioning (#14).

Covers Encoder / DenseDecoder / CountDecoder with n_cov > 0, InformedVAE's
threading of the cov argument through encode/decode/forward,
predict_counterfactual determinism and likelihood-guard behavior, and the
swap_condition helper in bio.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pyvae.bio import swap_condition
from pyvae.components import CountDecoder, DenseDecoder, Encoder
from pyvae.models import InformedVAE


# Encoder shape and error tests

def test_encoder_fc_mean_widens_when_n_cov_positive():
    """With n_cov=3, fc_mean input width becomes n_pathways + 3."""
    adj = torch.randint(0, 2, size=(20, 8), dtype=torch.float32)
    enc = Encoder(adj=adj, latent_dim=4, n_cov=3)
    assert enc.fc_mean.in_features == 8 + 3
    assert enc.fc_log_var.in_features == 8 + 3
    assert enc.n_cov == 3


def test_encoder_forward_with_cov_returns_correct_shapes():
    """Encoder forward with valid cov returns (mu, log_var, h) with expected shapes.

    Critically, h is the RAW (batch, n_pathways) tensor, NOT concatenated with cov.
    """
    adj = torch.randint(0, 2, size=(20, 8), dtype=torch.float32)
    enc = Encoder(adj=adj, latent_dim=4, n_cov=3)

    x = torch.randn(5, 20)
    cov = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]],
        dtype=torch.float32,
    )
    mu, log_var, h = enc(x, cov)

    assert mu.shape == (5, 4)
    assert log_var.shape == (5, 4)
    assert h.shape == (5, 8), (
        f"h must be RAW pathway activations (batch, n_pathways), "
        f"NOT concatenated with cov; got shape {h.shape}"
    )


def test_encoder_raises_when_n_cov_positive_but_cov_missing():
    """n_cov > 0 forward without cov produces a clear ValueError, not a shape error."""
    adj = torch.randint(0, 2, size=(20, 8), dtype=torch.float32)
    enc = Encoder(adj=adj, latent_dim=4, n_cov=3)
    x = torch.randn(5, 20)
    with pytest.raises(ValueError, match="n_cov=3"):
        enc(x)


def test_encoder_n_cov_zero_matches_pre_issue_14_behavior():
    """n_cov=0 default: encoder builds identical layers to before this issue."""
    adj = torch.randint(0, 2, size=(20, 8), dtype=torch.float32)
    enc = Encoder(adj=adj, latent_dim=4)
    assert enc.n_cov == 0
    assert enc.fc_mean.in_features == 8
    assert enc.fc_log_var.in_features == 8


# DenseDecoder and CountDecoder shape tests with cov

def test_dense_decoder_with_cov_adds_covariate_layer():
    """DenseDecoder with n_cov > 0 creates cov_decoder with no bias."""
    dec = DenseDecoder(latent_dim=4, n_pathways=8, n_genes=20, n_cov=3)
    assert hasattr(dec, "cov_decoder")
    assert dec.cov_decoder.in_features == 3
    assert dec.cov_decoder.out_features == 20
    assert dec.cov_decoder.bias is None  # bias=False per issue spec


def test_dense_decoder_forward_with_cov_returns_correct_shape():
    """DenseDecoder forward accepts cov and returns (batch, n_genes)."""
    dec = DenseDecoder(latent_dim=4, n_pathways=8, n_genes=20, n_cov=3)
    z = torch.randn(5, 4)
    cov = torch.zeros(5, 3)
    cov[:, 0] = 1  # one-hot
    recon = dec(z, cov)
    assert recon.shape == (5, 20)


def test_count_decoder_with_cov_preserves_softmax_property():
    """CountDecoder with cov: rows still sum to 1 (softmax after adding cov to logits)."""
    dec = CountDecoder(latent_dim=4, n_pathways=8, n_genes=20, n_cov=3)
    z = torch.randn(5, 4)
    cov = torch.zeros(5, 3)
    cov[:, 1] = 1  # one-hot
    px_scale = dec(z, cov)
    assert px_scale.shape == (5, 20)
    # Softmax over gene axis: each row sums to 1
    torch.testing.assert_close(
        px_scale.sum(dim=1),
        torch.ones(5),
        rtol=1e-5, atol=1e-5,
    )
    assert (px_scale >= 0).all()
    assert (px_scale <= 1).all()


def test_count_decoder_cov_actually_changes_output():
    """Sanity: two cov values on the same z produce different px_scale.

    Otherwise the covariate layer was silently doing nothing.
    """
    torch.manual_seed(0)
    dec = CountDecoder(latent_dim=4, n_pathways=8, n_genes=20, n_cov=3)
    z = torch.randn(2, 4)

    cov_a = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
    cov_b = torch.tensor([[0., 0., 1.], [0., 0., 1.]])
    out_a = dec(z, cov_a)
    out_b = dec(z, cov_b)

    # Different covariates on same z must produce different outputs.
    assert not torch.allclose(out_a, out_b)


# InformedVAE with n_cov > 0

def test_informed_vae_n_cov_threads_through_all_components():
    """n_cov flows to encoder AND decoder; encoder.fc_mean input widens."""
    adj = torch.randint(0, 2, size=(20, 8), dtype=torch.float32)
    model = InformedVAE(adj=adj, latent_dim=4, seed=42, n_cov=3)

    assert model.n_cov == 3
    assert model.encoder.n_cov == 3
    assert model.encoder.fc_mean.in_features == 8 + 3  
    assert model.decoder.n_cov == 3


def test_informed_vae_forward_with_cov():
    """Full forward with n_cov > 0 returns correctly shaped mu, log_var, h, recon."""
    adj = torch.randint(0, 2, size=(20, 8), dtype=torch.float32)
    model = InformedVAE(adj=adj, latent_dim=4, seed=42, n_cov=3)

    x = torch.randn(5, 20)
    cov = torch.zeros(5, 3)
    cov[:, 0] = 1

    recon, mu, log_var, h = model(x, cov)
    assert recon.shape == (5, 20)
    assert mu.shape == (5, 4)
    assert log_var.shape == (5, 4)
    assert h.shape == (5, 8)  # RAW, not concatenated


def test_informed_vae_forward_raises_when_cov_missing():
    """n_cov > 0 forward without cov raises clear error, not shape mismatch."""
    adj = torch.randint(0, 2, size=(20, 8), dtype=torch.float32)
    model = InformedVAE(adj=adj, latent_dim=4, seed=42, n_cov=3)

    x = torch.randn(5, 20)
    with pytest.raises(ValueError, match="n_cov=3"):
        model(x)


# predict_counterfactual

def test_predict_counterfactual_is_deterministic():
    """Two calls with identical inputs return identical output.

    Confirms the method uses the posterior mean, not a stochastic z sample.
    """
    torch.manual_seed(0)
    adj = torch.randint(0, 2, size=(20, 8), dtype=torch.float32)
    model = InformedVAE(
        adj=adj, latent_dim=4, seed=42, likelihood="nb", n_cov=2
    )

    x = torch.randn(5, 20)
    library = torch.full((5, 1), 1000.0)
    cov_from = torch.tensor(
        [[1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.]]
    )
    cov_to = torch.tensor(
        [[0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.]]
    )

    out_1 = model.predict_counterfactual(x, library, cov_from, cov_to)
    out_2 = model.predict_counterfactual(x, library, cov_from, cov_to)

    torch.testing.assert_close(out_1, out_2)


def test_predict_counterfactual_output_shape_and_range():
    """Output is (batch, n_genes) and non-negative (px_scale * library)."""
    adj = torch.randint(0, 2, size=(20, 8), dtype=torch.float32)
    model = InformedVAE(
        adj=adj, latent_dim=4, seed=42, likelihood="nb", n_cov=2
    )

    x = torch.randn(5, 20)
    library = torch.full((5, 1), 1000.0)
    cov_from = torch.zeros(5, 2)
    cov_from[:, 0] = 1
    cov_to = torch.zeros(5, 2)
    cov_to[:, 1] = 1

    predicted = model.predict_counterfactual(x, library, cov_from, cov_to)

    assert predicted.shape == (5, 20)
    # px_scale is a softmax output; predicted = px_scale * library must be
    # non-negative and each row must sum to library.
    assert (predicted >= 0).all()
    torch.testing.assert_close(
        predicted.sum(dim=1),
        torch.full((5,), 1000.0),
        rtol=1e-4, atol=1e-4,
    )


def test_predict_counterfactual_rejects_gaussian_model():
    """predict_counterfactual on likelihood='gaussian' raises clear ValueError."""
    adj = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    model = InformedVAE(
        adj=adj, latent_dim=1, seed=42, likelihood="gaussian", n_cov=2
    )

    x = torch.randn(2, 2)
    library = torch.full((2, 1), 100.0)
    cov_from = torch.tensor([[1., 0.], [1., 0.]])
    cov_to = torch.tensor([[0., 1.], [0., 1.]])

    with pytest.raises(ValueError, match="requires likelihood_kind == 'nb'"):
        model.predict_counterfactual(x, library, cov_from, cov_to)


def test_predict_counterfactual_uses_swapped_covariate():
    """Sanity: predictions under cov_to differ from predictions under cov_from.

    If the swap did nothing (bug), the "counterfactual" would equal the identity
    prediction, defeating the whole point of the method.
    """
    torch.manual_seed(0)
    adj = torch.randint(0, 2, size=(20, 8), dtype=torch.float32)
    model = InformedVAE(
        adj=adj, latent_dim=4, seed=42, likelihood="nb", n_cov=2
    )

    x = torch.randn(5, 20)
    library = torch.full((5, 1), 1000.0)
    cov_ctrl = torch.zeros(5, 2)
    cov_ctrl[:, 0] = 1
    cov_stim = torch.zeros(5, 2)
    cov_stim[:, 1] = 1

    # "Identity" counterfactual: encode with control, decode with control.
    identity = model.predict_counterfactual(x, library, cov_ctrl, cov_ctrl)
    # True counterfactual: encode with control, decode with stimulated.
    swapped = model.predict_counterfactual(x, library, cov_ctrl, cov_stim)

    # The two must differ; if they don't, the covariate isn't influencing decode.
    assert not torch.allclose(identity, swapped)


# swap_condition

def _fake_cov() -> pd.DataFrame:
    """Build a synthetic covariate DataFrame with cell_type + condition one-hots."""
    return pd.DataFrame({
        "cell_type_CD4T": [1, 0, 1],
        "cell_type_CD8T": [0, 1, 0],
        "condition_control": [1, 1, 1],
        "condition_stimulated": [0, 0, 0],
    })


def test_swap_condition_flips_condition_columns():
    cov = _fake_cov()
    cov_to = swap_condition(cov, from_label="control", to_label="stimulated")
    assert (cov_to["condition_control"] == 0).all()
    assert (cov_to["condition_stimulated"] == 1).all()


def test_swap_condition_preserves_cell_type_columns():
    cov = _fake_cov()
    cov_to = swap_condition(cov, from_label="control", to_label="stimulated")
    assert (cov_to["cell_type_CD4T"] == cov["cell_type_CD4T"]).all()
    assert (cov_to["cell_type_CD8T"] == cov["cell_type_CD8T"]).all()


def test_swap_condition_does_not_mutate_input():
    """Original DataFrame must be untouched after the call."""
    cov = _fake_cov()
    original_control = cov["condition_control"].copy()
    _ = swap_condition(cov, from_label="control", to_label="stimulated")
    assert (cov["condition_control"] == original_control).all()


def test_swap_condition_round_trip_returns_original():
    """control -> stimulated -> control returns the original condition columns."""
    cov = _fake_cov()
    swapped = swap_condition(cov, from_label="control", to_label="stimulated")
    round_trip = swap_condition(swapped, from_label="stimulated", to_label="control")
    assert (round_trip["condition_control"] == cov["condition_control"]).all()
    assert (round_trip["condition_stimulated"] == cov["condition_stimulated"]).all()


def test_swap_condition_raises_on_missing_column():
    """Typo in to_label raises KeyError with helpful message."""
    cov = _fake_cov()
    with pytest.raises(KeyError, match="pd.get_dummies"):
        swap_condition(cov, from_label="control", to_label="typo")