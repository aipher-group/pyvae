"""Tests for the Phase 1c wiring: helpers and option threading.

Coverage plan:

- as_float_tensor: accepts torch / numpy / pandas / scipy sparse; rejects
  anything else with a message that names the offending argument.
- _make_dec_out: returns nn.Linear when adj is None, InformedLinear otherwise;
  the masked variant has mask_bias=False so per-gene bias stays free; validates
  the adj shape.
- Encoder wiring: each of the four keyword-only options reaches the underlying
  InformedLinear.
- InformedVAE wiring: informed_decoder produces a masked dec_out; nonneg_encoder
  reaches only the encoder; every option is stored as an attribute; both
  Gaussian and NB variants accept the new options; a model with all options
  turned on trains without errors.
- Backward compatibility: the pre-Phase-1c constructor call still works and
  produces the same shapes as before.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from pyvae.components import (
    CountDecoder,
    DenseDecoder,
    Encoder,
    _make_dec_out,
    as_float_tensor,
)
from pyvae.layers import InformedLinear
from pyvae.models import InformedVAE


# ---------- Fixtures ----------


def make_adj(n_genes: int = 20, n_pathways: int = 3) -> torch.Tensor:
    """Adjacency (n_genes, n_pathways): three non-overlapping pathways."""
    adj = torch.zeros(n_genes, n_pathways)
    adj[0:10, 0] = 1
    adj[5:15, 1] = 1
    adj[10:20, 2] = 1
    return adj


# ---------- as_float_tensor ----------


class TestAsFloatTensor:
    def test_torch_tensor_float_passthrough(self):
        x = torch.randn(5, 3)
        out = as_float_tensor(x)
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32

    def test_torch_tensor_int_casts_to_float(self):
        x = torch.arange(6).reshape(3, 2)
        out = as_float_tensor(x)
        assert out.dtype == torch.float32

    def test_numpy_array_becomes_tensor(self):
        x = np.random.randn(5, 3).astype(np.float64)
        out = as_float_tensor(x)
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32
        assert out.shape == (5, 3)

    def test_numpy_int_array_becomes_float_tensor(self):
        x = np.arange(6).reshape(3, 2)
        out = as_float_tensor(x)
        assert out.dtype == torch.float32

    def test_pandas_dataframe_becomes_tensor(self):
        df = pd.DataFrame(np.random.randn(5, 3), columns=["a", "b", "c"])
        out = as_float_tensor(df)
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32
        assert out.shape == (5, 3)

    def test_scipy_sparse_becomes_dense_tensor(self):
        from scipy import sparse
        x = sparse.csr_matrix(np.eye(4, dtype=np.float32))
        out = as_float_tensor(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 4)
        assert out.dtype == torch.float32
        assert torch.allclose(out, torch.eye(4))

    def test_unknown_type_raises_typeerror_naming_argument(self):
        with pytest.raises(TypeError, match="expression"):
            as_float_tensor("not an array", name="expression")

    def test_default_name_appears_in_error(self):
        with pytest.raises(TypeError, match="input"):
            as_float_tensor(42)


# ---------- _make_dec_out ----------


class TestMakeDecOut:
    def test_returns_nn_linear_when_adj_is_none(self):
        layer = _make_dec_out(n_pathways=5, n_genes=20, adj=None)
        assert isinstance(layer, nn.Linear)
        assert layer.in_features == 5
        assert layer.out_features == 20

    def test_returns_informed_linear_when_adj_provided(self):
        adj = make_adj()
        layer = _make_dec_out(n_pathways=3, n_genes=20, adj=adj)
        assert isinstance(layer, InformedLinear)

    def test_informed_dec_out_has_mask_bias_disabled(self):
        """Decoder-side masked layer must keep per-gene bias free."""
        adj = make_adj()
        layer = _make_dec_out(n_pathways=3, n_genes=20, adj=adj)
        assert layer.mask_bias_enabled is False

    def test_informed_dec_out_uses_linear_activation(self):
        """Decoder's masked output layer has no activation of its own."""
        adj = make_adj()
        layer = _make_dec_out(n_pathways=3, n_genes=20, adj=adj)
        assert layer.activation == "linear"

    def test_informed_dec_out_forward_shape(self):
        """Forward pass produces (batch, n_genes) as expected by the decoder."""
        adj = make_adj()
        layer = _make_dec_out(n_pathways=3, n_genes=20, adj=adj)
        h_prime = torch.randn(64, 3)
        out = layer(h_prime)
        assert out.shape == (64, 20)

    def test_shape_validation_raises_on_mismatch(self):
        adj = torch.zeros(15, 3)  # wrong shape
        with pytest.raises(ValueError, match="adj must have shape"):
            _make_dec_out(n_pathways=3, n_genes=20, adj=adj)

    def test_init_flag_is_passed_through(self):
        """`init="fan_in"` reaches the underlying InformedLinear."""
        adj = make_adj()
        layer = _make_dec_out(n_pathways=3, n_genes=20, adj=adj, init="fan_in")
        assert layer.init_kind == "fan_in"


# ---------- Encoder wiring ----------


class TestEncoderWiring:
    def test_encoder_default_reproduces_current_behaviour(self):
        adj = make_adj()
        enc = Encoder(adj=adj, latent_dim=4, n_cov=0)
        assert enc.informed.init_kind == "xavier"
        assert enc.informed.normalize_kind == "none"
        assert enc.informed.nonneg is False
        assert enc.informed.standardize_input_enabled is False

    def test_encoder_forwards_init_flag(self):
        adj = make_adj()
        enc = Encoder(adj=adj, latent_dim=4, n_cov=0, init="fan_in")
        assert enc.informed.init_kind == "fan_in"

    def test_encoder_forwards_normalize_flag(self):
        adj = make_adj()
        enc = Encoder(adj=adj, latent_dim=4, n_cov=0, normalize="batch")
        assert enc.informed.normalize_kind == "batch"

    def test_encoder_forwards_nonneg_flag(self):
        adj = make_adj()
        enc = Encoder(adj=adj, latent_dim=4, n_cov=0, nonneg=True)
        assert enc.informed.nonneg is True

    def test_encoder_forwards_standardize_input_flag(self):
        adj = make_adj()
        enc = Encoder(adj=adj, latent_dim=4, n_cov=0, standardize_input=True)
        assert enc.informed.standardize_input_enabled is True


# ---------- Decoder wiring ----------


class TestDecoderWiring:
    @pytest.mark.parametrize("decoder_cls", [DenseDecoder, CountDecoder])
    def test_decoder_default_has_dense_dec_out(self, decoder_cls):
        dec = decoder_cls(latent_dim=4, n_pathways=3, n_genes=20, n_cov=0)
        assert isinstance(dec.dec_out, nn.Linear)
        assert not isinstance(dec.dec_out, InformedLinear)

    @pytest.mark.parametrize("decoder_cls", [DenseDecoder, CountDecoder])
    def test_decoder_with_adj_has_informed_dec_out(self, decoder_cls):
        adj = make_adj()
        dec = decoder_cls(latent_dim=4, n_pathways=3, n_genes=20, n_cov=0, adj=adj)
        assert isinstance(dec.dec_out, InformedLinear)
        assert dec.dec_out.mask_bias_enabled is False


# ---------- InformedVAE wiring ----------


class TestInformedVAEWiring:
    def test_default_construction_stores_default_flags(self):
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4)
        assert model.init_kind == "xavier"
        assert model.normalize_kind == "none"
        assert model.informed_decoder is False
        assert model.nonneg_encoder is False
        assert model.standardize_input_enabled is False

    def test_default_produces_dense_decoder_dec_out(self):
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4)
        assert isinstance(model.decoder.dec_out, nn.Linear)
        assert not isinstance(model.decoder.dec_out, InformedLinear)

    def test_informed_decoder_produces_masked_dec_out(self):
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, informed_decoder=True)
        assert isinstance(model.decoder.dec_out, InformedLinear)
        assert model.decoder.dec_out.mask_bias_enabled is False

    def test_nonneg_encoder_does_not_imply_nonneg_decoder(self):
        """The two sides are separate flags and neither implies the other.

        The original version of this test asserted that nonneg "must not reach
        the decoder" because "a non-negative decoder cannot repress". That
        reasoning is wrong: the pathway layer is a tanh and may be negative, so
        a positive-weight `dec_out` still lowers a gene when its unit falls.
        What the constraint actually does is move the direction from the weight
        to the unit's activity, which is what gives a pathway's sign a meaning.
        The invariant worth protecting is that the flags stay independent.
        """
        adj = make_adj()
        model = InformedVAE(
            adj=adj,
            latent_dim=4,
            informed_decoder=True,
            nonneg_encoder=True,
        )
        assert model.encoder.informed.nonneg is True
        assert isinstance(model.decoder.dec_out, InformedLinear)
        assert model.decoder.dec_out.nonneg is False, (
            "nonneg_encoder must not silently constrain the decoder; "
            "nonneg_decoder is its own flag"
        )

    def test_nonneg_decoder_reaches_decoder_only(self):
        """And the mirror image: nonneg_decoder must not constrain the encoder."""
        adj = make_adj()
        model = InformedVAE(
            adj=adj,
            latent_dim=4,
            informed_decoder=True,
            nonneg_decoder=True,
        )
        assert model.decoder.dec_out.nonneg is True
        assert model.encoder.informed.nonneg is False

    def test_nonneg_decoder_can_still_lower_a_gene(self):
        """The claim that replaced the old comment, asserted rather than argued.

        With every unit->gene weight positive, a gene's logit must still fall
        when its unit's activity falls. If this ever failed, the constraint
        really would cost expressiveness and the old comment would have been
        right.
        """
        adj = make_adj()
        model = InformedVAE(
            adj=adj,
            latent_dim=4,
            informed_decoder=True,
            nonneg_decoder=True,
        )
        w = model.decoder.dec_out.effective_weight()
        live = model.decoder.dec_out.mask > 0
        assert (w[live] > 0).all(), "every live decoder weight should be positive"

        h_up = torch.full((1, adj.shape[1]), 0.9)
        h_down = torch.full((1, adj.shape[1]), -0.9)
        out_up = model.decoder.dec_out(h_up)
        out_down = model.decoder.dec_out(h_down)
        annotated = live.any(dim=1)
        assert (out_down[0, annotated] < out_up[0, annotated]).all(), (
            "a positive-weight decoder must still lower an annotated gene when "
            "its units fall"
        )

    def test_nonneg_decoder_requires_informed_decoder(self):
        """A dense decoder has no units to orient, so the combination is refused."""
        adj = make_adj()
        with pytest.raises(ValueError, match="requires informed_decoder"):
            InformedVAE(adj=adj, latent_dim=4, nonneg_decoder=True)

    def test_standardize_input_reaches_encoder_only(self):
        """standardize_input is an encoder-side fix; decoder input is a latent."""
        adj = make_adj()
        model = InformedVAE(
            adj=adj,
            latent_dim=4,
            informed_decoder=True,
            standardize_input=True,
        )
        assert model.encoder.informed.standardize_input_enabled is True
        assert isinstance(model.decoder.dec_out, InformedLinear)
        assert model.decoder.dec_out.standardize_input_enabled is False

    def test_normalize_reaches_encoder_only(self):
        """normalize applies to the encoder's pre-activation only."""
        adj = make_adj()
        model = InformedVAE(
            adj=adj,
            latent_dim=4,
            informed_decoder=True,
            normalize="batch",
        )
        assert model.encoder.informed.normalize_kind == "batch"
        assert isinstance(model.decoder.dec_out, InformedLinear)
        assert model.decoder.dec_out.normalize_kind == "none"

    def test_init_reaches_both_encoder_and_decoder(self):
        """init is a scale/shape choice that both sides can share."""
        adj = make_adj()
        model = InformedVAE(
            adj=adj,
            latent_dim=4,
            informed_decoder=True,
            init="fan_in",
        )
        assert model.encoder.informed.init_kind == "fan_in"
        assert isinstance(model.decoder.dec_out, InformedLinear)
        assert model.decoder.dec_out.init_kind == "fan_in"


# ---------- Both likelihood variants accept new options ----------


class TestBothLikelihoodsAcceptOptions:
    @pytest.mark.parametrize("likelihood", ["gaussian", "nb"])
    def test_variant_supports_all_new_options(self, likelihood):
        """Both Gaussian and NB variants build successfully with all options on."""
        adj = make_adj()
        model = InformedVAE(
            adj=adj,
            latent_dim=4,
            likelihood=likelihood,
            init="fan_in",
            normalize="batch",
            informed_decoder=True,
            nonneg_encoder=True,
            standardize_input=True,
        )
        assert model.init_kind == "fan_in"
        assert model.normalize_kind == "batch"
        assert model.informed_decoder is True
        assert model.nonneg_encoder is True
        assert model.standardize_input_enabled is True


# ---------- End-to-end: model with all options trains without errors ----------


class TestEndToEndSanity:
    def test_gaussian_model_with_all_options_forward_and_backward(self):
        """A gaussian model with every option on runs a forward + backward cleanly."""
        adj = make_adj()
        model = InformedVAE(
            adj=adj,
            latent_dim=4,
            likelihood="gaussian",
            init="fan_in",
            normalize="batch",
            informed_decoder=True,
            nonneg_encoder=True,
            standardize_input=True,
        )
        model.train()
        x = torch.randn(64, 20)
        recon, mu, log_var, h = model(x)
        loss = model.loss(x, recon, mu, log_var, h)
        loss.backward()
        # No NaN / inf in gradients anywhere.
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"

    def test_nb_model_with_all_options_forward_and_backward(self):
        """An NB model with every option on runs a forward + backward cleanly."""
        adj = make_adj()
        model = InformedVAE(
            adj=adj,
            latent_dim=4,
            likelihood="nb",
            init="fan_in",
            normalize="batch",
            informed_decoder=True,
            nonneg_encoder=True,
            standardize_input=True,
        )
        model.train()
        # NB path uses counts and library, not x directly.
        counts = torch.randint(0, 20, (64, 20)).float()
        library = counts.sum(dim=1, keepdim=True)
        # Encoder still needs an input.
        x = torch.log1p(counts)
        recon, mu, log_var, h = model(x)
        loss = model.loss(x, recon, mu, log_var, h, counts=counts, library=library)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


# ---------- Backward compatibility ----------


class TestBackwardCompatibility:
    def test_pre_phase_1c_constructor_still_works(self):
        """The pre-Phase-1c call signature (positional args only) still works."""
        adj = make_adj()
        model = InformedVAE(
            adj=adj,
            latent_dim=4,
            seed=42,
            l2_lambda=1e-5,
            beta=1.0,
            likelihood="gaussian",
            n_cov=0,
        )
        assert isinstance(model, InformedVAE)
        assert model.n_pathways == 3

    def test_forward_return_signature_unchanged(self):
        """forward still returns (recon, mu, log_var, h) of the same shapes."""
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4)
        x = torch.randn(8, 20)
        out = model(x)
        assert len(out) == 4
        recon, mu, log_var, h = out
        assert recon.shape == (8, 20)
        assert mu.shape == (8, 4)
        assert log_var.shape == (8, 4)
        assert h.shape == (8, 3)


# ---------- Encoder mask ablation ----------


class TestEncoderMaskAblation:
    """`informed_encoder=False` drops the encoder mask and nothing else.

    The decoder's mask has had an ablation flag since Phase 1c; the encoder's
    did not, so "what is the encoder mask worth" had no measurement. These fix
    the flag's meaning: same layer, same init, same normalisation, same sign
    options, annotation removed.
    """

    def test_default_encoder_is_masked(self):
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4)
        assert model.informed_encoder is True
        assert torch.equal(model.encoder.informed.mask, adj.T)

    def test_dense_encoder_has_an_all_ones_mask(self):
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, informed_encoder=False)
        mask = model.encoder.informed.mask
        assert mask.shape == adj.T.shape
        assert torch.equal(mask, torch.ones_like(mask))
        # Orphan pathways keep a free bias too: with no mask they are no
        # longer orphans, because every unit reads every gene.
        assert torch.equal(model.encoder.informed.bias_mask,
                           torch.ones(adj.shape[1]))

    def test_masked_unit_ignores_non_member_genes(self):
        # The property every encoder-side read-out rests on, and the one
        # `integrated_gradients` reports: a unit's activation has exactly zero
        # gradient with respect to a gene not annotated to it.
        adj = make_adj()
        enc = Encoder(adj=adj, latent_dim=4, n_cov=0)
        x = torch.randn(8, adj.shape[0], requires_grad=True)
        enc.informed(x)[:, 0].sum().backward()
        g = x.grad.abs().sum(0)
        members = adj[:, 0] > 0
        assert torch.all(g[~members] == 0)
        assert torch.all(g[members] > 0)

    def test_dense_unit_reads_every_gene(self):
        adj = make_adj()
        enc = Encoder(adj=adj, latent_dim=4, n_cov=0, use_mask=False)
        x = torch.randn(8, adj.shape[0], requires_grad=True)
        enc.informed(x)[:, 0].sum().backward()
        assert torch.all(x.grad.abs().sum(0) > 0)

    def test_integrated_gradients_sees_the_same_difference(self):
        from pyvae.interpret import integrated_gradients

        adj = make_adj()
        x = torch.randn(6, adj.shape[0]).abs()
        members = adj[:, 0] > 0

        masked = InformedVAE(adj=adj, latent_dim=4, seed=0)
        a = integrated_gradients(masked, x=x, pathway_idx=0, steps=8)
        assert torch.all(a[:, ~members] == 0)
        assert torch.any(a[:, members] != 0)

        dense = InformedVAE(adj=adj, latent_dim=4, seed=0,
                            informed_encoder=False)
        b = integrated_gradients(dense, x=x, pathway_idx=0, steps=8)
        assert torch.any(b[:, ~members] != 0)

    def test_the_two_masks_are_independent(self):
        adj = make_adj()
        model = InformedVAE(adj=adj, latent_dim=4, informed_decoder=True,
                            informed_encoder=False)
        enc_mask = model.encoder.informed.mask
        dec_mask = model.decoder.dec_out.mask
        assert torch.equal(enc_mask, torch.ones_like(enc_mask))
        assert (dec_mask == 0).any(), "the decoder mask should still be a mask"

    def test_nonneg_encoder_without_the_mask_raises(self):
        adj = make_adj()
        with pytest.raises(ValueError, match="requires informed_encoder"):
            InformedVAE(adj=adj, latent_dim=4, nonneg_encoder=True,
                        informed_encoder=False)

    def test_the_new_flag_does_not_change_the_default_model(self):
        adj = make_adj()
        a = InformedVAE(adj=adj, latent_dim=4, seed=0)
        b = InformedVAE(adj=adj, latent_dim=4, seed=0, informed_encoder=True)
        sa, sb = a.state_dict(), b.state_dict()
        assert sa.keys() == sb.keys()
        assert all(torch.equal(sa[k], sb[k]) for k in sa)
