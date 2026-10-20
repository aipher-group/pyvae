"""Where a covariate's effect lives, and what each placement guarantees.

`InformedVAE` has two routes from a covariate to a gene:

    cov -> fc_mean -> z -> dec_latent -> h' -> dec_out      (through the mask)
    cov -> cov_decoder                                      (dense, after it)

`cov_mode` chooses whether the second exists at all, `cov_in_encoder` whether the
first does, and `nonneg_decoder` whether a unit's sign fixes its genes'
direction. The tests here pin the three properties that make the pathway-space
read-out trustworthy, each of which was found by measurement rather than
reasoning:

1. `cov_mode="pathway"` removes the dense term, so nothing bypasses the mask.
2. `cov_in_encoder=False` makes `z` independent of the covariate, so the
   decoder-side term is the *whole* conditional effect rather than one of two.
3. `pathway_contrast` equals what the forward pass actually does, and refuses to
   answer in the configurations where it would be only part of the story.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pyvae import InformedVAE, unit_contrast, unit_sign_structure

N_GENES, N_PATHWAYS, N_COV, LATENT = 40, 8, 3, 4


def make_adj(seed: int = 0) -> torch.Tensor:
    """A dense-enough adjacency that every unit has members and every gene a unit."""
    g = torch.Generator().manual_seed(seed)
    adj = (torch.rand(N_GENES, N_PATHWAYS, generator=g) > 0.45).float()
    adj[:, 0] = 1.0  # one unit that certainly has members
    adj[0, :] = 1.0  # one gene that certainly has units
    return adj


def build(**kw) -> InformedVAE:
    opts = dict(
        adj=make_adj(),
        latent_dim=LATENT,
        n_cov=N_COV,
        likelihood="nb",
        normalize="batch",
        informed_decoder=True,
        nonneg_encoder=True,
        standardize_input=True,
    )
    opts.update(kw)
    return InformedVAE(**opts)


def onehot(level: int, n: int = 1) -> torch.Tensor:
    return torch.eye(N_COV)[[level] * n]


class TestCovMode:
    def test_gene_mode_builds_the_dense_bypass(self):
        m = build(cov_mode="gene")
        assert hasattr(m.decoder, "cov_decoder")
        assert not hasattr(m.decoder, "cov_path")
        assert m.decoder.cov_decoder.weight.shape == (N_GENES, N_COV)

    def test_pathway_mode_builds_a_pathway_term_and_no_gene_term(self):
        m = build(cov_mode="pathway")
        assert hasattr(m.decoder, "cov_path")
        assert not hasattr(m.decoder, "cov_decoder"), (
            "the two are mutually exclusive: leaving the dense term in place "
            "would let the covariate bypass the mask anyway"
        )
        assert m.decoder.cov_path.weight.shape == (N_PATHWAYS, N_COV)
        assert m.decoder.cov_path.bias is None

    def test_pathway_mode_actually_changes_the_output(self):
        m = build(cov_mode="pathway").eval()
        z = torch.randn(5, LATENT)
        with torch.no_grad():
            a = m.decode(z, onehot(0, 5))
            b = m.decode(z, onehot(1, 5))
        assert not torch.allclose(a, b)

    def test_pathway_mode_keeps_the_softmax_property(self):
        m = build(cov_mode="pathway").eval()
        with torch.no_grad():
            out = m.decode(torch.randn(5, LATENT), onehot(2, 5))
        assert torch.allclose(out.sum(dim=1), torch.ones(5), atol=1e-5)

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="cov_mode"):
            build(cov_mode="genes")

    def test_pathway_mode_needs_a_covariate(self):
        with pytest.raises(ValueError, match="requires n_cov"):
            InformedVAE(adj=make_adj(), latent_dim=LATENT, n_cov=0,
                        informed_decoder=True, cov_mode="pathway")


class TestCovInEncoder:
    def test_default_lets_the_covariate_reach_the_latent(self):
        m = build().eval()
        x = torch.rand(6, N_GENES)
        with torch.no_grad():
            mu_a = m.encode(x, onehot(0, 6))[0]
            mu_b = m.encode(x, onehot(1, 6))[0]
        assert not torch.allclose(mu_a, mu_b)
        assert m.encoder.fc_mean.in_features == N_PATHWAYS + N_COV

    def test_closed_route_makes_the_latent_independent_of_the_covariate(self):
        m = build(cov_in_encoder=False).eval()
        x = torch.rand(6, N_GENES)
        with torch.no_grad():
            mu_a, lv_a, _ = m.encode(x, onehot(0, 6))
            mu_b, lv_b, _ = m.encode(x, onehot(2, 6))
        assert torch.equal(mu_a, mu_b), "z must not depend on cov"
        assert torch.equal(lv_a, lv_b)
        assert m.encoder.fc_mean.in_features == N_PATHWAYS

    def test_closed_route_still_accepts_cov(self):
        """The model needs cov for the decoder, so the encoder must tolerate it."""
        m = build(cov_in_encoder=False).eval()
        with torch.no_grad():
            recon, mu, lv, h = m(torch.rand(4, N_GENES), onehot(1, 4))
        assert recon.shape == (4, N_GENES)
        assert mu.shape == (4, LATENT)
        assert h.shape == (4, N_PATHWAYS)


class TestPathwayContrast:
    def test_equals_the_forward_pass_pre_activation_difference(self):
        """The read-out is only worth having if it is what the model does."""
        m = build(cov_mode="pathway", cov_in_encoder=False).eval()
        a, b = onehot(0), onehot(1)
        got = m.pathway_contrast(a, b)

        x = torch.rand(9, N_GENES)
        with torch.no_grad():
            mu = m.encode(x, onehot(0, 9))[0]
            pre_a = m.decoder.dec_latent(mu) + m.decoder.cov_path(onehot(0, 9))
            pre_b = m.decoder.dec_latent(mu) + m.decoder.cov_path(onehot(1, 9))
        expected = (pre_b - pre_a)
        assert expected.shape == (9, N_PATHWAYS)
        # Cell-independent by construction: every row is the same vector.
        assert torch.allclose(expected, expected[0].expand_as(expected), atol=1e-6)
        assert torch.allclose(got, expected[0], atol=1e-6)

    def test_is_antisymmetric_and_zero_for_a_null_contrast(self):
        m = build(cov_mode="pathway", cov_in_encoder=False)
        a, b = onehot(0), onehot(1)
        assert torch.allclose(m.pathway_contrast(a, b),
                              -m.pathway_contrast(b, a), atol=1e-7)
        assert torch.allclose(m.pathway_contrast(a, a),
                              torch.zeros(N_PATHWAYS), atol=1e-7)

    def test_accepts_a_batch_and_reduces_by_the_mean(self):
        m = build(cov_mode="pathway", cov_in_encoder=False)
        single = m.pathway_contrast(onehot(0), onehot(1))
        batched = m.pathway_contrast(onehot(0, 7), onehot(1, 7))
        assert torch.allclose(single, batched, atol=1e-7)

    def test_refuses_gene_mode(self):
        m = build(cov_mode="gene")
        with pytest.raises(ValueError, match='cov_mode="pathway"'):
            m.pathway_contrast(onehot(0), onehot(1))

    def test_refuses_an_open_encoder_route(self):
        """Because then it is one of two routes, and reporting it is a mistake."""
        m = build(cov_mode="pathway", cov_in_encoder=True)
        with pytest.raises(ValueError, match="cov_in_encoder=False"):
            m.pathway_contrast(onehot(0), onehot(1))


class TestUnitContrastDecomposition:
    def test_gene_mode_splits_and_most_of_it_bypasses_the_mask(self):
        m = build(cov_mode="gene").eval()
        res = unit_contrast(m, torch.rand(12, N_GENES), onehot(0, 12),
                            onehot(1, 12))
        s = res["summary"]
        assert 0.0 < s["masked_share"] < 1.0
        assert s["bypass_l2"] > 0
        assert np.allclose(res["genes"]["total"],
                           res["genes"]["masked"] + res["genes"]["bypass"])

    def test_pathway_mode_leaves_nothing_outside_the_mask(self):
        m = build(cov_mode="pathway", cov_in_encoder=False).eval()
        res = unit_contrast(m, torch.rand(12, N_GENES), onehot(0, 12),
                            onehot(1, 12))
        assert res["summary"]["bypass_l2"] == 0.0
        assert res["summary"]["masked_share"] == pytest.approx(1.0)

    def test_pre_activation_shift_is_cell_independent(self):
        """The diagnostic that says the inputs really were a two-level contrast."""
        m = build(cov_mode="gene").eval()
        res = unit_contrast(m, torch.rand(20, N_GENES), onehot(0, 20),
                            onehot(1, 20))
        assert res["summary"]["delta_pre_sd_max"] < 1e-5

    def test_agrees_with_pathway_contrast_where_both_are_valid(self):
        m = build(cov_mode="pathway", cov_in_encoder=False).eval()
        res = unit_contrast(m, torch.rand(15, N_GENES), onehot(0, 15),
                            onehot(1, 15))
        direct = m.pathway_contrast(onehot(0), onehot(1)).cpu().numpy()
        assert np.allclose(direct, res["units"]["delta_pre"].to_numpy(), atol=1e-5)

    def test_accepts_tensors_and_frames_alike(self):
        """torch.Tensor has a `.values` method; a duck-type check gets it wrong."""
        pd = pytest.importorskip("pandas")
        m = build(cov_mode="gene").eval()
        x = torch.rand(10, N_GENES)
        a, b = onehot(0, 10), onehot(1, 10)
        from_tensor = unit_contrast(m, x, a, b)["units"]
        from_frame = unit_contrast(m, pd.DataFrame(x.numpy()),
                                   pd.DataFrame(a.numpy()),
                                   pd.DataFrame(b.numpy()))["units"]
        assert np.allclose(from_tensor.to_numpy(), from_frame.to_numpy())

    def test_requires_a_masked_decoder(self):
        m = InformedVAE(adj=make_adj(), latent_dim=LATENT, n_cov=N_COV,
                        likelihood="nb", informed_decoder=False)
        with pytest.raises(ValueError, match="informed_decoder=True"):
            unit_contrast(m, torch.rand(4, N_GENES), onehot(0, 4), onehot(1, 4))


class TestSaturationDoesNotDestroyTheSign:
    """`tanh(pre + d) - tanh(pre)` must survive a saturated unit.

    This is the one place in the read-out where float32 is not good enough. The
    difference is a catastrophic cancellation, and once |pre| passes about 4 it
    rounds to exactly 0 in single precision -- so a saturated unit reports *no*
    direction rather than a small one, and a caller checking that each unit's
    sign matches its genes' sees a mismatch that the architecture guarantees
    cannot happen. Measured on a real Kang fit before the fix: 9 units of 1,615,
    which is exactly the gap between a reported 0.9944 and 1.0000.
    """

    N_CELLS = 16

    @staticmethod
    def _saturated_model():
        """A model whose units sit deep in the flat tail of `tanh`.

        Large `dec_latent` weights push every pre-activation well past the
        knee; a small `cov_path` keeps the contrast itself tiny, which is the
        regime where the cancellation bites.
        """
        m = build(cov_mode="pathway", cov_in_encoder=False,
                  nonneg_decoder=True).eval()
        with torch.no_grad():
            m.decoder.dec_latent.weight.mul_(60.0)
            m.decoder.dec_latent.bias.add_(30.0)
            m.decoder.cov_path.weight.mul_(1e-3)
        return m

    def _reference(self, m, x, a, b):
        """What the answer is, computed independently and in float64."""
        with torch.no_grad():
            pa = (m.decoder.dec_latent(m.encode(x, a)[0])
                  + m.decoder.cov_path(a)).double()
            pb = (m.decoder.dec_latent(m.encode(x, b)[0])
                  + m.decoder.cov_path(b)).double()
            exact = (torch.tanh(pb) - torch.tanh(pa)).mean(0).numpy()
            # The same quantity the old code produced: the tanh taken in
            # float32 and only then widened.
            single = ((torch.tanh(pb.float()) - torch.tanh(pa.float()))
                      .double().mean(0).numpy())
        return exact, single

    def test_the_setup_really_is_saturated(self):
        """Guards the other two: without saturation they pass for free."""
        m = self._saturated_model()
        res = unit_contrast(m, torch.rand(self.N_CELLS, N_GENES),
                            onehot(0, self.N_CELLS), onehot(1, self.N_CELLS))
        assert res["summary"]["saturated_frac_max"] > 0.9, (
            "this fixture no longer saturates, so the tests below would pass "
            "for the wrong reason")

    def test_the_old_single_precision_route_really_did_lose_the_shift(self):
        """The bug, pinned. If this stops failing, the regression test below
        has nothing left to catch and both can go."""
        m = self._saturated_model()
        x = torch.rand(self.N_CELLS, N_GENES)
        a, b = onehot(0, self.N_CELLS), onehot(1, self.N_CELLS)
        exact, single = self._reference(m, x, a, b)
        lost = int(((exact != 0) & (single == 0)).sum())
        assert lost > 0, (
            "float32 no longer rounds any of these shifts to zero, so this "
            "fixture is not exercising the cancellation")

    def test_unit_contrast_matches_the_float64_reference(self):
        m = self._saturated_model()
        x = torch.rand(self.N_CELLS, N_GENES)
        a, b = onehot(0, self.N_CELLS), onehot(1, self.N_CELLS)
        exact, _ = self._reference(m, x, a, b)
        got = unit_contrast(m, x, a, b)["units"]["delta_post"].to_numpy()
        assert np.allclose(got, exact, rtol=1e-9, atol=0.0), (
            "delta_post disagrees with an independent float64 computation; the "
            "difference must be taken after widening, not before")

    def test_no_unit_writes_against_its_own_shift(self):
        m = self._saturated_model()
        res = unit_contrast(m, torch.rand(self.N_CELLS, N_GENES),
                            onehot(0, self.N_CELLS), onehot(1, self.N_CELLS))
        u = res["units"]
        disagree = int((np.sign(u["delta_pre"]) * np.sign(u["delta_post"])
                        < 0).sum())
        assert disagree == 0, (
            f"{disagree} units move their genes opposite to their own shift; "
            "tanh is monotone, so this is impossible")


class TestUnitSignStructure:
    def test_free_signed_decoder_has_no_pathway_direction(self):
        m = build(cov_mode="gene")
        s = unit_sign_structure(m, min_genes=5)
        assert 0.2 < s["dec_out_frac_negative"] < 0.8, (
            "an unconstrained decoder should be near a coin flip"
        )
        assert s["encoder_frac_negative"] == 0.0, "nonneg_encoder is the control"

    def test_nonneg_decoder_gives_every_unit_a_direction(self):
        m = build(cov_mode="pathway", cov_in_encoder=False, nonneg_decoder=True)
        s = unit_sign_structure(m, min_genes=5)
        assert s["dec_out_frac_negative"] == 0.0
        assert s["frac_pos_median"] == 1.0
        assert s["units_90pct_one_sign"] == 1.0
        assert s["units_no_orientation"] == 0.0

    def test_requires_a_masked_decoder(self):
        m = InformedVAE(adj=make_adj(), latent_dim=LATENT, n_cov=N_COV,
                        likelihood="nb", informed_decoder=False)
        with pytest.raises(ValueError, match="informed_decoder=True"):
            unit_sign_structure(m)


class TestDefaultsUnchanged:
    """Every new option defaults to the old behaviour, including the RNG stream.

    The three flags add parameters and change layer widths when set. If any of
    them perturbed construction when *unset*, every stored checkpoint and every
    golden regression number in this suite would silently shift.
    """

    @pytest.mark.parametrize("n_cov", [0, N_COV])
    @pytest.mark.parametrize("likelihood", ["gaussian", "nb"])
    def test_state_dict_shape_is_untouched(self, n_cov, likelihood):
        adj = make_adj()
        m = InformedVAE(adj=adj, latent_dim=LATENT, n_cov=n_cov,
                        likelihood=likelihood, informed_decoder=True,
                        nonneg_encoder=True, standardize_input=True,
                        normalize="batch")
        keys = {k: tuple(v.shape) for k, v in m.state_dict().items()}
        assert "decoder.cov_path.weight" not in keys
        assert m.cov_mode == "gene"
        assert m.cov_in_encoder is True
        assert m.nonneg_decoder is False
        assert m.decoder.dec_out.nonneg is False
        expected_fc_in = N_PATHWAYS + n_cov if n_cov else N_PATHWAYS
        assert keys["encoder.fc_mean.weight"] == (LATENT, expected_fc_in)
        if n_cov:
            assert keys["decoder.cov_decoder.weight"] == (N_GENES, n_cov)

    def test_weights_are_bit_identical_across_two_constructions(self):
        """Same seed, same stream: the refactor must not reorder parameter creation."""
        a = InformedVAE(adj=make_adj(), latent_dim=LATENT, n_cov=N_COV, seed=7,
                        likelihood="nb", informed_decoder=True,
                        nonneg_encoder=True, normalize="batch")
        b = InformedVAE(adj=make_adj(), latent_dim=LATENT, n_cov=N_COV, seed=7,
                        likelihood="nb", informed_decoder=True,
                        nonneg_encoder=True, normalize="batch")
        for k, v in a.state_dict().items():
            assert torch.equal(v, b.state_dict()[k]), k
