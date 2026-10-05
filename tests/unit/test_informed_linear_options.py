"""Tests for the InformedLinear options introduced in issue #21 phase 1b.

Coverage plan mirrors Carlos's issue section 2.1:

  * Each new option in isolation — fan_in, normalize, nonneg, standardize_input,
    mask_bias.
  * The effective_weight() contract — reads what the forward actually uses.
  * Guard tests — invalid activation / init / normalize raise, naming what
    was passed.

Defaults reproducing current behaviour is already pinned by
tests/unit/test_refactor_regression.py's golden loss value; we don't re-do
that work here.

Design notes for anyone extending this file later:

  * Fixture weights stay small (0.05, not 1.0). A weight of 1.0 summed across
    ~16 members saturates tanh and every unit outputs the same value, so a
    test written that way is measuring nothing about the layer.
  * BatchNorm with default eps produces std ≈ 0.9956, not exactly 1.0.
    Assertions on normalised standard deviation use rtol=0.05.
  * Where a forward pass is expected to be a specific numeric value, the
    adjacency + input + expected output are all constructed by hand so the
    result is derivable on paper — never asserted against a captured number.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pyvae.layers import InformedLinear


# ---------- Fixtures ----------


def make_adj(n_genes: int = 20, n_pathways: int = 3, orphan_last: bool = False) -> torch.Tensor:
    """Adjacency (n_genes, n_pathways) with three overlapping pathways.

    - pathway 0: genes 0-9    (10 genes)
    - pathway 1: genes 5-14   (10 genes, overlaps pathway 0 by 5)
    - pathway 2: genes 10-19  (10 genes)

    If orphan_last is True, pathway 2 has zero members instead — useful for
    checking that the fan_in initialiser handles n_in == 0 without producing
    inf weights.
    """
    adj = torch.zeros(n_genes, n_pathways)
    adj[0:10, 0] = 1
    adj[5:15, 1] = 1
    if not orphan_last:
        adj[10:20, 2] = 1
    return adj


def make_batch(n_cells: int = 64, n_genes: int = 20, seed: int = 0) -> torch.Tensor:
    """Random input matrix; standard normal, deterministic."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_cells, n_genes, generator=g)


# ---------- Guard tests: invalid arguments raise, naming what was passed ----------


class TestInputValidation:
    def test_invalid_activation_names_the_offender(self):
        adj = make_adj()
        with pytest.raises(ValueError, match="activation must be one of"):
            InformedLinear(adj, activation="softplus")

    def test_invalid_init_names_the_offender(self):
        adj = make_adj()
        with pytest.raises(ValueError, match="init must be one of"):
            InformedLinear(adj, init="he")

    def test_invalid_normalize_names_the_offender(self):
        adj = make_adj()
        with pytest.raises(ValueError, match="normalize must be one of"):
            InformedLinear(adj, normalize="l2")


# ---------- effective_weight(): matches what forward uses ----------


class TestEffectiveWeight:
    def test_effective_weight_equals_weight_times_mask_when_nonneg_off(self):
        """With nonneg=False, effective_weight is just weight * mask."""
        adj = make_adj()
        layer = InformedLinear(adj, activation="linear")
        w_eff = layer.effective_weight()
        w_raw_masked = layer.weight * layer.mask
        assert torch.allclose(w_eff, w_raw_masked)

    def test_effective_weight_equals_softplus_weight_times_mask_when_nonneg_on(self):
        """With nonneg=True, effective_weight is softplus(weight) * mask."""
        adj = make_adj()
        layer = InformedLinear(adj, activation="linear", nonneg=True)
        w_eff = layer.effective_weight()
        w_expected = torch.nn.functional.softplus(layer.weight) * layer.mask
        assert torch.allclose(w_eff, w_expected)

    def test_effective_weight_matches_forward_pass_output(self):
        """The forward pass must use effective_weight() — verify by reproducing it."""
        adj = make_adj()
        layer = InformedLinear(adj, activation="linear", mask_bias=True)
        x = make_batch()
        with torch.no_grad():
            out_forward = layer(x)
            w_eff = layer.effective_weight()
            b = layer.bias * layer.bias_mask
            out_manual = x @ w_eff.T + b
        assert torch.allclose(out_forward, out_manual, atol=1e-6)


# ---------- fan_in init: per-unit bound, zero for orphan units ----------


class TestFanInInit:
    def test_fan_in_zeroes_units_with_no_members(self):
        """A pathway with zero members must have zero weights, not inf."""
        adj = make_adj(orphan_last=True)  # pathway 2 has no members
        layer = InformedLinear(adj, activation="linear", init="fan_in")
        # All weights for the orphan unit should be exactly 0.
        assert torch.all(layer.weight[2] == 0.0), (
            "fan_in must zero units with n_in == 0, not blow up with sqrt(3/0) = inf."
        )
        # No inf, no nan anywhere.
        assert torch.isfinite(layer.weight).all()

    def test_fan_in_bound_is_per_unit_and_matches_formula(self):
        """Each live unit's weights fall in [-sqrt(3/n_in), +sqrt(3/n_in)]."""
        adj = make_adj()
        layer = InformedLinear(adj, activation="linear", init="fan_in")
        n_in_per_unit = layer.mask.sum(dim=1)  # shape (out_f,)
        for j in range(layer.out_f):
            n_in = int(n_in_per_unit[j].item())
            if n_in == 0:
                continue
            bound = (3.0 / n_in) ** 0.5
            live_weights = layer.weight[j][layer.mask[j].bool()]
            # Uniform draws on [-bound, bound] fit exactly in that range.
            assert live_weights.max().item() <= bound + 1e-6
            assert live_weights.min().item() >= -bound - 1e-6

    def test_fan_in_produces_similar_preactivation_variance_across_units(self):
        """Fan-in init should produce roughly-equal pre-activation variance per unit.

        Xavier gives every unit the same scale regardless of how many genes
        it reads, so a unit reading 100 genes gets larger pre-activations
        than a unit reading 3. Fan-in normalises for that.
        """
        # Build an adjacency with very different pathway sizes.
        adj = torch.zeros(50, 3)
        adj[0:5, 0] = 1     # small pathway: 5 members
        adj[5:25, 1] = 1    # medium: 20 members
        adj[25:50, 2] = 1   # large: 25 members
        layer = InformedLinear(adj, activation="linear", init="fan_in")
        x = torch.randn(2000, 50)
        with torch.no_grad():
            pre = x @ layer.effective_weight().T  # (n_cells, 3), no bias, no activation
        per_unit_std = pre.std(dim=0)
        # All three should be close to 1.0 (uniform var 1 → std 1).
        # Relatively loose tolerance because 2000 samples is a small sample.
        assert per_unit_std.max() / per_unit_std.min() < 1.5, (
            f"fan_in should equalise per-unit pre-activation scale; got stds "
            f"{per_unit_std.tolist()} with ratio "
            f"{per_unit_std.max().item() / per_unit_std.min().item():.2f}"
        )


# ---------- normalize: standardises pre-activation ----------


class TestNormalize:
    def test_batchnorm_standardises_preactivation(self):
        """With normalize='batch' and activation='linear', each unit's output
        should have batch mean ≈ 0 and batch std ≈ 1."""
        adj = make_adj()
        layer = InformedLinear(adj, activation="linear", normalize="batch")
        layer.train()  # BatchNorm needs train mode to actually normalise
        x = make_batch(n_cells=256)
        out = layer(x)
        # BatchNorm uses eps → std ≈ 0.9956, not exactly 1.
        assert torch.allclose(out.mean(dim=0), torch.zeros(layer.out_f), atol=1e-5)
        assert torch.allclose(
            out.std(dim=0, unbiased=False),
            torch.ones(layer.out_f),
            rtol=0.05,
        )

    def test_layernorm_standardises_per_row(self):
        """LayerNorm standardises across features per sample, not across batch."""
        adj = make_adj()
        layer = InformedLinear(adj, activation="linear", normalize="layer")
        x = make_batch(n_cells=256)
        out = layer(x)
        # For each row (each cell), mean across pathways ≈ 0, std ≈ 1.
        assert torch.allclose(out.mean(dim=1), torch.zeros(256), atol=1e-5)
        assert torch.allclose(out.std(dim=1, unbiased=False), torch.ones(256), rtol=0.05)


# ---------- nonneg: weights positive, monotone, gradients finite ----------


class TestNonneg:
    def test_effective_weight_is_nonnegative_on_live_positions(self):
        """Every position where the mask is 1 has effective_weight >= 0."""
        adj = make_adj()
        layer = InformedLinear(adj, activation="linear", nonneg=True)
        w_eff = layer.effective_weight()
        live = layer.mask.bool()
        assert (w_eff[live] > 0).all(), (
            "nonneg should keep every live weight strictly positive (softplus > 0)."
        )

    def test_effective_weight_stays_positive_after_gradient_step(self):
        """Even after an optimiser step, softplus(w) * mask remains positive."""
        adj = make_adj()
        layer = InformedLinear(adj, activation="linear", nonneg=True)
        opt = torch.optim.Adam(layer.parameters(), lr=0.1)
        x = make_batch(n_cells=64)
        # A few update steps with random targets — deliberately push weights around.
        for _ in range(20):
            out = layer(x)
            loss = ((out * torch.randn_like(out)).sum(dim=1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        w_eff = layer.effective_weight()
        live = layer.mask.bool()
        assert (w_eff[live] > 0).all()
        assert torch.isfinite(w_eff).all()

    def test_output_is_monotone_in_a_member_gene_when_nonneg(self):
        """Increasing a member gene's expression must not decrease its unit's output."""
        adj = make_adj()
        layer = InformedLinear(adj, activation="linear", nonneg=True)
        # Pick a cell and a member gene of pathway 0.
        member_gene = 3  # gene 3 is in pathway 0 (members 0-9)
        x_low = torch.zeros(1, 20)
        x_high = torch.zeros(1, 20)
        x_high[0, member_gene] = 1.0
        with torch.no_grad():
            out_low = layer(x_low)
            out_high = layer(x_high)
        # Unit 0 must go up (weight is positive, member gene went from 0 to 1).
        assert out_high[0, 0] >= out_low[0, 0], (
            "nonneg unit must be monotone-increasing in each of its member genes."
        )

    def test_nonneg_init_preserves_target_magnitude(self):
        """After init with nonneg, softplus(weight) magnitude ~ target magnitude
        of the chosen init (not collapsed onto softplus(0) = 0.693)."""
        adj = make_adj()
        layer = InformedLinear(adj, activation="linear", nonneg=True, init="fan_in")
        w_eff = layer.effective_weight()
        live = layer.mask.bool()
        live_magnitudes = w_eff[live]
        # Live magnitudes must be spread across a range like fan_in produces,
        # not all clustered near 0.693 (which is what naive softplus(0) would give).
        assert live_magnitudes.max().item() > 0.05, (
            "nonneg init lost the fan_in scale; every weight is near zero."
        )
        # And critically NOT clustered at 0.693 (softplus(0)) — that would mean
        # the init was applied post-softplus rather than pre-softplus.
        assert live_magnitudes.mean().item() < 0.5, (
            f"nonneg init collapsed onto softplus(0) ~ 0.693; got mean "
            f"{live_magnitudes.mean().item():.3f}."
        )


# ---------- standardize_input: loud gene can't dominate its unit ----------


class TestStandardizeInput:
    def test_standardize_input_prevents_loud_gene_from_dominating(self):
        """A deliberately loud gene should not dominate the unit's output when
        standardize_input is on, compared to the same weights without it."""
        adj = make_adj()
        layer = InformedLinear(adj, activation="linear", standardize_input=True)
        layer.train()
        # Deliberately loud gene 0 with mean=100, others standard normal.
        x = torch.randn(256, 20)
        x[:, 0] = x[:, 0] * 50 + 100  # gene 0 is 50x the scale, off-centred
        # Compare against a layer with no standardisation and the same weights.
        layer_no_norm = InformedLinear(adj, activation="linear", standardize_input=False)
        with torch.no_grad():
            layer_no_norm.weight.copy_(layer.weight)
            layer_no_norm.bias.copy_(layer.bias)
        out_with_norm = layer(x)
        out_no_norm = layer_no_norm(x)
        # Without normalisation, unit 0 (which reads gene 0) will have huge
        # magnitude driven by gene 0. With normalisation, it should be much smaller.
        assert out_with_norm[:, 0].abs().max() < out_no_norm[:, 0].abs().max() / 5, (
            "standardize_input should stop a loud gene from dominating its unit's output."
        )


# ---------- mask_bias: on and off ----------


class TestMaskBias:
    def test_mask_bias_true_zeroes_orphan_pathway_bias(self):
        """When mask_bias=True and a pathway has no members, its output gets zero bias."""
        adj = make_adj(orphan_last=True)  # pathway 2 has no members
        layer = InformedLinear(adj, activation="linear", mask_bias=True)
        # Force a non-zero bias to make the test non-trivial.
        with torch.no_grad():
            layer.bias.fill_(1.0)
        # Zero input: output = bias * bias_mask.
        x = torch.zeros(1, 20)
        with torch.no_grad():
            out = layer(x)
        # Pathways 0 and 1 have members -> bias is applied -> out is 1.
        # Pathway 2 is orphan -> bias_mask is 0 -> out is 0.
        assert out[0, 0].item() == pytest.approx(1.0)
        assert out[0, 1].item() == pytest.approx(1.0)
        assert out[0, 2].item() == pytest.approx(0.0)

    def test_mask_bias_false_leaves_orphan_pathway_bias_free(self):
        """When mask_bias=False, every pathway's bias is respected, orphan or not."""
        adj = make_adj(orphan_last=True)
        layer = InformedLinear(adj, activation="linear", mask_bias=False)
        with torch.no_grad():
            layer.bias.fill_(1.0)
        x = torch.zeros(1, 20)
        with torch.no_grad():
            out = layer(x)
        # All three should now have bias=1.0.
        assert out[0, 0].item() == pytest.approx(1.0)
        assert out[0, 1].item() == pytest.approx(1.0)
        assert out[0, 2].item() == pytest.approx(1.0)


# ---------- Mask is respected in forward and backward regardless of options ----------


class TestMaskIntegrity:
    @pytest.mark.parametrize("nonneg", [False, True])
    @pytest.mark.parametrize("init", ["xavier", "fan_in"])
    def test_backward_never_grows_weights_where_mask_is_zero(self, nonneg, init):
        """Non-member positions must stay at zero effective weight after training."""
        adj = make_adj()
        layer = InformedLinear(adj, activation="linear", init=init, nonneg=nonneg)
        opt = torch.optim.Adam(layer.parameters(), lr=0.1)
        x = make_batch(n_cells=64)
        # Non-degenerate loss: avoid .sum() which would produce a constant
        # gradient on a linear layer; use random-weighted sum instead.
        for _ in range(10):
            out = layer(x)
            loss = ((out * torch.randn_like(out)).sum(dim=1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        w_eff = layer.effective_weight()
        dead = ~layer.mask.bool()
        assert torch.all(w_eff[dead] == 0), (
            f"Non-member positions must stay at 0 in effective_weight after training "
            f"(init={init}, nonneg={nonneg}). "
            f"Max abs value at dead positions: {w_eff[dead].abs().max().item()}"
        )
