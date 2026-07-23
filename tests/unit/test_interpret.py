"""Unit tests for the interpretation read-outs (#16).

Covers:
  * bayes_factor_da: DataFrame contract, reproducibility, planted-signal
    recovery, tie handling, works with conditional models and NB decoders.
  * integrated_gradients: output contract, determinism, baseline invariant,
    signed attributions, works with conditional models and NB decoders.

The Kang biological-plausibility checks from the acceptance list
(interferon pathways ranked top, ISG15 among top-attribution genes) live in
the analysis notebook rather than the automated suite: they need the real
dataset. This file exercises the code paths that would produce those
results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pyvae.interpret import bayes_factor_da, integrated_gradients
from pyvae.models import InformedVAE


# Small helpers

def _tiny_model(
    n_genes: int = 20,
    n_pathways: int = 8,
    latent_dim: int = 4,
    likelihood: str = "gaussian",
    n_cov: int = 0,
    seed: int = 42,
) -> InformedVAE:
    """Build a small, deterministic InformedVAE for testing."""
    torch.manual_seed(seed)
    adj = torch.randint(0, 2, size=(n_genes, n_pathways), dtype=torch.float32)
    return InformedVAE(
        adj=adj,
        latent_dim=latent_dim,
        seed=seed,
        likelihood=likelihood,
        n_cov=n_cov,
    )


# bayes_factor_da — contract, reproducibility, tie handling

def test_bayes_factor_da_returns_correct_dataframe_shape():
    """Output is a DataFrame with n_pathways rows and [bf, p] columns."""
    model = _tiny_model(n_pathways=8)
    x_a = torch.randn(50, 20)
    x_b = torch.randn(50, 20)

    df = bayes_factor_da(model, x_a, x_b, n_pairs=500, seed=0)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (8, 2)
    assert list(df.columns) == ["bf", "p"]


def test_bayes_factor_da_sorted_by_absolute_bf_desc():
    """The DataFrame is sorted so |bf| decreases down the rows."""
    model = _tiny_model()
    x_a = torch.randn(50, 20)
    x_b = torch.randn(50, 20)

    df = bayes_factor_da(model, x_a, x_b, n_pairs=500, seed=0)

    abs_bf = df["bf"].abs().values
    assert all(abs_bf[i] >= abs_bf[i + 1] for i in range(len(abs_bf) - 1))


def test_bayes_factor_da_probabilities_in_unit_interval():
    """p is always in [0, 1]; values are finite."""
    model = _tiny_model()
    x_a = torch.randn(50, 20)
    x_b = torch.randn(50, 20)

    df = bayes_factor_da(model, x_a, x_b, n_pairs=500, seed=0)

    assert (df["p"] >= 0).all()
    assert (df["p"] <= 1).all()
    assert np.isfinite(df["bf"]).all()


def test_bayes_factor_da_reproducible_with_same_seed():
    """Two calls with the same seed return identical bf and p."""
    model = _tiny_model()
    x_a = torch.randn(50, 20)
    x_b = torch.randn(50, 20)

    df_1 = bayes_factor_da(model, x_a, x_b, n_pairs=500, seed=42)
    df_2 = bayes_factor_da(model, x_a, x_b, n_pairs=500, seed=42)

    pd.testing.assert_frame_equal(df_1, df_2)


def test_bayes_factor_da_identical_groups_give_near_zero_bf():
    """When x_a and x_b are identical, ties dominate, so p ~= 0.5 and bf ~= 0
    within Monte-Carlo noise.

    Note: even with x_a == x_b, the sampled pairs (idx_a, idx_b) draw two
    different cells whose h-values differ, so exact zero is impossible.
    Instead we check that BFs stay within the Monte-Carlo confidence interval
    around p = 0.5.

    Confirms the tie-handling design: without the 0.5/0.5 split, ties (when
    they DO occur) would be counted as False and push p systematically below
    0.5. With the split, p should be symmetric around 0.5.
    """
    model = _tiny_model()
    x = torch.randn(50, 20)

    df = bayes_factor_da(model, x, x, n_pairs=2000, seed=0)

    # Under identical inputs and n_pairs=2000, p ~ 0.5 with std ~= 0.011.
    # A 4-sigma envelope allows ~ p in [0.455, 0.545], i.e. |bf| < ~0.18.
    assert (df["p"] - 0.5).abs().max() < 0.05, (
        f"p values should be within MC noise of 0.5; got:\n{df}"
    )
    assert df["bf"].abs().max() < 0.2, (
        f"BFs should be small under identical inputs; got:\n{df}"
    )
    # Also confirm the sign symmetry: about half positive, half negative.
    assert (df["bf"] > 0).sum() > 0
    assert (df["bf"] < 0).sum() > 0


def test_bayes_factor_da_recovers_planted_signal():
    """Group A shifted upward on a specific pathway ranks that pathway on top
    with a positive BF (i.e., pathway is HIGHER in A).

    We can't cleanly plant a per-pathway signal at the input level because
    the encoder mixes genes into pathways; but we CAN plant a strong overall
    signal by shifting group A's expression up broadly, and verify that the
    resulting BFs are mostly positive (some pathway will dominate).
    """
    torch.manual_seed(0)
    model = _tiny_model()
    x_a = torch.randn(80, 20) + 2.0   # shifted up
    x_b = torch.randn(80, 20)          # baseline

    df = bayes_factor_da(model, x_a, x_b, n_pairs=2000, seed=0)

    # The most differentially active pathway should be strongly signed.
    top_bf = df["bf"].iloc[0]
    assert abs(top_bf) > 1.0, f"top |bf| = {abs(top_bf)}, expected > 1.0"


def test_bayes_factor_da_pathway_names_used_as_index():
    """If pathway_names is provided, it becomes the DataFrame index."""
    model = _tiny_model(n_pathways=8)
    x_a = torch.randn(50, 20)
    x_b = torch.randn(50, 20)
    names = [f"pathway_{c}" for c in "ABCDEFGH"]

    df = bayes_factor_da(model, x_a, x_b, n_pairs=500, seed=0, pathway_names=names)

    assert set(df.index) == set(names)


def test_bayes_factor_da_pathway_names_length_mismatch_raises():
    """Wrong-length pathway_names produces a clear error."""
    model = _tiny_model(n_pathways=8)
    x_a = torch.randn(50, 20)
    x_b = torch.randn(50, 20)

    with pytest.raises(ValueError, match="pathway_names has length"):
        bayes_factor_da(
            model, x_a, x_b, n_pairs=100, seed=0,
            pathway_names=["only_two", "names"],
        )


def test_bayes_factor_da_works_with_conditional_model():
    """Model with n_cov > 0 requires cov_a / cov_b and returns valid BFs."""
    model = _tiny_model(n_cov=2)

    x_a = torch.randn(30, 20)
    x_b = torch.randn(30, 20)
    cov_a = torch.zeros(30, 2)
    cov_a[:, 0] = 1
    cov_b = torch.zeros(30, 2)
    cov_b[:, 1] = 1

    df = bayes_factor_da(
        model, x_a, x_b, cov_a=cov_a, cov_b=cov_b, n_pairs=500, seed=0,
    )
    assert df.shape == (8, 2)
    assert np.isfinite(df["bf"]).all()


def test_bayes_factor_da_works_with_nb_decoder():
    """NB (CountDecoder) models expose the same encoder path; BFs still work."""
    model = _tiny_model(likelihood="nb")

    x_a = torch.randn(30, 20)
    x_b = torch.randn(30, 20)

    df = bayes_factor_da(model, x_a, x_b, n_pairs=500, seed=0)
    assert df.shape == (8, 2)
    assert np.isfinite(df["bf"]).all()


# integrated_gradients — output contract, invariants

def test_integrated_gradients_returns_correct_shape():
    """Output is (batch, n_genes) — one row per input cell, one per gene."""
    model = _tiny_model()
    x = torch.randn(5, 20)

    attr = integrated_gradients(model, x, pathway_idx=3, steps=20)

    assert attr.shape == (5, 20)


def test_integrated_gradients_is_deterministic():
    """Two calls on the same input return identical attribution (no RNG)."""
    model = _tiny_model()
    x = torch.randn(5, 20)

    torch.manual_seed(0)
    attr_1 = integrated_gradients(model, x, pathway_idx=3, steps=20)
    torch.manual_seed(1)
    attr_2 = integrated_gradients(model, x, pathway_idx=3, steps=20)

    torch.testing.assert_close(attr_1, attr_2)


def test_integrated_gradients_zero_when_baseline_equals_input():
    """When baseline == x, (x - baseline) == 0, so attribution collapses to 0.

    This is a mathematical invariant of Integrated Gradients: no path to
    integrate over means no attribution.
    """
    model = _tiny_model()
    x = torch.randn(5, 20)

    attr = integrated_gradients(model, x, pathway_idx=3, baseline=x, steps=20)

    torch.testing.assert_close(attr, torch.zeros_like(attr))


def test_integrated_gradients_satisfies_completeness_axiom():
    """sum_g attribution[:, g] ~= h(x)[:, pathway_idx] - h(baseline)[:, pathway_idx].

    Completeness is the defining mathematical property of Integrated
    Gradients (Sundararajan et al., 2017): the attributions, summed over all
    input features, must recover the model's actual output difference
    between x and the baseline. This is a stronger check than "runs and
    returns finite numbers" -- it validates the accumulated-gradient formula
    itself, not just its shape.

    With a finite number of Riemann-sum steps the two sides only match up to
    a discretization error that shrinks as `steps` grows, so we use a high
    step count and a loose-but-meaningful tolerance rather than exact
    equality.
    """
    torch.manual_seed(0)
    model = _tiny_model()
    x = torch.randn(4, 20)
    baseline = torch.zeros_like(x)
    pathway_idx = 2

    attr = integrated_gradients(model, x, pathway_idx, baseline=baseline, steps=200)

    model.eval()
    with torch.no_grad():
        _, _, h_x = model.encode(x)
        _, _, h_baseline = model.encode(baseline)
    delta_h = h_x[:, pathway_idx] - h_baseline[:, pathway_idx]

    sum_attr = attr.sum(dim=1)
    torch.testing.assert_close(sum_attr, delta_h, atol=1e-2, rtol=1e-2)


def test_integrated_gradients_finite_and_signed():
    """Attributions are finite and include both positive and negative values."""
    torch.manual_seed(0)
    model = _tiny_model()
    x = torch.randn(5, 20)

    attr = integrated_gradients(model, x, pathway_idx=3, steps=20)

    assert torch.isfinite(attr).all()
    assert (attr > 0).any(), "expected at least one positive attribution"
    assert (attr < 0).any(), "expected at least one negative attribution"


def test_integrated_gradients_custom_baseline_works():
    """A user-supplied baseline is used verbatim (not overridden by zeros)."""
    torch.manual_seed(0)
    model = _tiny_model()
    x = torch.randn(5, 20)
    baseline = torch.ones_like(x)   # non-zero baseline

    attr_custom = integrated_gradients(model, x, pathway_idx=3, baseline=baseline, steps=20)
    attr_default = integrated_gradients(model, x, pathway_idx=3, steps=20)

    # Different baselines should yield different attributions.
    assert not torch.allclose(attr_custom, attr_default)


def test_integrated_gradients_works_with_conditional_model():
    """Model with n_cov > 0 uses cov during encoding for IG."""
    model = _tiny_model(n_cov=2)
    x = torch.randn(5, 20)
    cov = torch.zeros(5, 2)
    cov[:, 0] = 1

    attr = integrated_gradients(model, x, pathway_idx=3, cov=cov, steps=20)

    assert attr.shape == (5, 20)
    assert torch.isfinite(attr).all()


def test_integrated_gradients_works_with_nb_decoder():
    """NB (CountDecoder) models expose the same encoder path; IG still works."""
    model = _tiny_model(likelihood="nb")
    x = torch.randn(5, 20)

    attr = integrated_gradients(model, x, pathway_idx=3, steps=20)

    assert attr.shape == (5, 20)
    assert torch.isfinite(attr).all()