"""Tests for pathway_unit_fidelity — the encoder-side fidelity diagnostic.

Approach: plant three units by hand, whose behaviour is known ahead of time,
and check that the function correctly identifies each. This is more informative
than testing on a trained model, because a trained model's fidelity is an
empirical result we would then be asserting rather than deriving.

The synthetic setup, once for the whole file:

- 20 genes, 3 pathways, 200 cells.
- Adjacency:
    pathway 0 members: genes 0-9    (10 genes)
    pathway 1 members: genes 5-14   (10 genes, overlaps pathway 0 by 5)
    pathway 2 members: genes 10-19  (10 genes)
- Expression matrix: each cell has three latent "pathway scores" drawn i.i.d.
  N(0, 1). Each gene's expression = mean(scores of pathways it belongs to)
  + small noise. This gives a matrix where each pathway's mean-z-scored
  member expression tracks its own latent score by construction.

The fake encoder we build has one masked linear layer with hand-crafted
weights, and returns h = layer(x) (no activation, no BatchNorm — those are
irrelevant for fidelity, which is a Pearson correlation and is invariant
to affine transforms of h).

Unit 0: weights uniformly +1 on its pathway's members (should track)
Unit 1: weights uniformly -1 on its pathway's members (should be inverted)
Unit 2: weights placed on the WRONG genes (pathway 0's members instead of
        pathway 2's) — the function should still index this unit by pathway 2's
        adjacency column, so the correlation with pathway 2's mean-z is ~0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from pyvae import pathway_unit_fidelity


# ---------- Synthetic fixtures ----------


N_CELLS = 200
N_GENES = 20
N_PATHWAYS = 3
SEED = 42


def build_adj() -> np.ndarray:
    """Adjacency (n_genes, n_pathways): three overlapping pathways.

    - pathway 0: genes 0-9
    - pathway 1: genes 5-14  (overlaps pathway 0 by 5 genes)
    - pathway 2: genes 10-19
    """
    adj = np.zeros((N_GENES, N_PATHWAYS), dtype=np.float32)
    adj[0:10, 0] = 1
    adj[5:15, 1] = 1
    adj[10:20, 2] = 1
    return adj


def build_expression(adj: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Build (x, scores).

    scores : (n_cells, n_pathways) latent pathway activity per cell.
    x      : (n_cells, n_genes) expression = mean(scores of pathways
             the gene belongs to) + noise.

    Constructed so that pathway j's mean-z-scored member expression
    tracks scores[:, j] closely — which is the ground truth a well-behaved
    encoder unit should track.
    """
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(N_CELLS, N_PATHWAYS)).astype(np.float32)

    # Each gene = mean of pathway scores it belongs to, + small noise.
    x = np.zeros((N_CELLS, N_GENES), dtype=np.float32)
    for g in range(N_GENES):
        member_pathways = np.nonzero(adj[g])[0]
        if len(member_pathways) == 0:
            continue
        x[:, g] = scores[:, member_pathways].mean(axis=1)

    x = x + rng.normal(scale=0.05, size=x.shape).astype(np.float32)
    return x, scores


class PlantedEncoderModel(nn.Module):
    """Fake InformedVAE-shaped model.

    Has just enough surface to be usable by pathway_unit_fidelity:
    - .encode(x, cov=None) -> (mu, log_var, h)
    - .n_cov attribute
    - .parameters() (for device detection)

    The hidden layer's weights are set by the caller so we can plant units
    with known behaviour.
    """

    def __init__(self, planted_weights: np.ndarray, n_cov: int = 0):
        super().__init__()
        n_genes, n_units = planted_weights.shape
        self.n_cov = n_cov
        # A dummy linear that maps x -> h. Weights: (n_units, n_genes) in PyTorch layout.
        self.hidden = nn.Linear(n_genes, n_units, bias=False)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.from_numpy(planted_weights.T).float())
        # Dummy heads so we can produce (mu, log_var) — content doesn't matter.
        self.mu_head = nn.Linear(n_units, 4, bias=False)
        self.log_var_head = nn.Linear(n_units, 4, bias=False)

    def encode(self, x, cov=None):
        h = self.hidden(x)
        mu = self.mu_head(h)
        log_var = self.log_var_head(h)
        return mu, log_var, h


@pytest.fixture
def synthetic_data():
    adj = build_adj()
    x, scores = build_expression(adj)
    adj_df = pd.DataFrame(
        adj,
        index=[f"gene_{i}" for i in range(N_GENES)],
        columns=["pathway_0", "pathway_1", "pathway_2"],
    )
    return x, adj_df, scores


# ---------- The three planted units ----------


def test_unit_that_tracks_its_pathway_has_high_positive_correlation(synthetic_data):
    """A unit whose weights uniformly sum its pathway's members should track."""
    x, adj_df, _ = synthetic_data
    adj = adj_df.values

    # Unit 0: +1 on pathway 0's members (genes 0-9)
    # Unit 1 and Unit 2: also set to +1 on their pathways for a full model.
    # But we only assert on unit 0 in this test.
    weights = np.zeros((N_GENES, N_PATHWAYS), dtype=np.float32)
    weights[:, 0] = adj[:, 0]        # unit 0 tracks pathway 0
    weights[:, 1] = adj[:, 1]        # unit 1 tracks pathway 1 (not tested here)
    weights[:, 2] = adj[:, 2]        # unit 2 tracks pathway 2 (not tested here)

    model = PlantedEncoderModel(weights)

    fid = pathway_unit_fidelity(model, x, adj_df, min_genes=5)

    assert fid.loc["pathway_0", "corr"] > 0.8, (
        f"A unit summing its pathway's members should track it strongly. "
        f"Got corr={fid.loc['pathway_0', 'corr']:.3f}"
    )
    assert fid.loc["pathway_0", "sign"] == 1
    assert fid.loc["pathway_0", "abs_corr"] > 0.8


def test_unit_that_is_inverted_has_high_negative_correlation(synthetic_data):
    """A unit with negative weights on its pathway's members should anti-correlate."""
    x, adj_df, _ = synthetic_data
    adj = adj_df.values

    weights = np.zeros((N_GENES, N_PATHWAYS), dtype=np.float32)
    weights[:, 0] = adj[:, 0]         # unit 0 tracks pathway 0 (not tested here)
    weights[:, 1] = -adj[:, 1]        # unit 1 is INVERTED on pathway 1
    weights[:, 2] = adj[:, 2]         # unit 2 tracks pathway 2

    model = PlantedEncoderModel(weights)

    fid = pathway_unit_fidelity(model, x, adj_df, min_genes=5)

    assert fid.loc["pathway_1", "corr"] < -0.8, (
        f"A unit with negative weights on its pathway's members should be "
        f"strongly anti-correlated. Got corr={fid.loc['pathway_1', 'corr']:.3f}"
    )
    assert fid.loc["pathway_1", "sign"] == -1
    assert fid.loc["pathway_1", "abs_corr"] > 0.8


def test_unit_that_ignores_its_pathway_has_low_correlation(synthetic_data):
    """A unit whose weights are on the wrong genes should not track its pathway."""
    x, adj_df, _ = synthetic_data
    adj = adj_df.values

    weights = np.zeros((N_GENES, N_PATHWAYS), dtype=np.float32)
    weights[:, 0] = adj[:, 0]         # unit 0 tracks pathway 0
    weights[:, 1] = adj[:, 1]         # unit 1 tracks pathway 1
    # Unit 2: point at pathway 0's members (genes 0-9) even though we'll
    # index it as pathway 2's unit. Fidelity for pathway 2 should be low.
    weights[:, 2] = adj[:, 0]

    model = PlantedEncoderModel(weights)

    fid = pathway_unit_fidelity(model, x, adj_df, min_genes=5)

    assert abs(fid.loc["pathway_2", "corr"]) < 0.4, (
        f"A unit whose weights are on the wrong genes should not track its "
        f"claimed pathway. Got corr={fid.loc['pathway_2', 'corr']:.3f}"
    )


# ---------- Structural tests: the return-value contract ----------


def test_returns_dataframe_indexed_by_pathway_names(synthetic_data):
    """The output is a DataFrame with pathway_name index and four expected columns."""
    x, adj_df, _ = synthetic_data
    weights = adj_df.values.astype(np.float32)  # any weights will do
    model = PlantedEncoderModel(weights)

    fid = pathway_unit_fidelity(model, x, adj_df, min_genes=5)

    assert isinstance(fid, pd.DataFrame)
    assert list(fid.index) == ["pathway_0", "pathway_1", "pathway_2"]
    assert set(fid.columns) == {"n_genes", "corr", "abs_corr", "sign"}
    assert fid.index.name == "pathway"


def test_n_genes_column_matches_adjacency_column_sums(synthetic_data):
    """n_genes reports the true count of annotated genes per pathway."""
    x, adj_df, _ = synthetic_data
    weights = adj_df.values.astype(np.float32)
    model = PlantedEncoderModel(weights)

    fid = pathway_unit_fidelity(model, x, adj_df, min_genes=5)
    expected = adj_df.sum(axis=0).astype(int)

    for p in adj_df.columns:
        assert fid.loc[p, "n_genes"] == expected[p]


def test_pathway_below_min_genes_returns_nan_correlation():
    """A pathway with fewer than min_genes members gets corr=NaN and sign=0."""
    adj = np.zeros((N_GENES, N_PATHWAYS), dtype=np.float32)
    adj[0:10, 0] = 1
    adj[10:20, 1] = 1
    adj[0:3, 2] = 1   # only 3 genes — below any reasonable min_genes threshold
    adj_df = pd.DataFrame(
        adj,
        index=[f"gene_{i}" for i in range(N_GENES)],
        columns=["p0", "p1", "p_small"],
    )

    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N_CELLS, N_GENES)).astype(np.float32)

    weights = adj.astype(np.float32)
    model = PlantedEncoderModel(weights)

    fid = pathway_unit_fidelity(model, x, adj_df, min_genes=10)

    assert np.isnan(fid.loc["p_small", "corr"])
    assert fid.loc["p_small", "sign"] == 0


# ---------- Guard tests ----------


def test_raises_when_gene_axis_mismatches(synthetic_data):
    """A gene-count mismatch between x and adj must raise, not silently produce garbage."""
    x, adj_df, _ = synthetic_data
    weights = adj_df.values.astype(np.float32)
    model = PlantedEncoderModel(weights)

    # x has 20 genes; hand adj with 25.
    bad_adj = pd.DataFrame(
        np.zeros((25, N_PATHWAYS), dtype=np.float32),
        columns=list(adj_df.columns),
    )
    with pytest.raises(ValueError, match="Gene axis mismatch"):
        pathway_unit_fidelity(model, x, bad_adj, min_genes=5)


def test_raises_when_covariate_missing_for_conditional_model(synthetic_data):
    """A model with n_cov > 0 must be given cov, not silently encode without it."""
    x, adj_df, _ = synthetic_data
    weights = adj_df.values.astype(np.float32)
    model = PlantedEncoderModel(weights, n_cov=2)

    with pytest.raises(ValueError, match="n_cov=2"):
        pathway_unit_fidelity(model, x, adj_df, cov=None, min_genes=5)


def test_raises_when_covariate_given_for_unconditional_model(synthetic_data):
    """A model with n_cov = 0 must not be handed a cov, since it will be ignored."""
    x, adj_df, _ = synthetic_data
    weights = adj_df.values.astype(np.float32)
    model = PlantedEncoderModel(weights, n_cov=0)

    fake_cov = np.zeros((N_CELLS, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="Pass cov=None"):
        pathway_unit_fidelity(model, x, adj_df, cov=fake_cov, min_genes=5)
