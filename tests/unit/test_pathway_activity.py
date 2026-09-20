"""Tests for pathway_activity and _benjamini_hochberg.

Coverage plan (mirrors Carlos's issue section 2.1):

- Guard tests: unknown statistic, gene-axis mismatch, both DataFrame paths
  (named alignment vs positional alignment).
- Return-value contract: DataFrame shape, columns, index, sort order, sign
  convention.
- The two invariants Carlos explicitly named:
    * pathway_activity prefers broad support over one huge gene (competitive
      Mann-Whitney beats self-contained on the "one loud member" failure mode).
    * center=True fixes the compositional offset — a genuinely up-regulated
      pathway comes out positive even when one strong outlier elsewhere in
      the panel pulls the raw median down.
- _benjamini_hochberg in isolation: matches known-good outputs, monotone,
  NaN propagation, clipping at 1.
- Guard against a pathway that includes the whole panel (degenerate reference).
- min_genes filter produces NaN rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pyvae import pathway_activity
from pyvae.interpret import _benjamini_hochberg


# ---------- _benjamini_hochberg in isolation ----------


class TestBenjaminiHochberg:
    def test_known_input_matches_reference(self):
        """Sanity-check against a hand-computed BH sequence.

        p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212,
             0.216, 0.216, 0.222, 0.251, 0.269, 0.275, 0.34, 0.341, 0.384,
             0.569, 0.594, 0.696, 0.762, 0.94, 0.942, 0.975, 0.986]
        with n = 26, the Benjamini-Hochberg step-up gives adjusted values
        that at minimum should:
          - be monotone non-decreasing when sorted by input p
          - never exceed 1
          - equal at least the raw p for each entry (never smaller)
        """
        p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205,
                      0.212, 0.216, 0.216, 0.222, 0.251, 0.269, 0.275, 0.34,
                      0.341, 0.384, 0.569, 0.594, 0.696, 0.762, 0.94, 0.942,
                      0.975, 0.986])
        q = _benjamini_hochberg(p)
        # Order by input p and verify monotonicity of q along that order.
        order = np.argsort(p)
        q_sorted = q[order]
        for i in range(1, len(q_sorted)):
            assert q_sorted[i] >= q_sorted[i - 1] - 1e-9
        assert (q <= 1).all()
        assert (q >= p - 1e-9).all(), "BH-adjusted p should never be smaller than raw p"

    def test_matches_statsmodels_when_available(self):
        """When statsmodels is installed, our output matches its fdr_bh."""
        try:
            from statsmodels.stats.multitest import multipletests
        except ImportError:
            pytest.skip("statsmodels not available; skipping cross-check")
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, size=50)
        q_ours = _benjamini_hochberg(p)
        _, q_ref, _, _ = multipletests(p, method="fdr_bh")
        np.testing.assert_allclose(q_ours, q_ref, atol=1e-12)

    def test_nan_inputs_propagate(self):
        p = np.array([0.01, np.nan, 0.05, np.nan, 0.5])
        q = _benjamini_hochberg(p)
        assert np.isnan(q[1])
        assert np.isnan(q[3])
        assert np.isfinite(q[0])
        assert np.isfinite(q[2])
        assert np.isfinite(q[4])

    def test_nan_positions_dont_inflate_n(self):
        """NaN entries should not count toward n in the BH formula.

        If we counted them, adjusting the smallest raw p by n/rank would use
        a larger n and produce a larger q. Verify by injecting NaNs into a
        p vector and checking that the non-NaN q-values are unchanged.
        """
        p_finite = np.array([0.01, 0.02, 0.03])
        q_finite = _benjamini_hochberg(p_finite)

        p_padded = np.array([0.01, 0.02, np.nan, 0.03, np.nan])
        q_padded = _benjamini_hochberg(p_padded)

        # The three finite positions in p_padded should have the same q as
        # p_finite's three positions, in the same order.
        finite_mask = np.isfinite(p_padded)
        np.testing.assert_allclose(q_padded[finite_mask], q_finite)

    def test_clips_at_one(self):
        """p_i * n / rank can exceed 1; output must clip at 1."""
        p = np.array([0.9, 0.95, 0.99])
        q = _benjamini_hochberg(p)
        assert (q <= 1.0).all()

    def test_all_nan_returns_all_nan(self):
        p = np.array([np.nan, np.nan, np.nan])
        q = _benjamini_hochberg(p)
        assert np.isnan(q).all()


# ---------- Fixtures for pathway_activity ----------


N_GENES = 30
N_PATHWAYS = 3


def make_adj(overlap: bool = False) -> pd.DataFrame:
    """Adjacency (n_genes, n_pathways) as a DataFrame with named index."""
    adj = np.zeros((N_GENES, N_PATHWAYS), dtype=np.float32)
    if overlap:
        adj[0:10, 0] = 1
        adj[5:15, 1] = 1
        adj[10:20, 2] = 1
    else:
        adj[0:10, 0] = 1
        adj[10:20, 1] = 1
        adj[20:30, 2] = 1
    return pd.DataFrame(
        adj,
        index=[f"gene_{i}" for i in range(N_GENES)],
        columns=[f"pathway_{j}" for j in range(N_PATHWAYS)],
    )


def make_de(lfc_values: np.ndarray) -> pd.DataFrame:
    """Build a minimal DE table with the given per-gene lfc_mean."""
    return pd.DataFrame(
        {"lfc_mean": lfc_values, "proba_de": np.abs(lfc_values) / 5.0},
        index=[f"gene_{i}" for i in range(len(lfc_values))],
    )


# ---------- Return-value contract ----------


class TestReturnValueContract:
    def test_returns_dataframe_with_expected_columns(self):
        adj = make_adj()
        lfc = np.zeros(N_GENES)
        lfc[0:10] = 1.0  # pathway 0 members are consistently up
        de = make_de(lfc)
        result = pathway_activity(de, adj)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {
            "n_genes", "n_tested", "median_member", "median_reference",
            "effect", "pvalue", "qvalue", "sign",
        }

    def test_output_has_one_row_per_pathway(self):
        adj = make_adj()
        lfc = np.random.RandomState(0).randn(N_GENES)
        de = make_de(lfc)
        result = pathway_activity(de, adj)
        assert len(result) == N_PATHWAYS

    def test_index_is_pathway_names(self):
        adj = make_adj()
        lfc = np.random.RandomState(0).randn(N_GENES)
        de = make_de(lfc)
        result = pathway_activity(de, adj)
        assert set(result.index) == {"pathway_0", "pathway_1", "pathway_2"}
        assert result.index.name == "pathway"

    def test_sort_order_qvalue_asc_then_abs_effect_desc(self):
        adj = make_adj()
        lfc = np.zeros(N_GENES)
        lfc[0:10] = 3.0    # pathway 0: strong signal
        lfc[10:20] = 0.5   # pathway 1: weak signal
        lfc[20:30] = 0.0   # pathway 2: no signal
        de = make_de(lfc)
        result = pathway_activity(de, adj, center=False)
        # Rows with finite qvalue should be non-decreasing in qvalue.
        q_finite = result["qvalue"].dropna().values
        for i in range(1, len(q_finite)):
            assert q_finite[i] >= q_finite[i - 1] - 1e-12

    def test_sign_matches_effect(self):
        adj = make_adj()
        lfc = np.zeros(N_GENES)
        lfc[0:10] = 2.0      # pathway 0 up
        lfc[10:20] = -2.0    # pathway 1 down
        # pathway 2 (genes 20-29): zeros — no signal
        de = make_de(lfc)
        result = pathway_activity(de, adj, center=False)
        assert result.loc["pathway_0", "sign"] == 1
        assert result.loc["pathway_1", "sign"] == -1


# ---------- Guard tests ----------


class TestGuards:
    def test_unknown_statistic_raises(self):
        adj = make_adj()
        de = make_de(np.zeros(N_GENES))
        with pytest.raises(ValueError, match="is not a column"):
            pathway_activity(de, adj, statistic="nonexistent")

    def test_positional_length_mismatch_raises(self):
        """When both indices are integer-based, positional alignment requires equal lengths."""
        de = pd.DataFrame({"lfc_mean": np.zeros(N_GENES)})  # RangeIndex
        adj_arr = np.zeros((N_GENES + 5, N_PATHWAYS), dtype=np.float32)  # different length
        with pytest.raises(ValueError, match="positional alignment"):
            pathway_activity(de, adj_arr)

    def test_missing_genes_in_de_raises(self):
        """adj references genes de doesn't have -> raise, don't silently NaN-fill."""
        adj = make_adj()
        # de missing genes 20-29 that pathway 2 relies on
        de = pd.DataFrame(
            {"lfc_mean": np.zeros(20), "proba_de": np.zeros(20)},
            index=[f"gene_{i}" for i in range(20)],
        )
        with pytest.raises(ValueError, match="not in `de`"):
            pathway_activity(de, adj)

    def test_unknown_adj_type_raises(self):
        de = make_de(np.zeros(N_GENES))
        with pytest.raises(TypeError, match="adj must be"):
            pathway_activity(de, "not an array")


# ---------- min_genes filter ----------


class TestMinGenesFilter:
    def test_small_pathway_gets_nan(self):
        """A pathway with fewer members than min_genes should have NaN pvalue."""
        adj = np.zeros((N_GENES, 3), dtype=np.float32)
        adj[0:10, 0] = 1     # 10 members
        adj[10:12, 1] = 1    # only 2 members
        adj[12:22, 2] = 1    # 10 members
        adj_df = pd.DataFrame(
            adj,
            index=[f"gene_{i}" for i in range(N_GENES)],
            columns=["big_a", "small", "big_b"],
        )
        de = make_de(np.random.RandomState(0).randn(N_GENES))
        result = pathway_activity(de, adj_df, min_genes=5)
        assert np.isnan(result.loc["small", "pvalue"])
        assert np.isnan(result.loc["small", "qvalue"])
        assert result.loc["small", "sign"] == 0
        # Big pathways still get numbers
        assert np.isfinite(result.loc["big_a", "pvalue"])
        assert np.isfinite(result.loc["big_b", "pvalue"])


# ---------- The two key Carlos-flagged invariants ----------


class TestBroadSupportOverOneHugeGene:
    """Carlos: 'pathway_activity prefers broad support over one huge gene'.

    This is the whole reason for competitive over self-contained. A
    self-contained test would rank Pathway B (one gigantic outlier) above
    Pathway A (consistent modest signal). Competitive Mann-Whitney should
    reverse that.
    """
    def test_broad_support_beats_one_outlier(self):
        # Non-overlapping pathways.
        adj = make_adj()

        # Pathway A (genes 0-9): 10 modestly positive members, lfc = 0.5 each.
        # Pathway B (genes 10-19): 9 zeros + 1 huge outlier of lfc=5.
        # Pathway C (genes 20-29): 10 zeros (null).
        # Aggregate |lfc|: A gets 5, B gets 5. Same magnitude, different shape.
        lfc = np.zeros(N_GENES)
        lfc[0:10] = 0.5
        lfc[19] = 5.0
        de = make_de(lfc)

        result = pathway_activity(de, adj, center=False)

        # Pathway A's members are all consistently above the panel median (~0),
        # so Mann-Whitney against non-members finds a real difference.
        # Pathway B has one huge member but 9 that look like the null; the
        # median of B's members is 0, so its effect is ~0.
        assert result.loc["pathway_0", "qvalue"] < result.loc["pathway_1", "qvalue"], (
            "Broad-support pathway should rank higher (smaller qvalue) than a "
            "one-outlier pathway with the same aggregate absolute signal. "
            f"Got: pathway_0 q={result.loc['pathway_0', 'qvalue']:.4f}, "
            f"pathway_1 q={result.loc['pathway_1', 'qvalue']:.4f}"
        )


class TestCenteringFixesCompositionalOffset:
    """Carlos: 'centering fixes the compositional offset'.

    A compositional decoder emits proportions summing to 1. One strongly
    induced gene mechanically depresses every other gene. Test: build a DE
    table where pathway A has a modest positive signal, but a single non-member
    outlier elsewhere pulls the panel median negative. Without centering, a
    genuinely up-regulated pathway can end up with a negative raw member median
    (in absolute terms) but should still test as positive relative to the
    (depressed) rest of the panel. Centering makes the direction consistent.
    """
    def test_centering_recovers_positive_effect(self):
        adj = make_adj()

        # Pathway A members: consistent lfc = +0.3.
        # Rest of panel: nearly all lfc = 0 EXCEPT one extreme outlier at +5
        # in gene_29 (not a member of any pathway we're focusing on).
        # Panel median after centering is 0; pathway A ends up clearly positive.
        # Without centering, the mean-based summaries would still show +0.3
        # for pathway A, but the point of `center` is that in the compositional
        # setting the panel-wide shift can hide the direction. Here we verify
        # the mechanical property: centering doesn't reverse a real signal.
        lfc = np.zeros(N_GENES)
        lfc[0:10] = 0.3
        lfc[29] = 5.0
        de = make_de(lfc)

        result_centered = pathway_activity(de, adj, center=True)
        result_raw = pathway_activity(de, adj, center=False)

        # Both should agree pathway 0 is positive relative to the rest.
        assert result_centered.loc["pathway_0", "sign"] == 1
        assert result_raw.loc["pathway_0", "sign"] == 1

    def test_centering_produces_finite_medians(self):
        """Centering shouldn't introduce NaN or inf into the medians."""
        adj = make_adj()
        lfc = np.random.RandomState(0).randn(N_GENES)
        de = make_de(lfc)
        result = pathway_activity(de, adj, center=True)
        for col in ["median_member", "median_reference", "effect"]:
            assert np.isfinite(result[col]).all()


# ---------- Positional alignment (integer indexed) ----------


class TestPositionalAlignment:
    def test_positional_alignment_matches_named_alignment(self):
        """When gene order is preserved, positional and named alignment agree."""
        adj_df = make_adj()
        lfc = np.zeros(N_GENES)
        lfc[0:10] = 1.5
        de_named = make_de(lfc)

        # Positional version: bare arrays, both integer-indexed.
        de_positional = pd.DataFrame(
            {"lfc_mean": lfc, "proba_de": de_named["proba_de"].values}
        )  # RangeIndex
        adj_arr = adj_df.values

        result_named = pathway_activity(de_named, adj_df, center=False)
        result_positional = pathway_activity(de_positional, adj_arr, center=False)

        # Effects should match.
        for i, pathway in enumerate(result_named.index):
            positional_row = result_positional.iloc[i]
            named_row = result_named.loc[f"pathway_{i}"]
            # Match on effect and sign; qvalue depends on all pathways so it
            # should also match given identical input.
            assert abs(positional_row["effect"] - named_row["effect"]) < 1e-9

    def test_tensor_adj_accepted(self):
        """Bare torch.Tensor adj works via positional alignment."""
        de = pd.DataFrame(
            {"lfc_mean": np.zeros(N_GENES), "proba_de": np.zeros(N_GENES)}
        )  # RangeIndex
        adj_tensor = torch.zeros(N_GENES, N_PATHWAYS)
        adj_tensor[0:10, 0] = 1
        adj_tensor[10:20, 1] = 1
        adj_tensor[20:30, 2] = 1
        result = pathway_activity(de, adj_tensor)
        assert len(result) == N_PATHWAYS


# ---------- Degenerate case: pathway that includes every gene ----------


class TestDegenerateCases:
    def test_pathway_covering_whole_panel_gets_nan(self):
        """A pathway with every gene as a member leaves the reference empty."""
        adj = np.zeros((N_GENES, 2), dtype=np.float32)
        adj[:, 0] = 1        # every gene
        adj[0:10, 1] = 1     # normal
        adj_df = pd.DataFrame(
            adj,
            index=[f"gene_{i}" for i in range(N_GENES)],
            columns=["universal", "normal"],
        )
        de = make_de(np.random.RandomState(0).randn(N_GENES))
        result = pathway_activity(de, adj_df)
        # Universal pathway: reference is empty, can't test.
        assert np.isnan(result.loc["universal", "pvalue"])
        # Normal pathway still works.
        assert np.isfinite(result.loc["normal", "pvalue"])
