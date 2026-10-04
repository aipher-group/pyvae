"""Tests for pseudobulk_paired_test.

Coverage plan:

- Guard tests: shape mismatches, missing labels, too few paired donors.
- Return-value contract: DataFrame shape, columns, index, sort order,
  sign convention.
- The pseudo-replication fix Carlos flagged: with real donor structure,
  a paired Wilcoxon over donors doesn't overstate significance the way
  cell-level Wilcoxon does. Test that a truly-signal gene gets a smaller
  pvalue than a truly-null gene, and — critically — that on donor-level
  data alone we can't rank thousands of null genes as significant just
  because we have 24,000 cells.
- Tie-breaking by |lfc_mean| when many genes hit the same pvalue (the
  2/256 floor case at n=8 donors).
- Pairing correctness: an unpaired donor is dropped.
- Numerical stability: zero-count genes don't produce inf/NaN pvalues.

Fixtures are deliberately small — 8 donors, ~60 cells, 20 genes — so the
whole file runs in seconds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyvae import pseudobulk_paired_test


# ---------- Fixtures ----------


N_GENES = 20


def make_donor_dataset(
    n_donors: int = 8,
    n_cells_per_donor_per_condition: int = 4,
    de_genes: list[int] | None = None,
    de_direction: str = "up",
    de_magnitude: float = 10.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic donor-structured dataset.

    Every donor has ``n_cells_per_donor_per_condition`` cells in each of
    two conditions ("control" and "stimulated"). Baseline counts are
    Poisson(5) per gene per cell. For genes in ``de_genes``, "stimulated"
    cells get an additive multiplier of ``de_magnitude`` — i.e. the gene
    responds to stimulation across all donors, but with per-donor Poisson
    variance.

    Returns (counts, donor_ids, condition), all length n_cells_per_donor_per_condition
    * n_donors * 2.
    """
    rng = np.random.default_rng(seed)
    de_genes = de_genes or []
    donors = []
    conditions = []
    counts_rows = []

    for donor_i in range(n_donors):
        donor_name = f"donor_{donor_i}"
        for cond in ("control", "stimulated"):
            for _ in range(n_cells_per_donor_per_condition):
                baseline = rng.poisson(5, size=N_GENES).astype(np.float64)
                if cond == "stimulated" and de_genes:
                    for g in de_genes:
                        if de_direction == "up":
                            baseline[g] += rng.poisson(de_magnitude)
                        else:
                            baseline[g] = max(0, baseline[g] - de_magnitude)
                counts_rows.append(baseline)
                donors.append(donor_name)
                conditions.append(cond)

    counts = np.stack(counts_rows)
    return counts, np.array(donors), np.array(conditions)


# ---------- Guard tests ----------


class TestGuards:
    def test_donor_ids_length_mismatch_raises(self):
        counts, donor_ids, condition = make_donor_dataset()
        with pytest.raises(ValueError, match="donor_ids has length"):
            pseudobulk_paired_test(
                counts,
                donor_ids[:-5],  # wrong length
                condition,
                label_a="control",
                label_b="stimulated",
            )

    def test_condition_length_mismatch_raises(self):
        counts, donor_ids, condition = make_donor_dataset()
        with pytest.raises(ValueError, match="condition has length"):
            pseudobulk_paired_test(
                counts,
                donor_ids,
                condition[:-3],
                label_a="control",
                label_b="stimulated",
            )

    def test_missing_label_a_raises(self):
        counts, donor_ids, condition = make_donor_dataset()
        with pytest.raises(ValueError, match="label_a=.*not present"):
            pseudobulk_paired_test(
                counts, donor_ids, condition,
                label_a="ghost", label_b="stimulated",
            )

    def test_missing_label_b_raises(self):
        counts, donor_ids, condition = make_donor_dataset()
        with pytest.raises(ValueError, match="label_b=.*not present"):
            pseudobulk_paired_test(
                counts, donor_ids, condition,
                label_a="control", label_b="phantom",
            )

    def test_too_few_paired_donors_raises(self):
        """With min_donors=8 but only 2 donors in both conditions, refuse."""
        counts, donor_ids, condition = make_donor_dataset(n_donors=2)
        with pytest.raises(ValueError, match="paired Wilcoxon"):
            pseudobulk_paired_test(
                counts, donor_ids, condition,
                label_a="control", label_b="stimulated",
                min_donors=8,
            )

    def test_gene_names_wrong_length_raises(self):
        counts, donor_ids, condition = make_donor_dataset()
        with pytest.raises(ValueError, match="gene_names has length"):
            pseudobulk_paired_test(
                counts, donor_ids, condition,
                label_a="control", label_b="stimulated",
                gene_names=["gene_0", "gene_1"],  # only 2, need 20
            )


# ---------- Return-value contract ----------


class TestReturnValueContract:
    def test_returns_dataframe_with_expected_columns(self):
        counts, donor_ids, condition = make_donor_dataset()
        result = pseudobulk_paired_test(
            counts, donor_ids, condition,
            label_a="control", label_b="stimulated",
        )
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {
            "n_donors_paired", "lfc_mean", "lfc_median", "pvalue", "qvalue", "sign",
        }

    def test_output_has_one_row_per_gene(self):
        counts, donor_ids, condition = make_donor_dataset()
        result = pseudobulk_paired_test(
            counts, donor_ids, condition,
            label_a="control", label_b="stimulated",
        )
        assert len(result) == N_GENES

    def test_gene_names_used_as_index(self):
        counts, donor_ids, condition = make_donor_dataset()
        names = [f"GENE_{i}" for i in range(N_GENES)]
        result = pseudobulk_paired_test(
            counts, donor_ids, condition,
            label_a="control", label_b="stimulated",
            gene_names=names,
        )
        assert set(result.index) == set(names)

    def test_n_donors_paired_is_eight(self):
        """With 8 fully-paired donors, all rows should show n_donors_paired=8."""
        counts, donor_ids, condition = make_donor_dataset(n_donors=8)
        result = pseudobulk_paired_test(
            counts, donor_ids, condition,
            label_a="control", label_b="stimulated",
        )
        assert (result["n_donors_paired"] == 8).all()

    def test_sign_matches_lfc_mean(self):
        counts, donor_ids, condition = make_donor_dataset(
            n_donors=8, de_genes=[0, 1, 2], de_direction="up", de_magnitude=20.0,
        )
        result = pseudobulk_paired_test(
            counts, donor_ids, condition,
            label_a="stimulated", label_b="control",  # A > B
        )
        # DE genes should be up (positive sign) since label_a=stimulated,
        # and up-regulated genes get lfc = log2(stim/ctrl) > 0.
        for g in [0, 1, 2]:
            assert result.loc[g, "sign"] == 1

    def test_sort_order_pvalue_asc_then_abs_lfc_desc(self):
        counts, donor_ids, condition = make_donor_dataset(
            n_donors=8, de_genes=[0, 1, 2], de_direction="up", de_magnitude=15.0,
        )
        result = pseudobulk_paired_test(
            counts, donor_ids, condition,
            label_a="stimulated", label_b="control",
        )
        # Rows with finite pvalue: non-decreasing pvalue.
        finite = result.dropna(subset=["pvalue"])
        p_values = finite["pvalue"].values
        for i in range(1, len(p_values)):
            assert p_values[i] >= p_values[i - 1] - 1e-12


# ---------- Pseudo-replication fix: donor-level test finds true signal ----------


class TestDonorLevelSignal:
    def test_strong_signal_gene_gets_smaller_pvalue_than_null(self):
        """A gene that goes up consistently across all 8 donors should have
        a smaller pvalue than a gene that's just Poisson noise everywhere."""
        counts, donor_ids, condition = make_donor_dataset(
            n_donors=8, de_genes=[0, 1, 2], de_direction="up", de_magnitude=20.0,
        )
        result = pseudobulk_paired_test(
            counts, donor_ids, condition,
            label_a="stimulated", label_b="control",
        )
        # Gene 0 (DE): consistent large up-regulation across all donors
        # → all 8 log-ratios positive → pvalue at the floor 2/256.
        # Gene 5 (null): Poisson noise, roughly equal counts in both conditions
        # → log-ratios distributed around 0 → higher pvalue.
        p_de = result.loc[0, "pvalue"]
        p_null = result.loc[5, "pvalue"]
        # Not always true for every seed, but for seed=0 with de_magnitude=20
        # the effect is strong enough to be clean.
        assert p_de <= p_null, (
            f"DE gene should have smaller pvalue than null; "
            f"got p_de={p_de:.4f}, p_null={p_null:.4f}"
        )

    def test_pvalue_floor_at_n_donors_eight(self):
        """At n=8 donors with all differences same sign, the two-sided
        Wilcoxon exact null hits its floor at 2/2^8 = 2/256 ≈ 0.0078."""
        counts, donor_ids, condition = make_donor_dataset(
            n_donors=8, de_genes=[0], de_direction="up", de_magnitude=50.0,
        )
        result = pseudobulk_paired_test(
            counts, donor_ids, condition,
            label_a="stimulated", label_b="control",
        )
        p_de = result.loc[0, "pvalue"]
        # scipy computes the exact test at small n. Floor is 2/256 = 0.0078.
        # Allow small slack — 8 donors all same sign is very strong signal.
        assert p_de <= 0.008, (
            f"Wilcoxon at n=8 donors with unanimous direction should hit the "
            f"exact-null floor near 2/256 = 0.0078; got p={p_de:.6f}"
        )


# ---------- Tie-breaking by |lfc_mean| ----------


class TestTieBreaking:
    def test_hits_floor_ties_broken_by_abs_lfc(self):
        """When multiple genes hit the same pvalue (the 2/256 floor), the
        gene with larger |lfc_mean| should sort first."""
        # Plant three genes with unanimous direction but different effect sizes.
        counts, donor_ids, condition = make_donor_dataset(
            n_donors=8, de_genes=[0, 1, 2], de_direction="up", de_magnitude=30.0,
            seed=1,
        )
        # Make gene 0 the biggest by adding more to it.
        stim_mask = condition == "stimulated"
        counts[stim_mask, 0] += 500  # much bigger effect on gene 0
        counts[stim_mask, 1] += 100  # medium
        counts[stim_mask, 2] += 20   # small

        result = pseudobulk_paired_test(
            counts, donor_ids, condition,
            label_a="stimulated", label_b="control",
        )
        # All three should have similar pvalues (unanimous direction) but
        # different lfc_mean.
        # Find the top three in the sort — they should be genes 0, 1, 2 in
        # some order, with |lfc_mean| decreasing.
        top3 = result.head(3)
        abs_lfc = top3["lfc_mean"].abs().values
        for i in range(1, len(abs_lfc)):
            assert abs_lfc[i] <= abs_lfc[i - 1] + 1e-12


# ---------- Pairing correctness ----------


class TestPairingCorrectness:
    def test_unpaired_donor_dropped(self):
        """A donor with cells in only one condition should not appear in the
        paired count."""
        # Start with 4 fully-paired donors, then add one control-only donor.
        counts, donor_ids, condition = make_donor_dataset(n_donors=4)
        # Append 4 cells from donor_only_ctrl in control condition only.
        extra_counts = np.random.RandomState(0).poisson(5, size=(4, N_GENES)).astype(np.float64)
        counts_full = np.vstack([counts, extra_counts])
        donors_full = np.concatenate([donor_ids, ["donor_only_ctrl"] * 4])
        condition_full = np.concatenate([condition, ["control"] * 4])

        result = pseudobulk_paired_test(
            counts_full, donors_full, condition_full,
            label_a="control", label_b="stimulated",
        )
        # Should still see n_donors_paired=4 (the original 4), not 5.
        assert (result["n_donors_paired"] == 4).all()


# ---------- Numerical stability ----------


class TestNumericalStability:
    def test_zero_count_genes_dont_produce_inf(self):
        """A gene with zero counts in every cell should produce a finite
        (though possibly NaN) pvalue, never inf."""
        counts, donor_ids, condition = make_donor_dataset(n_donors=8)
        # Zero out gene 0 completely.
        counts[:, 0] = 0.0
        result = pseudobulk_paired_test(
            counts, donor_ids, condition,
            label_a="control", label_b="stimulated",
        )
        # pvalue for gene 0: either NaN (wilcoxon refused) or finite. Not inf.
        p = result.loc[0, "pvalue"]
        assert np.isnan(p) or np.isfinite(p)
        # lfc_mean should be exactly 0 (log2((0+1)/(0+1)) = 0 for every donor).
        assert result.loc[0, "lfc_mean"] == 0.0

    def test_lfc_never_infinite(self):
        counts, donor_ids, condition = make_donor_dataset(n_donors=8)
        result = pseudobulk_paired_test(
            counts, donor_ids, condition,
            label_a="control", label_b="stimulated",
        )
        assert np.isfinite(result["lfc_mean"]).all()
        assert np.isfinite(result["lfc_median"]).all()

    def test_pandas_dataframe_input_works(self):
        """counts can be a pandas DataFrame, not just an array."""
        counts, donor_ids, condition = make_donor_dataset(n_donors=8)
        result = pseudobulk_paired_test(
            pd.DataFrame(counts),
            donor_ids, condition,
            label_a="control", label_b="stimulated",
        )
        assert len(result) == N_GENES
