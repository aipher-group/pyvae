"""The null that separates the annotation's content from its sparsity.

A masked layer beating a dense one shows only that sparsity helps. To ask whether
the *biology* in the gene->pathway map matters, you need a map with the same
sparsity statistics and no biology. `permute_adjacency` shuffles gene rows, which
preserves every pathway's size exactly and destroys which genes co-occur.

What must hold for the comparison to mean anything: set sizes identical, unit
names unchanged (the point is whether the name still predicts), the result
seed-deterministic, and co-membership actually broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pyvae import permute_adjacency

N_GENES, N_PATHWAYS = 60, 9


def make_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Deliberately uneven set sizes, so "sizes preserved" is a real constraint.
    cols = []
    for j in range(N_PATHWAYS):
        col = np.zeros(N_GENES)
        col[rng.choice(N_GENES, size=3 + 5 * j, replace=False)] = 1.0
        cols.append(col)
    return pd.DataFrame(np.array(cols).T,
                        index=[f"g{i}" for i in range(N_GENES)],
                        columns=[f"p{j}" for j in range(N_PATHWAYS)])


class TestPreservedStatistics:
    def test_every_set_size_is_identical(self):
        adj = make_frame()
        perm = permute_adjacency(adj, seed=1)
        pd.testing.assert_series_equal(perm.sum(axis=0), adj.sum(axis=0))

    def test_the_multiset_of_gene_degrees_is_identical(self):
        """Individual genes swap degrees; the distribution does not change."""
        adj = make_frame()
        perm = permute_adjacency(adj, seed=1)
        assert sorted(perm.sum(axis=1)) == sorted(adj.sum(axis=1))

    def test_total_density_is_identical(self):
        adj = make_frame()
        assert permute_adjacency(adj, seed=3).to_numpy().sum() == adj.to_numpy().sum()

    def test_unit_names_and_gene_names_are_kept(self):
        """A permuted unit keeps its label on purpose: that is the question."""
        adj = make_frame()
        perm = permute_adjacency(adj, seed=1)
        assert list(perm.columns) == list(adj.columns)
        assert list(perm.index) == list(adj.index)


class TestWhatIsAndIsNotDestroyed:
    """A row permutation **relabels the genes**. That is a tighter null than it
    first appears, and the difference matters for how the result is described.

    Preserved: every set size, the degree multiset, and the *entire*
    pathway-by-pathway overlap matrix — because each gene's whole membership row
    travels together, so the partition structure over gene slots is untouched and
    only the slot-to-gene-identity mapping is randomised. The two arms therefore
    differ in nothing except which genes sit where.

    Destroyed: which *named* genes share a unit, and hence any biology a unit
    could learn.
    """

    def test_membership_actually_moves(self):
        adj = make_frame()
        assert not permute_adjacency(adj, seed=1).equals(adj)

    def test_pathway_overlap_structure_is_preserved_exactly(self):
        """Not a weakness of the null: it is what makes the arms matched."""
        adj = make_frame()
        perm = permute_adjacency(adj, seed=1)
        real = adj.to_numpy().T @ adj.to_numpy()
        shuffled = perm.to_numpy().T @ perm.to_numpy()
        assert np.array_equal(real, shuffled), (
            "rows move as units, so pathway-pathway overlap cannot change; if "
            "this ever fails the permutation is no longer a relabelling and the "
            "arms differ in topology as well as in biology"
        )

    def test_named_genes_no_longer_share_the_same_units(self):
        """The statistic the null does destroy."""
        adj = make_frame()
        perm = permute_adjacency(adj, seed=1)
        real = adj.to_numpy() @ adj.to_numpy().T
        shuffled = perm.to_numpy() @ perm.to_numpy().T
        assert not np.array_equal(real, shuffled)
        # It is a symmetric relabelling of the same matrix, so the multiset of
        # co-membership counts survives while every named pair is reassigned.
        assert sorted(real.ravel()) == sorted(shuffled.ravel())

    def test_each_units_membership_is_reassigned(self):
        adj = make_frame()
        perm = permute_adjacency(adj, seed=1)
        moved = [c for c in adj.columns if not adj[c].equals(perm[c])]
        assert len(moved) == N_PATHWAYS, (
            "every unit should end up with a different gene set"
        )


class TestDeterminism:
    def test_same_seed_gives_the_same_map(self):
        adj = make_frame()
        assert permute_adjacency(adj, seed=5).equals(permute_adjacency(adj, seed=5))

    def test_different_seeds_give_different_maps(self):
        adj = make_frame()
        assert not permute_adjacency(adj, seed=5).equals(
            permute_adjacency(adj, seed=6))

    def test_the_input_is_not_mutated(self):
        adj = make_frame()
        before = adj.copy()
        permute_adjacency(adj, seed=1)
        pd.testing.assert_frame_equal(adj, before)


class TestInputTypes:
    def test_numpy_round_trips(self):
        adj = make_frame().to_numpy()
        perm = permute_adjacency(adj, seed=1)
        assert isinstance(perm, np.ndarray)
        assert perm.shape == adj.shape
        assert np.array_equal(perm.sum(axis=0), adj.sum(axis=0))

    def test_torch_round_trips_and_keeps_dtype(self):
        adj = torch.as_tensor(make_frame().to_numpy(), dtype=torch.float32)
        perm = permute_adjacency(adj, seed=1)
        assert isinstance(perm, torch.Tensor)
        assert perm.dtype == adj.dtype
        assert torch.equal(perm.sum(dim=0), adj.sum(dim=0))

    def test_all_three_types_agree_on_the_same_seed(self):
        frame = make_frame()
        from_frame = permute_adjacency(frame, seed=11).to_numpy()
        from_numpy = permute_adjacency(frame.to_numpy(), seed=11)
        from_torch = permute_adjacency(
            torch.as_tensor(frame.to_numpy()), seed=11).numpy()
        assert np.array_equal(from_frame, from_numpy)
        assert np.array_equal(from_frame, from_torch)


class TestUsableAsAModelMask:
    def test_a_permuted_map_builds_the_same_shaped_model(self):
        """The arms must differ in the adjacency and in nothing else."""
        from pyvae import InformedVAE

        adj = make_frame()
        real = torch.as_tensor(adj.to_numpy(), dtype=torch.float32)
        perm = torch.as_tensor(permute_adjacency(adj, seed=2).to_numpy(),
                               dtype=torch.float32)
        a = InformedVAE(adj=real, latent_dim=4, n_cov=3, likelihood="nb",
                        informed_decoder=True, seed=1)
        b = InformedVAE(adj=perm, latent_dim=4, n_cov=3, likelihood="nb",
                        informed_decoder=True, seed=1)
        assert {k: tuple(v.shape) for k, v in a.state_dict().items()} == {
            k: tuple(v.shape) for k, v in b.state_dict().items()}
        assert not torch.equal(a.encoder.informed.mask, b.encoder.informed.mask)
        assert torch.equal(a.encoder.informed.mask.sum(dim=1),
                           b.encoder.informed.mask.sum(dim=1)), (
            "per-unit live-weight counts must match, or the arms differ in "
            "capacity as well as in biology"
        )
