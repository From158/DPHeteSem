"""Functional tests for the public release interface."""

import numpy as np
import pytest
import scipy.sparse as sp

from dphetesem import ReleaseConfig, release_semantic_views


def make_view(node_count: int, offset: int) -> sp.csr_matrix:
    rows = []
    cols = []
    for node in range(node_count):
        neighbor = (node + offset) % node_count
        if node != neighbor:
            rows.extend((node, neighbor))
            cols.extend((neighbor, node))
    data = np.ones(len(rows), dtype=np.float64)
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(node_count, node_count))
    matrix.data[:] = 1.0
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    return matrix


def test_release_is_deterministic_and_sparse() -> None:
    views = (make_view(24, 1), make_view(24, 3))
    config = ReleaseConfig(epsilon=2.0, num_blocks=4, candidate_nodes=4, seed=9)
    first = release_semantic_views(views, config)
    second = release_semantic_views(views, config)

    np.testing.assert_array_equal(first.partition, second.partition)
    assert np.isclose(first.accounting.composed_epsilon, config.epsilon)
    for left, right in zip(first.released_views, second.released_views):
        assert (left != right).nnz == 0
        assert left.shape == (24, 24)
        assert (left != left.T).nnz == 0
        assert np.count_nonzero(left.diagonal()) == 0
        assert set(np.unique(left.data)).issubset({1.0})


def test_zero_partition_fraction_uses_valid_random_partition() -> None:
    views = (make_view(18, 1),)
    config = ReleaseConfig(
        epsilon=1.0,
        num_blocks=3,
        partition_fraction=0.0,
        candidate_nodes=3,
        seed=5,
    )
    result = release_semantic_views(views, config)
    assert set(result.partition.tolist()) == {0, 1, 2}
    assert result.accounting.partition_epsilon == 0.0


def test_asymmetric_input_is_rejected() -> None:
    asymmetric = sp.csr_matrix(([1.0], ([0], [1])), shape=(4, 4))
    with pytest.raises(ValueError, match="symmetric"):
        release_semantic_views(
            (asymmetric,),
            ReleaseConfig(epsilon=1.0, num_blocks=2),
        )

def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="same shape"):
        release_semantic_views(
            (make_view(10, 1), make_view(12, 1)),
            ReleaseConfig(epsilon=1.0, num_blocks=2),
        )
