"""Core differentially private semantic-view release mechanism."""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .config import ReleaseConfig


@dataclass(frozen=True)
class PrivacyAccounting:
    """Sequential-composition record for one multi-view release."""

    total_epsilon: float
    partition_epsilon: float
    block_count_epsilons: Tuple[float, ...]
    degree_epsilons: Tuple[float, ...]

    @property
    def composed_epsilon(self) -> float:
        return (
            self.partition_epsilon
            + sum(self.block_count_epsilons)
            + sum(self.degree_epsilons)
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_epsilon": self.total_epsilon,
            "partition_epsilon": self.partition_epsilon,
            "block_count_epsilons": list(self.block_count_epsilons),
            "degree_epsilons": list(self.degree_epsilons),
            "composed_epsilon": self.composed_epsilon,
        }


@dataclass(frozen=True)
class ReleaseResult:
    """Released graphs and the DP quantities required for reproducibility."""

    released_views: Tuple[sp.csr_matrix, ...]
    partition: np.ndarray
    noisy_degrees: Tuple[np.ndarray, ...]
    calibrated_block_counts: Tuple[np.ndarray, ...]
    accounting: PrivacyAccounting
    config: ReleaseConfig


@dataclass(frozen=True)
class _PrivateStatistics:
    partition: np.ndarray
    noisy_degrees: Tuple[np.ndarray, ...]
    calibrated_block_counts: Tuple[np.ndarray, ...]
    accounting: PrivacyAccounting


def release_semantic_views(
    views: Sequence[sp.spmatrix], config: ReleaseConfig
) -> ReleaseResult:
    """Release node-aligned undirected semantic views.

    Original views are accessed only while producing the DP statistics. Graph
    instantiation receives the private statistics object and cannot query the
    original adjacency matrices.
    """

    config.validate()
    normalized = _validate_and_normalize_views(views)
    if config.num_blocks > normalized[0].shape[0]:
        raise ValueError("num_blocks cannot exceed the number of nodes")

    rng = np.random.default_rng(config.seed)
    private_stats = _release_private_statistics(normalized, config, rng)
    released = _instantiate_views(private_stats, config, rng)

    return ReleaseResult(
        released_views=tuple(released),
        partition=private_stats.partition.copy(),
        noisy_degrees=tuple(x.copy() for x in private_stats.noisy_degrees),
        calibrated_block_counts=tuple(
            x.copy() for x in private_stats.calibrated_block_counts
        ),
        accounting=private_stats.accounting,
        config=config,
    )


def _validate_and_normalize_views(
    views: Sequence[sp.spmatrix],
) -> Tuple[sp.csr_matrix, ...]:
    if not views:
        raise ValueError("at least one semantic view is required")

    expected_shape = views[0].shape
    normalized: List[sp.csr_matrix] = []
    for index, view in enumerate(views, start=1):
        if view.ndim != 2 or view.shape[0] != view.shape[1]:
            raise ValueError(f"view {index} must be square")
        if view.shape != expected_shape:
            raise ValueError("all views must have the same shape and node ordering")

        matrix = view.tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        if (matrix != matrix.T).nnz:
            raise ValueError(f"view {index} must be symmetric")

        matrix.setdiag(0)
        matrix.eliminate_zeros()
        matrix.data = np.ones(matrix.nnz, dtype=np.float64)
        normalized.append(matrix)

    return tuple(normalized)


def _allocate_budget(epsilon: float, partition_fraction: float, block_fraction: float, view_count: int) -> PrivacyAccounting:
    partition_epsilon = epsilon * partition_fraction
    per_view_epsilon = (epsilon - partition_epsilon) / view_count
    block_epsilon = per_view_epsilon * block_fraction
    degree_epsilon = per_view_epsilon - block_epsilon
    accounting = PrivacyAccounting(
        total_epsilon=epsilon,
        partition_epsilon=partition_epsilon,
        block_count_epsilons=tuple(block_epsilon for _ in range(view_count)),
        degree_epsilons=tuple(degree_epsilon for _ in range(view_count)),
    )
    if not np.isclose(accounting.composed_epsilon, epsilon, rtol=0.0, atol=1e-12):
        raise RuntimeError("privacy allocation does not compose to epsilon")
    return accounting


def _release_private_statistics(
    views: Sequence[sp.csr_matrix],
    config: ReleaseConfig,
    rng: np.random.Generator,
) -> _PrivateStatistics:
    accounting = _allocate_budget(
        config.epsilon,
        config.partition_fraction,
        config.block_fraction,
        len(views),
    )
    node_count = views[0].shape[0]
    partition = _release_partition(
        views,
        node_count,
        config.num_blocks,
        accounting.partition_epsilon,
        rng,
    )

    noisy_degrees: List[np.ndarray] = []
    calibrated_counts: List[np.ndarray] = []
    for view, count_epsilon, degree_epsilon in zip(
        views,
        accounting.block_count_epsilons,
        accounting.degree_epsilons,
    ):
        true_counts = _block_edge_counts(view, partition, config.num_blocks)
        private_counts = true_counts + rng.laplace(
            loc=0.0,
            scale=1.0 / count_epsilon,
            size=true_counts.shape,
        )
        private_counts = np.maximum(private_counts, 0.0)

        true_degrees = np.asarray(view.sum(axis=1)).reshape(-1)
        private_degrees = true_degrees + rng.laplace(
            loc=0.0,
            scale=2.0 / degree_epsilon,
            size=true_degrees.shape,
        )
        private_degrees = np.maximum(private_degrees, 0.0)

        calibrated = _calibrate_counts(
            private_counts,
            private_degrees,
            partition,
            config.num_blocks,
        )
        noisy_degrees.append(private_degrees)
        calibrated_counts.append(calibrated)

    return _PrivateStatistics(
        partition=partition,
        noisy_degrees=tuple(noisy_degrees),
        calibrated_block_counts=tuple(calibrated_counts),
        accounting=accounting,
    )


def _release_partition(
    views: Sequence[sp.csr_matrix],
    node_count: int,
    block_count: int,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if epsilon <= 0.0:
        labels = np.arange(node_count, dtype=np.int32) % block_count
        rng.shuffle(labels)
        return labels

    aggregate_degree = np.zeros(node_count, dtype=np.float64)
    for view in views:
        aggregate_degree += np.asarray(view.sum(axis=1)).reshape(-1)

    noisy_sketch = aggregate_degree + rng.laplace(
        loc=0.0,
        scale=2.0 / epsilon,
        size=node_count,
    )
    order = np.argsort(noisy_sketch, kind="mergesort")
    labels = np.empty(node_count, dtype=np.int32)
    for rank, node in enumerate(order):
        labels[node] = min(block_count - 1, rank * block_count // node_count)
    return labels


def _block_edge_counts(
    view: sp.csr_matrix, partition: np.ndarray, block_count: int
) -> np.ndarray:
    upper = sp.triu(view, k=1, format="coo")
    left = partition[upper.row]
    right = partition[upper.col]
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    counts = np.zeros((block_count, block_count), dtype=np.float64)
    np.add.at(counts, (low, high), 1.0)
    return counts


def _calibrate_counts(
    noisy_counts: np.ndarray,
    noisy_degrees: np.ndarray,
    partition: np.ndarray,
    block_count: int,
) -> np.ndarray:
    upper = np.triu(noisy_counts)
    degree_mass = float(noisy_degrees.sum())
    edge_mass = float(upper.sum())
    ratio = min(1.0, degree_mass / (2.0 * edge_mass + 1e-12))
    calibrated = np.rint(upper * ratio).astype(np.int64)

    sizes = np.bincount(partition, minlength=block_count)
    for left in range(block_count):
        for right in range(left, block_count):
            if left == right:
                capacity = sizes[left] * (sizes[left] - 1) // 2
            else:
                capacity = sizes[left] * sizes[right]
            calibrated[left, right] = min(calibrated[left, right], capacity)
    return calibrated


def _instantiate_views(
    private_stats: _PrivateStatistics,
    config: ReleaseConfig,
    rng: np.random.Generator,
) -> Tuple[sp.csr_matrix, ...]:
    aggregate_degree = np.sum(np.vstack(private_stats.noisy_degrees), axis=0)
    released: List[sp.csr_matrix] = []
    for view_degree, block_counts in zip(
        private_stats.noisy_degrees,
        private_stats.calibrated_block_counts,
    ):
        released.append(
            _instantiate_one_view(
                private_stats.partition,
                view_degree,
                aggregate_degree,
                block_counts,
                config,
                rng,
            )
        )
    return tuple(released)


def _instantiate_one_view(
    partition: np.ndarray,
    view_degree: np.ndarray,
    aggregate_degree: np.ndarray,
    block_counts: np.ndarray,
    config: ReleaseConfig,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    node_count = partition.size
    block_nodes = [
        np.flatnonzero(partition == block).astype(np.int64)
        for block in range(config.num_blocks)
    ]
    row_chunks: List[np.ndarray] = []
    col_chunks: List[np.ndarray] = []

    for left in range(config.num_blocks):
        for right in range(left, config.num_blocks):
            target_edges = int(block_counts[left, right])
            if target_edges <= 0:
                continue
            candidates = _candidate_pool(
                block_nodes[left],
                block_nodes[right],
                view_degree,
                aggregate_degree,
                target_edges,
                config.candidate_nodes,
                same_block=left == right,
            )
            if candidates.size == 0:
                continue

            first = candidates[:, 0]
            second = candidates[:, 1]
            weights = (
                (1.0 - config.coupling)
                * view_degree[first]
                * view_degree[second]
                + config.coupling
                * aggregate_degree[first]
                * aggregate_degree[second]
            )
            weights = np.maximum(weights, 0.0) + 1e-12
            probabilities = weights / weights.sum()
            sample_size = min(target_edges, candidates.shape[0])
            selected = rng.choice(
                candidates.shape[0],
                size=sample_size,
                replace=False,
                p=probabilities,
            )
            edges = candidates[selected]
            row_chunks.extend((edges[:, 0], edges[:, 1]))
            col_chunks.extend((edges[:, 1], edges[:, 0]))

    if not row_chunks:
        return sp.csr_matrix((node_count, node_count), dtype=np.float32)

    rows = np.concatenate(row_chunks)
    cols = np.concatenate(col_chunks)
    data = np.ones(rows.size, dtype=np.float32)
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(node_count, node_count))
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    return matrix


def _candidate_pool(
    left_nodes: np.ndarray,
    right_nodes: np.ndarray,
    view_degree: np.ndarray,
    aggregate_degree: np.ndarray,
    target_edges: int,
    initial_limit: int,
    same_block: bool,
) -> np.ndarray:
    if left_nodes.size == 0 or right_nodes.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    limit = min(max(1, initial_limit), max(left_nodes.size, right_nodes.size))
    while True:
        top_left = _top_nodes(left_nodes, view_degree, aggregate_degree, limit)
        top_right = _top_nodes(right_nodes, view_degree, aggregate_degree, limit)
        pairs = set()

        for first in left_nodes:
            for second in top_right:
                _add_candidate(pairs, int(first), int(second), same_block)
        for first in top_left:
            for second in right_nodes:
                _add_candidate(pairs, int(first), int(second), same_block)

        if len(pairs) >= target_edges:
            break
        if limit >= max(left_nodes.size, right_nodes.size):
            break
        limit = min(max(left_nodes.size, right_nodes.size), limit * 2)

    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(sorted(pairs), dtype=np.int64)


def _top_nodes(
    nodes: np.ndarray,
    view_degree: np.ndarray,
    aggregate_degree: np.ndarray,
    limit: int,
) -> np.ndarray:
    count = min(limit, nodes.size)
    if count == nodes.size:
        return nodes
    scores = view_degree[nodes] + aggregate_degree[nodes]
    selected = np.argpartition(scores, -count)[-count:]
    return nodes[selected]


def _add_candidate(pairs: set, first: int, second: int, same_block: bool) -> None:
    if first == second:
        return
    if same_block or first > second:
        first, second = min(first, second), max(first, second)
    pairs.add((first, second))
