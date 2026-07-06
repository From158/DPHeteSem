"""Create two small node-aligned semantic views for the quick-start example."""

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def sample_view(node_count: int, groups: np.ndarray, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    rows = []
    cols = []
    for first in range(node_count):
        for second in range(first + 1, node_count):
            probability = 0.20 if groups[first] == groups[second] else 0.025
            if rng.random() < probability:
                rows.extend((first, second))
                cols.extend((second, first))
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(node_count, node_count))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    groups = np.arange(args.nodes) % 4
    first = sample_view(args.nodes, groups, args.seed)
    second = sample_view(args.nodes, np.roll(groups, 7), args.seed + 1)
    sp.save_npz(args.output / "view_1.npz", first)
    sp.save_npz(args.output / "view_2.npz", second)
    print(f"Wrote demo views to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
