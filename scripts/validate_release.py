"""Validate sparse graph invariants and privacy accounting in a release folder."""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    args = parser.parse_args()

    metadata_path = args.release_dir / "release_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    accounting = metadata["privacy_accounting"]
    if not np.isclose(
        accounting["total_epsilon"],
        accounting["composed_epsilon"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("composed epsilon does not match total epsilon")

    expected_nodes = int(metadata["node_count"])
    for index in range(1, int(metadata["view_count"]) + 1):
        path = args.release_dir / f"released_view_{index}.npz"
        matrix = sp.load_npz(path).tocsr()
        if matrix.shape != (expected_nodes, expected_nodes):
            raise ValueError(f"unexpected shape in {path.name}: {matrix.shape}")
        if (matrix != matrix.T).nnz:
            raise ValueError(f"released view is asymmetric: {path.name}")
        if np.count_nonzero(matrix.diagonal()):
            raise ValueError(f"released view has self-loops: {path.name}")
        if matrix.nnz and not np.all(matrix.data == 1):
            raise ValueError(f"released view is not binary: {path.name}")

    print(
        f"Validated {metadata['view_count']} released views over "
        f"{expected_nodes} nodes; composed epsilon="
        f"{accounting['composed_epsilon']:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
