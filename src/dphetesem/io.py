"""Sparse input and release-artifact output utilities."""

import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import scipy.sparse as sp

from .release import ReleaseResult


def load_sparse_views(paths: Iterable[Path]) -> Tuple[sp.csr_matrix, ...]:
    """Load SciPy sparse matrices without changing their node order."""

    return tuple(sp.load_npz(str(path)).tocsr() for path in paths)


def save_release(result: ReleaseResult, output_dir: Path) -> None:
    """Write released views, shared partition, and accounting metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, view in enumerate(result.released_views, start=1):
        sp.save_npz(output_dir / f"released_view_{index}.npz", view)
    np.save(output_dir / "partition.npy", result.partition)

    metadata = {
        "format_version": 1,
        "config": result.config.to_dict(),
        "privacy_accounting": result.accounting.to_dict(),
        "node_count": int(result.partition.size),
        "view_count": len(result.released_views),
        "released_undirected_edges": [
            int(view.nnz // 2) for view in result.released_views
        ],
    }
    (output_dir / "release_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
