"""Command-line interface for DPHeteSem."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .config import ReleaseConfig
from .io import load_sparse_views, save_release
from .release import release_semantic_views


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dphetesem",
        description="Release node-aligned semantic graph views with differential privacy.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    release_parser = subparsers.add_parser("release", help="release sparse semantic views")
    release_parser.add_argument("--views", nargs="+", type=Path, required=True)
    release_parser.add_argument("--output", type=Path, required=True)
    release_parser.add_argument("--epsilon", type=float, default=1.0)
    release_parser.add_argument("--blocks", type=int, default=8)
    release_parser.add_argument("--partition-fraction", type=float, default=0.2)
    release_parser.add_argument("--block-fraction", type=float, default=0.45)
    release_parser.add_argument("--coupling", type=float, default=0.5)
    release_parser.add_argument("--candidate-nodes", type=int, default=64)
    release_parser.add_argument("--seed", type=int, default=2024)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "release":
        raise RuntimeError(f"unsupported command: {args.command}")

    config = ReleaseConfig(
        epsilon=args.epsilon,
        num_blocks=args.blocks,
        partition_fraction=args.partition_fraction,
        block_fraction=args.block_fraction,
        coupling=args.coupling,
        candidate_nodes=args.candidate_nodes,
        seed=args.seed,
    )
    views = load_sparse_views(args.views)
    result = release_semantic_views(views, config)
    save_release(result, args.output)
    print(
        f"Released {len(result.released_views)} views over "
        f"{result.partition.size} nodes to {args.output}"
    )
    print(
        "Composed epsilon: "
        f"{result.accounting.composed_epsilon:.12g}"
    )
    return 0
