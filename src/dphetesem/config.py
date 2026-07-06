"""Configuration objects for semantic-view release."""

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class ReleaseConfig:
    """Parameters controlling privacy allocation and graph synthesis.

    The total budget is split into one shared partition release and two
    releases per semantic view: block counts and node degrees.
    """

    epsilon: float = 1.0
    num_blocks: int = 8
    partition_fraction: float = 0.2
    block_fraction: float = 0.45
    coupling: float = 0.5
    candidate_nodes: int = 64
    seed: int = 2024

    def validate(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be at least one")
        if not 0.0 <= self.partition_fraction < 1.0:
            raise ValueError("partition_fraction must be in [0, 1)")
        if not 0.0 < self.block_fraction < 1.0:
            raise ValueError("block_fraction must be in (0, 1)")
        if not 0.0 <= self.coupling <= 1.0:
            raise ValueError("coupling must be in [0, 1]")
        if self.candidate_nodes < 1:
            raise ValueError("candidate_nodes must be positive")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)
