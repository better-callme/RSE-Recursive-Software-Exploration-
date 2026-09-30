"""ForgeMind configuration: every experimental constant lives here."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Mapping


@dataclasses.dataclass(frozen=True)
class ScoringWeights:
    """Weights for soft ranking. Hard gates are never weighted."""

    test_pass_rate: float = 1.0
    performance: float = 0.2
    memory: float = 0.1
    complexity: float = 0.15
    depth_cost: float = 0.05
    repeated_failure_penalty: float = 0.25
    stagnation_penalty: float = 0.3


@dataclasses.dataclass(frozen=True)
class ResourceLimits:
    wall_timeout_seconds: float = 10.0
    cpu_limit_seconds: int = 8
    memory_limit_mb: int = 512
    process_limit: int = 64
    max_output_bytes: int = 1_000_000


@dataclass(frozen=True)
class ForgeMindConfig:
    """Immutable engine configuration."""

    random_seed: int = 0

    # Search shape
    beam_width: int = 3
    branching_factor: int = 3
    max_depth: int = 6
    max_nodes: int = 200
    max_agent_calls: int = 100
    max_verifications: int = 100
    max_wall_clock_seconds: float = 300.0

    # Sandbox resources
    resources: ResourceLimits = field(default_factory=ResourceLimits)

    # Stagnation
    stagnation_window: int = 4
    stagnation_epsilon: float = 1e-3

    # Scoring
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)

    # Recovery
    max_repair_attempts_per_node: int = 2

    # Behaviour switches
    enable_optimizer: bool = True
    enable_adversarial_tier: bool = True
    reverify_duplicates: bool = False  # explicit opt-in for redundant verification

    logging_verbose: bool = False

    def __post_init__(self) -> None:
        if self.beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.branching_factor < 1:
            raise ValueError("branching_factor must be >= 1")
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")

    def as_dict(self) -> Mapping[str, object]:
        """Deterministic serialization of the full configuration."""
        def conv(value: object) -> object:
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                return {f.name: conv(getattr(value, f.name)) for f in fields(value)}
            return value

        return MappingProxyType(
            {f.name: conv(getattr(self, f.name)) for f in fields(self)}
        )
