"""Scoring policy: guidance score always; acceptance score only on hard pass."""

from __future__ import annotations

import math

from forgemind.core.config import ScoringWeights
from forgemind.verification.results import ComponentScores, HardGateResult, VerificationResult


class ScoringPolicy:
    def __init__(self, weights: ScoringWeights) -> None:
        self._w = weights

    def score(
        self,
        verification: VerificationResult,
        gate: HardGateResult,
        *,
        depth: int,
        repeated_failures: int = 0,
        stagnated: bool = False,
    ) -> ComponentScores:
        w = self._w
        correctness = verification.tests_pass_rate
        # Deterministic soft components: measured wall-time/memory vary run to
        # run (OS jitter), so they are excluded from scoring in v0.1 and
        # recorded as raw evidence only. Neutral 0.5 keeps weights stable.
        performance = 0.5
        memory = 0.5
        complexity = max(0.0, 1.0 - depth / 10.0)
        maintainability = 0.5  # placeholder metric until quality analyzers exist
        search_cost = max(0.0, 1.0 - depth * 0.05)

        penalty = (
            min(repeated_failures, 4) / 4.0 * w.repeated_failure_penalty
            + (w.stagnation_penalty if stagnated else 0.0)
        )

        guidance = (
            w.test_pass_rate * correctness
            + w.performance * performance
            + w.memory * memory
            + w.complexity * complexity
            + w.depth_cost * search_cost
        ) / (
            w.test_pass_rate + w.performance + w.memory + w.complexity + w.depth_cost
        )
        guidance = max(0.0, guidance - penalty)

        total = guidance if gate.passed else 0.0
        return ComponentScores(
            hard_pass=gate.passed,
            correctness=correctness,
            performance=performance,
            memory=memory,
            complexity=complexity,
            maintainability=maintainability,
            search_cost=search_cost,
            guidance=guidance,
            total=total,
        )

    @staticmethod
    def _runtime_score(runtime_ms: float) -> float:
        if runtime_ms <= 0:
            return 0.5
        return math.exp(-runtime_ms / 5000.0)

    @staticmethod
    def _memory_score(peak_mb: float) -> float:
        if peak_mb <= 0:
            return 0.5
        return math.exp(-peak_mb / 512.0)
