"""Metrics and experiment result serialization."""

from __future__ import annotations

import dataclasses
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import forgemind
from forgemind.search.engine import SearchResult


@dataclass(frozen=True)
class RunRecord:
    runner: str                 # "forgemind" | "baseline"
    problem_id: str
    success: bool
    accepted: bool
    test_pass_rate: float
    wall_clock_seconds: float
    agent_calls: int
    verifications: int
    candidates: int
    nodes_explored: int
    nodes_pruned: int
    duplicates: int
    backtracks: int
    stagnation_events: int
    max_depth: int
    token_estimate: int
    termination_reason: str
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    records: tuple[RunRecord, ...]

    @property
    def success_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.success) / len(self.records)

    @property
    def average_cost(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.wall_clock_seconds for r in self.records) / len(self.records)

    def cost_per_success(self) -> float:
        wins = [r for r in self.records if r.success]
        if not wins:
            return float("inf")
        return sum(r.wall_clock_seconds for r in wins) / len(wins)

    def compare(self, baseline: "BenchmarkResult") -> Mapping[str, float]:
        return {
            "delta_success": self.success_rate - baseline.success_rate,
            "cost_ratio": (
                self.average_cost / baseline.average_cost
                if baseline.average_cost > 0 else float("inf")
            ),
            "forgemind_success_rate": self.success_rate,
            "baseline_success_rate": baseline.success_rate,
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "forgemind_version": forgemind.__version__,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "records": [
                {**dataclasses.asdict(r), "extra": dict(r.extra)}
                for r in self.records
            ],
            "success_rate": self.success_rate,
            "average_cost_seconds": self.average_cost,
        }


def record_from_search(problem_id: str, result: SearchResult) -> RunRecord:
    node = result.accepted_node
    pass_rate = 0.0
    accepted = result.success
    if node is not None and node.verification is not None:
        pass_rate = node.verification.tests_pass_rate
    elif not result.graph.nodes:
        pass_rate = 0.0
    else:
        # best guidance among all verified nodes for partial-credit reporting
        rates = [n.verification.tests_pass_rate for n in result.graph.nodes.values()
                 if n.verification is not None]
        pass_rate = max(rates) if rates else 0.0
    m = result.metrics
    return RunRecord(
        runner="forgemind",
        problem_id=problem_id,
        success=result.success,
        accepted=result.success,
        test_pass_rate=pass_rate,
        wall_clock_seconds=result.elapsed_seconds,
        agent_calls=m.agent_calls,
        verifications=m.verifications,
        candidates=m.candidates_created,
        nodes_explored=m.nodes_explored,
        nodes_pruned=m.pruned,
        duplicates=m.duplicates,
        backtracks=m.backtracks,
        stagnation_events=m.stagnation_events,
        max_depth=m.max_depth_reached,
        token_estimate=m.token_estimate,
        termination_reason=result.termination.reason,
    )


def _raise_frozen_bench(self: object, name: str, value: object) -> None:
    raise TypeError(f"{type(self).__name__} is immutable ({name})")


for _cls in (RunRecord,):
    _cls.__setattr__ = _raise_frozen_bench  # type: ignore[method-assign]
