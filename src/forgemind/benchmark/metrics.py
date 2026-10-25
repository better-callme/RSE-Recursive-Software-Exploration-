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


@dataclass(frozen=True)
class SuiteResult:
    """One benchmark run across a named problem suite, both arms.

    ``per_problem`` rows pair the ForgeMind and baseline RunRecords for one
    registry problem id. Because two problems can share a ProblemSpec
    (``stats`` vs ``stats-trap``), each record's ``extra`` carries the
    registry name under ``"suite_problem"``.

    M8 determinism note: the suite is deterministic — the engine's
    seeded RNG is never consulted, so different seeds produce identical
    records. There is deliberately no seed-replication parameter: a
    parameter that doesn't influence execution is not an experimental
    variable. Replication returns when RSE genuinely has randomized
    search policies; until then the replication axis is problem
    diversity, and the primary cost comparison is the deterministic
    totals in aggregate(), not the wall-clock ratio.
    """

    problem_names: tuple[str, ...]
    forgemind: BenchmarkResult
    baseline: BenchmarkResult

    @property
    def per_problem(self) -> list[dict[str, Any]]:
        rows = []
        fm_by_problem = {r.extra.get("suite_problem"): r
                         for r in self.forgemind.records}
        base_by_problem = {r.extra.get("suite_problem"): r
                           for r in self.baseline.records}
        for name in self.problem_names:
            fm = fm_by_problem.get(name)
            base = base_by_problem.get(name)
            rows.append({
                "problem": name,
                "forgemind": ({**dataclasses.asdict(fm),
                               "extra": dict(fm.extra)} if fm else None),
                "baseline": ({**dataclasses.asdict(base),
                               "extra": dict(base.extra)} if base else None),
            })
        return rows

    def aggregate(self) -> Mapping[str, Any]:
        fm, base = self.forgemind, self.baseline
        fm_walls = [r.wall_clock_seconds for r in fm.records]
        base_walls = [r.wall_clock_seconds for r in base.records]
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0

        def total(records: tuple[RunRecord, ...], field: str) -> float:
            return float(sum(getattr(r, field) for r in records))

        # M8: deterministic cost totals are the primary comparison —
        # seed- and jitter-independent. The wall-clock ratio is kept as
        # an environment-sensitive observation, never a pass/fail
        # criterion; cost_ratio_noisy flags sub-second means, where
        # scheduler/cache/startup noise dominates.
        return {
            "problems": float(len(self.problem_names)),
            "forgemind_success_rate": fm.success_rate,
            "baseline_success_rate": base.success_rate,
            "delta_success": fm.success_rate - base.success_rate,
            "cost_ratio": (mean(fm_walls) / mean(base_walls)
                           if mean(base_walls) > 0 else float("inf")),
            "cost_ratio_noisy": bool(mean(fm_walls) < 1.0
                                     or mean(base_walls) < 1.0),
            "forgemind_agent_calls": total(fm.records, "agent_calls"),
            "baseline_agent_calls": total(base.records, "agent_calls"),
            "forgemind_verifications": total(fm.records, "verifications"),
            "baseline_verifications": total(base.records, "verifications"),
            "forgemind_total_tokens": total(fm.records, "token_estimate"),
            "baseline_total_tokens": total(base.records, "token_estimate"),
            "forgemind_nodes_created": total(fm.records, "candidates"),
            "baseline_nodes_created": total(base.records, "candidates"),
            "forgemind_nodes_explored": total(fm.records, "nodes_explored"),
            "baseline_nodes_explored": total(base.records, "nodes_explored"),
            "forgemind_nodes_pruned": total(fm.records, "nodes_pruned"),
            "baseline_nodes_pruned": total(base.records, "nodes_pruned"),
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "forgemind_version": forgemind.__version__,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "problem_names": list(self.problem_names),
            "per_problem": self.per_problem,
            "aggregate": self.aggregate(),
        }
