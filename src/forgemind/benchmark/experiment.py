"""ExperimentRunner: runs ForgeMind and baseline over problem sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from forgemind.benchmark.baseline import run_baseline
from forgemind.benchmark.metrics import (
    BenchmarkResult,
    RunRecord,
    record_from_search,
)
from forgemind.core.config import ForgeMindConfig
from forgemind.core.history import History
from forgemind.core.problem import ProblemSpec
from forgemind.recovery.failure import RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee


class ExperimentRunner:
    def __init__(self, config: ForgeMindConfig | None = None) -> None:
        self.config = config or ForgeMindConfig()
        self.env = SubprocessExecutionEnvironment(self.config.resources)
        self._test_provider = None

    def _referee(self, spec: ProblemSpec) -> Referee:
        referee = Referee(spec, self.env, self.config)
        # Wire test sources for any spec exposing test_files (e.g. StatsProblem).
        provider = getattr(spec, "test_files", None) or self._test_provider
        if callable(provider):
            referee.attach_test_provider(provider)
        return referee

    def run_forgemind(self, problems: Sequence[ProblemSpec]) -> BenchmarkResult:
        records: list[RunRecord] = []
        for spec in problems:
            engine = SearchEngine(
                spec, self.config,
                referee=self._referee(spec),
                recovery=RecoveryPolicy(),
                history=History(),
            )
            result = engine.run()
            records.append(record_from_search(spec.problem_id, result))
        return BenchmarkResult(records=tuple(records))

    def run_baseline(self, problems: Sequence[ProblemSpec]) -> BenchmarkResult:
        records: list[RunRecord] = []
        for spec in problems:
            outcome = run_baseline(spec, self._referee(spec), self.config)
            pass_rate = 0.0
            if outcome.node is not None and outcome.node.verification is not None:
                pass_rate = outcome.node.verification.tests_pass_rate
            records.append(RunRecord(
                runner="baseline",
                problem_id=spec.problem_id,
                success=outcome.success,
                accepted=outcome.success,
                test_pass_rate=pass_rate,
                wall_clock_seconds=outcome.elapsed_seconds,
                agent_calls=1,
                verifications=1,
                candidates=1,
                nodes_explored=1,
                nodes_pruned=0,
                duplicates=0,
                backtracks=0,
                stagnation_events=0,
                max_depth=1,
                token_estimate=0,
                termination_reason=outcome.termination.reason,
            ))
        return BenchmarkResult(records=tuple(records))

    @staticmethod
    def save_report(result: BenchmarkResult, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(result.to_json_dict(), fh, indent=2, sort_keys=True)
        return out
