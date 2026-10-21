"""ExperimentRunner: runs ForgeMind and baseline over problem sets."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Sequence

from forgemind.agents.mocks import MockBuilder, MockCritic
from forgemind.benchmark.baseline import run_baseline
from forgemind.benchmark.metrics import (
    BenchmarkResult,
    RunRecord,
    SuiteResult,
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
        # Test sources normally come from spec.test_sources (H2); a
        # runner-level provider, when set, is an explicit override.
        if callable(self._test_provider):
            referee.attach_test_provider(self._test_provider)
        return referee

    def run_forgemind(
        self,
        problems: Sequence[ProblemSpec],
        builders: Sequence[MockBuilder | None] | None = None,
    ) -> BenchmarkResult:
        records: list[RunRecord] = []
        for i, spec in enumerate(problems):
            builder = builders[i] if builders is not None else None
            engine = SearchEngine(
                spec, self.config,
                referee=self._referee(spec),
                builder=builder,
                recovery=RecoveryPolicy(),
                history=History(),
            )
            result = engine.run()
            records.append(record_from_search(spec.problem_id, result))
        return BenchmarkResult(records=tuple(records))

    def run_baseline(
        self,
        problems: Sequence[ProblemSpec],
        builders: Sequence[MockBuilder | None] | None = None,
        critic: MockCritic | None = None,
    ) -> BenchmarkResult:
        records: list[RunRecord] = []
        for i, spec in enumerate(problems):
            builder = builders[i] if builders is not None else None
            outcome = run_baseline(spec, self._referee(spec), self.config,
                                   builder=builder, critic=critic)
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

    def run_suite(self, problem_names: Sequence[str]) -> SuiteResult:
        """Run both arms over the named registry problems.

        Each problem resolves to its ProblemSpec *and* its configured mock
        builder (e.g. ``stats-trap`` uses a wrong-first builder). Records are
        tagged with the registry name in ``extra["suite_problem"]`` so arms
        sharing a ProblemSpec stay distinguishable.
        """
        from forgemind.benchmark.problems import problem_builder, problem_spec

        names = list(problem_names)
        specs = [problem_spec(n) for n in names]
        builders: list[MockBuilder] = [problem_builder(n) for n in names]

        fm_records = []
        for name, spec, builder in zip(names, specs, builders):
            engine = SearchEngine(
                spec, self.config,
                referee=self._referee(spec),
                builder=builder,
                recovery=RecoveryPolicy(),
                history=History(),
            )
            result = engine.run()
            fm_records.append(dataclasses.replace(
                record_from_search(spec.problem_id, result),
                extra={"suite_problem": name},
            ))

        base_records = []
        for name, spec, builder in zip(names, specs, builders):
            outcome = run_baseline(spec, self._referee(spec), self.config,
                                   builder=builder)
            pass_rate = 0.0
            if outcome.node is not None and outcome.node.verification is not None:
                pass_rate = outcome.node.verification.tests_pass_rate
            base_records.append(RunRecord(
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
                extra={"suite_problem": name},
            ))

        return SuiteResult(
            problem_names=tuple(names),
            forgemind=BenchmarkResult(records=tuple(fm_records)),
            baseline=BenchmarkResult(records=tuple(base_records)),
        )

    @staticmethod
    def save_report(result: BenchmarkResult, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(result.to_json_dict(), fh, indent=2, sort_keys=True)
        return out

    @staticmethod
    def save_suite_report(result: SuiteResult, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(result.to_json_dict(), fh, indent=2, sort_keys=True)
        return out
