"""Benchmark problem registry.

Each entry maps a CLI/API problem id to (problem_factory, builder_factory).

``stats`` is the friendly case: the mock builder emits the correct variant
first, so a single proposal can succeed.

``stats-trap`` is the *deceptive* case: the same ProblemSpec, but the mock
builder emits wrong variants first (mean truncation, then bad empty-list
handling). The baseline arm takes variant 0 and fails; the ForgeMind arm
must use beam search, failure recording and repair to converge. This is the
experimental manipulation that lets the benchmark discriminate between the
arms — with deterministic mocks, no other source of variation exists.
"""

from __future__ import annotations

from typing import Callable

from forgemind.agents.mocks import MockBuilder
from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.problem import ProblemSpec


ProblemFactory = Callable[[], object]
BuilderFactory = Callable[[], MockBuilder]

PROBLEMS: dict[str, tuple[ProblemFactory, BuilderFactory]] = {
    "stats": (StatsProblem, MockBuilder),
    "stats-trap": (StatsProblem, lambda: MockBuilder(correct_first=False)),
}


def list_problem_names() -> list[str]:
    return sorted(PROBLEMS)


def problem_spec(name: str) -> ProblemSpec:
    """Build the ProblemSpec for a registered problem id (KeyError if unknown)."""
    problem = PROBLEMS[name][0]()
    spec = problem.spec
    if hasattr(problem, "test_files"):
        object.__setattr__(spec, "_test_provider_ref", problem.test_files)
    return spec


def problem_builder(name: str) -> MockBuilder:
    """Build the mock builder configured for a registered problem id."""
    return PROBLEMS[name][1]()
