"""Benchmark-layer mock-kit registry (H5).

Maps a benchmark registry name to the MockKit configured for it. This
module lives in the benchmark layer on purpose: the generic engine never
imports it, and call sites (CLI, API, benchmark runner) resolve kits and
pass explicit agents into SearchEngine.

Keyed by registry name, NOT by spec.problem_id: ``stats`` and
``stats-trap`` share one ProblemSpec and differ only in builder variant
order, so the spec alone cannot select the kit.
"""

from __future__ import annotations

from forgemind.agents.mocks import MockKit


def stats_kit(*, correct_first: bool = True) -> MockKit:
    return MockKit(
        module_path="solution.py", module="solution",
        correct_first=correct_first,
    )


def mock_kit_for(problem_id: str) -> MockKit:
    """Return the mock kit for a benchmark registry problem id."""
    if problem_id == "stats":
        return stats_kit()
    if problem_id == "stats-trap":
        return stats_kit(correct_first=False)
    raise KeyError(f"no mock kit for problem {problem_id!r}")
