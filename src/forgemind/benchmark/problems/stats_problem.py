"""The first benchmark problem: descriptive statistics.

compute_stats(data) -> dict(count, sum, min, max, mean)
Edge cases: empty list, single value, negatives, duplicates, large values.
"""

from __future__ import annotations

from dataclasses import dataclass

from forgemind.core.problem import AcceptanceTest, ProblemSpec

_ACCEPTANCE_SOURCE = '''
def _check(result):
    assert isinstance(result, dict), type(result)
    for key in ("count", "sum", "min", "max", "mean"):
        assert key in result, f"missing {key}"


def test_normal():
    from solution import compute_stats
    r = compute_stats([1, 2, 3, 4])
    _check(r)
    assert r["count"] == 4 and r["sum"] == 10 and r["min"] == 1
    assert r["max"] == 4 and abs(r["mean"] - 2.5) < 1e-9


def test_empty():
    from solution import compute_stats
    r = compute_stats([])
    _check(r)
    assert r["count"] == 0 and r["sum"] == 0


def test_single():
    from solution import compute_stats
    r = compute_stats([7])
    assert r["count"] == 1 and r["mean"] == 7 and r["min"] == r["max"] == 7


def test_negative():
    from solution import compute_stats
    r = compute_stats([-5, -1, -9])
    assert r["min"] == -9 and r["max"] == -1 and r["sum"] == -15
    assert abs(r["mean"] + 5.0) < 1e-9


def test_duplicates():
    from solution import compute_stats
    r = compute_stats([3, 3, 3])
    assert r["count"] == 3 and r["min"] == r["max"] == r["mean"] == 3


def test_large_values():
    from solution import compute_stats
    big = 10**12
    r = compute_stats([big, -big, big])
    assert r["sum"] == big and r["max"] == big and r["min"] == -big
'''


@dataclass(frozen=True)
class StatsProblem:
    """ProblemSpec + its acceptance-test sources (Referee materializes them)."""

    spec: ProblemSpec = ProblemSpec(
        problem_id="stats-v1",
        description=(
            "Implement compute_stats(data) returning descriptive statistics: "
            "count, sum, min, max, mean. Must handle empty lists explicitly."
        ),
        requirements=(
            "function named compute_stats accepting a list of integers",
            "return dict with keys count,sum,min,max,mean",
            "empty list returns count=0 without raising",
            "mean must be true division (float) when count > 0",
        ),
        acceptance_tests=(
            AcceptanceTest("normal_list", "test_normal"),
            AcceptanceTest("empty_list", "test_empty"),
            AcceptanceTest("single_value", "test_single"),
            AcceptanceTest("negative_numbers", "test_negative"),
            AcceptanceTest("duplicates", "test_duplicates"),
            AcceptanceTest("large_values", "test_large_values"),
        ),
        interface_constraints={"compute_stats": "function: compute_stats(data)"},
        performance_constraints={"max_runtime_ms": 2000.0},
        dependency_constraints=(),
        language="python",
        runtime="python3",
    )

    def test_files(self) -> dict[str, str]:
        return {"__fm_acceptance_tests.py": _ACCEPTANCE_SOURCE}
