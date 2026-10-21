"""H2: acceptance test sources are part of the spec's hashed identity."""

import pytest

from forgemind.benchmark.problems import problem_spec
from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.problem import AcceptanceTest, ProblemSpec


def _spec(**overrides) -> ProblemSpec:
    kw = dict(
        problem_id="p1",
        description="d",
        requirements=("r",),
        acceptance_tests=(AcceptanceTest("t1", "test_t1"),),
    )
    kw.update(overrides)
    return ProblemSpec(**kw)


def test_spec_hash_covers_test_sources():
    a = _spec(test_sources={"t.py": "def test_t1():\n    pass\n"})
    b = _spec(test_sources={"t.py": "def test_t1():\n    assert False\n"})
    assert a.spec_hash != b.spec_hash


def test_spec_hash_changes_when_sources_added():
    assert _spec().spec_hash != _spec(
        test_sources={"t.py": "def test_t1():\n    pass\n"}).spec_hash


def test_spec_hash_test_sources_order_independent():
    a = _spec(test_sources={"a.py": "x", "b.py": "y"})
    b = _spec(test_sources={"b.py": "y", "a.py": "x"})
    assert a.spec_hash == b.spec_hash


def test_problem_spec_carries_sources_without_smuggling():
    spec = problem_spec("stats")
    assert spec.test_sources  # sources are IN the spec
    assert "__fm_acceptance_tests.py" in spec.test_sources
    # The old mechanism punched a hidden attribute through the freeze.
    assert not hasattr(spec, "_test_provider_ref")


def test_stats_problem_test_files_match_spec_sources():
    problem = StatsProblem()
    assert problem.test_files() == dict(problem.spec.test_sources)
