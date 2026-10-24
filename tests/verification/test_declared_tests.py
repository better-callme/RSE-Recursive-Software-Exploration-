"""M2: every test declared by the spec must actually execute.

Declared ⊆ executed, checked by the referee from the harness's
structured evidence — never inferred from counts or output text.
"""

import pytest

from forgemind.core.config import ForgeMindConfig
from forgemind.core.problem import AcceptanceTest, ProblemSpec
from forgemind.core.state import CodebaseState
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee, hard_gate


def _spec(test_ids, source):
    return ProblemSpec(
        problem_id="m2-probe",
        description="d",
        requirements=("r",),
        acceptance_tests=tuple(
            AcceptanceTest(f"case_{tid}", tid) for tid in test_ids),
        interface_constraints={"compute_stats": "f(data)"},
        test_sources={"__fm_acceptance_tests.py": source},
    )


def _state():
    return CodebaseState(
        spec_hash="a" * 64,
        files={"solution.py": "def compute_stats(data):\n    return data\n"})


def _referee(spec):
    config = ForgeMindConfig()
    return Referee(spec, SubprocessExecutionEnvironment(config.resources),
                   config)


def test_missing_declared_test_fails():
    """The core gap: a declared test the source omits must fail."""
    spec = _spec(("test_a", "test_b"), "def test_a():\n    assert True\n")
    v = _referee(spec).verify(_state())
    assert v.tests_total == 1  # counts stay truthful about what ran
    assert not v.acceptance_passed
    assert "Missing declared acceptance tests: test_b" in v.violations
    gate = hard_gate(v)
    assert not gate.passed
    assert "requiredTests" in gate.failed_gates()


def test_extra_passing_test_allowed():
    """Superset, not equality: extras that pass are harmless."""
    spec = _spec(("test_a",),
                 "def test_a():\n    assert True\n"
                 "def test_extra():\n    assert True\n")
    v = _referee(spec).verify(_state())
    assert v.acceptance_passed
    assert v.tests_total == 2


def test_extra_failing_test_fails():
    """Extras were always policed by failure counting; guard that."""
    spec = _spec(("test_a",),
                 "def test_a():\n    assert True\n"
                 "def test_extra():\n    assert False\n")
    v = _referee(spec).verify(_state())
    assert not v.acceptance_passed


def test_all_declared_executed_passes():
    spec = _spec(("test_a", "test_b"),
                 "def test_a():\n    assert True\n"
                 "def test_b():\n    assert True\n")
    v = _referee(spec).verify(_state())
    assert v.acceptance_passed


def test_missing_executed_evidence_fails_closed(monkeypatch):
    """Evidence without the executed field: missing = all declared."""
    spec = _spec(("test_a",), "def test_a():\n    assert True\n")
    referee = _referee(spec)
    monkeypatch.setattr(
        Referee, "_read_evidence",
        staticmethod(lambda result_path, nonce: {
            "acceptance": {"total": 1, "passed": 1, "failed": 0,
                           "details": []}}))
    v = referee.verify(_state())
    assert not v.acceptance_passed
    assert "Missing declared acceptance tests: test_a" in v.violations


def test_unimportable_test_module_fails():
    """A test module that fails to import collects nothing, so its
    declared tests never execute — acceptance must fail."""
    spec = _spec(("test_a",),
                 "import definitely_missing_module_xyz\n"
                 "def test_a():\n    assert True\n")
    v = _referee(spec).verify(_state())
    assert not v.acceptance_passed
    assert v.failure_type == "IMPORT_ERROR"
