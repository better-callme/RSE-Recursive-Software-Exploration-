"""Verification layer tests: real subprocess execution."""

import pytest

from forgemind.core.proposal import CodePatchProposal
from forgemind.core.state import CodebaseState
from forgemind.verification.referee import hard_gate
from forgemind.verification.scoring import ScoringPolicy
from forgemind.core.config import ScoringWeights
from forgemind.verification.results import HardGateResult, VerificationResult

GOOD = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t / len(data)}\n"
)

BAD_MEAN = GOOD.replace("t / len(data)", "t // len(data)")
NO_EMPTY = "\n".join(l for l in GOOD.splitlines() if "if not data" not in l
                     and "'mean': None}" not in l) + "\n"


def state_with(code: str) -> CodebaseState:
    return CodebaseState(spec_hash="a" * 64, files={"solution.py": code})


def test_correct_code_passes(referee):
    v = referee.verify(state_with(GOOD))
    assert v.static_gate_passed and v.acceptance_passed
    assert v.tests_failed == 0


def test_incorrect_mean_fails(referee):
    v = referee.verify(state_with(BAD_MEAN))
    assert not v.acceptance_passed
    assert v.failure_type == "TEST_FAILURE"
    assert v.tests_failed > 0


def test_syntax_error_fails_static(referee):
    v = referee.verify(state_with("def broken(:\n"))
    assert not v.static_gate_passed
    assert v.failure_type == "SYNTAX_ERROR"


def test_runtime_exception_fails(referee):
    v = referee.verify(state_with(NO_EMPTY))  # raises on empty input
    assert not v.acceptance_passed
    assert any("test_empty" in d for d in v.violations)


def test_forbidden_constructs_flagged(referee):
    evil = GOOD + "\ndef helper():\n    eval('1')\n"
    v = referee.verify(state_with(evil))
    # eval is caught by static tier (defense-in-depth)
    assert not v.static_gate_passed or any("forbidden call" in x for x in v.violations)


def test_hard_gate_requires_all_gates():
    ok = VerificationResult(static_gate_passed=True, acceptance_passed=True,
                            regression_passed=True, tests_total=6, tests_passed=6)
    gate = hard_gate(ok)
    assert gate.passed

    failing_tests = VerificationResult(static_gate_passed=True,
                                       acceptance_passed=False,
                                       regression_passed=True,
                                       tests_total=6, tests_passed=5)
    gate = hard_gate(failing_tests)
    assert not gate.passed
    assert "requiredTests" in gate.failed_gates()


def test_partial_failure_cannot_be_accepted():
    """90% pass rate must still fail the hard gate."""
    partial = VerificationResult(static_gate_passed=True, acceptance_passed=False,
                                 regression_passed=True,
                                 tests_total=10, tests_passed=9)
    gate = hard_gate(partial)
    assert not gate.passed
    scorer = ScoringPolicy(ScoringWeights())
    scores = scorer.score(partial, gate, depth=1)
    # Guidance reflects progress; acceptance total stays zero.
    assert scores.guidance > 0.0
    assert scores.total == 0.0
    assert not scores.hard_pass


def test_scoring_prefers_higher_pass_rate_as_guidance():
    scorer = ScoringPolicy(ScoringWeights())
    gate_fail = HardGateResult(gates={"requiredTests": False})
    low = scorer.score(
        VerificationResult(static_gate_passed=True, tests_total=100, tests_passed=2),
        gate_fail, depth=1)
    high = scorer.score(
        VerificationResult(static_gate_passed=True, tests_total=100, tests_passed=99),
        gate_fail, depth=1)
    assert high.guidance > low.guidance
