"""C3 tests: the hard gate must honor the adversarial tier whenever
adversarial evidence was expected, and all_tiers_passed must mean it."""

from forgemind.verification.referee import hard_gate
from forgemind.verification.results import VerificationResult


def _result(**kw) -> VerificationResult:
    base = dict(
        static_gate_passed=True,
        acceptance_passed=True,
        adversarial_passed=True,
        regression_passed=True,
        tests_total=6,
        tests_passed=6,
        # M1: fixtures must state their structured verdicts explicitly —
        # the fail-closed defaults (False) would otherwise flip these.
        contract_passed=True,
        static_analysis_complete=True,
    )
    base.update(kw)
    return VerificationResult(**base)


def test_hard_gate_enforces_adversarial_when_expected():
    gate = hard_gate(_result(adversarial_passed=False),
                     adversarial_expected=True)
    assert not gate.passed
    assert "adversarial" in gate.failed_gates()


def test_hard_gate_passes_when_adversarial_expected_and_passed():
    gate = hard_gate(_result(), adversarial_expected=True)
    assert gate.passed


def test_hard_gate_adversarial_not_expected_by_default():
    # Callers that never ran the adversarial tier (the baseline's first
    # pass, the engine's main path) keep their previous behavior: the
    # adversarial component is vacuously true unless expected.
    gate = hard_gate(_result(adversarial_passed=False))
    assert gate.passed


def test_all_tiers_passed_true_only_when_everything_passes():
    assert _result().all_tiers_passed


def test_all_tiers_passed_false_on_adversarial_failure():
    # Regression for the tautology: the old implementation computed
    # (adversarial_passed or True) is not False — always True.
    assert not _result(adversarial_passed=False).all_tiers_passed


def test_all_tiers_passed_false_on_regression_failure():
    assert not _result(regression_passed=False).all_tiers_passed


def test_all_tiers_passed_false_with_violations():
    assert not _result(violations=("anything",)).all_tiers_passed
