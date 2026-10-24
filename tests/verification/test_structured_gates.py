"""M1: gates consume structured evidence, never violation text.

The hard gate is a pure decision function over the canonical predicates
(results.gate_predicates). The referee owns all observation — static
analysis, contract analysis, test execution — and produces structured
evidence. Nothing in the gate inspects human-readable diagnostics.
"""

from forgemind.core.config import ForgeMindConfig
from forgemind.core.problem import AcceptanceTest, ProblemSpec
from forgemind.core.state import CodebaseState
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee, hard_gate
from forgemind.verification.results import VerificationResult
from forgemind.verification.static import StaticVerifier

TRIVIAL_TESTS = "def test_trivial():\n    assert True\n"


def _spec(entry: str = "compute_stats") -> ProblemSpec:
    return ProblemSpec(
        problem_id="m1-probe",
        description="d",
        requirements=("r",),
        acceptance_tests=(AcceptanceTest("t1", "test_trivial"),),
        interface_constraints={entry: "f(data)"},
        test_sources={"__fm_acceptance_tests.py": TRIVIAL_TESTS},
    )


def _state(code: str) -> CodebaseState:
    return CodebaseState(spec_hash="a" * 64, files={"solution.py": code})


def _referee(spec: ProblemSpec) -> Referee:
    config = ForgeMindConfig()
    return Referee(spec, SubprocessExecutionEnvironment(config.resources), config)


def _result(**kw) -> VerificationResult:
    base = dict(
        static_gate_passed=True,
        contract_passed=True,
        acceptance_passed=True,
        adversarial_passed=True,
        regression_passed=True,
        static_analysis_complete=True,
    )
    base.update(kw)
    return VerificationResult(**base)


def test_contract_gate_fails_when_entry_missing_despite_green_tests():
    """Acceptance green, entry absent: the old gate passed vacuously."""
    v = _referee(_spec()).verify(
        _state("def something_else(data):\n    return data\n"))
    assert v.acceptance_passed  # trivial tests never import the entry
    assert not v.contract_passed
    gate = hard_gate(v)
    assert not gate.passed
    assert "contract" in gate.failed_gates()


def test_contract_gate_is_spec_driven():
    """The entry name comes from the spec (H5), not a hardcoded literal."""
    referee = _referee(_spec(entry="describe_numbers"))
    good = referee.verify(_state("def describe_numbers(data):\n    return {}\n"))
    assert good.contract_passed
    wrong = referee.verify(_state("def compute_stats(data):\n    return {}\n"))
    assert not wrong.contract_passed


def test_gate_ignores_forbidden_text():
    """Killer test: violation text claims a forbidden call, but the
    structured evidence says clean — the gate must pass."""
    v = _result(violations=("solution.py: forbidden call 'eval()' line 3",))
    assert hard_gate(v).passed


def test_gate_fails_on_structured_forbidden_despite_clean_text():
    """Mirror killer test: clean text, structured finding — must fail."""
    v = _result(violations=("everything looks fine",),
                forbidden_constructs=(
                    "solution.py: forbidden call 'system()' line 1",))
    gate = hard_gate(v)
    assert not gate.passed
    assert "security" in gate.failed_gates()


def test_static_split_syntax_is_not_safety():
    s = StaticVerifier().verify({"a.py": "def broken(:\n"})
    assert not s.syntax_valid
    assert s.forbidden_constructs == ()
    assert not s.analysis_complete
    assert s.violations  # syntax diagnostic still present


def test_static_split_forbidden_is_safety():
    s = StaticVerifier().verify({"a.py": "def f():\n    eval('1')\n"})
    assert s.syntax_valid
    assert s.forbidden_constructs
    assert s.analysis_complete


def test_malformed_dangerous_source_fails_security_via_completeness():
    """Unparseable source containing dangerous text: nothing is
    established, so security must fail — not pass on an empty list."""
    code = "def broken(:\n    eval('1')\n"
    s = StaticVerifier().verify({"a.py": code})
    assert s.forbidden_constructs == ()
    assert not s.analysis_complete
    gate = hard_gate(_result(static_gate_passed=False,
                             static_analysis_complete=False))
    assert not gate.passed
    assert "security" in gate.failed_gates()


def test_referee_malformed_dangerous_source_fails_security():
    v = _referee(_spec()).verify(_state("def broken(:\n    eval('1')\n"))
    assert not v.static_gate_passed
    assert not v.static_analysis_complete
    assert v.forbidden_constructs == ()
    gate = hard_gate(v)
    assert "security" in gate.failed_gates()


def test_referee_good_state_carries_structured_evidence():
    v = _referee(_spec()).verify(
        _state("def compute_stats(data):\n    return data\n"))
    assert v.contract_passed
    assert v.forbidden_constructs == ()
    assert v.static_analysis_complete
    assert hard_gate(v).passed


def test_referee_eval_state_fails_security():
    v = _referee(_spec()).verify(
        _state("def compute_stats(data):\n    eval('1')\n    return data\n"))
    assert v.forbidden_constructs
    gate = hard_gate(v)
    assert not gate.passed
    assert "security" in gate.failed_gates()


def test_consistency_gate_level_contract_true_passes():
    """Gate interpretation is independent of referee computation."""
    assert hard_gate(_result(contract_passed=True)).passed


def test_consistency_referee_cannot_fabricate_contract():
    """The referee cannot produce contract_passed=True for a state
    lacking the entry."""
    v = _referee(_spec()).verify(_state("def nope(data):\n    return data\n"))
    assert not v.contract_passed


def test_consistency_security_pass_adversarial_fail_not_accepted():
    v = _result(adversarial_passed=False)
    assert not v.all_tiers_passed
    gate = hard_gate(v, adversarial_expected=True)
    assert not gate.passed
    assert "adversarial" in gate.failed_gates()


def test_centralization_accept_implies_no_failed_gates():
    assert _result().all_tiers_passed
    assert hard_gate(_result()).failed_gates() == ()
    # Documented strict superset (C3 behavior): diagnostics present with
    # all gates green fails all_tiers_passed but not the gate itself.
    v = _result(violations=("anything",))
    assert not v.all_tiers_passed
    assert hard_gate(v).passed
