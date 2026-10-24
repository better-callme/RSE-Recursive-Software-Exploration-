"""M3: verification tiers are isolated — one subprocess per tier.

A tier's crash or timeout fails that tier only; counts are per-tier;
scoring correctness is the acceptance rate; workdirs, evidence files
and nonces are per-tier.
"""

import json
import signal

from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig, ScoringWeights
from forgemind.core.problem import AcceptanceTest, ProblemSpec
from forgemind.core.proposal import AttackTestProposal
from forgemind.core.state import CodebaseState
from forgemind.recovery.failure import RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee, hard_gate
from forgemind.verification.results import ExecutionResult
from forgemind.verification.scoring import ScoringPolicy

GOOD = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t / len(data)}\n"
)


def _stats_referee():
    config = ForgeMindConfig()
    referee = Referee(StatsProblem().spec,
                      SubprocessExecutionEnvironment(config.resources), config)
    referee.attach_test_provider(StatsProblem().test_files)
    return referee


def _state(code=GOOD):
    return CodebaseState(spec_hash="a" * 64, files={"solution.py": code})


def _attack(name, code):
    return AttackTestProposal(agent_role="critic", test_name=name,
                              hypothesis="h", test_code=code)


class AdversarialTimeoutEnvironment:
    """Time out ONLY adversarial-tier subprocesses (M3 policy test)."""

    def __init__(self, delegate):
        self._delegate = delegate

    def materialize_state(self, state):
        return self._delegate.materialize_state(state)

    def cleanup(self, workdir):
        return self._delegate.cleanup(workdir)

    def execute(self, workdir, command, limits=None):
        try:
            sections = json.loads(command[2]).get("sections", {})
        except (ValueError, IndexError):
            sections = {}
        if "adversarial" in sections:
            return ExecutionResult(
                exit_code=-signal.SIGKILL, stdout="", stderr="",
                timed_out=True, runtime_ms=0.0,
                killed_signal=int(signal.SIGKILL), limit_hit="timeout")
        return self._delegate.execute(workdir, command, limits)


def test_adversarial_import_failure_leaves_acceptance_intact():
    """Discriminator: an adversarial crash must not erase acceptance."""
    attack = _attack("attack_broken",
                     "import definitely_missing_module_xyz\n"
                     "def attack_broken():\n    assert True\n")
    v = _stats_referee().verify(_state(), attack_tests=[attack])
    assert v.acceptance_passed
    assert (v.acceptance_total, v.acceptance_passes) == (6, 6)
    assert not v.adversarial_passed
    assert v.failure_type == "IMPORT_ERROR"
    gate = hard_gate(v, adversarial_expected=True)
    assert not gate.passed
    assert "adversarial" in gate.failed_gates()
    assert "requiredTests" not in gate.failed_gates()


def test_regression_crash_leaves_acceptance_intact():
    v = _stats_referee().verify(
        _state(), regression_tests={"r.py": "raise RuntimeError('boom')\n"})
    assert v.acceptance_passed
    assert not v.regression_passed
    assert v.failure_type == "INTERNAL_ERROR"
    gate = hard_gate(v)
    assert "regression" in gate.failed_gates()


def test_per_tier_counts_reconcile():
    attack = _attack("attack_ok",
                     "def attack_ok():\n    assert True\n")
    v = _stats_referee().verify(
        _state(), attack_tests=[attack],
        regression_tests={"r.py": "def regress_ok():\n    assert True\n"})
    assert (v.acceptance_total, v.acceptance_passes,
            v.acceptance_failures) == (6, 6, 0)
    assert (v.adversarial_total, v.adversarial_passes,
            v.adversarial_failures) == (1, 1, 0)
    assert (v.regression_total, v.regression_passes,
            v.regression_failures) == (1, 1, 0)
    assert (v.tests_total, v.tests_passed, v.tests_failed) == (8, 8, 0)
    assert v.acceptance_pass_rate == 1.0


def test_scoring_uses_acceptance_rate():
    """Discriminator: passing acceptance but failing attacks is fully
    correct — attack outcomes are not correctness points."""
    attack = _attack("attack_mean_wrong",
                     "def attack_mean_wrong():\n"
                     "    from solution import compute_stats\n"
                     "    assert compute_stats([1, 2, 3])['mean'] == 999\n")
    v = _stats_referee().verify(_state(), attack_tests=[attack])
    assert v.acceptance_pass_rate == 1.0
    assert v.tests_pass_rate < 1.0  # summed rate still mixes tiers
    gate = hard_gate(v, adversarial_expected=True)
    assert not gate.passed
    scores = ScoringPolicy(ScoringWeights()).score(v, gate, depth=1)
    assert scores.correctness == 1.0


def test_adversarial_only_timeout_attributed():
    env = AdversarialTimeoutEnvironment(SubprocessExecutionEnvironment())
    config = ForgeMindConfig()
    referee = Referee(StatsProblem().spec, env, config)
    referee.attach_test_provider(StatsProblem().test_files)
    attack = _attack("attack_ok", "def attack_ok():\n    assert True\n")
    v = referee.verify(_state(), attack_tests=[attack])
    assert v.timed_out
    assert v.timed_out_tiers == ("adversarial",)
    assert v.acceptance_passed
    assert not v.adversarial_passed
    assert v.failure_type == "TIMEOUT"


def test_engine_adversarial_only_timeout_quarantines(spec):
    """M3 policy, engine level: _verify with attacks times out on the
    adversarial tier only → H4 retry → quarantine, even though
    acceptance passed throughout. The verdict is inconclusive, and
    quarantine (not rejection) is the honest outcome."""
    env = AdversarialTimeoutEnvironment(SubprocessExecutionEnvironment())
    cfg = ForgeMindConfig(random_seed=7)
    referee = Referee(spec, env, cfg)
    referee.attach_test_provider(StatsProblem().test_files)
    eng = SearchEngine(spec, cfg, referee=referee,
                       recovery=RecoveryPolicy())
    res = eng.run()
    assert not res.success
    assert res.quarantined
    assert res.termination.reason == "quarantined"


def test_stdout_stderr_labeled_per_tier():
    attack = _attack("attack_ok", "def attack_ok():\n    assert True\n")
    v = _stats_referee().verify(
        _state(), attack_tests=[attack],
        regression_tests={"r.py": "def regress_ok():\n    assert True\n"})
    for field in (v.stdout, v.stderr):
        headers = ["=== ACCEPTANCE %s ===" % ("STDOUT" if field is v.stdout
                                               else "STDERR"),
                   "=== ADVERSARIAL %s ===" % ("STDOUT" if field is v.stdout
                                               else "STDERR"),
                   "=== REGRESSION %s ===" % ("STDOUT" if field is v.stdout
                                              else "STDERR")]
        positions = [field.index(h) for h in headers]
        assert positions == sorted(positions)


def test_cross_tier_filesystem_isolation():
    """Acceptance test writes a sentinel; the adversarial tier must
    not see it — fresh workdir, not shared state."""
    spec = ProblemSpec(
        problem_id="m3-probe",
        description="d",
        requirements=("r",),
        acceptance_tests=(AcceptanceTest("writer", "test_writer"),),
        interface_constraints={"compute_stats": "f(data)"},
        test_sources={"t.py": "def test_writer():\n"
                                "    open('secret.txt', 'w').write('x')\n"
                                "    assert True\n"},
    )
    config = ForgeMindConfig()
    referee = Referee(spec, SubprocessExecutionEnvironment(config.resources),
                      config)
    attack = _attack("attack_no_sentinel",
                     "def attack_no_sentinel():\n"
                     "    import os\n"
                     "    assert not os.path.exists('secret.txt')\n")
    v = referee.verify(_state(), attack_tests=[attack])
    assert v.acceptance_passed
    assert v.adversarial_passed


def test_later_tier_cannot_rewrite_earlier_verdict():
    """Temporal isolation, stronger than fail-closed: each tier's
    evidence is read immediately after that tier runs, so a later
    tier overwriting an earlier tier's evidence file — even with a
    forged FAILURE — cannot change the earlier verdict. (A wrong
    nonce would fail authentication anyway, per C1; this test pins
    the ordering: the verdict is already captured.) Guards against
    a future refactor that defers evidence reading until all tiers
    finish, which would reintroduce cross-tier interference."""
    attack = _attack(
        "attack_forge",
        "def attack_forge():\n"
        "    import glob, json\n"
        "    for path in glob.glob('/tmp/forgemind-result-*/result_acceptance.json'):\n"
        "        with open(path, 'w') as fh:\n"
        "            json.dump({'nonce': 'forged-nonce', 'sections': "
        "                       {'acceptance': {'total': 1, 'passed': 0, "
        "                        'failed': 1, 'details': ['forged'], "
        "                        'executed': []}}}, fh)\n"
        "    assert True\n")
    v = _stats_referee().verify(_state(), attack_tests=[attack])
    assert v.adversarial_passed  # the forging tier's own tests passed
    assert v.acceptance_passed  # the forged failure did not take effect
    assert (v.acceptance_total, v.acceptance_passes) == (6, 6)
