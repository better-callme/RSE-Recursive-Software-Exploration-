"""Greedy Trap under the new opt-in strategies (keyless, mock builders).

The landscape is the SAME as test_greedy_trap.py — same proposals,
same verification, same ranking inputs — only the frontier strategy
and/or the guidance mode change. Each combination must still reach
GOOD, and greedy must still die in the trap under both scorers.

Note on negative controls: these integration tests are outcome tests,
not difference tests — beam also wins this trap, so reverting UCT to
beam (or Bayesian to pass-rate) would still pass them. The
difference is proven at the unit level (test_uct_strategy.py:
test_uct_differs_from_beam_on_revisited_node; test_bayesian_scoring.py:
0/5 gives 1/7 Bayesian vs 0.0 pass-rate). What these tests prove is
that the new switches do not break the search: the trap still resolves
under every combination.
"""

import importlib.util
import os
import sys

from forgemind.agents.mocks import MockArchitect
from forgemind.agents.projections import (
    build_architect_projection,
    build_builder_projection,
)
from forgemind.core.config import ForgeMindConfig, ScoringWeights
from forgemind.core.state import CodebaseState
from forgemind.core.transition import TransitionEngine
from forgemind.recovery.failure import RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.search.node import NodeStatus
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee, hard_gate
from forgemind.verification.scoring import ScoringPolicy

# ----------------------------------------------------------------------
# Reuse of the Greedy Trap fixtures (test-only, same pattern as 2B)
# ----------------------------------------------------------------------

_GT_PATH = os.path.join(os.path.dirname(__file__), "test_greedy_trap.py")
_gt_spec = importlib.util.spec_from_file_location("trap_fixtures", _GT_PATH)
_gt = importlib.util.module_from_spec(_gt_spec)
sys.modules["trap_fixtures"] = _gt
_gt_spec.loader.exec_module(_gt)

SPEC = _gt.SPEC
TEST_FILES = _gt.TEST_FILES
GOOD = _gt.GOOD
TRAP_MAPPING = _gt.TRAP_MAPPING
TrapBuilder = _gt.TrapBuilder
PrecisionCritic = _gt.PrecisionCritic
ATTACK = _gt.ATTACK
A_CODE = _gt.A_CODE
C_CODE = _gt.C_CODE
M_CODE = _gt.M_CODE
B_CODE = _gt.B_CODE
WRONG_EMPTY = _gt.WRONG_EMPTY
WRONG_MEAN = _gt.WRONG_MEAN


def _rse_run_combo(*, beam_width, frontier_strategy, guidance_mode):
    cfg = ForgeMindConfig(
        random_seed=0, enable_optimizer=False, beam_width=beam_width,
        frontier_strategy=frontier_strategy, guidance_mode=guidance_mode)
    env = SubprocessExecutionEnvironment()
    referee = Referee(SPEC, env, cfg)
    referee.attach_test_provider(TEST_FILES)
    eng = SearchEngine(SPEC, cfg, builder=TrapBuilder(TRAP_MAPPING),
                       critic=PrecisionCritic(), referee=referee,
                       recovery=RecoveryPolicy())
    return eng, eng.run()


def _assert_trap_resolved(result):
    assert result.success
    assert result.termination.reason == "goal_accepted"
    accepted = [n for n in result.graph.nodes.values()
                if n.status is NodeStatus.ACCEPTED]
    assert len(accepted) == 1
    assert accepted[0].state.files["solution.py"] == GOOD
    by_code = {}
    for n in result.graph.nodes.values():
        by_code.setdefault(n.state.files.get("solution.py"), n)
    for code in (A_CODE, C_CODE, M_CODE, B_CODE):
        assert code in by_code, "RSE must actually visit every branch"


def test_trap_uct_pass_rate():
    for beam_width in (1, 3):
        _, result = _rse_run_combo(
            beam_width=beam_width, frontier_strategy="uct",
            guidance_mode="pass_rate")
        _assert_trap_resolved(result)


def test_trap_beam_bayesian():
    for beam_width in (1, 3):
        _, result = _rse_run_combo(
            beam_width=beam_width, frontier_strategy="beam",
            guidance_mode="bayesian")
        _assert_trap_resolved(result)


def test_trap_uct_bayesian():
    for beam_width in (1, 3):
        _, result = _rse_run_combo(
            beam_width=beam_width, frontier_strategy="uct",
            guidance_mode="bayesian")
        _assert_trap_resolved(result)


def _greedy_run(*, rank, scorer):
    """Ordinary greedy patching (mirror of test_greedy_trap._greedy_run,
    but with the scorer as a parameter so greedy can be run under the
    Bayesian guidance too). Commits to the single best child and never
    looks back."""
    cfg = ForgeMindConfig(random_seed=0)
    env = SubprocessExecutionEnvironment()
    referee = Referee(SPEC, env, cfg)
    referee.attach_test_provider(TEST_FILES)
    trans = TransitionEngine(SPEC)
    architect = MockArchitect()
    builder = TrapBuilder(TRAP_MAPPING)

    state = CodebaseState(spec_hash=SPEC.spec_hash, files={})
    arch = architect.propose(build_architect_projection(SPEC, state), {})[0]
    committed = []
    verifications = 0
    depth = 0
    for _ in range(20):  # budget guard; the trap ends far earlier
        proj = build_builder_projection(SPEC, state, architecture=arch,
                                        failures=(), critiques=(),
                                        attempt_number=0)
        proposals = builder.propose(proj, {})
        if not proposals:
            return False, committed, verifications
        ranked = []
        for order, proposal in enumerate(proposals):
            tresult = trans.apply(state, proposal)
            if not tresult.ok:
                continue
            verification = referee.verify(tresult.new_state,
                                          attack_tests=(ATTACK,))
            verifications += 1
            gate = hard_gate(verification, adversarial_expected=True)
            scores = scorer.score(verification, gate, depth=depth + 1)
            if gate.passed:
                return True, committed + [tresult.new_state.files["solution.py"]], verifications
            key = (scores.guidance if rank == "guidance"
                   else verification.acceptance_passes)
            ranked.append((-key, order, tresult.new_state))
        if not ranked:
            return False, committed, verifications
        ranked.sort()
        state = ranked[0][2]
        committed.append(state.files["solution.py"])
        depth += 1
    return False, committed, verifications


def test_greedy_still_dies_under_new_scorers():
    # Greedy commits to A and dies under BOTH rankings and BOTH
    # guidance modes: the new switches change RSE's selection/scoring,
    # not the landscape — and the landscape is what kills greedy.
    scorers = {
        "pass_rate": ScoringPolicy(ScoringWeights()),
        "bayesian": ScoringPolicy(ScoringWeights(), guidance_mode="bayesian"),
    }
    for name, scorer in scorers.items():
        for rank in ("guidance", "test_count"):
            success, committed, _ = _greedy_run(rank=rank, scorer=scorer)
            assert not success, f"greedy-by-{rank}/{name} must not escape"
            assert committed[0] == A_CODE
            assert committed[-1] in (WRONG_EMPTY, WRONG_MEAN)
            assert GOOD not in committed
