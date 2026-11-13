"""Experiment 1 -- the Greedy Trap (highest priority).

Question: does RSE reach a solution that ordinary greedy patching cannot
reach, despite both receiving EXACTLY the same candidate proposals, the
same verification, and the same ranking?

Landscape (all admitted nodes pass every acceptance test -- they are
"attractive" -- but fail the precision attack, so none is accepted
outright; the winner hides one level deeper):

    ROOT -> A (admitted) -> fail1, fail2   (dead end)
         -> C (admitted) -> fail3          (dead end)
         -> M (admitted) -> B (admitted) -> GOOD (accepted)

Greedy (both rankings below) picks A first, commits, walks into the
dead end, and dies: it never returns to C, M, or B.
RSE picks A first too -- same ranking, same first move -- but retains
the frontier, survives both dead ends, goes deeper through M to B, and
accepts GOOD.

Two honest framing notes, both verified by the test itself:

1. "B looks worse immediately" is TRUE in guidance terms, and only
   because B is deeper: RSE's soft scoring terms are depth-only
   (performance/memory/maintainability are neutral 0.5 constants, and
   admitted nodes all have correctness 1.0), so same-depth admitted
   siblings tie EXACTLY and rank by creation order. B (depth 2) scores
   below A (depth 1). The test asserts this measured gap.
2. A partial-credit trap (B passing 3/6 tests) is OUTSIDE RSE's search
   space by design: the hard gate admits only fully acceptance-passing
   states. The trap therefore lives in the admitted space -- nodes that
   pass every acceptance test yet dead-end -- which is the stronger
   claim: even when greedy and RSE agree on every ranking, commitment
   loses and retention wins.

Greedy baseline: a small harness around the PRODUCTION TransitionEngine,
Referee, and ScoringPolicy (no invented scores, no fake verification).
It differs from RSE in exactly one way: it commits to the single best
child and never retains a frontier, backtracks, or repairs. Two
rankings are tried (guidance, and raw acceptance-pass count); both fail.
"""

from __future__ import annotations

from forgemind.agents.mocks import MockArchitect
from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig, ScoringWeights
from forgemind.core.history import EventType
from forgemind.core.proposal import (
    AttackTestProposal,
    CodePatchProposal,
    CritiqueProposal,
)
from forgemind.agents.projections import (
    build_architect_projection,
    build_builder_projection,
)
from forgemind.core.state import CodebaseState
from forgemind.core.transition import TransitionEngine
from forgemind.recovery.failure import RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.search.node import NodeStatus
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee, hard_gate
from forgemind.verification.scoring import ScoringPolicy

SPEC = StatsProblem().spec
TEST_FILES = StatsProblem().test_files

GOOD = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t / len(data)}\n"
)

_F32_HEADER = (
    "import struct\n\n\n"
    "def _f32(x):\n"
    "    return struct.unpack('f', struct.pack('f', x))[0]\n\n\n"
)

_F32_BODY = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': _f32(t / len(data))}\n"
)


def _f32_code(marker: str) -> str:
    return f"# {marker}\n" + _F32_HEADER + _F32_BODY


WRONG_EMPTY = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        raise ValueError('empty input')\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t / len(data)}\n"
)

WRONG_MEAN = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t // len(data)}\n"
)

WRONG_COUNT = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data) + 1, 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t / len(data)}\n"
)

A_CODE = _f32_code("# MARK-A")
C_CODE = _f32_code("# MARK-C")
M_CODE = _f32_code("# MARK-M")
B_CODE = _f32_code("# MARK-B")

ATTACK = AttackTestProposal(
    agent_role="critic",
    test_name="attack_mean_precision",
    hypothesis="float32 mean breaks precision",
    test_code=(
        "def attack_mean_precision():\n"
        "    from solution import compute_stats\n"
        "    r = compute_stats([-3, -3, 5, 5, 5])\n"
        "    assert abs(r['mean'] - 1.8) < 1e-9\n"
    ),
)


class TrapBuilder:
    """Marker-keyed landscape builder (stateless, so the SAME builder
    feeds RSE and both greedy variants)."""

    role = "builder"

    def __init__(self, mapping):
        self._mapping = dict(mapping)

    def propose(self, projection, request_context):
        current = projection.target_files.get("solution.py", "")
        codes = None
        for marker, cs in self._mapping.items():
            if marker and marker in current:
                codes = cs
                break
        if codes is None and not current:
            codes = self._mapping.get("", ())
        if not codes:
            return ()
        out = []
        for code in codes:
            if "solution.py" in projection.target_files:
                out.append(CodePatchProposal(
                    agent_role=self.role, intent="trap variant",
                    rationale="trap", estimated_cost=1.0,
                    modified_files={"solution.py": code}))
            else:
                out.append(CodePatchProposal(
                    agent_role=self.role, intent="trap variant",
                    rationale="trap", estimated_cost=1.0,
                    created_files={"solution.py": code}))
        return tuple(out)


TRAP_MAPPING = {
    "": [A_CODE, C_CODE, M_CODE],
    "# MARK-A": [WRONG_EMPTY, WRONG_MEAN],
    "# MARK-C": [WRONG_COUNT],
    "# MARK-M": [B_CODE],
    "# MARK-B": [GOOD],
}


class PrecisionCritic:
    role = "critic"

    def propose(self, projection, request_context):
        critique = CritiqueProposal(
            agent_role=self.role, intent="adversarial review", issues=(),
            suggested_repair_directions=(), confidence=0.8)
        return (critique,), (ATTACK,)


def _rse_run(*, beam_width):
    cfg = ForgeMindConfig(random_seed=0, enable_optimizer=False,
                          beam_width=beam_width)
    env = SubprocessExecutionEnvironment()
    referee = Referee(SPEC, env, cfg)
    referee.attach_test_provider(TEST_FILES)
    eng = SearchEngine(SPEC, cfg, builder=TrapBuilder(TRAP_MAPPING),
                       critic=PrecisionCritic(), referee=referee,
                       recovery=RecoveryPolicy())
    return eng, eng.run()


def _greedy_run(*, rank):
    """Ordinary greedy patching on production machinery. At each step it
    evaluates every proposal with the real transition + referee +
    scorer, commits to the single best child by `rank`, and never looks
    back. Returns (success, committed_solution_codes, verifications)."""
    cfg = ForgeMindConfig(random_seed=0)
    env = SubprocessExecutionEnvironment()
    referee = Referee(SPEC, env, cfg)
    referee.attach_test_provider(TEST_FILES)
    scorer = ScoringPolicy(ScoringWeights())
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


def test_greedy_trap_rse_wins_where_greedy_dies():
    # --- RSE, sharp beam (one node per round) -------------------------
    eng, result = _rse_run(beam_width=1)
    assert result.success
    assert result.termination.reason == "goal_accepted"
    accepted = [n for n in result.graph.nodes.values()
                if n.status is NodeStatus.ACCEPTED]
    assert len(accepted) == 1
    assert accepted[0].state.files["solution.py"] == GOOD

    # RSE explored every branch, including the deeper, worse-looking one.
    by_code = {}
    for n in result.graph.nodes.values():
        by_code.setdefault(n.state.files.get("solution.py"), n)
    for code in (A_CODE, C_CODE, M_CODE, B_CODE):
        assert code in by_code, "RSE must actually visit every branch"
    # B genuinely looks worse than A by the production guidance score:
    # depth is the only soft differentiator among admitted nodes.
    assert by_code[B_CODE].scores.guidance < by_code[A_CODE].scores.guidance
    # ... and RSE's first move agrees with greedy's: the difference is
    # what happens AFTER the dead end, not the ranking.
    selected = [e.metadata.get("node") for e in eng.history.events
                if e.event_type is EventType.NODE_SELECTED]
    first_pick = result.graph.nodes[selected[1]].state.files["solution.py"]
    assert first_pick == A_CODE

    # --- RSE with the DEFAULT beam also wins (not a beam_width artifact)
    _, result3 = _rse_run(beam_width=3)
    assert result3.success
    assert result3.accepted_node.state.files["solution.py"] == GOOD

    # --- Greedy dies under BOTH rankings -------------------------------
    for rank in ("guidance", "test_count"):
        success, committed, verifications = _greedy_run(rank=rank)
        assert not success, f"greedy-by-{rank} must not escape the trap"
        # Greedy's whole journey is the A branch and its dead end.
        assert committed[0] == A_CODE
        assert committed[-1] in (WRONG_EMPTY, WRONG_MEAN)
        assert GOOD not in committed

    # --- the fingerprint ------------------------------------------------
    report = {
        "rse_success": result.success,
        "rse_nodes_explored": eng.metrics.nodes_explored,
        "rse_candidates_created": eng.metrics.candidates_created,
        "rse_verifications": eng.metrics.verifications,
        "rse_duplicates": eng.metrics.duplicates,
        "rse_backtracks": eng.metrics.backtracks,
        "rse_max_depth": eng.metrics.max_depth_reached,
        "rse_termination": result.termination.reason,
        "greedy_guidance": _greedy_run(rank="guidance"),
        "greedy_test_count": _greedy_run(rank="test_count"),
    }
    assert report["rse_nodes_explored"] >= 5  # root, A, C, M, B
    assert report["rse_backtracks"] >= 1  # dead ends reinserted for recovery
    # (The report dict is kept for readability; pytest tests return None.)
