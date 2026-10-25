"""M6: stagnation needs a complete per-branch window; prunes are not backtracks.

Self-contained staging (modeled on M4's helpers): X variants fail the
float32-precision attack and are admitted; GOOD passes it and is accepted.
Guidance shifts by ~0.0117 per depth (measured), so the branch tests use
stagnation_epsilon=0.05 — wide enough for the depth shift, tight enough
that the window/branch logic is what is under test. The detector unit
tests pin epsilon semantics precisely.
"""

from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig
from forgemind.core.history import EventType
from forgemind.core.proposal import (
    AttackTestProposal,
    CodePatchProposal,
    CritiqueProposal,
)
from forgemind.recovery.failure import FailureType, RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee

_HEADER = (
    "import struct\n\n\n"
    "def _f32(x):\n"
    "    return struct.unpack('f', struct.pack('f', x))[0]\n\n\n"
)

GOOD = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t / len(data)}\n"
)

_X_BODY = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': _f32(t / len(data))}\n"
)


def _x_code(marker):
    return f"# variant {marker}\n" + _HEADER + _X_BODY


def _proposals(srcs, target_files):
    if "solution.py" in target_files:
        return tuple(
            CodePatchProposal(
                agent_role="builder", intent="m6 variant",
                rationale="m6", estimated_cost=1.0,
                modified_files={"solution.py": s})
            for s in srcs
        )
    return tuple(
        CodePatchProposal(
            agent_role="builder", intent="m6 variant",
            rationale="m6", estimated_cost=1.0,
            created_files={"solution.py": s})
        for s in srcs
    )


class PrecisionCritic:
    """One attack: float32 mean precision. X fails it, GOOD passes."""

    role = "critic"

    def propose(self, projection, request_context):
        attack = AttackTestProposal(
            agent_role=self.role,
            test_name="attack_mean_precision",
            hypothesis="float32 mean breaks precision",
            test_code=(
                "def attack_mean_precision():\n"
                "    from solution import compute_stats\n"
                "    r = compute_stats([-3, -3, 5, 5, 5])\n"
                "    assert abs(r['mean'] - 1.8) < 1e-9\n"
            ),
        )
        critique = CritiqueProposal(
            agent_role=self.role, intent="adversarial review", issues=(),
            suggested_repair_directions=(), confidence=0.8)
        return (critique,), (attack,)


class EarlyAcceptBuilder:
    """Root -> [A, B] (admitted); any marked parent -> GOOD (accepted)."""

    role = "builder"

    def propose(self, projection, request_context):
        current = projection.target_files.get("solution.py", "")
        if not current:
            return _proposals([_x_code("A"), _x_code("B")],
                              projection.target_files)
        return _proposals([GOOD], projection.target_files)


class BranchBuilder:
    """Root -> [A, B]; A -> [A1, A2, A3]; any other marked parent -> fresh X.

    Marker matching is exact on the first line: "# variant A" must not
    match "# variant A1".
    """

    role = "builder"

    def __init__(self):
        self._n = 0

    def propose(self, projection, request_context):
        current = projection.target_files.get("solution.py", "")
        first = current.split("\n", 1)[0] if current else ""
        if not current:
            return _proposals([_x_code("A"), _x_code("B")],
                              projection.target_files)
        if first == "# variant A":
            return _proposals([_x_code("A1"), _x_code("A2"), _x_code("A3")],
                              projection.target_files)
        self._n += 1
        return _proposals([_x_code(f"Z{self._n}")],
                          projection.target_files)


def _engine(spec, env, builder, critic, **cfg_kwargs):
    cfg = ForgeMindConfig(random_seed=7, **cfg_kwargs)
    referee = Referee(spec, env, cfg)
    referee.attach_test_provider(StatsProblem().test_files)
    return SearchEngine(spec, cfg, builder=builder, critic=critic,
                        referee=referee, recovery=RecoveryPolicy())


def _stagnation_nodes(eng):
    return [e.metadata["node"] for e in eng.history.events
            if e.event_type is EventType.STAGNATION_DETECTED]


def _stagnation_records(eng):
    return [r for r in eng._all_records
            if r.failure_type is FailureType.STAGNATION]


def test_two_siblings_are_not_stagnation(spec):
    """M6: default window=4, two admitted siblings, run accepts early.

    Pre-fix the second sibling triggers stagnation (2-sample rule) and
    counts a backtrack. Post-fix each sibling is its own branch with one
    sample: no event, no record, no backtrack.
    """
    eng = _engine(spec, SubprocessExecutionEnvironment(),
                  EarlyAcceptBuilder(), PrecisionCritic())
    result = eng.run()
    assert result.success
    assert _stagnation_nodes(eng) == []
    assert _stagnation_records(eng) == []
    assert eng.metrics.stagnation_events == 0
    assert eng.metrics.backtracks == 0


def test_branch_window_fires_on_third_descendant(spec):
    """M6 window=3: branch A accumulates [A, A1, A2, ...]; the first
    stagnation event is on A2 (depth 2, the branch's 3rd admitted
    descendant) — never on a depth-1 node — and backtracks stay 0 even
    though stagnation fired (a prune is not a revisit)."""
    eng = _engine(spec, SubprocessExecutionEnvironment(),
                  BranchBuilder(), PrecisionCritic(),
                  stagnation_window=3, stagnation_epsilon=0.05)
    eng.run()
    nodes = _stagnation_nodes(eng)
    assert nodes, "expected at least one stagnation event"
    depth1 = {n.node_id for n in eng.graph.nodes.values() if n.depth == 1}
    assert not (set(nodes) & depth1), "no stagnation on depth-1 nodes"
    assert eng.graph.nodes[nodes[0]].depth == 2
    assert eng.metrics.backtracks == 0


def test_default_window_fires_on_fourth_descendant(spec):
    """M6 default window=4 at Kashi's A/B example: A and B never share a
    window, so the first stagnation event is on A3 (depth 2, the
    branch's 4th admitted descendant), never on a depth-1 node."""
    eng = _engine(spec, SubprocessExecutionEnvironment(),
                  BranchBuilder(), PrecisionCritic(),
                  stagnation_epsilon=0.05)
    eng.run()
    nodes = _stagnation_nodes(eng)
    assert nodes, "expected at least one stagnation event"
    depth1 = {n.node_id for n in eng.graph.nodes.values() if n.depth == 1}
    assert not (set(nodes) & depth1), "no stagnation on depth-1 nodes"
    assert eng.graph.nodes[nodes[0]].depth == 2
    assert eng.metrics.backtracks == 0
