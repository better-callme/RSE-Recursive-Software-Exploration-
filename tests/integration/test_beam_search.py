"""H1 integration: level discipline beats guidance.

Scenario: the root produces two depth-1 states, X and Y, that pass the
hard gate but fail the adversarial tier (so both land in the frontier).
Expanding X produces a depth-2 state G whose guidance (7/8 tests) is
HIGHER than Y's (6/8). Greedy best-first — the pre-H1 behavior — would
expand G before Y. A beam search must finish the depth-1 level first:
the NODE_SELECTED order must be root, X, Y.
"""

from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig
from forgemind.core.history import EventType
from forgemind.core.proposal import (
    AttackTestProposal,
    CodePatchProposal,
    CritiqueProposal,
)
from forgemind.recovery.failure import RecoveryPolicy
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

# Float32 mean, correct empty handling: fails attack A1 only.
X_CODE = "# variant X\n" + _HEADER + (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': _f32(t / len(data))}\n"
)

# Float32 mean AND empty returns mean 0 instead of None: fails A1 and A2.
Y_CODE = "# variant Y\n" + _HEADER + (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': 0}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': _f32(t / len(data))}\n"
)

# Same profile as X (fails A1 only), produced by expanding X.
G_CODE = "# variant G\n" + X_CODE.split("\n", 1)[1]


class TwoAttackCritic:
    role = "critic"

    def propose(self, projection, request_context):
        attacks = (
            AttackTestProposal(
                agent_role=self.role,
                test_name="attack_duplicates_and_negatives",
                hypothesis="duplicates or negatives break min/max/mean",
                test_code=(
                    "def attack_duplicates_and_negatives():\n"
                    "    from solution import compute_stats\n"
                    "    r = compute_stats([-3, -3, 5, 5, 5])\n"
                    "    assert r['min'] == -3 and r['max'] == 5\n"
                    "    assert abs(r['mean'] - 1.8) < 1e-9\n"
                    "    assert r['count'] == 5\n"
                ),
            ),
            AttackTestProposal(
                agent_role=self.role,
                test_name="attack_empty_mean_none",
                hypothesis="empty input must return mean None",
                test_code=(
                    "def attack_empty_mean_none():\n"
                    "    from solution import compute_stats\n"
                    "    r = compute_stats([])\n"
                    "    assert r['mean'] is None\n"
                ),
            ),
        )
        critique = CritiqueProposal(
            agent_role=self.role, intent="adversarial review", issues=(),
            suggested_repair_directions=(), confidence=0.8)
        return (critique,), attacks


class LevelBuilder:
    """Content-keyed builder: root -> [X, Y]; X -> [G]; Y -> [GOOD].

    Below the root, solution.py already exists, so proposals must EDIT
    it — the transition contract rejects creating an existing file."""

    role = "builder"

    def propose(self, projection, request_context):
        current = projection.target_files.get("solution.py", "")
        if "# variant Y" in current:
            srcs = [GOOD]
        elif "# variant X" in current:
            srcs = [G_CODE]
        elif current:
            srcs = [GOOD]
        else:
            srcs = [X_CODE, Y_CODE]
        if "solution.py" in projection.target_files:
            return tuple(
                CodePatchProposal(
                    agent_role=self.role, intent="level variant",
                    rationale="beam scenario", estimated_cost=1.0,
                    modified_files={"solution.py": s})
                for s in srcs
            )
        return tuple(
            CodePatchProposal(
                agent_role=self.role, intent="level variant",
                rationale="beam scenario", estimated_cost=1.0,
                created_files={"solution.py": s})
            for s in srcs
        )


def test_beam_expands_level_before_deeper_higher_guidance_node(spec):
    config = ForgeMindConfig()
    referee = Referee(spec, SubprocessExecutionEnvironment(config.resources),
                      config)
    referee.attach_test_provider(StatsProblem().test_files)
    engine = SearchEngine(
        spec, config, builder=LevelBuilder(), critic=TwoAttackCritic(),
        referee=referee, recovery=RecoveryPolicy(),
    )
    result = engine.run()
    assert result.success

    graph = result.graph
    root_id = next(nid for nid, n in graph.nodes.items() if n.depth == 0)
    x_id, y_id = graph.children[root_id]
    selected = [
        e.metadata["node"] for e in engine.history.events
        if e.event_type is EventType.NODE_SELECTED
    ]
    # Beam discipline: Y (depth 1) is selected before G (depth 2) even
    # though G's guidance is higher. Best-first would pick G third.
    assert selected == [root_id, x_id, y_id]
    # Sanity on the scenario itself: G really was created and inserted,
    # and really does outscore Y — otherwise the test proves nothing.
    g_nodes = [n for n in graph.nodes.values()
               if "# variant G" in n.state.files.get("solution.py", "")]
    assert g_nodes
    y_node = graph.get(y_id)
    assert g_nodes[0].scores.guidance > y_node.scores.guidance
