"""M5: ACCEPTED is terminal — the run's final winner only.

A node that passes the hard gate but fails the adversarial gate
is GATE_PASSED while admitted to the frontier, never ACCEPTED.
Successful run: exactly one ACCEPTED node, equal to the
delivered node. Unsuccessful run: zero ACCEPTED nodes.
"""

from pathlib import Path

from forgemind.agents.mocks import MockBuilder, MockCritic
from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig
from forgemind.core.proposal import (
    AttackTestProposal,
    CodePatchProposal,
    CritiqueProposal,
    OptimizationProposal,
)
from forgemind.recovery.failure import RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.search.node import NodeStatus
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


class SingleXBuilder:
    """Root -> one X variant (passes acceptance, fails the attack);
    any marked parent -> nothing (run ends frontier_empty)."""

    role = "builder"

    def propose(self, projection, request_context):
        current = projection.target_files.get("solution.py", "")
        if "# variant" in current:
            return ()
        src = _x_code("M1")
        if "solution.py" in projection.target_files:
            return (CodePatchProposal(
                agent_role=self.role, intent="m5 variant",
                rationale="m5", estimated_cost=1.0,
                modified_files={"solution.py": src}),)
        return (CodePatchProposal(
            agent_role=self.role, intent="m5 variant",
            rationale="m5", estimated_cost=1.0,
            created_files={"solution.py": src}),)


class GoodBuilder:
    """Root -> GOOD; marked parents -> nothing (run ends on accept)."""

    role = "builder"

    def propose(self, projection, request_context):
        current = projection.target_files.get("solution.py", "")
        if "# variant" in current or current:
            return ()
        return (CodePatchProposal(
            agent_role=self.role, intent="m5 good",
            rationale="m5", estimated_cost=1.0,
            created_files={"solution.py": "# variant G\n" + GOOD}),)


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


class CommentOptimizer:
    """Passing 'optimization': appends a comment, changing nothing."""

    role = "optimizer"

    def propose(self, projection, request_context):
        src = projection.verified_files["solution.py"]
        patch = CodePatchProposal(
            agent_role=self.role, intent="comment",
            modified_files={"solution.py": src + "\n# optimized\n"},
            rationale="annotate",
        )
        return (OptimizationProposal(agent_role=self.role, patch=patch,
                                     target_metric="readability"),)


def _engine(spec, builder, critic, optimizer=None, **cfg_kwargs):
    cfg = ForgeMindConfig(random_seed=7, **cfg_kwargs)
    referee = Referee(spec, SubprocessExecutionEnvironment(cfg.resources),
                      cfg)
    referee.attach_test_provider(StatsProblem().test_files)
    return SearchEngine(spec, cfg, builder=builder, critic=critic,
                        optimizer=optimizer, referee=referee,
                        recovery=RecoveryPolicy())


def _accepted_nodes(eng):
    return [n for n in eng.graph.nodes.values()
            if n.status is NodeStatus.ACCEPTED]


def test_adversarial_failed_admitted_node_is_gate_passed(spec):
    """Discriminator: gate-passed, adversarial-failed, admitted,
    run ends with no winner -> zero ACCEPTED, node GATE_PASSED."""
    eng = _engine(spec, SingleXBuilder(), PrecisionCritic())
    res = eng.run()
    assert not res.success
    assert res.accepted_node is None
    assert res.termination.reason == "frontier_empty"
    assert _accepted_nodes(eng) == []
    admitted = [n for n in eng.graph.nodes.values()
                if "# variant M1" in n.state.files.get("solution.py", "")]
    assert len(admitted) == 1
    assert admitted[0].status is NodeStatus.GATE_PASSED


def test_successful_run_has_exactly_one_accepted(spec):
    """Inverse: a successful run has exactly one ACCEPTED node,
    and it is the delivered node."""
    eng = _engine(spec, GoodBuilder(), PrecisionCritic(),
                  enable_optimizer=False)
    res = eng.run()
    assert res.success
    assert res.accepted_node is not None
    accepted = _accepted_nodes(eng)
    assert len(accepted) == 1
    assert accepted[0].node_id == res.accepted_node.node_id


def test_optimizer_promotion_leaves_original_gate_passed(spec):
    """Passing optimization: delivered opt node is ACCEPTED, its
    original is GATE_PASSED. Pre-fix the original is ACCEPTED
    too (count 2), so this discriminates on the optimizer path."""
    eng = _engine(spec, MockBuilder(), MockCritic(),
                  optimizer=CommentOptimizer())
    res = eng.run()
    assert res.success
    accepted = res.accepted_node
    assert accepted is not None
    assert "# optimized" in accepted.state.files["solution.py"]
    original = eng.graph.nodes[accepted.parent_node_id]
    assert original.status is NodeStatus.GATE_PASSED
    assert len(_accepted_nodes(eng)) == 1


def test_gate_passed_enum_exists_and_distinct():
    assert NodeStatus.GATE_PASSED.value == "GATE_PASSED"
    assert NodeStatus.GATE_PASSED is not NodeStatus.ACCEPTED
    assert NodeStatus.GATE_PASSED is not NodeStatus.REJECTED


def test_accepted_is_observational_not_control_flow():
    """Mechanical (Kashi's requirement): no production code
    branches on ACCEPTED, and exactly one ACCEPTED assignment
    exists in the search engine (the promotion site). This is
    what makes the fix behavior-preserving: if nothing reads
    the label for control flow, search decisions cannot change.
    The baseline's synthetic record assignment is out of scope
    (never admitted to a frontier, never scanned as a result)
    and excluded from the engine-site count."""
    src_root = Path(__file__).parents[2] / "src" / "forgemind"
    branch_pats = ("== NodeStatus.ACCEPTED", "is NodeStatus.ACCEPTED",
                   "!= NodeStatus.ACCEPTED", "case NodeStatus.ACCEPTED")
    offenders = []
    for path in sorted(src_root.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if any(p in line for p in branch_pats):
                offenders.append(f"{path}:{lineno}: {line.strip()}")
    assert offenders == []
    engine_lines = (src_root / "search" / "engine.py").read_text().splitlines()
    assigns = [l.strip() for l in engine_lines
               if "status=NodeStatus.ACCEPTED" in l
               or "status = NodeStatus.ACCEPTED" in l]
    assert len(assigns) == 1
