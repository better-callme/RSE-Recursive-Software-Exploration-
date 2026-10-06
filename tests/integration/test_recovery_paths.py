"""Targeted tests for recovery/backtracking/stagnation execution paths."""

from __future__ import annotations

from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig
from forgemind.core.history import EventType, History
from forgemind.core.proposal import (
    AttackTestProposal,
    CodePatchProposal,
    CritiqueProposal,
)
from forgemind.core.state import CodebaseState
from forgemind.recovery.failure import RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.search.node import SearchNode
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee


BAD_STATS = (
    "def compute_stats(data):\n"
    "    total = sum(data)\n"
    "    n = len(data)\n"
    "    return {'count': n, 'sum': total, 'min': min(data),\n"
    "            'max': max(data), 'mean': total / n}\n"
)

GOOD_STATS = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    total = sum(data)\n"
    "    n = len(data)\n"
    "    return {'count': n, 'sum': total, 'min': min(data),\n"
    "            'max': max(data), 'mean': total / n}\n"
)


class RecoveringBuilder:
    role = "builder"

    def __init__(self) -> None:
        self.repair_calls = 0

    def propose(self, projection, request_context):
        return (CodePatchProposal(created_files={"solution.py": BAD_STATS}),)

    def repair(self, projection, failure_evidence):
        self.repair_calls += 1
        return (CodePatchProposal(created_files={"solution.py": GOOD_STATS}),)


class FailingRepairBuilder(RecoveringBuilder):
    def repair(self, projection, failure_evidence):
        self.repair_calls += 1
        return (CodePatchProposal(created_files={"solution.py": BAD_STATS}),)


class AdversarialRepairBuilder(RecoveringBuilder):
    def propose(self, projection, request_context):
        src = GOOD_STATS + "\nADVERSARIAL_GUARD = False\n"
        return (CodePatchProposal(created_files={"solution.py": src}),)

    def repair(self, projection, failure_evidence):
        self.repair_calls += 1
        src = GOOD_STATS + "\nADVERSARIAL_GUARD = True\n"
        return (CodePatchProposal(created_files={"solution.py": src}),)


class GuardCritic:
    role = "critic"

    def propose(self, projection, request_context):
        critique = CritiqueProposal(
            agent_role=self.role,
            intent="guard-check",
            issues=(),
            suggested_repair_directions=(),
            confidence=0.9,
        )
        attack = AttackTestProposal(
            agent_role=self.role,
            test_name="attack_guard_switch",
            hypothesis="guard must be enabled during recovery",
            test_code=(
                "def attack_guard_switch():\n"
                "    import solution\n"
                "    assert getattr(solution, 'ADVERSARIAL_GUARD', False) is True\n"
            ),
        )
        return (critique,), (attack,)


def _referee(spec, config):
    ref = Referee(spec, SubprocessExecutionEnvironment(config.resources), config)
    ref.attach_test_provider(StatsProblem().test_files)
    return ref


def _engine(spec, config, *, builder, critic):
    return SearchEngine(
        spec,
        config,
        builder=builder,
        critic=critic,
        referee=_referee(spec, config),
        recovery=RecoveryPolicy(),
        history=History(),
    )


def test_recovery_branch_executes_and_recovers(spec):
    config = ForgeMindConfig(
        enable_optimizer=False,
        enable_adversarial_tier=False,
        max_repair_attempts_per_node=2,
        max_nodes=20,
        max_verifications=20,
    )
    builder = RecoveringBuilder()
    engine = _engine(spec, config, builder=builder, critic=GuardCritic())
    result = engine.run()

    assert result.success
    assert builder.repair_calls == 1
    assert any(
        e.metadata.get("type") == "TEST_FAILURE"
        for e in engine.history.of_type(EventType.FAILURE_RECORDED)
    )
    assert any(
        e.metadata.get("action") == "CRITIC_ANALYSIS"
        for e in engine.history.of_type(EventType.RECOVERY_ACTION)
    )


def test_recovery_is_bounded_when_repairs_keep_failing(spec):
    config = ForgeMindConfig(
        enable_optimizer=False,
        enable_adversarial_tier=False,
        max_repair_attempts_per_node=2,
        max_nodes=20,
        max_verifications=20,
    )
    builder = FailingRepairBuilder()
    engine = _engine(spec, config, builder=builder, critic=GuardCritic())
    result = engine.run()

    assert not result.success
    assert builder.repair_calls == 2
    assert any(
        e.metadata.get("outcome") == "repair_limit_reached"
        for e in engine.history.of_type(EventType.RECOVERY_ACTION)
    )
    assert result.termination.reason in ("frontier_empty", "max_nodes", "max_verifications")


def test_adversarial_failure_triggers_repair_instead_of_silent_progress(spec):
    config = ForgeMindConfig(
        enable_optimizer=False,
        enable_adversarial_tier=True,
        max_repair_attempts_per_node=2,
        max_nodes=20,
        max_verifications=30,
    )
    builder = AdversarialRepairBuilder()
    engine = _engine(spec, config, builder=builder, critic=GuardCritic())
    result = engine.run()

    assert result.success
    assert builder.repair_calls >= 1
    assert any(
        e.metadata.get("type") == "ADVERSARIAL_FAILURE"
        for e in engine.history.of_type(EventType.FAILURE_RECORDED)
    )
    assert not engine.history.of_type(EventType.NODE_INSERTED)


def test_stagnation_backtrack_prunes_frontier_and_resets_branch(spec):
    config = ForgeMindConfig(enable_optimizer=False)
    engine = _engine(spec, config, builder=RecoveringBuilder(), critic=GuardCritic())
    engine.detector.epsilon = 1.0

    root = SearchNode(
        node_id=engine.graph.next_node_id(),
        state=CodebaseState(spec_hash=spec.spec_hash, files={}),
        parent_node_id=None,
        candidate_id=None,
        depth=0,
        creation_index=0,
    )
    engine.graph.add(root)
    c1 = SearchNode(
        node_id=engine.graph.next_node_id(),
        state=CodebaseState(spec_hash=spec.spec_hash, files={"a.py": "x"}),
        parent_node_id=root.node_id,
        candidate_id="c1",
        depth=1,
        creation_index=1,
    )
    c2 = SearchNode(
        node_id=engine.graph.next_node_id(),
        state=CodebaseState(spec_hash=spec.spec_hash, files={"b.py": "y"}),
        parent_node_id=root.node_id,
        candidate_id="c2",
        depth=2,
        creation_index=2,
    )
    engine.graph.add(c1)
    engine.graph.add(c2)
    engine.frontier.insert(c1, guidance=0.1)
    engine.frontier.insert(c2, guidance=0.2)

    engine._check_stagnation(c1, guidance=0.5)
    engine._check_stagnation(c2, guidance=0.5)

    assert engine.metrics.stagnation_events == 1
    assert engine.metrics.backtracks == 1
    assert engine.metrics.pruned >= 1
    assert len(engine.frontier) == 1
    assert engine.frontier.peek_best() is not None
    assert engine.frontier.peek_best().node_id == c1.node_id
    assert engine.history.of_type(EventType.BACKTRACK)
