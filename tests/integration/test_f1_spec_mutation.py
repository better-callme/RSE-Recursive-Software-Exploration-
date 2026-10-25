"""F1 regression: a spec-mutation proposal sent through the REAL engine.

Before the fix, the transition layer correctly rejected the proposal
with violation="PROBLEM_SPEC_MUTATION", but the engine's mapping to
FailureType raised AttributeError because the enum member did not
exist (and, behind that, RecoveryPolicy.action_for would have raised
KeyError for the missing map row). This test pins the end-to-end
behavior: rejection, correct failure type, PRUNE recovery, unchanged
state, no erroneous acceptance, and search continuing to the good
candidate in the same run.
"""

from __future__ import annotations

from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig
from forgemind.core.history import EventType
from forgemind.core.proposal import CodePatchProposal
from forgemind.recovery.failure import (
    FailureType,
    RecoveryAction,
    RecoveryPolicy,
)
from forgemind.search.engine import SearchEngine
from forgemind.search.node import NodeStatus
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee

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


class NoAttackCritic:
    role = "critic"

    def propose(self, projection, request_context):
        return (), ()


class SpecAttackBuilder:
    """Root only: a spec-mutation proposal, then the good solution.
    Nothing below the root, so the run's continuation is the GOOD
    candidate's acceptance, not further branching."""

    role = "builder"

    def propose(self, projection, request_context):
        if projection.target_files:
            return ()
        return (
            CodePatchProposal(
                agent_role=self.role, intent="spec mutation",
                rationale="f1", estimated_cost=1.0,
                created_files={"problem_spec.json": "{}\n"}),
            CodePatchProposal(
                agent_role=self.role, intent="good solution",
                rationale="f1", estimated_cost=1.0,
                created_files={"solution.py": GOOD}),
        )


def test_f1_spec_mutation_through_engine():
    cfg = ForgeMindConfig(random_seed=0, enable_optimizer=False,
                          branching_factor=2)
    env = SubprocessExecutionEnvironment()
    referee = Referee(SPEC, env, cfg)
    referee.attach_test_provider(TEST_FILES)
    eng = SearchEngine(SPEC, cfg, builder=SpecAttackBuilder(),
                       critic=NoAttackCritic(), referee=referee,
                       recovery=RecoveryPolicy())
    result = eng.run()  # pre-fix: AttributeError here

    # The run continues past the rejection and accepts GOOD.
    assert result.success
    assert result.termination.reason == "goal_accepted"

    # The rejection is recorded with the correct type, stage, recovery.
    recs = [r for r in eng._all_records
            if r.failure_type is FailureType.PROBLEM_SPEC_MUTATION]
    assert len(recs) == 1
    assert recs[0].stage == "transition"
    assert recs[0].recovery is RecoveryAction.PRUNE
    assert any(e.event_type is EventType.TRANSITION_REJECTED
               for e in eng.history.events)

    # State unchanged: no node carries the spec file, root still empty.
    assert not any("problem_spec.json" in n.state.files
                   for n in result.graph.nodes.values())
    root = [n for n in result.graph.nodes.values()
            if n.parent_node_id is None][0]
    assert root.state.files == {}

    # Exactly one ACCEPTED node, and it is the good solution.
    accepted = [n for n in result.graph.nodes.values()
                if n.status is NodeStatus.ACCEPTED]
    assert len(accepted) == 1
    assert accepted[0].state.files["solution.py"] == GOOD
    assert result.accepted_node is accepted[0]
