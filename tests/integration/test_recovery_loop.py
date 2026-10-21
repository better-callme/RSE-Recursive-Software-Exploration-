"""C2 tests: failure-driven recovery must actually happen.

Before the fix, the loop was never closed: MockBuilder.repair() had no
call sites, builder projections never carried prior_failures, recovery
actions were recorded but never enacted, and a node whose expansion
produced no accepted/frontier child died on the spot.
"""

from forgemind.agents.mocks import MockBuilder, MockCritic
from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig
from forgemind.core.history import EventType
from forgemind.core.proposal import CodePatchProposal
from forgemind.recovery.failure import FailureType, RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee

GOOD = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t / len(data)}\n"
)
BAD_MEAN = GOOD.replace("t / len(data)", "t // len(data)")


def make_engine(spec, builder, **config_overrides):
    config = ForgeMindConfig(**config_overrides)
    referee = Referee(spec, SubprocessExecutionEnvironment(config.resources),
                      config)
    referee.attach_test_provider(StatsProblem().test_files)
    return SearchEngine(
        spec, config, builder=builder, critic=MockCritic(),
        referee=referee, recovery=RecoveryPolicy(),
    )


class RepairSpyBuilder(MockBuilder):
    """Trap builder (wrong variants first) that counts repair() calls."""

    def __init__(self):
        super().__init__(correct_first=False)
        self.repair_calls = 0
        self.repair_evidence = []

    def repair(self, projection, failure_evidence):
        self.repair_calls += 1
        self.repair_evidence.append(failure_evidence)
        return super().repair(projection, failure_evidence)


def test_recovery_via_repair_closes_the_loop(spec):
    """branching_factor=1 + trap builder: only the bad-mean variant is
    ever proposed. It fails, the critic analyzes, the root is re-expanded
    in repair mode, and repair() supplies the correct implementation."""
    builder = RepairSpyBuilder()
    engine = make_engine(spec, builder, branching_factor=1)
    result = engine.run()
    assert result.success
    assert builder.repair_calls == 1
    # The repair turn received real failure evidence, not an empty string.
    assert builder.repair_evidence and builder.repair_evidence[0]
    assert engine.metrics.backtracks >= 1
    kinds = [e.event_type for e in engine.history.events]
    assert EventType.BACKTRACK in kinds
    assert EventType.RECOVERY_ACTION in kinds
    accepted = result.accepted_node
    assert accepted is not None and accepted.verification is not None
    assert accepted.verification.acceptance_passed


def test_no_repair_budget_no_recovery(spec):
    """Control: identical setup with max_repair_attempts_per_node=0 must
    FAIL — proving the recovery mechanism, not something else, is what
    closed the loop in the test above."""
    builder = RepairSpyBuilder()
    engine = make_engine(spec, builder, branching_factor=1,
                         max_repair_attempts_per_node=0)
    result = engine.run()
    assert not result.success
    assert result.termination.reason == "frontier_empty"
    assert builder.repair_calls == 0


class EvidenceDrivenBuilder:
    """No repair() method: proposes badly until the projection carries
    prior failures, then proposes correctly. Exercises the propose()
    fallback and proves failures actually reach the builder."""

    role = "builder"

    def __init__(self):
        self.projections = []

    def propose(self, projection, request_context):
        self.projections.append(projection)
        src = GOOD if projection.prior_failures else BAD_MEAN
        return (CodePatchProposal(
            agent_role=self.role, intent="evidence-driven attempt",
            rationale="bad until failures are visible", estimated_cost=1.0,
            created_files={"solution.py": src}),)


def test_failures_reach_builder_projection_and_retry_recovers(spec):
    builder = EvidenceDrivenBuilder()
    engine = make_engine(spec, builder, branching_factor=1)
    result = engine.run()
    assert result.success
    assert len(builder.projections) == 2
    assert builder.projections[0].prior_failures == ()
    failures = builder.projections[1].prior_failures
    assert failures
    assert failures[0].failure_type is FailureType.TEST_FAILURE


class StubbornBuilder:
    """Emits a fresh broken variant on every call (distinct states, all
    failing): recovery must try exactly max_repair_attempts_per_node
    times and then give up — never loop forever."""

    role = "builder"

    def __init__(self):
        self.calls = 0

    def propose(self, projection, request_context):
        self.calls += 1
        src = BAD_MEAN + f"\n# attempt {self.calls}\n"
        return (CodePatchProposal(
            agent_role=self.role, intent="stubborn variant",
            rationale="always wrong, always new", estimated_cost=1.0,
            created_files={"solution.py": src}),)


def test_repair_cap_enforced_and_search_terminates(spec):
    builder = StubbornBuilder()
    engine = make_engine(spec, builder, branching_factor=1,
                         max_repair_attempts_per_node=2)
    result = engine.run()
    assert not result.success
    assert result.termination.reason == "frontier_empty"
    # 1 initial expansion + exactly 2 repair re-expansions of the root.
    root_selections = [
        e for e in engine.history.events
        if e.event_type is EventType.NODE_SELECTED
        and e.metadata.get("depth") == 0
    ]
    assert len(root_selections) == 3
    assert builder.calls == 3
    assert engine.metrics.backtracks == 2
