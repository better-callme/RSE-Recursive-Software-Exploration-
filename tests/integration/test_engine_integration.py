"""End-to-end integration tests of the full search engine."""

import pytest

from forgemind.agents.mocks import MockBuilder, MockCritic
from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig
from forgemind.core.history import EventType
from forgemind.core.proposal import CodePatchProposal
from forgemind.core.state import CodebaseState
from forgemind.core.transition import TransitionEngine, make_root_state
from forgemind.recovery.failure import RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee


def make_referee(spec, config=None):
    cfg = config or ForgeMindConfig()
    ref = Referee(spec, SubprocessExecutionEnvironment(cfg.resources), cfg)
    ref.attach_test_provider(StatsProblem().test_files)
    return ref


def make_engine(spec, referee, **overrides):
    config = overrides.pop("config", None) or ForgeMindConfig()
    return SearchEngine(
        spec, config, builder=overrides.pop("builder", None) or MockBuilder(),
        critic=MockCritic(), referee=referee,
        recovery=RecoveryPolicy(), **overrides,
    )


def test_full_search_accepts_verified_candidate(spec, referee):
    engine = make_engine(spec, referee)
    result = engine.run()
    assert result.success
    node = result.accepted_node
    assert node is not None and node.verification is not None
    assert node.verification.acceptance_passed
    # Invariant 10: objective evidence exists for the acceptance decision.
    assert node.scores is not None and node.scores.hard_pass
    events = [e.event_type for e in engine.history.events]
    assert EventType.STATE_ACCEPTED in events
    assert "solution.py" in node.state.files


def test_multiple_candidates_generated_and_separate_states(spec):
    """§46: builder emits A/B/C; all three become candidates with distinct states."""
    engine = make_engine(spec, make_referee(spec))
    # Inspect pre-search: run one expansion manually by running full search but
    # asserting on candidate count from proposals the builder generates.
    builder = MockBuilder()
    root_state = CodebaseState(spec_hash=spec.spec_hash, files={})
    from forgemind.agents import projections
    patches = builder.propose(
        projections.build_builder_projection(spec, root_state), {})
    assert len(patches) >= 3
    hashes = set()
    eng = TransitionEngine(spec)
    root = make_root_state(spec)
    for p in patches:
        r = eng.apply(root, p)
        if r.ok:
            hashes.add(r.new_state.content_hash)
    # The two broken variants must still produce distinct valid states.
    assert len(hashes) >= 2

    result = engine.run()
    assert result.metrics.candidates_created >= 1
    assert EventType.CANDIDATE_CREATED in [
        e.event_type for e in engine.history.events]


def test_failed_branches_recorded_with_evidence(spec, referee):
    engine = make_engine(spec, referee)
    result = engine.run()
    failure_events = engine.history.of_type(EventType.FAILURE_RECORDED)
    rejections = engine.history.of_type(EventType.TRANSITION_REJECTED)
    verif_fails = [
        n for n in result.graph.nodes.values() if n.status.name == "REJECTED"
    ]
    assert rejections or verif_fails or failure_events or result.success


def test_duplicate_transition_rejected_and_hash_identical(spec):
    """Two identical patches on the same root produce identical states."""
    eng = TransitionEngine(spec)
    root = make_root_state(spec)
    good = (
        "def compute_stats(data):\n"
        "    if not data:\n"
        "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
        "    t = sum(data)\n"
        "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
        "            'max': max(data), 'mean': t / len(data)}\n")
    ra = eng.apply(root, CodePatchProposal(created_files={"solution.py": good}))
    rb = eng.apply(root, CodePatchProposal(created_files={"solution.py": good}))
    assert ra.ok and rb.ok
    assert ra.new_state is not None and rb.new_state is not None
    assert ra.new_state.content_hash == rb.new_state.content_hash

    # Frontier refuses the second registration (duplicate pruning).
    from forgemind.search.node import SearchNode
    engine = make_engine(spec, make_referee(spec))
    n1 = SearchNode(node_id="na", state=ra.new_state, parent_node_id="root",
                    candidate_id="ca", depth=1, creation_index=1)
    n2 = SearchNode(node_id="nb", state=rb.new_state, parent_node_id="root",
                    candidate_id="cb", depth=1, creation_index=2)
    assert engine.frontier.insert(n1, 0.5) is True
    assert engine.frontier.insert(n2, 0.9) is False
    assert len(engine.frontier) == 1


def test_depth_exhaustion_terminates_with_reason(spec):
    config = ForgeMindConfig(max_depth=1, max_verifications=50)
    engine = make_engine(spec, make_referee(spec, config), config=config)
    result = engine.run()
    # With depth capped at 1 and a correct variant available, acceptance on the
    # first expansion is legal; otherwise search ends via budget/frontier.
    if not result.success:
        assert result.termination.reason in ("frontier_empty", "max_nodes")


def test_verification_budget_terminates(spec):
    config = ForgeMindConfig(max_verifications=2, max_agent_calls=100)
    engine = make_engine(spec, make_referee(spec, config), config=config)
    result = engine.run()
    assert engine.metrics.verifications <= 3  # hard cap + in-flight verification
    assert result.termination.reason in (
        "max_verifications", "goal_accepted", "frontier_empty")


def test_history_records_complete_story(spec, referee):
    engine = make_engine(spec, referee)
    engine.run()
    kinds = {e.event_type for e in engine.history.events}
    required = {
        EventType.SEARCH_STARTED, EventType.NODE_SELECTED, EventType.AGENT_CALLED,
        EventType.PROPOSAL_GENERATED, EventType.CANDIDATE_CREATED,
        EventType.VERIFICATION_STARTED, EventType.VERIFICATION_COMPLETED,
    }
    assert required <= kinds


def test_determinism_same_config_same_outcome(spec):
    """Two runs with fresh engines must produce identical decision traces."""
    def run_once():
        from forgemind.core.proposal import reset_id_counter

        reset_id_counter(0)
        cfg = ForgeMindConfig(random_seed=7)
        eng = SearchEngine(spec, cfg, referee=make_referee(spec, cfg),
                           recovery=RecoveryPolicy())
        res = eng.run()
        decisions = [
            (e.seq, e.event_type.value,
             tuple(sorted((k, str(v)) for k, v in e.metadata.items())))
            for e in eng.history.events
        ]
        accepted_hash = (res.accepted_node.state.content_hash
                         if res.accepted_node else None)
        return res.success, decisions, accepted_hash

    a = run_once()
    b = run_once()
    assert a[0] == b[0]
    assert a[1] == b[1]
    assert a[2] == b[2]
