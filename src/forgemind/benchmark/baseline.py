"""Baseline: single deterministic proposal, verified once. No branching."""

from __future__ import annotations

import time
from dataclasses import dataclass

from forgemind.agents.mocks import MockBuilder, MockCritic
from forgemind.agents.projections import build_critic_projection
from forgemind.core.config import ForgeMindConfig
from forgemind.core.problem import ProblemSpec
from forgemind.core.tokens import estimate_tokens_for_proposals
from forgemind.core.proposal import Candidate, CodePatchProposal
from forgemind.core.state import CodebaseState
from forgemind.core.transition import TransitionEngine
from forgemind.search.engine import TerminationInfo
from forgemind.search.node import NodeStatus, SearchGraph, SearchNode
from forgemind.verification.referee import Referee, hard_gate


@dataclass(frozen=True)
class BaselineResult:
    success: bool
    node: SearchNode | None
    termination: TerminationInfo
    elapsed_seconds: float
    # M8: measured, not hardcoded — the experiment record reports these
    # instead of the old 1/1/0 constants.
    agent_calls: int = 0
    verifications: int = 0
    token_estimate: int = 0


def run_baseline(
    spec: ProblemSpec,
    referee: Referee,
    config: ForgeMindConfig | None = None,
    builder: MockBuilder | None = None,
    critic: MockCritic | None = None,
) -> BaselineResult:
    config = config or ForgeMindConfig()
    builder = builder or MockBuilder()
    critic = critic or MockCritic()
    started = time.monotonic()
    graph = SearchGraph()

    root_state = CodebaseState(spec_hash=spec.spec_hash, files={})
    root = SearchNode(
        node_id=graph.next_node_id(), state=root_state,
        parent_node_id=None, candidate_id=None, depth=0, creation_index=0,
        status=NodeStatus.VALID,
    )
    proposals = builder.propose(
        _minimal_builder_projection(spec, root_state),
        {"source_node_id": root.node_id},
    )
    # M8: count what actually happened. The builder ran (1 agent call)
    # and generated proposals (token estimate over ALL of them, matching
    # the engine's rule — not just the one taken below).
    agent_calls = 1
    token_estimate = estimate_tokens_for_proposals(proposals)
    # Baseline takes exactly ONE proposal (variant 0) — no branching/backtracking.
    proposal: CodePatchProposal = next(
        p for p in proposals if isinstance(p, CodePatchProposal))
    candidate = Candidate(parent_node_id=root.node_id, proposal=proposal)
    tresult = TransitionEngine(spec).apply(root_state, proposal)
    if not tresult.ok:
        return BaselineResult(False, None, TerminationInfo("fatal", detail=tresult.reason),
                              time.monotonic() - started,
                              agent_calls=agent_calls, verifications=0,
                              token_estimate=token_estimate)
    state = tresult.new_state
    assert state is not None
    node = SearchNode(
        node_id=graph.next_node_id(), state=state,
        parent_node_id=root.node_id, candidate_id=candidate.candidate_id,
        depth=1, creation_index=1,
    )
    verification = referee.verify(state)
    verifications = 1
    gate = hard_gate(verification)
    success = gate.passed
    termination_reason = "goal_accepted" if gate.passed else "single_shot_failed"
    # Tier 3 (adversarial), mirroring the ForgeMind engine's adversarial gate,
    # so "success" means the same thing in both arms and the measured delta
    # is attributable to search, not to a different verification bar.
    if gate.passed and config.enable_adversarial_tier:
        proj = build_critic_projection(spec, state, verification)
        critiques, attacks = critic.propose(proj, {"source_node_id": root.node_id})
        agent_calls += 1  # the critic ran, whether or not it found attacks
        token_estimate += (estimate_tokens_for_proposals(critiques)
                           + estimate_tokens_for_proposals(attacks))
        if attacks:
            adv = referee.verify(state, attack_tests=attacks)
            verifications += 1
            success = adv.adversarial_passed
            if not success:
                termination_reason = "adversarial_failed"
    node = node.with_updates(
        verification=verification,
        status=NodeStatus.ACCEPTED if success else NodeStatus.REJECTED,
    )
    return BaselineResult(
        success=success,
        node=node,
        termination=TerminationInfo(termination_reason),
        elapsed_seconds=time.monotonic() - started,
        agent_calls=agent_calls,
        verifications=verifications,
        token_estimate=token_estimate,
    )


def _minimal_builder_projection(spec: ProblemSpec, state: CodebaseState):
    from forgemind.agents.projections import build_builder_projection

    return build_builder_projection(spec, state)


def _raise_frozen_bl(self: object, name: str, value: object) -> None:
    raise TypeError(f"{type(self).__name__} is immutable ({name})")


for _cls in (BaselineResult,):
    _cls.__setattr__ = _raise_frozen_bl  # type: ignore[method-assign]
