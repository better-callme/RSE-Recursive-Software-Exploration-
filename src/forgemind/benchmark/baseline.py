"""Baseline: single deterministic proposal, verified once. No branching."""

from __future__ import annotations

import time
from dataclasses import dataclass

from forgemind.agents.mocks import MockBuilder
from forgemind.core.config import ForgeMindConfig
from forgemind.core.problem import ProblemSpec
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


def run_baseline(
    spec: ProblemSpec,
    referee: Referee,
    config: ForgeMindConfig | None = None,
    builder: MockBuilder | None = None,
) -> BaselineResult:
    config = config or ForgeMindConfig()
    builder = builder or MockBuilder()
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
    # Baseline takes exactly ONE proposal (variant 0) — no branching/backtracking.
    proposal: CodePatchProposal = next(
        p for p in proposals if isinstance(p, CodePatchProposal))
    candidate = Candidate(parent_node_id=root.node_id, proposal=proposal)
    tresult = TransitionEngine(spec).apply(root_state, proposal)
    if not tresult.ok:
        return BaselineResult(False, None, TerminationInfo("fatal", detail=tresult.reason),
                              time.monotonic() - started)
    state = tresult.new_state
    assert state is not None
    node = SearchNode(
        node_id=graph.next_node_id(), state=state,
        parent_node_id=root.node_id, candidate_id=candidate.candidate_id,
        depth=1, creation_index=1,
    )
    verification = referee.verify(state)
    gate = hard_gate(verification)
    node = node.with_updates(
        verification=verification,
        status=NodeStatus.ACCEPTED if gate.passed else NodeStatus.REJECTED,
    )
    return BaselineResult(
        success=gate.passed,
        node=node,
        termination=TerminationInfo("goal_accepted" if gate.passed else "single_shot_failed"),
        elapsed_seconds=time.monotonic() - started,
    )


def _minimal_builder_projection(spec: ProblemSpec, state: CodebaseState):
    from forgemind.agents.projections import build_builder_projection

    return build_builder_projection(spec, state)


def _raise_frozen_bl(self: object, name: str, value: object) -> None:
    raise TypeError(f"{type(self).__name__} is immutable ({name})")


for _cls in (BaselineResult,):
    _cls.__setattr__ = _raise_frozen_bl  # type: ignore[method-assign]
