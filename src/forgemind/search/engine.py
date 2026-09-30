"""SearchEngine: the deterministic loop that integrates everything.

Agents propose; this engine alone decides. See spec §57.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from forgemind.agents.projections import (
    build_architect_projection,
    build_builder_projection,
    build_critic_projection,
    build_optimizer_projection,
)
from forgemind.agents.mocks import (
    MockArchitect,
    MockBuilder,
    MockCritic,
    MockOptimizer,
)
from forgemind.core.config import ForgeMindConfig
from forgemind.core.history import EventType, History
from forgemind.core.problem import ProblemSpec
from forgemind.core.proposal import (
    AttackTestProposal,
    Candidate,
    CodePatchProposal,
    CritiqueProposal,
    Proposal,
)
from forgemind.core.state import CodebaseState
from forgemind.core.transition import TransitionEngine
from forgemind.recovery.failure import FailureType, RecoveryAction, RecoveryPolicy
from forgemind.search.node import (
    BeamSearchStrategy,
    NodeStatus,
    SearchFrontier,
    SearchGraph,
    SearchNode,
)
from forgemind.search.stagnation import StagnationDetector
from forgemind.verification.referee import Referee, hard_gate
from forgemind.verification.scoring import ScoringPolicy
from forgemind.verification.results import VerificationResult


@dataclass(frozen=True)
class TerminationInfo:
    reason: str  # goal_accepted | frontier_empty | max_depth | max_nodes |
    #              max_verifications | max_agent_calls | wall_clock | fatal
    accepted_node_id: str | None = None
    detail: str = ""


@dataclass
class SearchMetrics:
    nodes_explored: int = 0
    candidates_created: int = 0
    verifications: int = 0
    agent_calls: int = 0
    duplicates: int = 0
    pruned: int = 0
    backtracks: int = 0
    stagnation_events: int = 0
    max_depth_reached: int = 0
    token_estimate: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(vars(self))


class SearchEngine:
    def __init__(
        self,
        spec: ProblemSpec,
        config: ForgeMindConfig | None = None,
        *,
        architect: MockArchitect | None = None,
        builder: MockBuilder | None = None,
        critic: MockCritic | None = None,
        optimizer: MockOptimizer | None = None,
        referee: Referee | None = None,
        recovery: RecoveryPolicy | None = None,
        history: History | None = None,
    ) -> None:
        self.spec = spec
        self.config = config or ForgeMindConfig()
        self.architect = architect or MockArchitect()
        self.builder = builder or MockBuilder()
        self.critic = critic or MockCritic()
        self.optimizer = optimizer or MockOptimizer()
        self.referee = referee or Referee(spec, _default_env(), self.config)
        self.recovery = recovery or RecoveryPolicy()
        self.history = history or History()
        self.graph = SearchGraph()
        self.frontier = SearchFrontier(
            capacity=max(self.config.max_nodes, 8),
            beam_width=self.config.beam_width,
        )
        self.strategy = BeamSearchStrategy(self.config.beam_width)
        self.scorer = ScoringPolicy(self.config.scoring_weights)
        self.detector = StagnationDetector(
            window=self.config.stagnation_window,
            epsilon=self.config.stagnation_epsilon,
        )
        self.metrics = SearchMetrics()
        self._attack_tests: list[AttackTestProposal] = []
        self._critiques: tuple[CritiqueProposal, ...] = ()
        self._architecture = None
        self._failure_counts: dict[str, int] = {}
        self._accepted: SearchNode | None = None
        self._regression_tests: dict[str, str] = {}

    # ------------------------------------------------------------------
    def run(self) -> "SearchResult":
        started = time.monotonic()
        root_state = CodebaseState(spec_hash=self.spec.spec_hash, files={})
        root = SearchNode(
            node_id=self.graph.next_node_id(), state=root_state,
            parent_node_id=None, candidate_id=None, depth=0,
            creation_index=0, status=NodeStatus.VALID,
        )
        self.graph.add(root)
        self.frontier.insert(root, guidance=1.0)  # root always explorable
        self.history.append(EventType.SEARCH_STARTED, problem=self.spec.problem_id)

        termination = self._loop(started)

        result = SearchResult(
            termination=termination,
            graph=self.graph,
            history=self.history,
            metrics=self.metrics,
            elapsed_seconds=time.monotonic() - started,
        )
        self.history.append(
            EventType.SEARCH_TERMINATED, reason=termination.reason,
            accepted=termination.accepted_node_id or "",
        )
        return result

    # ------------------------------------------------------------------
    def _budget_exceeded(self, started: float) -> TerminationInfo | None:
        if getattr(self, "_accepted", None) is not None:
            return TerminationInfo("goal_accepted", self._accepted.node_id)
        m = self.metrics
        if len(self.frontier) == 0 and m.nodes_explored > 0:
            return TerminationInfo("frontier_empty")
        if m.nodes_explored >= self.config.max_nodes:
            return TerminationInfo("max_nodes")
        if m.verifications >= self.config.max_verifications:
            return TerminationInfo("max_verifications")
        if m.agent_calls >= self.config.max_agent_calls:
            return TerminationInfo("max_agent_calls")
        if time.monotonic() - started >= self.config.max_wall_clock_seconds:
            return TerminationInfo("wall_clock")
        return None

    def _loop(self, started: float) -> TerminationInfo:
        while True:
            exhausted = self._budget_exceeded(started)
            if exhausted is not None:
                return exhausted

            node = self.strategy.select_next(self.frontier, self.graph)
            if node is None:
                return TerminationInfo("frontier_empty")

            self.history.append(EventType.NODE_SELECTED, node=node.node_id,
                                depth=node.depth, hash=node.state.content_hash[:12])
            self.metrics.nodes_explored += 1
            self.metrics.max_depth_reached = max(self.metrics.max_depth_reached,
                                                 node.depth)

            if node.depth >= self.config.max_depth:
                rec = self.recovery.record(
                    state_hash=node.state.content_hash, candidate_id="",
                    failure_type=FailureType.DEPTH_EXHAUSTED, stage="search",
                    evidence=f"depth {node.depth} reached",
                )
                self._record_failure(node, rec)
                continue

            self._expand_node(node)

    def _expand_node(self, node: SearchNode) -> None:
        # Architect once at root.
        if self._architecture is None:
            arch_proj = build_architect_projection(self.spec, node.state)
            proposals = self._call_agent(self.architect, arch_proj,
                                         source_node_id=node.node_id)
            if proposals:
                self._architecture = proposals[0]

        builder_proj = build_builder_projection(
            self.spec, node.state, architecture=self._architecture,
            critiques=self._critiques, attempt_number=len(self._critiques),
        )
        patch_proposals = self._call_agent(
            self.builder, builder_proj, source_node_id=node.node_id)

        surviving: list[tuple[Candidate, CodebaseState]] = []
        for gen_index, proposal in enumerate(patch_proposals[
                : self.config.branching_factor]):
            candidate = Candidate(
                parent_node_id=node.node_id, proposal=proposal,
                estimated_cost=proposal.estimated_cost,
                generation_index=gen_index,
            )
            self.history.append(EventType.CANDIDATE_CREATED,
                                candidate=candidate.candidate_id,
                                parent=node.node_id)
            self.metrics.candidates_created += 1

            engine = TransitionEngine(self.spec)
            tresult = engine.apply(node.state, proposal)
            if not tresult.ok:
                ftype = (FailureType.PROBLEM_SPEC_MUTATION
                         if tresult.violation == "PROBLEM_SPEC_MUTATION"
                         else FailureType.SYNTAX_ERROR
                         if tresult.violation == "SYNTAX"
                         else FailureType.CONTRACT_VIOLATION)
                rec = self.recovery.record(
                    state_hash=node.state.content_hash,
                    candidate_id=candidate.candidate_id,
                    failure_type=ftype, stage="transition",
                    evidence=tresult.reason,
                )
                self.history.append(EventType.TRANSITION_REJECTED,
                                    candidate=candidate.candidate_id,
                                    reason=tresult.reason)
                self._record_failure(node, rec)
                continue

            new_state: CodebaseState = tresult.new_state  # type: ignore[assignment]
            self.history.append(EventType.TRANSITION_APPLIED,
                                candidate=candidate.candidate_id,
                                hash=new_state.content_hash[:12])

            # Duplicate detection via content hash (Invariant 6).
            if not self.frontier.remember_state(new_state.content_hash):
                self.metrics.duplicates += 1
                self.history.append(EventType.DUPLICATE_DETECTED,
                                    hash=new_state.content_hash[:12],
                                    candidate=candidate.candidate_id)
                rec = self.recovery.record(
                    state_hash=new_state.content_hash,
                    candidate_id=candidate.candidate_id,
                    failure_type=FailureType.DUPLICATE_STATE, stage="search",
                    evidence="state content hash already closed",
                )
                self._record_failure(node, rec)
                continue

            child = SearchNode(
                node_id=self.graph.next_node_id(),
                state=new_state,
                parent_node_id=node.node_id,
                candidate_id=candidate.candidate_id,
                depth=node.depth + 1,
                creation_index=len(self.graph),
                status=NodeStatus.UNVERIFIED,
            )
            self.graph.add(child)
            verification = self._verify(child)
            gate = hard_gate(verification)
            repeated = self._failure_counts.get(child.state.content_hash, 0)
            scores = self.scorer.score(verification, gate, depth=child.depth,
                                       repeated_failures=repeated)
            child = child.with_updates(
                verification=verification,
                scores=scores,
                status=(NodeStatus.ACCEPTED if gate.passed else NodeStatus.REJECTED),
            )
            self.graph.update(child)

            if gate.passed and self._adversarial_gate(child):
                self.history.append(EventType.STATE_ACCEPTED, node=child.node_id,
                                    hash=child.state.content_hash[:12],
                                    score=scores.total)
                # Optimizer round on the verified solution (regression-checked).
                if self.config.enable_optimizer:
                    child = self._optimize(child)
                self.frontier.remove(child.node_id)
                self.graph.update(child.with_updates(status=NodeStatus.ACCEPTED))
                self._accepted = child
                return

            # Failed: record failure + recovery-driven follow-up.
            if not gate.passed:
                ftype = FailureType(verification.failure_type)
                rec = self.recovery.record(
                    state_hash=child.state.content_hash,
                    candidate_id=candidate.candidate_id,
                    failure_type=ftype, stage="dynamic" if ftype is FailureType.TEST_FAILURE else "adversarial",
                    evidence="; ".join(verification.violations)[:500] or "gate failed",
                )
                self._record_failure(child, rec)
                action = self.recovery.action_for(ftype)
                if action in (RecoveryAction.CRITIC_ANALYSIS, RecoveryAction.BUILDER_REPAIR):
                    self._run_critic(child, verification)
                # Keep failed-but-informative states out of the frontier;
                # repair happens from the parent with evidence attached.
            else:
                inserted = self.frontier.insert(child, guidance=scores.guidance)
                if inserted:
                    self.history.append(EventType.NODE_INSERTED, node=child.node_id,
                                        guidance=round(scores.guidance, 6))
                self._check_stagnation(child, scores.guidance)

    # ------------------------------------------------------------------
    def _verify(self, node: SearchNode) -> VerificationResult:
        self.history.append(EventType.VERIFICATION_STARTED, node=node.node_id)
        self.metrics.verifications += 1
        result = self.referee.verify(
            node.state, attack_tests=self._attack_tests,
            regression_tests=self._regression_tests,
        )
        self.history.append(EventType.VERIFICATION_COMPLETED, node=node.node_id,
                            passed=result.acceptance_passed,
                            tests=f"{result.tests_passed}/{result.tests_total}")
        return result

    def _adversarial_gate(self, node: SearchNode) -> bool:
        """Tier-3: run critic attack tests against a hard-gate-passing state."""
        if not self.config.enable_adversarial_tier:
            return True
        if not self._attack_tests:
            self._run_critic(node, node.verification)
        if not self._attack_tests:
            return True
        self.metrics.verifications += 1
        adv_result = self.referee.verify(node.state,
                                         attack_tests=self._attack_tests)
        ok = adv_result.adversarial_passed
        if not ok:
            rec = self.recovery.record(
                state_hash=node.state.content_hash,
                candidate_id=node.candidate_id or "",
                failure_type=FailureType.ADVERSARIAL_FAILURE,
                stage="adversarial",
                evidence="; ".join(adv_result.violations)[:500],
            )
            self._record_failure(node, rec)
        return ok

    def _run_critic(self, node: SearchNode, verification: VerificationResult | None) -> None:
        proj = build_critic_projection(self.spec, node.state, verification)
        critiques, attacks = self._call_agent_pair(self.critic, proj,
                                                   source_node_id=node.node_id)
        self._critiques = self._critiques + critiques
        known = {a.test_name for a in self._attack_tests}
        self._attack_tests.extend(a for a in attacks if a.test_name not in known)
        self._pending_repairs = True

    def _optimize(self, node: SearchNode) -> SearchNode:
        metrics = {"runtime_ms": (node.verification.runtime_ms
                                  if node.verification else 0.0)}
        proj = build_optimizer_projection(self.spec, node.state, metrics)
        proposals = self._call_agent(self.optimizer, proj,
                                     source_node_id=node.node_id)
        if not proposals:
            return node
        proposal = proposals[0]
        engine = TransitionEngine(self.spec)
        tresult = engine.apply(node.state, proposal)
        if not tresult.ok:
            return node
        opt_state = tresult.new_state
        opt_node = SearchNode(
            node_id=self.graph.next_node_id(), state=opt_state,
            parent_node_id=node.node_id, candidate_id=None,
            depth=node.depth + 1, creation_index=len(self.graph),
        )
        self.graph.add(opt_node)
        verification = self._verify(opt_node)
        gate = hard_gate(verification)
        scores = self.scorer.score(verification, gate, depth=opt_node.depth)
        opt_node = opt_node.with_updates(
            verification=verification, scores=scores,
            status=NodeStatus.ACCEPTED if gate.passed else NodeStatus.REJECTED,
        )
        self.graph.update(opt_node)
        if not gate.passed:
            # REGRESSION: optimization rejected; keep original accepted node.
            rec = self.recovery.record(
                state_hash=opt_state.content_hash, candidate_id="",
                failure_type=FailureType.REGRESSION_FAILURE, stage="optimization",
                evidence="optimized state failed regression gates",
            )
            self._record_failure(opt_node, rec)
            return node
        self.history.append(EventType.STATE_ACCEPTED, node=opt_node.node_id,
                            optimized=True)
        return opt_node

    def _check_stagnation(self, node: SearchNode, guidance: float) -> None:
        if self.detector.record(guidance=guidance,
                                content_hash=node.state.content_hash):
            self.metrics.stagnation_events += 1
            self.history.append(EventType.STAGNATION_DETECTED, node=node.node_id)
            rec = self.recovery.record(
                state_hash=node.state.content_hash, candidate_id="",
                failure_type=FailureType.STAGNATION, stage="search",
                evidence="no measurable guidance improvement / oscillation",
            )
            self._record_failure(node, rec)
            # Recovery for stagnation per policy: terminate the stagnant branch.
            prunable = [
                e for e in self.frontier.top_k(len(self.frontier))
                if e.node_id.rsplit("-", 1)[-1].startswith(
                    node.node_id.rsplit("-", 1)[-1]) or e.depth > node.depth
            ]
            for entry in self.frontier.prune_guidance_below(entry_guidance_floor(guidance)):
                self.metrics.pruned += 1
                self.history.append(EventType.NODE_PRUNED, node=entry.node_id,
                                    reason="stagnation")
            self.metrics.backtracks += 1
            self.history.append(EventType.BACKTRACK, node=node.node_id,
                                action="prune_branch")

    def _record_failure(self, node: SearchNode, record) -> None:
        self._failure_counts[record.state_hash] = (
            self._failure_counts.get(record.state_hash, 0) + 1)
        self.history.append(EventType.FAILURE_RECORDED, type=record.failure_type.value,
                            node=node.node_id, stage=record.stage,
                            recovery=record.recovery.name)

    def _call_agent(self, agent, projection, *, source_node_id: str) -> tuple:
        self.metrics.agent_calls += 1
        ctx = {"source_node_id": source_node_id}
        proposals = agent.propose(projection, ctx)
        self.history.append(EventType.AGENT_CALLED, role=getattr(agent, "role", "?"))
        for p in proposals:
            self.history.append(EventType.PROPOSAL_GENERATED,
                                kind=type(p).__name__, agent=getattr(agent, "role", "?"),
                                id=getattr(p, "proposal_id", ""))
        self.metrics.token_estimate += sum(
            len(getattr(p, "rationale", "")) // 4 for p in proposals)
        return proposals

    def _call_agent_pair(self, agent, projection, *, source_node_id: str):
        self.metrics.agent_calls += 1
        ctx = {"source_node_id": source_node_id}
        critiques, attacks = agent.propose(projection, ctx)
        self.history.append(EventType.AGENT_CALLED, role=getattr(agent, "role", "?"))
        return critiques, attacks


def entry_guidance_floor(guidance: float) -> float:
    return guidance * 0.5


def _default_env():
    from forgemind.verification.environment import SubprocessExecutionEnvironment

    return SubprocessExecutionEnvironment()


@dataclass(frozen=True)
class SearchResult:
    termination: TerminationInfo
    graph: SearchGraph
    history: History
    metrics: SearchMetrics
    elapsed_seconds: float

    @property
    def accepted_node(self) -> SearchNode | None:
        if self.termination.accepted_node_id is None:
            return None
        return self.graph.nodes.get(self.termination.accepted_node_id)

    @property
    def accepted_state(self) -> CodebaseState | None:
        node = self.accepted_node
        return node.state if node else None

    @property
    def success(self) -> bool:
        return self.termination.reason == "goal_accepted"

    def summary(self) -> str:
        lines = [
            "SEARCH COMPLETE",
            f"outcome = {'ACCEPTED' if self.success else self.termination.reason}",
            f"nodes explored = {self.metrics.nodes_explored}",
            f"candidates = {self.metrics.candidates_created}",
            f"verifications = {self.metrics.verifications}",
            f"duplicates = {self.metrics.duplicates}",
            f"backtracks = {self.metrics.backtracks}",
            f"stagnation events = {self.metrics.stagnation_events}",
            f"max depth = {self.metrics.max_depth_reached}",
            f"elapsed = {self.elapsed_seconds:.2f}s",
        ]
        return "\n".join(lines)


# Engine-attached mutable session fields (explicit, not hidden globals).
SearchEngine.__init__.__defaults__  # keep linters calm
