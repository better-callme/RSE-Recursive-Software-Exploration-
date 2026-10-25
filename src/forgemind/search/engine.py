"""SearchEngine: the deterministic loop that integrates everything.

Agents propose; this engine alone decides. See spec §57.
"""

from __future__ import annotations

import random
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
    IdAllocator,
    Proposal,
    _current_allocator,
)
from forgemind.core.state import CodebaseState
from forgemind.core.tokens import estimate_tokens_for_proposals
from forgemind.core.transition import TransitionEngine
from forgemind.recovery.failure import (
    FailureRecord,
    FailureType,
    RecoveryAction,
    RecoveryPolicy,
)
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
from forgemind.verification.results import (
    VerificationOutcome,
    VerificationResult,
    classify_verification,
)


@dataclass(frozen=True)
class TerminationInfo:
    reason: str  # goal_accepted | frontier_empty | quarantined | max_depth |
    #              max_nodes | max_verifications | max_agent_calls |
    #              wall_clock | fatal
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
        # M6: one stagnation detector per branch (root-child lineage),
        # never a single global window — independent branches must not
        # share a window. Branch keys are tracked in _branch_keys.
        self._stagnation_detectors: dict[str, StagnationDetector] = {}
        self._branch_keys: dict[str, str] = {}
        self.metrics = SearchMetrics()
        self._attack_tests: list[AttackTestProposal] = []
        # M7: the attack set each node was verified against, keyed by
        # node id. The cached verdict belongs to that specific node, and
        # reuse requires the set to be identical — see _attack_set().
        self._verify_attack_sets: dict[str, frozenset[tuple[str, str]]] = {}
        self._critiques: tuple[CritiqueProposal, ...] = ()
        self._architecture = None
        self._failure_counts: dict[str, int] = {}
        # C2 recovery state: every recorded failure, kept per node and in
        # global record order, so failures can be fed back to the builder
        # and recovery actions can be enacted instead of only logged.
        self._failures_by_node: dict[str, list[FailureRecord]] = {}
        self._all_records: list[FailureRecord] = []
        self._repairs_used: dict[str, int] = {}
        self._repair_mode: set[str] = set()
        self._accepted: SearchNode | None = None
        self._regression_tests: dict[str, str] = {}
        # H4: engine-local RNG (never the global random module) and the
        # per-run quarantine set (nodes set aside for slowness, not failure).
        self._rng = random.Random(self.config.random_seed)
        self._quarantined: list[QuarantinedRecord] = []

    # ------------------------------------------------------------------
    def run(self) -> "SearchResult":
        # H4: fresh per-run ID allocator, seed-derived start (seed 0
        # preserves the legacy sequence). Installed for this run only and
        # reset in finally; ContextVar is per-thread, so concurrent
        # engines never interleave IDs.
        allocator = IdAllocator(start=self.config.random_seed % 10**8)
        token = _current_allocator.set(allocator)
        try:
            return self._run_once()
        finally:
            _current_allocator.reset(token)

    def _run_once(self) -> "SearchResult":
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
            quarantined=tuple(self._quarantined),
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
            return self._empty_frontier_termination()
        return self._hard_budget_exceeded(started)

    def _hard_budget_exceeded(self, started: float) -> TerminationInfo | None:
        """Budget caps only — no acceptance/frontier state. Safe to check
        mid-round, when beam members are already out of the frontier and
        an empty frontier does not mean the search is done."""
        m = self.metrics
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

            # H1: expand in beam rounds — up to beam_width nodes at the
            # shallowest frontier depth before any deeper node is touched.
            beam = self.strategy.select_beam(self.frontier, self.graph)
            if not beam:
                return self._empty_frontier_termination()

            for node in beam:
                if self._accepted is not None:
                    return TerminationInfo("goal_accepted",
                                           self._accepted.node_id)
                exhausted = self._hard_budget_exceeded(started)
                if exhausted is not None:
                    return exhausted

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
                    # M4: depth-exhaustion is a path/budget outcome, not
                    # a state verdict — release the seen claim (never
                    # un-closes an already-closed hash).
                    self.frontier.reopen(node.state.content_hash)
                    continue

                self._expand_node(node)

    def _expand_node(self, node: SearchNode) -> None:
        records_before = len(self._all_records)
        # Architect once at root.
        if self._architecture is None:
            arch_proj = build_architect_projection(self.spec, node.state)
            proposals = self._call_agent(self.architect, arch_proj,
                                         source_node_id=node.node_id)
            if proposals:
                self._architecture = proposals[0]

        failures = self._failures_for(node)
        builder_proj = build_builder_projection(
            self.spec, node.state, architecture=self._architecture,
            failures=failures,
            critiques=self._critiques, attempt_number=len(self._critiques),
        )
        if node.node_id in self._repair_mode:
            # C2: a previous expansion's failures demanded repair (or
            # critic analysis, which already ran inline); this re-expansion
            # is the builder's repair turn, evidence attached. Builders
            # without repair() fall back to propose() — the projection
            # still carries prior_failures, so the retry is informed.
            self._repair_mode.discard(node.node_id)
            evidence = "; ".join(r.evidence for r in failures)[:2000]
            patch_proposals = self._call_repair(
                builder_proj, evidence, source_node_id=node.node_id)
            if patch_proposals is None:
                patch_proposals = self._call_agent(
                    self.builder, builder_proj, source_node_id=node.node_id)
        else:
            patch_proposals = self._call_agent(
                self.builder, builder_proj, source_node_id=node.node_id)

        inserted_any = False
        best_child_guidance = 0.0
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

            # Duplicate detection via content hash (Invariant 6, M4):
            # seen = live (always skip); closed = decided (skip unless
            # reverify_duplicates opts into redundant re-verification).
            if not self.frontier.remember_state(new_state.content_hash):
                if not (self.frontier.is_closed(new_state.content_hash)
                        and self.config.reverify_duplicates):
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
                # else: reverify_duplicates bypasses a CLOSED hash —
                # fall through and create + verify again.

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
            # H4: timeout is inconclusive, not failure. Retry once with a
            # fresh subprocess (a new referee.verify() call); a second
            # timeout quarantines the child — no gate, no failure record,
            # no recovery action.
            if verification.timed_out:
                verification = self._verify(child)
                if verification.timed_out:
                    self._quarantine(child, verification)
                    continue
            gate = hard_gate(verification)
            repeated = self._failure_counts.get(child.state.content_hash, 0)
            scores = self.scorer.score(verification, gate, depth=child.depth,
                                       repeated_failures=repeated)
            best_child_guidance = max(best_child_guidance, scores.guidance)
            child = child.with_updates(
                verification=verification,
                scores=scores,
                # M5: gate pass is not acceptance — the adversarial
                # gate below still has to pass, and only the accept
                # path may grant ACCEPTED.
                status=(NodeStatus.GATE_PASSED if gate.passed else NodeStatus.REJECTED),
            )
            self.graph.update(child)

            if gate.passed and self._adversarial_gate(child):
                self.history.append(EventType.STATE_ACCEPTED, node=child.node_id,
                                    hash=child.state.content_hash[:12],
                                    score=scores.total)
                # Optimizer round on the verified solution (regression-checked).
                original = child
                if self.config.enable_optimizer:
                    child = self._optimize(child)
                self.frontier.remove(child.node_id)
                # M5: this promotion is the SOLE place a search child
                # becomes ACCEPTED (final winner only).
                self.graph.update(child.with_updates(status=NodeStatus.ACCEPTED))
                self._accepted = child
                # M4: acceptance is definitive (passing all gates
                # requires complete evidence) — close the final
                # accepted node's hash, plus the original's when the
                # optimizer replaced it (it too was decided).
                self.frontier.close(child.state.content_hash)
                if child is not original:
                    self.frontier.close(original.state.content_hash)
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
                # M4: close only on an authoritative, complete verdict;
                # incomplete evidence (crash/timeout) reopens instead.
                if verification.evidence_complete:
                    self.frontier.close(child.state.content_hash)
                else:
                    self.frontier.reopen(child.state.content_hash)
                action = self.recovery.action_for(ftype)
                if action in (RecoveryAction.CRITIC_ANALYSIS, RecoveryAction.BUILDER_REPAIR):
                    self._run_critic(child, verification)
                # Keep failed-but-informative states out of the frontier;
                # repair happens from the parent with evidence attached.
            else:
                # admit(), not insert(): the child's hash was registered
                # at creation for duplicate detection, and insert()
                # treats a registered hash as a duplicate (H1 finding —
                # before this, no child ever entered the frontier).
                inserted = self.frontier.admit(child, guidance=scores.guidance)
                inserted_any = inserted_any or inserted
                if inserted:
                    self.history.append(EventType.NODE_INSERTED, node=child.node_id,
                                        guidance=round(scores.guidance, 6))
                # M6: the child's branch is its parent's branch, or a new
                # branch (itself) when the parent is the root, which was
                # never admitted and so has no key.
                self._branch_keys[child.node_id] = self._branch_keys.get(
                    node.node_id, child.node_id)
                self._check_stagnation(child, scores.guidance)

        # No acceptance (that path returns above). If this expansion also
        # left nothing in the frontier, the branch would die here — check
        # whether recovery entitles the node to another expansion (C2).
        self._schedule_recovery(node, records_before,
                                inserted_any=inserted_any,
                                guidance=best_child_guidance)

    # ------------------------------------------------------------------
    # C2: failure-driven recovery, enacted.
    #
    # _REEXPAND_ACTIONS justify giving a node another expansion when its
    # branch would otherwise die. PRUNE / TERMINATE_BRANCH / NONE never
    # do: a duplicate-only expansion, for instance, carries no new
    # information and must not burn repair attempts.
    _REEXPAND_ACTIONS = frozenset({
        RecoveryAction.BUILDER_RETRY, RecoveryAction.BUILDER_REPAIR,
        RecoveryAction.CRITIC_ANALYSIS,
        RecoveryAction.ALTERNATIVE_IMPLEMENTATION, RecoveryAction.BACKTRACK,
    })
    # Actions whose re-expansion is the builder's repair turn (the critic
    # has already run inline for CRITIC_ANALYSIS, as before).
    _REPAIR_ACTIONS = frozenset({
        RecoveryAction.BUILDER_REPAIR, RecoveryAction.CRITIC_ANALYSIS,
    })

    def _schedule_recovery(self, node: SearchNode, records_before: int,
                           *, inserted_any: bool, guidance: float) -> None:
        if inserted_any:
            return  # a live continuation exists; recovery is per-branch
        records = self._all_records[records_before:]
        if not records:
            return
        actions = {r.recovery for r in records}
        if not actions & self._REEXPAND_ACTIONS:
            return
        used = self._repairs_used.get(node.node_id, 0)
        if used >= self.config.max_repair_attempts_per_node:
            return  # cap enforced: config.max_repair_attempts_per_node
        self._repairs_used[node.node_id] = used + 1
        accumulated = {r.recovery for r in self._failures_for(node)}
        if accumulated & self._REPAIR_ACTIONS:
            self._repair_mode.add(node.node_id)
        if self.frontier.reinsert(node, guidance):
            self.metrics.backtracks += 1
            self.history.append(EventType.BACKTRACK, node=node.node_id,
                                attempt=used + 1)
            self.history.append(
                EventType.RECOVERY_ACTION, node=node.node_id,
                actions=",".join(sorted(a.name for a in actions)))

    def _call_repair(self, projection, evidence: str, *,
                     source_node_id: str) -> tuple | None:
        """Invoke builder.repair() with _call_agent's bookkeeping.

        Returns None when the builder offers no repair(); the caller
        falls back to a propose() retry whose projection still carries
        prior_failures.
        """
        repair = getattr(self.builder, "repair", None)
        if repair is None:
            return None
        self.metrics.agent_calls += 1
        proposals = repair(projection, evidence)
        self.history.append(EventType.AGENT_CALLED,
                            role=getattr(self.builder, "role", "?"),
                            method="repair")
        for p in proposals:
            self.history.append(EventType.PROPOSAL_GENERATED,
                                kind=type(p).__name__,
                                agent=getattr(self.builder, "role", "?"),
                                id=getattr(p, "proposal_id", ""))
        self.metrics.token_estimate += estimate_tokens_for_proposals(proposals)
        return proposals

    # ------------------------------------------------------------------
    def _quarantine(self, child: SearchNode,
                    verification: VerificationResult) -> None:
        """Set a twice-timed-out child aside (H4). Not accepted, not a
        failure: no FailureRecord, no recovery action, never admitted to
        the frontier — so C2 recovery cannot reinsert its parent on its
        behalf. The quarantined NODE is never re-admitted within the
        run (re-admission only under a new engine instance, whose
        frontier starts fresh). M4: the HASH is not closed — a timeout
        is not a state verdict — so a future NEW node with the same
        hash may be verified again; loop prevention comes from the
        per-node quarantine plus run budgets, not hash closure."""
        outcome = classify_verification(verification, retries_exhausted=True)
        assert outcome is VerificationOutcome.QUARANTINED
        self.frontier.reopen(child.state.content_hash)
        self._quarantined.append(QuarantinedRecord(
            node_id=child.node_id, state_hash=child.state.content_hash,
            attempts=2))
        self.graph.update(child.with_updates(
            verification=verification, status=NodeStatus.QUARANTINED))
        self.history.append(EventType.QUARANTINED, node=child.node_id,
                            hash=child.state.content_hash[:12],
                            attempts=2, outcome=outcome.value)

    def _empty_frontier_termination(self) -> TerminationInfo:
        """Frontier empty: honest reason — quarantined when nodes were set
        aside for slowness rather than rejected, else frontier_empty."""
        if self._quarantined:
            return TerminationInfo("quarantined")
        return TerminationInfo("frontier_empty")

    def _attack_set(self) -> frozenset[tuple[str, str]]:
        """M7: the attack-set version — name + code is the execution
        identity of an attack. Any addition, removal, or content change
        yields a different set, so reuse below is exact, not heuristic."""
        return frozenset((a.test_name, a.test_code)
                         for a in self._attack_tests)

    def _verify(self, node: SearchNode) -> VerificationResult:
        self.history.append(EventType.VERIFICATION_STARTED, node=node.node_id)
        self.metrics.verifications += 1
        result = self.referee.verify(
            node.state, attack_tests=self._attack_tests,
            regression_tests=self._regression_tests,
        )
        # M7: record which attack set this result was produced against,
        # so the gate can tell reuse from invalidation.
        self._verify_attack_sets[node.node_id] = self._attack_set()
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
        # M7: reuse the stored verification's adversarial verdict only
        # when the attack set is identical to the set it was verified
        # against. The critic may have added attacks since _verify ran
        # (the no-attacks-yet case), in which case re-verification is
        # genuine and required. A missing or timed-out stored result
        # also falls back to re-verification (defensive; timeouts
        # quarantine before the gate, so this is unreachable via run()).
        verification = node.verification
        if (verification is not None and not verification.timed_out
                and self._verify_attack_sets.get(node.node_id)
                == self._attack_set()):
            ok = verification.adversarial_passed
            violations = verification.violations
            complete = verification.evidence_complete
        else:
            self.metrics.verifications += 1
            adv_result = self.referee.verify(node.state,
                                             attack_tests=self._attack_tests)
            ok = adv_result.adversarial_passed
            violations = adv_result.violations
            complete = adv_result.evidence_complete
        if not ok:
            rec = self.recovery.record(
                state_hash=node.state.content_hash,
                candidate_id=node.candidate_id or "",
                failure_type=FailureType.ADVERSARIAL_FAILURE,
                stage="adversarial",
                evidence="; ".join(violations)[:500],
            )
            self._record_failure(node, rec)
            # M4: this site settles its own verdict — close iff the
            # verdict's evidence was complete, else the hash stays seen
            # (live node in the frontier).
            if complete:
                self.frontier.close(node.state.content_hash)
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
        # C3: _verify ran the attack tests too; the gate must honor their
        # verdict, or an adversarially broken "optimization" would replace
        # the verified artifact as the accepted state.
        gate = hard_gate(verification,
                         adversarial_expected=bool(self._attack_tests))
        scores = self.scorer.score(verification, gate, depth=opt_node.depth)
        opt_node = opt_node.with_updates(
            verification=verification, scores=scores,
            # M5: same rule as _expand_node — gate pass is
            # GATE_PASSED; the caller promotes the delivered node.
            status=NodeStatus.GATE_PASSED if gate.passed else NodeStatus.REJECTED,
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
        # M6: `node` here is the admitted child. Its window belongs to
        # its branch, not to the run — independent branches never share
        # a window. Fallback is the child's own node_id (an own branch).
        branch = self._branch_keys.get(node.node_id, node.node_id)
        detector = self._stagnation_detectors.get(branch)
        if detector is None:
            detector = StagnationDetector(
                window=self.config.stagnation_window,
                epsilon=self.config.stagnation_epsilon,
            )
            self._stagnation_detectors[branch] = detector
        if detector.record(guidance=guidance,
                           content_hash=node.state.content_hash):
            self.metrics.stagnation_events += 1
            self.history.append(EventType.STAGNATION_DETECTED, node=node.node_id)
            rec = self.recovery.record(
                state_hash=node.state.content_hash, candidate_id="",
                failure_type=FailureType.STAGNATION, stage="search",
                evidence="no measurable guidance improvement / oscillation",
            )
            self._record_failure(node, rec)
            # M6: the response is record + prune weaker candidates
            # (guidance floor), NOT "terminate the stagnant branch":
            # the stagnant node itself survives the floor, and branch
            # termination is a separate search-policy decision.
            for entry in self.frontier.prune_guidance_below(entry_guidance_floor(guidance)):
                self.metrics.pruned += 1
                self.history.append(EventType.NODE_PRUNED, node=entry.node_id,
                                    reason="stagnation")
            # M6: no backtrack here — a prune is not a revisit.
            # backtracks counts reinsert revisits only.

    def _record_failure(self, node: SearchNode, record) -> None:
        self._failure_counts[record.state_hash] = (
            self._failure_counts.get(record.state_hash, 0) + 1)
        self._failures_by_node.setdefault(node.node_id, []).append(record)
        self._all_records.append(record)
        self.history.append(EventType.FAILURE_RECORDED, type=record.failure_type.value,
                            node=node.node_id, stage=record.stage,
                            recovery=record.recovery.name)

    def _failures_for(self, node: SearchNode) -> tuple[FailureRecord, ...]:
        """Failure evidence relevant to expanding `node`: records kept
        against the node itself plus those kept against its children
        (verification failures are recorded against the child, transition
        failures against the parent). Deterministic record order."""
        records = list(self._failures_by_node.get(node.node_id, ()))
        for child_id in self.graph.children.get(node.node_id, ()):
            records.extend(self._failures_by_node.get(child_id, ()))
        return tuple(records)

    def _call_agent(self, agent, projection, *, source_node_id: str) -> tuple:
        self.metrics.agent_calls += 1
        ctx = {"source_node_id": source_node_id}
        proposals = agent.propose(projection, ctx)
        self.history.append(EventType.AGENT_CALLED, role=getattr(agent, "role", "?"))
        for p in proposals:
            self.history.append(EventType.PROPOSAL_GENERATED,
                                kind=type(p).__name__, agent=getattr(agent, "role", "?"),
                                id=getattr(p, "proposal_id", ""))
        self.metrics.token_estimate += estimate_tokens_for_proposals(proposals)
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
class QuarantinedRecord:
    """A node set aside for slowness, not failure (H4). attempts is the
    number of timed-out verification attempts (1 initial + 1 retry)."""

    node_id: str
    state_hash: str
    attempts: int = 2


@dataclass(frozen=True)
class SearchResult:
    termination: TerminationInfo
    graph: SearchGraph
    history: History
    metrics: SearchMetrics
    elapsed_seconds: float
    quarantined: tuple[QuarantinedRecord, ...] = ()

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

