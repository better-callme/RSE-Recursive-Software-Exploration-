"""Failure records and recovery policy mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Mapping


class FailureType(str, Enum):
    CONTRACT_VIOLATION = "CONTRACT_VIOLATION"
    INVALID_PROPOSAL = "INVALID_PROPOSAL"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    IMPORT_ERROR = "IMPORT_ERROR"
    TEST_FAILURE = "TEST_FAILURE"
    ADVERSARIAL_FAILURE = "ADVERSARIAL_FAILURE"
    REGRESSION_FAILURE = "REGRESSION_FAILURE"
    TIMEOUT = "TIMEOUT"
    MEMORY_LIMIT = "MEMORY_LIMIT"
    PROCESS_LIMIT = "PROCESS_LIMIT"
    DUPLICATE_STATE = "DUPLICATE_STATE"
    CYCLE = "CYCLE"
    STAGNATION = "STAGNATION"
    DEPTH_EXHAUSTED = "DEPTH_EXHAUSTED"
    SEARCH_EXHAUSTED = "SEARCH_EXHAUSTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class RecoveryAction(Enum):
    BUILDER_RETRY = auto()
    CRITIC_ANALYSIS = auto()
    BUILDER_REPAIR = auto()
    ALTERNATIVE_IMPLEMENTATION = auto()
    REARCHITECT = auto()
    PRUNE = auto()
    BACKTRACK = auto()
    CHANGE_STRATEGY = auto()
    TERMINATE_BRANCH = auto()
    NONE = auto()


DEFAULT_RECOVERY_MAP: Mapping[FailureType, RecoveryAction] = {
    FailureType.CONTRACT_VIOLATION: RecoveryAction.PRUNE,
    FailureType.INVALID_PROPOSAL: RecoveryAction.BUILDER_RETRY,
    FailureType.SYNTAX_ERROR: RecoveryAction.BUILDER_RETRY,
    FailureType.IMPORT_ERROR: RecoveryAction.BUILDER_RETRY,
    FailureType.TEST_FAILURE: RecoveryAction.CRITIC_ANALYSIS,
    FailureType.ADVERSARIAL_FAILURE: RecoveryAction.BUILDER_REPAIR,
    FailureType.REGRESSION_FAILURE: RecoveryAction.BACKTRACK,
    FailureType.TIMEOUT: RecoveryAction.ALTERNATIVE_IMPLEMENTATION,
    FailureType.MEMORY_LIMIT: RecoveryAction.ALTERNATIVE_IMPLEMENTATION,
    FailureType.PROCESS_LIMIT: RecoveryAction.PRUNE,
    FailureType.DUPLICATE_STATE: RecoveryAction.PRUNE,
    FailureType.CYCLE: RecoveryAction.PRUNE,
    FailureType.STAGNATION: RecoveryAction.TERMINATE_BRANCH,
    FailureType.DEPTH_EXHAUSTED: RecoveryAction.BACKTRACK,
    FailureType.SEARCH_EXHAUSTED: RecoveryAction.NONE,
    FailureType.INTERNAL_ERROR: RecoveryAction.NONE,
}


@dataclass(frozen=True)
class FailureRecord:
    state_hash: str
    candidate_id: str
    failure_type: FailureType
    stage: str  # transition | static | dynamic | adversarial | search
    evidence: str
    severity: str = "high"  # low|medium|high|critical
    recovery: RecoveryAction = RecoveryAction.NONE
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


class RecoveryPolicy:
    """Deterministic R: FailureType -> RecoveryAction, overridable per type."""

    def __init__(
        self,
        overrides: Mapping[FailureType, RecoveryAction] | None = None,
    ) -> None:
        self._map: dict[FailureType, RecoveryAction] = dict(DEFAULT_RECOVERY_MAP)
        if overrides:
            self._map.update(dict(overrides))

    def action_for(self, failure_type: FailureType) -> RecoveryAction:
        return self._map[failure_type]

    def record(
        self,
        *,
        state_hash: str,
        candidate_id: str,
        failure_type: FailureType,
        stage: str,
        evidence: str,
        severity: str = "high",
        **metadata: object,
    ) -> FailureRecord:
        return FailureRecord(
            state_hash=state_hash,
            candidate_id=candidate_id,
            failure_type=failure_type,
            stage=stage,
            evidence=evidence,
            severity=severity,
            recovery=self.action_for(failure_type),
            metadata=metadata,
        )
