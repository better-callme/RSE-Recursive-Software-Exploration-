"""Append-only event history."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class EventType(str, Enum):
    SEARCH_STARTED = "SEARCH_STARTED"
    NODE_SELECTED = "NODE_SELECTED"
    AGENT_CALLED = "AGENT_CALLED"
    PROPOSAL_GENERATED = "PROPOSAL_GENERATED"
    CANDIDATE_CREATED = "CANDIDATE_CREATED"
    TRANSITION_APPLIED = "TRANSITION_APPLIED"
    TRANSITION_REJECTED = "TRANSITION_REJECTED"
    STATE_HASHED = "STATE_HASHED"
    DUPLICATE_DETECTED = "DUPLICATE_DETECTED"
    VERIFICATION_STARTED = "VERIFICATION_STARTED"
    VERIFICATION_COMPLETED = "VERIFICATION_COMPLETED"
    FAILURE_RECORDED = "FAILURE_RECORDED"
    NODE_INSERTED = "NODE_INSERTED"
    NODE_PRUNED = "NODE_PRUNED"
    STAGNATION_DETECTED = "STAGNATION_DETECTED"
    BACKTRACK = "BACKTRACK"
    RECOVERY_ACTION = "RECOVERY_ACTION"
    STATE_ACCEPTED = "STATE_ACCEPTED"
    SEARCH_TERMINATED = "SEARCH_TERMINATED"


@dataclass(frozen=True)
class HistoryEvent:
    seq: int
    event_type: EventType
    timestamp: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class History:
    """Append-only from the engine's perspective; thread-safe.

    Timestamps are recorded but excluded from determinism-sensitive comparisons.
    """

    def __init__(self) -> None:
        self._events: list[HistoryEvent] = []
        self._lock = threading.Lock()

    def append(self, event_type: EventType, **metadata: Any) -> HistoryEvent:
        with self._lock:
            event = HistoryEvent(
                seq=len(self._events),
                event_type=event_type,
                timestamp=time.time(),
                metadata=dict(metadata),
            )
            self._events.append(event)
            return event

    @property
    def events(self) -> tuple[HistoryEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def count(self, event_type: EventType) -> int:
        with self._lock:
            return sum(1 for e in self._events if e.event_type is event_type)

    def of_type(self, event_type: EventType) -> tuple[HistoryEvent, ...]:
        with self._lock:
            return tuple(e for e in self._events if e.event_type is event_type)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def render_trace(self) -> str:
        lines = []
        for e in self.events:
            meta = ", ".join(f"{k}={v}" for k, v in sorted(e.metadata.items()))
            lines.append(f"[{e.seq:04d}] {e.event_type.value} {meta}")
        return "\n".join(lines)
