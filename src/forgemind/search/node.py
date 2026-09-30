"""Search nodes, graph, frontier, strategy."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol

from forgemind.core.state import CodebaseState
from forgemind.verification.results import VerificationResult
from forgemind.verification.results import ComponentScores


class NodeStatus(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    VALID = "VALID"
    REJECTED = "REJECTED"
    ACCEPTED = "ACCEPTED"
    EXHAUSTED = "EXHAUSTED"
    CYCLE_DEAD_END = "CYCLE_DEAD_END"
    STAGNATED = "STAGNATED"


@dataclass(frozen=True)
class SearchNode:
    node_id: str
    state: CodebaseState
    parent_node_id: str | None  # None only for root
    candidate_id: str | None
    depth: int
    creation_index: int
    status: NodeStatus = NodeStatus.UNVERIFIED
    verification: VerificationResult | None = None
    scores: ComponentScores | None = None
    failure_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.parent_node_id is None and self.depth != 0:
            raise ValueError("only the root node may lack ancestry")
        if self.parent_node_id is not None and not self.parent_node_id:
            raise ValueError("non-root node requires valid ancestry")

    def with_updates(self, **changes: object) -> "SearchNode":
        payload = dict(
            node_id=self.node_id, state=self.state,
            parent_node_id=self.parent_node_id, candidate_id=self.candidate_id,
            depth=self.depth, creation_index=self.creation_index,
            status=self.status, verification=self.verification,
            scores=self.scores, failure_types=self.failure_types,
        )
        payload.update(changes)
        return SearchNode(**payload)  # type: ignore[arg-type]


class SearchGraph:
    """Nodes + edges; distinct nodes may share identical content hashes.

    Node IDs are allocated from a PER-GRAPH counter so identical experiments
    produce identical traces (no cross-run/global state).
    """

    def __init__(self) -> None:
        self.nodes: dict[str, SearchNode] = {}
        self.children: dict[str, list[str]] = {}
        self._ids = itertools.count(1)

    def next_node_id(self) -> str:
        return f"node-{next(self._ids):06d}"

    def add(self, node: SearchNode) -> SearchNode:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node id {node.node_id}")
        if node.parent_node_id is not None and node.parent_node_id not in self.nodes:
            raise ValueError(
                f"node {node.node_id} references unknown parent {node.parent_node_id}")
        self.nodes[node.node_id] = node
        if node.parent_node_id is not None:
            self.children.setdefault(node.parent_node_id, []).append(node.node_id)
        return node

    def update(self, node: SearchNode) -> None:
        if node.node_id not in self.nodes:
            raise ValueError("cannot update missing node")
        self.nodes[node.node_id] = node

    def get(self, node_id: str) -> SearchNode:
        return self.nodes[node_id]

    def ancestry(self, node_id: str) -> tuple[SearchNode, ...]:
        chain: list[SearchNode] = []
        current: SearchNode | None = self.nodes.get(node_id)
        guard = 0
        while current is not None and guard < 10_000:
            chain.append(current)
            pid = current.parent_node_id
            current = self.nodes.get(pid) if pid else None
            guard += 1
        return tuple(reversed(chain))

    def __len__(self) -> int:
        return len(self.nodes)


@dataclass(frozen=True)
class FrontierEntry:
    node_id: str
    guidance: float
    depth: int
    creation_index: int

    def sort_key(self) -> tuple[float, int, int, str]:
        # Deterministic tie-breaking: score desc, depth asc, insertion order, id.
        return (-self.guidance, self.depth, self.creation_index, self.node_id)


class SearchFrontier:
    """Bounded deterministic priority frontier with dedup by content hash."""

    def __init__(self, capacity: int = 64, beam_width: int = 3) -> None:
        if capacity < 1 or beam_width < 1:
            raise ValueError("frontier capacity/beam must be >= 1")
        self.capacity = capacity
        self.beam_width = beam_width
        self._entries: dict[str, FrontierEntry] = {}
        self._closed_hashes: set[str] = set()
        self._insert_counter = itertools.count(1)

    @property
    def closed_hashes(self) -> frozenset[str]:
        return frozenset(self._closed_hashes)

    def contains_state(self, content_hash: str) -> bool:
        return content_hash in self._closed_hashes

    def remember_state(self, content_hash: str) -> bool:
        """Register a hash as seen. Returns False if already present."""
        if content_hash in self._closed_hashes:
            return False
        self._closed_hashes.add(content_hash)
        return True

    def insert(self, node: SearchNode, guidance: float) -> bool:
        """Insert node if its state is novel. Returns False for duplicates."""
        if not self.remember_state(node.state.content_hash):
            return False
        entry = FrontierEntry(
            node_id=node.node_id, guidance=guidance, depth=node.depth,
            creation_index=node.creation_index,
        )
        self._entries[node.node_id] = entry
        self.prune_to_capacity()
        return True

    def remove(self, node_id: str) -> FrontierEntry | None:
        return self._entries.pop(node_id, None)

    def select(self) -> FrontierEntry | None:
        """Deterministically pick the best entry without removing it."""
        entries = sorted(self._entries.values(), key=FrontierEntry.sort_key)
        return entries[0] if entries else None

    def peek_best(self) -> FrontierEntry | None:
        entries = sorted(self._entries.values(), key=FrontierEntry.sort_key)
        return entries[0] if entries else None

    def top_k(self, k: int) -> tuple[FrontierEntry, ...]:
        return tuple(sorted(self._entries.values(), key=FrontierEntry.sort_key)[:k])

    def prune_to_capacity(self) -> tuple[FrontierEntry, ...]:
        ordered = sorted(self._entries.values(), key=FrontierEntry.sort_key)
        keep, dropped = ordered[: self.capacity], ordered[self.capacity:]
        for e in dropped:
            del self._entries[e.node_id]
        return tuple(dropped)

    def prune_guidance_below(self, threshold: float) -> tuple[FrontierEntry, ...]:
        drop = [e for e in self._entries.values() if e.guidance < threshold]
        for e in drop:
            del self._entries[e.node_id]
        return tuple(drop)

    def __len__(self) -> int:
        return len(self._entries)


class SearchStrategy(Protocol):
    def select_next(
        self,
        frontier: SearchFrontier,
        graph: SearchGraph,
    ) -> SearchNode | None: ...


class BeamSearchStrategy:
    """Bounded best-first beam selection. NOT A* — heuristic bounded search."""

    def __init__(self, beam_width: int = 3) -> None:
        self.beam_width = max(1, beam_width)

    def select_next(
        self,
        frontier: SearchFrontier,
        graph: SearchGraph,
    ) -> SearchNode | None:
        best = frontier.peek_best()
        if best is None:
            return None
        node = graph.get(best.node_id)
        frontier.remove(best.node_id)
        return node


def _raise_frozen_search(self: object, name: str, value: object) -> None:
    raise TypeError(f"{type(self).__name__} is immutable ({name})")


for _cls in (SearchNode, FrontierEntry):
    _cls.__setattr__ = _raise_frozen_search  # type: ignore[method-assign]
