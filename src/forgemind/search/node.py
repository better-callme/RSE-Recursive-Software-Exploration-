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
    QUARANTINED = "QUARANTINED"


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
    content_hash: str = ""

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
        # M4: seen and closed are different state-machine concepts.
        # SEEN = currently-live/claimed hashes (registered at creation,
        # released on every non-definitive terminal outcome). CLOSED =
        # hashes with a definitive, complete state verdict (terminal
        # for the run; only reverify_duplicates may re-verify them).
        self._seen_hashes: set[str] = set()
        self._closed_hashes: set[str] = set()
        self._insert_counter = itertools.count(1)

    @property
    def seen_hashes(self) -> frozenset[str]:
        return frozenset(self._seen_hashes)

    @property
    def closed_hashes(self) -> frozenset[str]:
        return frozenset(self._closed_hashes)

    def contains_state(self, content_hash: str) -> bool:
        """Closed-only: is this state decided (M4)?"""
        return content_hash in self._closed_hashes

    def is_closed(self, content_hash: str) -> bool:
        return content_hash in self._closed_hashes

    def remember_state(self, content_hash: str) -> bool:
        """Register a hash as seen. Returns False if already seen or
        closed (the caller distinguishes via is_closed)."""
        if (content_hash in self._seen_hashes
                or content_hash in self._closed_hashes):
            return False
        self._seen_hashes.add(content_hash)
        return True

    def close(self, content_hash: str) -> None:
        """Record a definitive, complete verdict for a hash."""
        self._seen_hashes.discard(content_hash)
        self._closed_hashes.add(content_hash)

    def reopen(self, content_hash: str) -> None:
        """Release a seen hash (non-definitive outcome: quarantine,
        depth-exhaustion, capacity-prune). Never un-closes."""
        self._seen_hashes.discard(content_hash)

    def insert(self, node: SearchNode, guidance: float) -> bool:
        """Insert node if its state is novel. Returns False for duplicates."""
        if not self.remember_state(node.state.content_hash):
            return False
        return self.admit(node, guidance)

    def admit(self, node: SearchNode, guidance: float) -> bool:
        """Place an already-registered node into the frontier.

        The engine registers every created state's hash for duplicate
        detection (remember_state), so insert() — which treats a
        registered hash as a duplicate — could never accept a search
        child: before H1 no child node had ever entered the frontier.
        admit() is the membership operation for those nodes; it returns
        False only if the node is already in the frontier."""
        if node.node_id in self._entries:
            return False
        entry = FrontierEntry(
            node_id=node.node_id, guidance=guidance, depth=node.depth,
            creation_index=node.creation_index,
            content_hash=node.state.content_hash,
        )
        self._entries[node.node_id] = entry
        self.prune_to_capacity()
        return True

    def remove(self, node_id: str) -> FrontierEntry | None:
        # Pure membership removal (selection, acceptance). Hash
        # effects are the engine's job: acceptance closes separately,
        # selection is not eviction.
        return self._entries.pop(node_id, None)

    def reinsert(self, node: SearchNode, guidance: float) -> bool:
        """Re-add a previously expanded node for a recovery re-attempt (C2).

        Bypasses the closed-hash novelty check in insert(): the state is
        known by design — recovery deliberately revisits it. Returns False
        if the node is already in the frontier.
        """
        if node.node_id in self._entries:
            return False
        self._entries[node.node_id] = FrontierEntry(
            node_id=node.node_id, guidance=guidance, depth=node.depth,
            creation_index=node.creation_index,
            content_hash=node.state.content_hash,
        )
        self.prune_to_capacity()
        return True

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
            # M4: eviction releases the seen claim (unless closed —
            # reopen never un-closes). A pruned incomplete-evidence
            # node must not linger as "seen" forever.
            self.reopen(e.content_hash)
        return tuple(dropped)

    def prune_guidance_below(self, threshold: float) -> tuple[FrontierEntry, ...]:
        drop = [e for e in self._entries.values() if e.guidance < threshold]
        for e in drop:
            del self._entries[e.node_id]
            self.reopen(e.content_hash)
        return tuple(drop)

    def __len__(self) -> int:
        return len(self._entries)


class SearchStrategy(Protocol):
    def select_next(
        self,
        frontier: SearchFrontier,
        graph: SearchGraph,
    ) -> SearchNode | None: ...

    def select_beam(
        self,
        frontier: SearchFrontier,
        graph: SearchGraph,
    ) -> tuple[SearchNode, ...]: ...


class BeamSearchStrategy:
    """Level-synchronous beam selection (H1). NOT A* — heuristic search.

    A beam round is up to `beam_width` nodes at the SHALLOWEST depth
    present in the frontier, in the frontier's deterministic sort order.
    Level discipline is the defining property: no deeper node is ever
    selected while a shallower one waits, no matter how high its
    guidance scores. (Before H1 this class selected a single global
    best node and beam_width was never read — greedy best-first wearing
    a beam label.)
    """

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

    def select_beam(
        self,
        frontier: SearchFrontier,
        graph: SearchGraph,
    ) -> tuple[SearchNode, ...]:
        """Remove and return one beam round (see class docstring)."""
        entries = frontier.top_k(len(frontier))
        if not entries:
            return ()
        min_depth = min(e.depth for e in entries)
        beam = [e for e in entries if e.depth == min_depth][: self.beam_width]
        nodes = []
        for entry in beam:
            frontier.remove(entry.node_id)
            nodes.append(graph.get(entry.node_id))
        return tuple(nodes)


def _raise_frozen_search(self: object, name: str, value: object) -> None:
    raise TypeError(f"{type(self).__name__} is immutable ({name})")


for _cls in (SearchNode, FrontierEntry):
    _cls.__setattr__ = _raise_frozen_search  # type: ignore[method-assign]
