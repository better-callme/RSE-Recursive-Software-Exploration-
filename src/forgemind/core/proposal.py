"""Typed proposals and candidates. Agents emit Proposals; the engine binds
them into Candidates; only TransitionEngine turns Candidates into new states."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Mapping

_id_source = [0]  # process-wide monotonic source; resettable for reproducibility


def reset_id_counter(start: int = 0) -> None:
    """Reset proposal/candidate ID allocation (for reproducible experiments)."""
    _id_source[0] = start


def _next_id(prefix: str) -> str:
    _id_source[0] += 1
    return f"{prefix}-{_id_source[0]:08d}"


@dataclass(frozen=True)
class FileEdit:
    """A single file operation proposed by an agent."""

    path: str
    content: str | None  # None for deletion


@dataclass(frozen=True)
class ProposalBase:
    """Shared immutable proposal metadata. Subclass via composition helpers;
    kept as a dataclass base for field reuse."""

    proposal_id: str = field(default_factory=lambda: _next_id("prop"))
    source_node_id: str = ""
    agent_role: str = "unknown"
    intent: str = ""
    rationale: str = ""
    estimated_cost: float = 0.0

    def __post_init__(self) -> None:
        if not self.proposal_id:
            raise ValueError("proposal_id must be non-empty")


@dataclass(frozen=True)
class ArchitectureProposal(ProposalBase):
    components: tuple[str, ...] = ()
    interfaces: Mapping[str, str] = field(default_factory=dict)
    dependency_graph: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    implementation_steps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.components:
            raise ValueError("architecture proposal requires components")
        object.__setattr__(self, "interfaces", dict(self.interfaces))


@dataclass(frozen=True)
class CodePatchProposal(ProposalBase):
    created_files: Mapping[str, str] = field(default_factory=dict)
    modified_files: Mapping[str, str] = field(default_factory=dict)
    deleted_files: tuple[str, ...] = ()
    dependency_changes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "created_files", dict(self.created_files))
        object.__setattr__(self, "modified_files", dict(self.modified_files))
        object.__setattr__(self, "dependency_changes", dict(self.dependency_changes))
        if not (self.created_files or self.modified_files or self.deleted_files):
            raise ValueError("code patch must touch at least one file")
        overlap = set(self.created_files) & set(self.modified_files)
        if overlap:
            raise ValueError(f"files both created and modified: {sorted(overlap)}")

    @property
    def is_empty(self) -> bool:
        return not (
            self.created_files or self.modified_files or self.deleted_files
            or self.dependency_changes
        )


@dataclass(frozen=True)
class Issue:
    location: str
    severity: str  # "low" | "medium" | "high" | "critical"
    description: str
    evidence: str


@dataclass(frozen=True)
class CritiqueProposal(ProposalBase):
    issues: tuple[Issue, ...] = ()
    suggested_repair_directions: tuple[str, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True)
class AttackTestProposal(ProposalBase):
    test_name: str = ""
    test_code: str = ""
    hypothesis: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.test_name or not self.test_code:
            raise ValueError("attack test requires a name and code")


@dataclass(frozen=True)
class OptimizationProposal(ProposalBase):
    patch: CodePatchProposal | None = None
    target_metric: str = "runtime"
    expected_improvement: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.patch is None:
            raise ValueError("optimization proposal requires a patch")


Proposal = (
    ArchitectureProposal
    | CodePatchProposal
    | CritiqueProposal
    | AttackTestProposal
    | OptimizationProposal
)


@dataclass(frozen=True)
class Candidate:
    """Bind(parent node, proposal). Not automatically valid."""

    candidate_id: str = field(default_factory=lambda: _next_id("cand"))
    parent_node_id: str = ""
    proposal: Proposal = field(default_factory=lambda: CodePatchProposal(
        proposal_id=_next_id("prop"), agent_role="none",
    ))
    estimated_cost: float = 0.0
    generation_index: int = 0

    def __post_init__(self) -> None:
        if not self.parent_node_id:
            raise ValueError("candidate must reference a parent node id")


# Keep frozen semantics strict.
def _raise_frozen_proposal(self: object, name: str, value: object) -> None:
    raise dataclasses.FrozenInstanceError(
        f"cannot modify immutable {type(self).__name__}.{name}"
    )


for _cls in (FileEdit, ProposalBase, ArchitectureProposal, CodePatchProposal,
             CritiqueProposal, AttackTestProposal, OptimizationProposal,
             Candidate):
    _cls.__setattr__ = _raise_frozen_proposal  # type: ignore[method-assign]
