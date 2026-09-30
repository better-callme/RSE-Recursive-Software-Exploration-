"""Immutable problem specification."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence


@dataclass(frozen=True)
class AcceptanceTest:
    """A single required test case (name maps to a test function id)."""

    name: str
    test_id: str  # e.g. "tests::test_empty_list"
    description: str = ""


@dataclass(frozen=True)
class ProblemSpec:
    """What ForgeMind is trying to solve. Immutable after creation.

    Search must never mutate this object; a proposal that attempts to relax
    it is rejected by the TransitionEngine.
    """

    problem_id: str
    description: str
    requirements: tuple[str, ...]
    acceptance_tests: tuple[AcceptanceTest, ...]
    interface_constraints: Mapping[str, str] = field(default_factory=dict)
    performance_constraints: Mapping[str, float] = field(default_factory=dict)
    dependency_constraints: tuple[str, ...] = ()
    language: str = "python"
    runtime: str = "python3"

    def __post_init__(self) -> None:
        if not self.problem_id:
            raise ValueError("problem_id must be non-empty")
        if not self.acceptance_tests:
            raise ValueError("a problem requires at least one acceptance test")
        # Defensive canonicalization of mappings.
        object.__setattr__(
            self, "interface_constraints", MappingProxyType(dict(self.interface_constraints))
        )
        object.__setattr__(
            self,
            "performance_constraints",
            MappingProxyType(dict(self.performance_constraints)),
        )

    @property
    def spec_hash(self) -> str:
        """Deterministic identity of the problem itself."""
        payload = {
            "problem_id": self.problem_id,
            "description": self.description,
            "requirements": list(self.requirements),
            "acceptance_tests": sorted(t.test_id for t in self.acceptance_tests),
            "interface_constraints": dict(sorted(self.interface_constraints.items())),
            "performance_constraints": dict(sorted(self.performance_constraints.items())),
            "dependency_constraints": sorted(self.dependency_constraints),
            "language": self.language,
            "runtime": self.runtime,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def test_ids(self) -> Sequence[str]:
        return tuple(t.test_id for t in self.acceptance_tests)


# Explicitly freeze dataclass mutation surface, including object.__setattr__.
def _raise_frozen(self: object, name: str, value: object) -> None:
    raise dataclasses.FrozenInstanceError(
        f"cannot modify immutable {type(self).__name__}.{name}"
    )


def _raise_frozen_delete(self: object, name: str) -> None:
    raise dataclasses.FrozenInstanceError(
        f"cannot delete immutable {type(self).__name__}.{name}"
    )


for _cls in (ProblemSpec, AcceptanceTest):
    _cls.__setattr__ = _raise_frozen  # type: ignore[method-assign]
    _cls.__delattr__ = _raise_frozen_delete  # type: ignore[method-assign]
