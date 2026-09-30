"""Deterministic projections: agents never see engine internals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

from forgemind.core.problem import ProblemSpec
from forgemind.core.proposal import (
    ArchitectureProposal,
    AttackTestProposal,
    CodePatchProposal,
    CritiqueProposal,
    OptimizationProposal,
)
from forgemind.recovery.failure import FailureRecord
from forgemind.verification.results import VerificationResult


@dataclass(frozen=True)
class ArchitectProjection:
    spec: ProblemSpec
    file_structure: Mapping[str, int]  # path -> line count
    interface_constraints: Mapping[str, str]
    failure_summary: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuilderProjection:
    spec: ProblemSpec
    architecture: ArchitectureProposal | None
    target_files: Mapping[str, str]
    prior_failures: tuple[FailureRecord, ...] = ()
    critiques: tuple[CritiqueProposal, ...] = ()
    attempt_number: int = 0


@dataclass(frozen=True)
class CriticProjection:
    spec: ProblemSpec
    relevant_code: Mapping[str, str]
    verification: VerificationResult | None
    contract_info: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizerProjection:
    spec: ProblemSpec
    verified_files: Mapping[str, str]
    benchmark_metrics: Mapping[str, float]
    performance_requirements: Mapping[str, float]


def build_architect_projection(spec: ProblemSpec, state) -> ArchitectProjection:
    return ArchitectProjection(
        spec=spec,
        file_structure={p: len(c.splitlines()) for p, c in sorted(state.files.items())},
        interface_constraints=dict(spec.interface_constraints),
        failure_summary=(),
    )


def build_builder_projection(
    spec: ProblemSpec,
    state,
    *,
    architecture: ArchitectureProposal | None = None,
    failures: tuple[FailureRecord, ...] = (),
    critiques: tuple[CritiqueProposal, ...] = (),
    attempt_number: int = 0,
) -> BuilderProjection:
    return BuilderProjection(
        spec=spec,
        architecture=architecture,
        target_files=dict(sorted(state.files.items())),
        prior_failures=tuple(failures),
        critiques=tuple(critiques),
        attempt_number=attempt_number,
    )


def build_critic_projection(
    spec: ProblemSpec,
    state,
    verification: VerificationResult | None = None,
) -> CriticProjection:
    return CriticProjection(
        spec=spec,
        relevant_code=dict(sorted(state.files.items())),
        verification=verification,
        contract_info=dict(spec.interface_constraints),
    )


def build_optimizer_projection(
    spec: ProblemSpec,
    state,
    metrics: Mapping[str, float],
) -> OptimizerProjection:
    return OptimizerProjection(
        spec=spec,
        verified_files=dict(sorted(state.files.items())),
        benchmark_metrics=dict(metrics),
        performance_requirements=dict(spec.performance_constraints),
    )


class Agent(Protocol):
    """Agents are stateless functions of projections to proposals."""

    role: str

    def propose(self, projection: object, request_context: Mapping[str, str]) -> tuple: ...


def _raise_frozen_proj(self: object, name: str, value: object) -> None:
    raise TypeError(f"{type(self).__name__} is immutable ({name})")


for _cls in (ArchitectProjection, BuilderProjection, CriticProjection,
             OptimizerProjection):
    _cls.__setattr__ = _raise_frozen_proj  # type: ignore[method-assign]
