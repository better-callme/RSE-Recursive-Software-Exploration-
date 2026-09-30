"""Execution result types and hard-gate/scoring structures."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of running code in an ExecutionEnvironment."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    runtime_ms: float
    peak_memory_mb: float = 0.0
    killed_signal: int | None = None
    limit_hit: str = ""  # "", "memory", "process", "output"


@dataclass(frozen=True)
class StaticGateResult:
    syntax_valid: bool
    forbidden_constructs: tuple[str, ...] = ()
    import_errors: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.syntax_valid and not self.violations


@dataclass(frozen=True)
class TestReport:
    """Parsed result of one test run."""

    total: int
    passed: int
    failed: int
    details: tuple[str, ...] = ()  # failing test ids / messages

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass(frozen=True)
class VerificationResult:
    """Structured evidence produced only by the Referee."""

    static_gate_passed: bool
    tests_total: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    acceptance_passed: bool = False
    adversarial_passed: bool = False
    regression_passed: bool = False
    runtime_ms: float = 0.0
    peak_memory_mb: float = 0.0
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    timed_out: bool = False
    violations: tuple[str, ...] = field(default_factory=tuple)
    failure_type: str = ""  # FailureType value when failed, else ""

    @property
    def tests_pass_rate(self) -> float:
        return self.tests_passed / self.tests_total if self.tests_total else 0.0

    @property
    def all_tiers_passed(self) -> bool:
        return (
            self.static_gate_passed
            and self.acceptance_passed
            and (self.adversarial_passed or True) is not False
            and not self.violations
        )


@dataclass(frozen=True)
class HardGateResult:
    """Binary eligibility. Broken software cannot be compensated by scores."""

    gates: Mapping[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(self.gates.values())

    def failed_gates(self) -> tuple[str, ...]:
        return tuple(sorted(k for k, v in self.gates.items() if not v))


@dataclass(frozen=True)
class ComponentScores:
    hard_pass: bool
    correctness: float  # test pass rate (guidance when hard_pass is False)
    performance: float
    memory: float
    complexity: float
    maintainability: float
    search_cost: float
    guidance: float
    total: float  # meaningful only when hard_pass


def _raise_frozen_res(self: object, name: str, value: object) -> None:
    raise dataclasses.FrozenInstanceError(
        f"cannot modify immutable {type(self).__name__}.{name}"
    )


for _cls in (ExecutionResult, StaticGateResult, VerificationResult,
             HardGateResult, ComponentScores):
    _cls.__setattr__ = _raise_frozen_res  # type: ignore[method-assign]
