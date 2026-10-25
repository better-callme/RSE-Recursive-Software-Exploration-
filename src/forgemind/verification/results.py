"""Execution result types and hard-gate/scoring structures."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
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
    # M1: False if any file failed to parse or safety enforcement was
    # skipped — the safety verdict is then "not established", never
    # "clean". The verifier always sets this explicitly.
    analysis_complete: bool = False

    @property
    def passed(self) -> bool:
        return self.syntax_valid and not self.violations


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
    # M3: the ACCEPTANCE subprocess's returncode, not an aggregate
    # verdict. returncode == 0 with adversarial == FAIL is not a
    # contradiction — different semantics.
    exit_code: int = -1
    timed_out: bool = False
    # M3: names of timed-out tiers, in tier order. timed_out is True
    # iff this is non-empty (any-tier timeout policy, explicit).
    timed_out_tiers: tuple[str, ...] = field(default_factory=tuple)
    # M4: True only when the verdict rests on authoritative, complete
    # evidence (no timeout, no crashed tier; static path: syntax
    # failure counts as complete, otherwise full analysis required).
    # Fail-closed default: a result that does not prove completeness
    # must not close a state hash (M4 close rule).
    evidence_complete: bool = False
    # M3 per-tier counts. They sum exactly to tests_total/passed/failed
    # (a crashed or timed-out tier contributes zeros). Field names use
    # _passes/_failures because _passed is already the verdict bool.
    acceptance_total: int = 0
    acceptance_passes: int = 0
    acceptance_failures: int = 0
    adversarial_total: int = 0
    adversarial_passes: int = 0
    adversarial_failures: int = 0
    regression_total: int = 0
    regression_passes: int = 0
    regression_failures: int = 0
    violations: tuple[str, ...] = field(default_factory=tuple)
    failure_type: str = ""  # FailureType value when failed, else ""
    # M1 structured gate inputs. contract_passed and
    # static_analysis_complete are fail-closed (the referee always sets
    # them; a hand-built result that omits them must not pass).
    # forbidden_constructs is an absence claim (empty default, like
    # violations) — meaningful only alongside analysis_complete=True.
    contract_passed: bool = False
    forbidden_constructs: tuple[str, ...] = field(default_factory=tuple)
    static_analysis_complete: bool = False

    @property
    def tests_pass_rate(self) -> float:
        # Summed all-tiers rate, kept for reporting compat. This is NOT
        # correctness (M3): scoring uses acceptance_pass_rate.
        return self.tests_passed / self.tests_total if self.tests_total else 0.0

    @property
    def acceptance_pass_rate(self) -> float:
        """M3: acceptance-tier pass rate — the correctness signal."""
        return (self.acceptance_passes / self.acceptance_total
                if self.acceptance_total else 0.0)

    @property
    def all_tiers_passed(self) -> bool:
        # The six canonical predicates (adversarial unconditional) plus
        # clean diagnostics — a strict superset of the gate conjunction,
        # so ACCEPT implies failed_gates() empty by construction. For
        # referee-produced results the diagnostics clause is redundant
        # (every non-empty violations list already implies a failed
        # predicate); it only constrains hand-built results.
        return (all(gate_predicates(self, adversarial_expected=True).values())
                and not self.violations)


@dataclass(frozen=True)
class HardGateResult:
    """Binary eligibility. Broken software cannot be compensated by scores."""

    gates: Mapping[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(self.gates.values())

    def failed_gates(self) -> tuple[str, ...]:
        return tuple(sorted(k for k, v in self.gates.items() if not v))


def gate_predicates(result: "VerificationResult", *,
                    adversarial_expected: bool = False) -> dict[str, bool]:
    """The ONE canonical gate definition (M1). hard_gate,
    all_tiers_passed and failed_gates() all derive from this — no gate
    reads violation text, and no second definition of "passed" exists.
    Security requires the safety analysis to have actually completed:
    a parse failure or skipped analysis can never masquerade as a
    clean bill of health via an empty forbidden_constructs."""
    return {
        "syntax": result.static_gate_passed,
        "contract": result.contract_passed,
        "requiredTests": result.acceptance_passed,
        "security": (result.static_analysis_complete
                     and not result.forbidden_constructs
                     and not result.timed_out),
        "regression": result.regression_passed,
        "adversarial": (result.adversarial_passed
                        if adversarial_expected else True),
    }


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


class VerificationOutcome(str, Enum):
    """First-class verification outcome (H4). A timeout is inconclusive,
    not failure; only retries-exhausted timeouts quarantine. Downstream
    code matches on this enum, not on strings."""

    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    QUARANTINED = "QUARANTINED"


def classify_verification(
    result: VerificationResult, *, retries_exhausted: bool,
) -> VerificationOutcome:
    """Map a verification result to its outcome. A timeout maps by retry
    state (INCONCLUSIVE while a retry remains, QUARANTINED once retries
    are exhausted); anything else maps by whether all tiers passed."""
    if result.timed_out:
        return (VerificationOutcome.QUARANTINED if retries_exhausted
                else VerificationOutcome.INCONCLUSIVE)
    return (VerificationOutcome.PASS if result.all_tiers_passed
            else VerificationOutcome.FAIL)


def _raise_frozen_res(self: object, name: str, value: object) -> None:
    raise dataclasses.FrozenInstanceError(
        f"cannot modify immutable {type(self).__name__}.{name}"
    )


for _cls in (ExecutionResult, StaticGateResult, VerificationResult,
             HardGateResult, ComponentScores):
    _cls.__setattr__ = _raise_frozen_res  # type: ignore[method-assign]
