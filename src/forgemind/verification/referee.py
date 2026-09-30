"""Referee: the ONLY component that produces authoritative verification.

Multi-tier: V1 static -> V2 dynamic acceptance -> V3 adversarial/regression.
Candidate code never executes in-process; everything runs through an
ExecutionEnvironment subprocess.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from forgemind.core.config import ForgeMindConfig
from forgemind.core.problem import ProblemSpec
from forgemind.core.proposal import AttackTestProposal
from forgemind.core.state import CodebaseState
from forgemind.recovery.failure import FailureType
from forgemind.verification.environment import ExecutionEnvironment
from forgemind.verification.results import (
    ExecutionResult,
    HardGateResult,
    VerificationResult,
)
from forgemind.verification.static import StaticVerifier

_HARNESS_TEMPLATE = textwrap.dedent(
    '''
    """Generated verification harness. Executes declared tests, emits JSON."""
    import json, sys, traceback

    def run(tests):
        total = passed = 0
        details = []
        for name, fn in tests:
            total += 1
            try:
                fn()
                passed += 1
            except Exception:
                details.append(name + ": " + traceback.format_exc(limit=3).splitlines()[-1])
        return {"total": total, "passed": passed, "failed": total - passed,
                "details": details}

    def load(module_name, prefixes):
        import importlib
        mod = importlib.import_module(module_name)
        return [
            (name, getattr(mod, name))
            for name in sorted(dir(mod))
            if name.startswith(prefixes) and callable(getattr(mod, name))
        ]

    if __name__ == "__main__":
        payload = json.loads(sys.argv[1])
        out = {}
        for section, module_name in payload.items():
            prefixes = ("test_", "attack_") if section != "regression" else ("regress_",)
            out[section] = run(load(module_name, prefixes))
        print("HARNESS_JSON:" + json.dumps(out))
    '''
)


@dataclass(frozen=True)
class TieredRun:
    acceptance: dict[str, int]
    adversarial: dict[str, int]
    regression: dict[str, int]


class TestProvider(Protocol):
    """A problem supplies its acceptance/regression test sources."""

    def test_files(self) -> Mapping[str, str]: ...


class Referee:
    def __init__(
        self,
        spec: ProblemSpec,
        environment: ExecutionEnvironment,
        config: ForgeMindConfig,
    ) -> None:
        self._spec = spec
        self._env = environment
        self._config = config
        self._static = StaticVerifier()
        self._test_provider = None

    def attach_test_provider(self, provider) -> None:
        """Register the callable returning {filename: test_source}."""
        self._test_provider = provider

    # ------------------------------------------------------------------
    @property
    def spec(self) -> ProblemSpec:
        return self._spec

    def verify(
        self,
        state: CodebaseState,
        *,
        attack_tests: Sequence[AttackTestProposal] = (),
        regression_tests: Mapping[str, str] | None = None,
    ) -> VerificationResult:
        """Full staged verification of one state."""
        # ---- Tier 1: static -------------------------------------------------
        static = self._static.verify(dict(state.files))
        if not static.passed:
            ftype = FailureType.SYNTAX_ERROR.value if not static.syntax_valid \
                else FailureType.CONTRACT_VIOLATION.value
            return VerificationResult(
                static_gate_passed=False,
                violations=tuple(static.violations),
                failure_type=ftype,
            )

        # ---- prepare sandbox -------------------------------------------------
        workdir = self._env.materialize_state(state)  # type: ignore[attr-defined]
        try:
            state.write_to_disk(workdir)
            with open(f"{workdir}/__fm_harness.py", "w", encoding="utf-8") as fh:
                fh.write(_HARNESS_TEMPLATE)

            sections: dict[str, str] = {"acceptance": "__fm_acceptance"}
            acceptance_src = "\n\n".join(
                src for src in self._problem_test_sources().values()
            ) or "pass\n"
            with open(f"{workdir}/__fm_acceptance.py", "w", encoding="utf-8") as fh:
                fh.write(acceptance_src)

            if attack_tests:
                sections["adversarial"] = "__fm_adversarial"
                adv_src = "\n\n".join(a.test_code for a in attack_tests)
                with open(f"{workdir}/__fm_adversarial.py", "w", encoding="utf-8") as fh:
                    fh.write(adv_src)
            if regression_tests:
                sections["regression"] = "__fm_regression"
                reg_src = "\n\n".join(regression_tests.values())
                with open(f"{workdir}/__fm_regression.py", "w", encoding="utf-8") as fh:
                    fh.write(reg_src)

            command = ["python3", "__fm_harness.py", json.dumps(sections)]
            result: ExecutionResult = self._env.execute(  # type: ignore[attr-defined]
                workdir, command, self._config.resources
            )
            return self._interpret(result, adversarial_expected=bool(attack_tests),
                                   regression_expected=bool(regression_tests))
        finally:
            cleanup = getattr(self._env, "cleanup", None)
            if cleanup is not None:
                cleanup(workdir)

    # ------------------------------------------------------------------
    def _problem_test_sources(self) -> Mapping[str, str]:
        if self._test_provider is not None:
            return dict(self._test_provider())
        provider = getattr(self._spec, "_test_provider_ref", None)
        if callable(provider):
            return dict(provider())
        provider = getattr(self._spec, "test_files", None)
        if callable(provider):
            return dict(provider())
        raise ValueError(
            "no test source available: attach_test_provider() or a "
            "spec with test_files() is required for dynamic verification")

    def _interpret(
        self,
        exec_result: ExecutionResult,
        *,
        adversarial_expected: bool,
        regression_expected: bool,
    ) -> VerificationResult:
        base = dict(
            runtime_ms=exec_result.runtime_ms,
            peak_memory_mb=exec_result.peak_memory_mb,
            stdout=exec_result.stdout[-4000:],
            stderr=exec_result.stderr[-4000:],
            exit_code=exec_result.exit_code,
            timed_out=exec_result.timed_out,
        )
        if exec_result.timed_out:
            return VerificationResult(static_gate_passed=True, failure_type=
                                      FailureType.TIMEOUT.value, **base)
        marker = "HARNESS_JSON:"
        line = next((l for l in exec_result.stdout.splitlines()
                     if l.startswith(marker)), None)
        if line is None:
            return VerificationResult(
                static_gate_passed=True,
                violations=("harness produced no structured result",),
                failure_type=FailureType.IMPORT_ERROR.value
                if "ImportError" in exec_result.stderr or "ModuleNotFoundError" in exec_result.stderr
                else FailureType.INTERNAL_ERROR.value,
                **base,
            )
        data = json.loads(line[len(marker):])

        acc = data.get("acceptance", {})
        adv = data.get("adversarial")
        reg = data.get("regression")

        acceptance_passed = acc.get("failed", 1) == 0 and acc.get("total", 0) > 0
        adversarial_passed = (adv.get("failed", 1) == 0) if adv is not None else True
        regression_passed = (reg.get("failed", 1) == 0) if reg is not None else True

        failures: list[str] = []
        ftype = ""
        if not acceptance_passed:
            ftype = FailureType.TEST_FAILURE.value
            failures.extend(acc.get("details", []))
        if adversarial_expected and not adversarial_passed:
            ftype = ftype or FailureType.ADVERSARIAL_FAILURE.value
            failures.extend((adv or {}).get("details", []))
        if regression_expected and not regression_passed:
            ftype = ftype or FailureType.REGRESSION_FAILURE.value
            failures.extend((reg or {}).get("details", []))

        return VerificationResult(
            static_gate_passed=True,
            tests_total=acc.get("total", 0) + (adv or {}).get("total", 0)
            + (reg or {}).get("total", 0),
            tests_passed=acc.get("passed", 0) + (adv or {}).get("passed", 0)
            + (reg or {}).get("passed", 0),
            tests_failed=acc.get("failed", 0) + (adv or {}).get("failed", 0)
            + (reg or {}).get("failed", 0),
            acceptance_passed=acceptance_passed,
            adversarial_passed=adversarial_passed,
            regression_passed=regression_passed,
            violations=tuple(failures[:20]),
            failure_type=ftype,
            **base,
        )


def hard_gate(result: VerificationResult) -> HardGateResult:
    """G(S) = syntax ∧ contract ∧ requiredTests ∧ security ∧ regression."""
    gates = {
        "syntax": result.static_gate_passed,
        "contract": not any("interface constraint" in v or "forbidden" in v
                            for v in result.violations),
        "requiredTests": result.acceptance_passed,
        "security": not any("forbidden call" in v or "forbidden import" in v
                            for v in result.violations) and not result.timed_out,
        "regression": result.regression_passed,
    }
    return HardGateResult(gates=gates)
