"""Referee: the ONLY component that produces authoritative verification.

Multi-tier: V1 static -> V2 dynamic acceptance -> V3 adversarial/regression.
Candidate code never executes in-process; everything runs through an
ExecutionEnvironment subprocess.

Evidence channel (C1): the harness writes its structured result as JSON
to a file in a referee-owned directory OUTSIDE the candidate workdir,
authenticated by a per-run random nonce. The nonce and result path are
scrubbed from sys.argv before any candidate module is imported, so
candidate code never sees them through normal channels. Candidate
stdout/stderr are diagnostics only and are NEVER parsed for the verdict.
Missing, corrupt or nonce-mismatched evidence fails closed.

Residual limit (documented honestly, same stance as the sandbox note in
environment.py): candidate code shares a process with the harness, so a
candidate doing deliberate stack-frame introspection is out of scope for
this channel; that is the future container backend's job.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import tempfile
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
    """Generated verification harness. Executes declared tests and writes
    the structured result to the referee-owned result file (never stdout)."""
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
        result_file = payload["result_file"]
        nonce = payload["nonce"]
        sections = payload["sections"]
        # Scrub BEFORE importing any test module: candidate code is imported
        # by the test modules and must never see the result path or nonce.
        sys.argv = [sys.argv[0]]
        out = {}
        for section, module_name in sections.items():
            prefixes = ("test_", "attack_") if section != "regression" else ("regress_",)
            out[section] = run(load(module_name, prefixes))
        # The verdict travels ONLY via this referee-owned file. Nothing
        # authoritative is printed: candidate stdout is untrusted.
        with open(result_file, "w", encoding="utf-8") as fh:
            json.dump({"nonce": nonce, "sections": out}, fh)
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
        result_dir: str | None = None
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

            # Referee-controlled evidence channel (C1): per-run result
            # directory outside the candidate workdir + random nonce.
            result_dir = tempfile.mkdtemp(prefix="forgemind-result-")
            result_path = os.path.join(result_dir, "result.json")
            nonce = secrets.token_hex(16)
            payload = json.dumps({
                "sections": sections,
                "result_file": result_path,
                "nonce": nonce,
            })
            command = ["python3", "__fm_harness.py", payload]
            result: ExecutionResult = self._env.execute(  # type: ignore[attr-defined]
                workdir, command, self._config.resources
            )
            return self._interpret(result, result_path=result_path, nonce=nonce,
                                   adversarial_expected=bool(attack_tests),
                                   regression_expected=bool(regression_tests))
        finally:
            cleanup = getattr(self._env, "cleanup", None)
            if cleanup is not None:
                cleanup(workdir)
            if result_dir is not None:
                shutil.rmtree(result_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    def _problem_test_sources(self) -> Mapping[str, str]:
        # An explicitly attached provider wins (a visible override at the
        # construction site). Otherwise the verdict must be a function of
        # the spec alone (H2): sources come from spec.test_sources, which
        # spec_hash covers. No hidden spec attributes are consulted.
        if self._test_provider is not None:
            return dict(self._test_provider())
        if self._spec.test_sources:
            return dict(self._spec.test_sources)
        raise ValueError(
            "no test source available: attach_test_provider() or a "
            "spec with test_sources is required for dynamic verification")

    @staticmethod
    def _read_evidence(result_path: str, nonce: str) -> dict | None:
        """Return the harness's section results, or None (fail-closed).

        Evidence is valid only if the result file exists, parses as a
        JSON object, carries THIS run's nonce, and has a dict of
        sections. Anything else — missing file, corrupt JSON, wrong
        nonce, wrong shape — is treated as no evidence at all.
        Candidate stdout/stderr play no part here by design (C1).
        """
        try:
            with open(result_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("nonce") != nonce:
            return None
        sections = payload.get("sections")
        return sections if isinstance(sections, dict) else None

    def _interpret(
        self,
        exec_result: ExecutionResult,
        *,
        result_path: str,
        nonce: str,
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
        data = self._read_evidence(result_path, nonce)
        if data is None:
            return VerificationResult(
                static_gate_passed=True,
                violations=("harness produced no valid structured result",),
                failure_type=FailureType.IMPORT_ERROR.value
                if "ImportError" in exec_result.stderr or "ModuleNotFoundError" in exec_result.stderr
                else FailureType.INTERNAL_ERROR.value,
                **base,
            )

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


def hard_gate(result: VerificationResult, *,
              adversarial_expected: bool = False) -> HardGateResult:
    """G(S) = syntax ∧ contract ∧ requiredTests ∧ security ∧ regression
    ∧ adversarial (adversarial only when the caller expected that tier's
    evidence, C3 — callers that never ran attack tests, e.g. the baseline's
    first pass, are unaffected)."""
    gates = {
        "syntax": result.static_gate_passed,
        "contract": not any("interface constraint" in v or "forbidden" in v
                            for v in result.violations),
        "requiredTests": result.acceptance_passed,
        "security": not any("forbidden call" in v or "forbidden import" in v
                            for v in result.violations) and not result.timed_out,
        "regression": result.regression_passed,
        "adversarial": result.adversarial_passed if adversarial_expected else True,
    }
    return HardGateResult(gates=gates)
