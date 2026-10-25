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

import ast
import json
import os
import secrets
import shutil
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from forgemind.core.config import ForgeMindConfig
from forgemind.core.problem import ProblemSpec, resolve_entry_point
from forgemind.core.proposal import AttackTestProposal
from forgemind.core.state import CodebaseState
from forgemind.recovery.failure import FailureType
from forgemind.verification.environment import ExecutionEnvironment
from forgemind.verification.results import (
    ExecutionResult,
    HardGateResult,
    VerificationResult,
    gate_predicates,
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
                "details": details,
                # M2: executed names, set-like by design (dir() yields
                # each name once; sorted for deterministic evidence).
                "executed": sorted({name for name, _ in tests})}

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
class _TierOutcome:
    """One tier's isolated result (M3). Internal to the referee."""
    name: str
    passed: bool
    total: int
    passes: int
    failures: int
    details: tuple[str, ...]
    executed: tuple[str, ...]
    timed_out: bool
    crashed: bool
    failure_type: str  # "" when passed
    runtime_ms: float
    peak_memory_mb: float
    stdout: str
    stderr: str
    exit_code: int


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
        # ---- Tier 1b: contract (M1) ----------------------------------------
        # Computed for every result, before branching, so the contract
        # verdict rides along on all paths (it never short-circuits the
        # run — dynamic evidence is still collected).
        contract_passed, contract_violation = self._check_contract(state)
        if not static.passed:
            ftype = FailureType.SYNTAX_ERROR.value if not static.syntax_valid \
                else FailureType.CONTRACT_VIOLATION.value
            violations = tuple(static.violations)
            if contract_violation is not None:
                violations += (contract_violation,)
            return VerificationResult(
                static_gate_passed=False,
                contract_passed=contract_passed,
                forbidden_constructs=tuple(static.forbidden_constructs),
                static_analysis_complete=static.analysis_complete,
                violations=violations,
                failure_type=ftype,
                # M4: a syntax failure IS complete evidence (the file
                # does not parse); otherwise completeness requires the
                # full static analysis to have run.
                evidence_complete=(not static.syntax_valid)
                or static.analysis_complete,
            )

        # ---- dynamic tiers (M3): one subprocess per tier ------------------
        # Each tier runs SEQUENTIALLY in its own fresh workdir holding
        # exactly the candidate state + the harness + that tier's test
        # module, with its own evidence file and nonce. A tier's crash
        # or timeout fails that tier only and never aborts the others.
        tiers: list[tuple[str, str, str]] = []
        acceptance_src = "\n\n".join(
            src for src in self._problem_test_sources().values()
        ) or "pass\n"
        tiers.append(("acceptance", "__fm_acceptance", acceptance_src))
        if attack_tests:
            tiers.append(("adversarial", "__fm_adversarial",
                          "\n\n".join(a.test_code for a in attack_tests)))
        if regression_tests:
            tiers.append(("regression", "__fm_regression",
                          "\n\n".join(regression_tests.values())))

        # Referee-controlled evidence channel (C1, per-tier in M3):
        # one result directory outside all workdirs, one file + nonce
        # per tier, so compromising one tier forges at most its own.
        result_dir = tempfile.mkdtemp(prefix="forgemind-result-")
        try:
            outcomes = [self._run_tier(state, name, module, source, result_dir)
                        for name, module, source in tiers]
            return self._assemble(
                outcomes,
                contract_passed=contract_passed,
                contract_violation=contract_violation,
                forbidden_constructs=tuple(static.forbidden_constructs),
                static_analysis_complete=static.analysis_complete)
        finally:
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

    def _check_contract(self, state: CodebaseState
                        ) -> tuple[bool, str | None]:
        """Referee-side interface check (M1): does the state actually
        define the spec's entry function? The transition engine checks
        proposals; this checks resulting states, including states that
        never went through a transition. Returns (passed, violation)."""
        try:
            entry = resolve_entry_point(self._spec)
        except ValueError as exc:
            return False, f"contract: spec names no usable entry point: {exc}"
        for path, content in sorted(state.files.items()):
            if not path.endswith(".py"):
                continue
            try:
                tree = ast.parse(content)
            except SyntaxError:
                continue  # syntax tier already fails these
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and node.name == entry:
                    return True, None
        return False, f"contract: entry function {entry!r} not defined in state"

    def _run_tier(self, state: CodebaseState, name: str, module: str,
                  source: str, result_dir: str) -> _TierOutcome:
        """Run ONE tier in isolation (M3): fresh workdir holding exactly
        the candidate state + the harness + this tier's test module,
        own evidence file + nonce. A crash or timeout fails this tier
        only; the caller still runs the remaining tiers."""
        workdir = self._env.materialize_state(state)  # type: ignore[attr-defined]
        try:
            state.write_to_disk(workdir)
            with open(f"{workdir}/__fm_harness.py", "w", encoding="utf-8") as fh:
                fh.write(_HARNESS_TEMPLATE)
            with open(f"{workdir}/{module}.py", "w", encoding="utf-8") as fh:
                fh.write(source)
            result_path = os.path.join(result_dir, f"result_{name}.json")
            nonce = secrets.token_hex(16)
            payload = json.dumps({
                "sections": {name: module},
                "result_file": result_path,
                "nonce": nonce,
            })
            exec_result: ExecutionResult = self._env.execute(  # type: ignore[attr-defined]
                workdir, ["python3", "__fm_harness.py", payload],
                self._config.resources
            )
            base = dict(
                name=name,
                runtime_ms=exec_result.runtime_ms,
                peak_memory_mb=exec_result.peak_memory_mb,
                stdout=exec_result.stdout[-4000:],
                stderr=exec_result.stderr[-4000:],
                exit_code=exec_result.exit_code,
            )
            if exec_result.timed_out:
                return _TierOutcome(
                    passed=False, total=0, passes=0, failures=0,
                    details=(), executed=(), timed_out=True, crashed=False,
                    failure_type=FailureType.TIMEOUT.value, **base)
            data = self._read_evidence(result_path, nonce)
            if data is None:
                return _TierOutcome(
                    passed=False, total=0, passes=0, failures=0,
                    details=(), executed=(), timed_out=False, crashed=True,
                    failure_type=FailureType.IMPORT_ERROR.value
                    if "ImportError" in exec_result.stderr
                    or "ModuleNotFoundError" in exec_result.stderr
                    else FailureType.INTERNAL_ERROR.value,
                    **base)
            section = data.get(name, {})
            total = section.get("total", 0)
            passed_n = section.get("passed", 0)
            failed_n = section.get("failed", 0)
            return _TierOutcome(
                passed=failed_n == 0 and total > 0,
                total=total, passes=passed_n, failures=failed_n,
                details=tuple(section.get("details", [])),
                executed=tuple(section.get("executed") or []),
                timed_out=False, crashed=False,
                failure_type="",
                **base)
        finally:
            cleanup = getattr(self._env, "cleanup", None)
            if cleanup is not None:
                cleanup(workdir)

    def _assemble(
        self,
        outcomes: list[_TierOutcome],
        *,
        contract_passed: bool,
        contract_violation: str | None,
        forbidden_constructs: tuple[str, ...],
        static_analysis_complete: bool,
    ) -> VerificationResult:
        """Merge per-tier outcomes into one verdict (M3, Revision 2
        invariants). Tiers are independent: one tier's crash or
        timeout never erases another tier's verdict."""
        by_name = {o.name: o for o in outcomes}
        acc = by_name["acceptance"]
        adv = by_name.get("adversarial")
        reg = by_name.get("regression")

        # M2: declared ⊆ executed, on the acceptance tier's evidence.
        # A crashed/timed-out acceptance tier has no evidence; its
        # failure is reported as crash/timeout, not as missing tests.
        missing: list[str] = []
        acceptance_passed = acc.passed
        if not acc.crashed and not acc.timed_out:
            declared = {t.test_id for t in self._spec.acceptance_tests}
            missing = sorted(declared - set(acc.executed))
            if missing:
                acceptance_passed = False
        adversarial_passed = adv.passed if adv is not None else True
        regression_passed = reg.passed if reg is not None else True

        # Per-tier failure types, in tier order; any timeout takes
        # precedence (the verdict is inconclusive, cf. H4).
        tier_failures = []
        if not acceptance_passed:
            tier_failures.append(
                acc.failure_type or FailureType.TEST_FAILURE.value)
        if adv is not None and not adversarial_passed:
            tier_failures.append(
                adv.failure_type or FailureType.ADVERSARIAL_FAILURE.value)
        if reg is not None and not regression_passed:
            tier_failures.append(
                reg.failure_type or FailureType.REGRESSION_FAILURE.value)
        timed_out_tiers = tuple(o.name for o in outcomes if o.timed_out)
        ftype = (FailureType.TIMEOUT.value if timed_out_tiers
                 else (tier_failures[0] if tier_failures else ""))

        failures: list[str] = []
        if not acceptance_passed:
            if missing:
                failures.append(
                    "Missing declared acceptance tests: " + ", ".join(missing))
            failures.extend(acc.details)
            if acc.crashed:
                failures.append(
                    "acceptance tier produced no valid structured result")
        if adv is not None and not adversarial_passed:
            failures.extend(adv.details)
            if adv.crashed:
                failures.append(
                    "adversarial tier produced no valid structured result")
        if reg is not None and not regression_passed:
            failures.extend(reg.details)
            if reg.crashed:
                failures.append(
                    "regression tier produced no valid structured result")
        contract_extra = (contract_violation,) if contract_violation else ()

        def labeled(field: str) -> str:
            # Diagnostics only (M1): explicit per-tier delimiters so
            # output is attributable by eye, never parsed for verdicts.
            parts = []
            for o in outcomes:
                parts.append(f"=== {o.name.upper()} {field} ===\n"
                             f"{getattr(o, field.lower())}")
            return "\n".join(parts)

        return VerificationResult(
            static_gate_passed=True,
            tests_total=sum(o.total for o in outcomes),
            tests_passed=sum(o.passes for o in outcomes),
            tests_failed=sum(o.failures for o in outcomes),
            acceptance_total=acc.total,
            acceptance_passes=acc.passes,
            acceptance_failures=acc.failures,
            adversarial_total=adv.total if adv else 0,
            adversarial_passes=adv.passes if adv else 0,
            adversarial_failures=adv.failures if adv else 0,
            regression_total=reg.total if reg else 0,
            regression_passes=reg.passes if reg else 0,
            regression_failures=reg.failures if reg else 0,
            acceptance_passed=acceptance_passed,
            adversarial_passed=adversarial_passed,
            regression_passed=regression_passed,
            contract_passed=contract_passed,
            forbidden_constructs=forbidden_constructs,
            static_analysis_complete=static_analysis_complete,
            timed_out=bool(timed_out_tiers),
            timed_out_tiers=timed_out_tiers,
            # M4: complete iff no tier timed out and every executed
            # tier produced valid evidence. Vacuous tiers (not run)
            # do not count as crashed; M2-missing counts as complete
            # (the tier ran; the failure is deterministic).
            evidence_complete=(not timed_out_tiers
                               and not any(o.crashed for o in outcomes)),
            runtime_ms=sum(o.runtime_ms for o in outcomes),
            peak_memory_mb=max((o.peak_memory_mb for o in outcomes),
                               default=0.0),
            stdout=labeled("STDOUT"),
            stderr=labeled("STDERR"),
            exit_code=acc.exit_code,
            violations=tuple(failures[:20]) + contract_extra,
            failure_type=ftype,
        )


def hard_gate(result: VerificationResult, *,
              adversarial_expected: bool = False) -> HardGateResult:
    """G(S) = syntax ∧ contract ∧ requiredTests ∧ security ∧ regression
    ∧ adversarial (adversarial only when the caller expected that tier's
    evidence, C3 — callers that never ran attack tests, e.g. the baseline's
    first pass, are unaffected)."""
    # Pure decision function (M1): the canonical predicates from
    # results.gate_predicates — no analysis here, and no violation text
    # is ever inspected.
    return HardGateResult(gates=gate_predicates(
        result, adversarial_expected=adversarial_expected))
