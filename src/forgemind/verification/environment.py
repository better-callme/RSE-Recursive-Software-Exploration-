"""Subprocess execution environment.

SECURITY NOTE (documented honestly): this environment provides resource
isolation / damage reduction via process groups, rlimits, tmpdirs and env
sanitization. It is NOT container/VM-grade hostile-code isolation. Production
ForgeMind should execute candidates in a container/gVisor/VM sandbox.

IMPLEMENTATION NOTE: CPython's subprocess may service Popen with preexec_fn
via vfork(), where the child shares the parent's address space until exec.
A preexec that calls setrlimit() under vfork therefore mutates the ENGINE's
own rlimits. To avoid this, rlimits are applied by a tiny POSIX shell
bootstrap (prlimit/set) executed via exec, never via preexec_fn. Session
creation uses start_new_session=True (kernel-side, vfork-safe).
"""

from __future__ import annotations

import os
import resource
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Protocol, Sequence

from forgemind.core.config import ResourceLimits
from forgemind.core.state import CodebaseState
from forgemind.verification.results import ExecutionResult

_SENSITIVE_ENV_PREFIXES = ("FORGEMIND_", "AWS_", "GITHUB_TOKEN", "OPENAI_API_KEY",
                           "ANTHROPIC_API_KEY", "HTTP_PROXY", "HTTPS_PROXY")


def _sanitized_env() -> dict[str, str]:
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    for key in ("TMPDIR",):
        if key in os.environ:
            env[key] = os.environ[key]
    # Allowlist policy: nothing else (and nothing sensitive) passes through.
    return env


def _rlimit_bootstrap_command(limits: ResourceLimits,
                              command: Sequence[str]) -> tuple[list[str], str]:
    """Build [sh, -c, bootstrap] that applies rlimits then execs the payload.

    Uses `prlimit` when available (util-linux), else a setrlimit python
    one-liner executed in the child AFTER fork+exec of a fresh interpreter,
    so no vfork address-space sharing can leak limits into the engine.
    """
    mem_bytes = limits.memory_limit_mb * 1024 * 1024
    cpu = max(1, int(limits.cpu_limit_seconds))
    fsize = limits.max_output_bytes * 10
    quoted = " ".join(shlex.quote(c) for c in command)
    inner = (
        "import resource, os, sys\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu + 1}))\n"
        f"resource.setrlimit(resource.RLIMIT_AS, ({mem_bytes}, {mem_bytes}))\n"
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
        f"resource.setrlimit(resource.RLIMIT_FSIZE, ({fsize}, {fsize}))\n"
        f"os.execvp(sys.argv[1], sys.argv[1:])\n"
    )
    # NOTE: RLIMIT_NPROC deliberately NOT set (cgroup pids.max interactions
    # make fork() fail on some kernels). Containment: wall-clock timeout +
    # process-group kill.
    return ["/bin/sh", "-c", f"exec python3 -c {shlex.quote(inner)} {quoted}"], inner


class ExecutionEnvironment(Protocol):
    def execute(
        self,
        workdir: str,
        command: Sequence[str],
        limits: ResourceLimits,
    ) -> ExecutionResult: ...


class SubprocessExecutionEnvironment:
    """POSIX subprocess execution with resource limits and group kill."""

    def __init__(self, default_limits: ResourceLimits | None = None) -> None:
        self._defaults = default_limits or ResourceLimits()

    def materialize_state(self, state: CodebaseState) -> str:
        root = tempfile.mkdtemp(prefix="forgemind-exec-")
        state.write_to_disk(root)
        return root

    def execute(
        self,
        workdir: str,
        command: Sequence[str],
        limits: ResourceLimits | None = None,
    ) -> ExecutionResult:
        lim = limits or self._defaults
        start = time.monotonic()
        proc: subprocess.Popen[bytes] | None = None
        output_limit = lim.max_output_bytes
        argv, _ = _rlimit_bootstrap_command(lim, list(command))
        try:
            proc = subprocess.Popen(
                argv,
                cwd=workdir,
                env=_sanitized_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # kernel-side setsid; vfork-safe
            )
            try:
                out, err = proc.communicate(timeout=lim.wall_timeout_seconds)
            except subprocess.TimeoutExpired:
                self._kill_group(proc)
                out, err = proc.communicate()
                elapsed = (time.monotonic() - start) * 1000.0
                return ExecutionResult(
                    exit_code=proc.returncode if proc.returncode is not None else -9,
                    stdout=out.decode("utf-8", "replace")[:output_limit],
                    stderr=err.decode("utf-8", "replace")[:output_limit],
                    timed_out=True,
                    runtime_ms=elapsed,
                    killed_signal=int(signal.SIGKILL),
                    limit_hit="timeout",
                )
            elapsed = (time.monotonic() - start) * 1000.0
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            return ExecutionResult(
                exit_code=proc.returncode or 0,
                stdout=out.decode("utf-8", "replace")[:output_limit],
                stderr=err.decode("utf-8", "replace")[:output_limit],
                timed_out=False,
                runtime_ms=elapsed,
                peak_memory_mb=max(0.0, usage.ru_maxrss / 1024.0),
                limit_hit="output" if len(out) >= output_limit else "",
            )
        finally:
            if proc is not None and proc.poll() is None:
                self._kill_group(proc)

    @staticmethod
    def _kill_group(proc: subprocess.Popen[bytes]) -> None:
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return
        # Safety: if session creation failed, the child shares OUR process
        # group; killing the group would terminate the engine itself.
        if pgid != os.getpgrp():
            try:
                os.killpg(pgid, signal.SIGKILL)
                return
            except (ProcessLookupError, PermissionError):
                pass
        try:
            proc.kill()
        except OSError:
            pass

    def cleanup(self, workdir: str) -> None:
        shutil.rmtree(workdir, ignore_errors=True)


class InMemoryExecutionEnvironment:
    """Test-only environment executing a fixed canned result per command tag.

    Used by unit tests to keep the Referee backend-agnostic; never used for
    real candidate verification.
    """

    def __init__(self, results_by_tag: dict[str, ExecutionResult]) -> None:
        self._results = results_by_tag
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def execute(
        self,
        workdir: str,
        command: Sequence[str],
        limits: ResourceLimits | None = None,
    ) -> ExecutionResult:
        self.calls.append((workdir, tuple(command)))
        for tag, result in self._results.items():
            if any(tag in part for part in command):
                return result
        return ExecutionResult(exit_code=1, stdout="", stderr="no canned result",
                               timed_out=False, runtime_ms=0.0)
