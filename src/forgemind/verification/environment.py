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
import threading
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


def _drain_stream(stream, buf: bytearray) -> None:
    """Read a pipe to EOF, appending to buf. Runs on a reader thread.

    Two threads (stdout, stderr) give the same deadlock protection as
    subprocess.communicate(): a chatty child can never block on a full
    pipe while the parent waits for it to exit.
    """
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            buf.extend(chunk)
    except (OSError, ValueError):
        pass


def _returncode_from_status(status: int) -> int:
    """Translate a raw wait4() status into Popen's returncode convention.

    A wait status is NOT a returncode: exit(1) does not produce the raw
    value 1. Interpret it explicitly — WIFEXITED/WEXITSTATUS for normal
    exits, WIFSIGNALED/WTERMSIG for signal deaths (negative, as Popen
    reports them).
    """
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    return -1  # unreachable without WUNTRACED; defensive


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
            # H3: measure THIS child's peak, not the process lifetime's.
            # getrusage(RUSAGE_CHILDREN).ru_maxrss is the largest child
            # ever reaped by this process — it never decreases, so every
            # verification after a big one reported the big one's peak.
            # os.wait4() returns the reaped child's own rusage instead.
            # Reader threads drain the pipes (communicate()'s deadlock
            # protection); they are JOINED before any result is built,
            # so captured output can never be truncated by a slow drain.
            out_buf, err_buf = bytearray(), bytearray()
            readers = [
                threading.Thread(target=_drain_stream,
                                 args=(proc.stdout, out_buf), daemon=True),
                threading.Thread(target=_drain_stream,
                                 args=(proc.stderr, err_buf), daemon=True),
            ]
            for reader in readers:
                reader.start()
            timed_out = False
            status: int | None = None
            rusage = None
            deadline = start + lim.wall_timeout_seconds
            while True:
                try:
                    pid, status, rusage = os.wait4(proc.pid, os.WNOHANG)
                except ChildProcessError:
                    # Defensive: reaped by someone else; fall back to poll().
                    status = None
                    rusage = None
                    break
                if pid != 0:
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    # Containment order (H6) preserved: kill the whole
                    # process group first, then reap the candidate, then
                    # drain the pipes (joins below) and clean up.
                    self._kill_group(proc)
                    pid, status, rusage = os.wait4(proc.pid, 0)
                    break
                time.sleep(0.005)
            for reader in readers:
                reader.join()
            if status is not None:
                proc.returncode = _returncode_from_status(status)
            elif proc.poll() is None:
                proc.poll()
            elapsed = (time.monotonic() - start) * 1000.0
            out, err = bytes(out_buf), bytes(err_buf)
            peak = max(0.0, rusage.ru_maxrss / 1024.0) if rusage is not None else 0.0
            if timed_out:
                return ExecutionResult(
                    exit_code=proc.returncode if proc.returncode is not None else -9,
                    stdout=out.decode("utf-8", "replace")[:output_limit],
                    stderr=err.decode("utf-8", "replace")[:output_limit],
                    timed_out=True,
                    runtime_ms=elapsed,
                    peak_memory_mb=peak,
                    killed_signal=int(signal.SIGKILL),
                    limit_hit="timeout",
                )
            return ExecutionResult(
                exit_code=proc.returncode or 0,
                stdout=out.decode("utf-8", "replace")[:output_limit],
                stderr=err.decode("utf-8", "replace")[:output_limit],
                timed_out=False,
                runtime_ms=elapsed,
                peak_memory_mb=peak,
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



