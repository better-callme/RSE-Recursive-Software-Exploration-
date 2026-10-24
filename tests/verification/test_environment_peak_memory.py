"""H3: peak_memory_mb is per-run (not the process-lifetime maximum),
wait4() statuses are interpreted into Popen-style returncodes, output is
complete before the result is built, and concurrent runs don't
cross-contaminate."""

import os
import signal
import tempfile
from concurrent.futures import ThreadPoolExecutor

from forgemind.core.config import ResourceLimits
from forgemind.verification.environment import SubprocessExecutionEnvironment


def _run(script, limits=None):
    env = SubprocessExecutionEnvironment()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "payload.py")
        with open(path, "w") as fh:
            fh.write(script)
        return env.execute(d, ["python3", "payload.py"],
                           limits or ResourceLimits(wall_timeout_seconds=10))


_ALLOC_80MB = (
    "n = 80 * 1024 * 1024\n"
    "b = bytearray(n)\n"
    "for i in range(0, n, 4096):\n"
    "    b[i] = 1\n"
    "print('allocated')\n"
)


def test_peak_memory_reports_actual_allocation():
    result = _run(_ALLOC_80MB)
    assert result.exit_code == 0
    # 80MB touched + interpreter overhead, under the 512MB cap.
    assert 60.0 <= result.peak_memory_mb <= 400.0


def test_peak_memory_is_per_run_not_process_maximum():
    """The H3 discriminator: a small run after a big one in the SAME
    process must report its own small peak, not the big child's peak
    (the old RUSAGE_CHILDREN value never decreased)."""
    big = _run(_ALLOC_80MB)
    small = _run("print('small')\n")
    assert big.peak_memory_mb > 60.0
    assert small.peak_memory_mb < 50.0
    assert small.peak_memory_mb < big.peak_memory_mb


def test_returncode_normal_exit_zero():
    result = _run("print('ok')\n")
    assert result.exit_code == 0
    assert not result.timed_out


def test_returncode_normal_exit_one():
    result = _run("import sys\nsys.exit(1)\n")
    assert result.exit_code == 1


def test_returncode_sigterm():
    result = _run("import os, signal\nos.kill(os.getpid(), signal.SIGTERM)\n")
    assert result.exit_code == -signal.SIGTERM


def test_returncode_sigkill():
    result = _run("import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n")
    assert result.exit_code == -signal.SIGKILL


def test_timeout_reports_killed_and_peak():
    result = _run("import time\ntime.sleep(30)\n",
                  ResourceLimits(wall_timeout_seconds=0.5))
    assert result.timed_out
    assert result.limit_hit == "timeout"
    assert result.killed_signal == int(signal.SIGKILL)
    assert result.exit_code == -signal.SIGKILL
    # The killed child's peak-before-kill, not the old silent 0.0.
    assert result.peak_memory_mb > 0.0


def test_output_complete_after_child_exit():
    """No drain race: all 20000 lines, including the tail, are present
    (a drain thread still running at result time would truncate)."""
    lines = [f"line-{i:05d}" for i in range(20000)]
    script = "".join(f"print({line!r})\n" for line in lines)
    result = _run(script)
    assert result.exit_code == 0
    assert result.stdout.splitlines() == lines


def test_concurrent_executions_do_not_cross_contaminate():
    """Four execute() calls at once: each result carries its own
    child's output and peak — no shared buffers, no shared rusage."""
    jobs = [
        ("big-a", _ALLOC_80MB + "print('marker-big-a')\n", 60.0, None),
        ("big-b", _ALLOC_80MB + "print('marker-big-b')\n", 60.0, None),
        ("small-c", "print('marker-small-c')\n", None, 50.0),
        ("small-d", "print('marker-small-d')\n", None, 50.0),
    ]

    def task(job):
        marker, script, _, _ = job
        return marker, _run(script)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = dict(pool.map(task, jobs))
    for marker, script, floor, ceiling in jobs:
        result = results[marker]
        assert result.exit_code == 0
        assert f"marker-{marker}" in result.stdout
        if floor is not None:
            assert result.peak_memory_mb > floor
        if ceiling is not None:
            assert result.peak_memory_mb < ceiling
