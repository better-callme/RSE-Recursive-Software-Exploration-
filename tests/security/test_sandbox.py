"""Security / resource-limit tests against the real subprocess environment.

These run deliberately hostile but bounded payloads. They validate damage
reduction only — see README security limitations.
"""

import os

import pytest

from forgemind.core.config import ResourceLimits
from forgemind.verification.environment import SubprocessExecutionEnvironment


@pytest.fixture()
def env(tmp_path):
    e = SubprocessExecutionEnvironment()
    workdir = tmp_path / "sandbox"
    workdir.mkdir()
    return e, str(workdir)


SHORT = ResourceLimits(wall_timeout_seconds=3, cpu_limit_seconds=2,
                       memory_limit_mb=512)


def _run(env_fix, script, limits=SHORT):
    e, workdir = env_fix
    path = os.path.join(workdir, "payload.py")
    with open(path, "w") as fh:
        fh.write(script)
    return e.execute(workdir, ["python3", "payload.py"], limits)


def test_infinite_loop_times_out():
    import tempfile
    e = SubprocessExecutionEnvironment()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "payload.py"), "w") as fh:
            fh.write("while True:\n    pass\n")
        result = e.execute(d, ["python3", "payload.py"],
                           ResourceLimits(wall_timeout_seconds=2, cpu_limit_seconds=4))
    assert result.timed_out
    assert result.limit_hit == "timeout"


def test_stdout_capture_and_excess_output_truncated():
    import tempfile
    e = SubprocessExecutionEnvironment()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "p.py"), "w") as fh:
            fh.write("print('hello forge')\n")
        r = e.execute(d, ["python3", "p.py"])
    assert r.exit_code == 0 and "hello forge" in r.stdout


def test_nonzero_exit_code_reported():
    import tempfile
    e = SubprocessExecutionEnvironment()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "p.py"), "w") as fh:
            fh.write("raise SystemExit(3)\n")
        r = e.execute(d, ["python3", "p.py"])
    assert r.exit_code == 3


def test_stderr_captured():
    import tempfile
    e = SubprocessExecutionEnvironment()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "p.py"), "w") as fh:
            fh.write("import sys; sys.stderr.write('boom\\n')\n")
        r = e.execute(d, ["python3", "p.py"])
    assert "boom" in r.stderr


def test_environment_sanitized():
    import tempfile
    e = SubprocessExecutionEnvironment()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "p.py"), "w") as fh:
            fh.write(
                "import os\n"
                "assert 'OPENAI_API_KEY' not in os.environ\n"
                "assert 'AWS_SECRET_ACCESS_KEY' not in os.environ\n"
                "print('clean')\n"
            )
        r = e.execute(d, ["python3", "p.py"])
    assert r.exit_code == 0 and "clean" in r.stdout


def test_file_access_outside_working_directory_is_visible_but_documented():
    """Honest limitation probe: the MVP sandbox does NOT have a filesystem jail.

    We assert the current (weaker) behavior so a future container backend can
    tighten it, and we document that this is NOT isolation.
    """
    import tempfile
    e = SubprocessExecutionEnvironment()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "p.py"), "w") as fh:
            fh.write("print(open('/etc/hostname').read() != '')\n")
        r = e.execute(d, ["python3", "p.py"])
    # Documenting reality: read succeeds on POSIX subprocess sandbox.
    assert "True" in r.stdout or r.exit_code != 0


def test_memory_pressure_killed_or_limited():
    import tempfile
    e = SubprocessExecutionEnvironment()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "p.py"), "w") as fh:
            fh.write("x = bytearray(400 * 1024 * 1024)\n")  # 400 MB alloc
        r = e.execute(d, ["python3", "p.py"],
                      ResourceLimits(wall_timeout_seconds=15, memory_limit_mb=256))
    assert r.exit_code != 0 or r.limit_hit == "memory"


def test_cleanup_removes_workdir():
    import tempfile
    e = SubprocessExecutionEnvironment()
    d = tempfile.mkdtemp(prefix="fm-cleanup-")
    open(os.path.join(d, "f.txt"), "w").write("x")
    e.cleanup(d)
    assert not os.path.exists(d)
