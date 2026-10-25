"""L4: CLI tests — main(argv) exercised directly, no CLI refactor."""

import json

import pytest

from forgemind.cli import build_parser, main


def test_run_stats(capsys):
    assert main(["run", "--problem", "stats"]) == 0
    out = capsys.readouterr().out
    assert "accepted file: solution.py" in out


def test_trace_stats(capsys):
    assert main(["trace", "--problem", "stats"]) == 0
    out = capsys.readouterr().out
    assert "SEARCH_STARTED" in out or "SEARCH_TERMINATED" in out


def test_benchmark_writes_report(tmp_path):
    out_file = tmp_path / "report.json"
    assert main(["benchmark", "--problem", "stats",
                 "--output", str(out_file)]) == 0
    raw = out_file.read_text()
    json.loads(raw)  # must be valid JSON
    assert "stats" in raw


def test_invalid_problem_exits():
    with pytest.raises(SystemExit):
        main(["run", "--problem", "bogus"])


def test_missing_command_exits():
    with pytest.raises(SystemExit):
        main([])


def test_serve_args_parse():
    args = build_parser().parse_args(
        ["serve", "--host", "127.0.0.1", "--port", "8123"])
    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 8123
