"""Tests for the ForgeMind HTTP API (stdlib-only server)."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.client import HTTPResponse

import pytest

from forgemind.api.server import run_server


@pytest.fixture(scope="module")
def api_base():
    server = run_server("127.0.0.1", 0)  # ephemeral port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _get(base: str, path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(base + path) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _post(base: str, path: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


import urllib.error  # noqa: E402


def test_health(api_base):
    status, payload = _get(api_base, "/api/health")
    assert status == 200
    assert payload["status"] == "ok"
    assert payload["version"]


def test_problems(api_base):
    status, payload = _get(api_base, "/api/problems")
    assert status == 200
    assert "stats" in payload["problems"]


def test_run(api_base):
    status, payload = _post(api_base, "/api/run", {"problem": "stats"})
    assert status == 200
    assert payload["record"]["runner"] == "forgemind"
    assert "SEARCH COMPLETE" in payload["summary"]
    assert isinstance(payload["accepted_files"], dict)


def test_trace(api_base):
    status, payload = _post(api_base, "/api/trace", {"problem": "stats"})
    assert status == 200
    assert payload["trace"]
    assert "SEARCH_STARTED" in payload["trace"] or "SEARCH_TERMINATED" in payload["trace"]
    assert "SEARCH COMPLETE" in payload["summary"]


def test_benchmark(api_base):
    status, payload = _post(api_base, "/api/benchmark", {})
    assert status == 200
    assert payload["problem_names"] == ["stats", "stats-trap"]
    assert len(payload["per_problem"]) == 2
    agg = payload["aggregate"]
    assert agg["delta_success"] is not None
    assert agg["forgemind_success_rate"] is not None
    assert agg["baseline_success_rate"] is not None


def test_benchmark_single_problem(api_base):
    status, payload = _post(api_base, "/api/benchmark",
                            {"problems": ["stats-trap"]})
    assert status == 200
    assert payload["problem_names"] == ["stats-trap"]
    row = payload["per_problem"][0]
    assert row["forgemind"]["success"] is True
    assert row["baseline"]["success"] is False


def test_benchmark_unknown_problem_is_404(api_base):
    status, payload = _post(api_base, "/api/benchmark",
                            {"problems": ["nope"]})
    assert status == 404
    assert "error" in payload


def test_unknown_problem_is_404(api_base):
    status, payload = _post(api_base, "/api/run", {"problem": "nope"})
    assert status == 404
    assert "error" in payload


def test_invalid_json_is_400(api_base):
    req = urllib.request.Request(api_base + "/api/run", data=b"{not json",
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req)
    except urllib.error.HTTPError as exc:
        assert exc.code == 400
    else:
        pytest.fail("expected HTTP 400")


def test_internal_error_is_generic_but_logged(api_base, monkeypatch, caplog):
    """L5: both sides of the security boundary. The client gets exactly
    a generic body (no exception class, no message); the server log
    gets the full traceback including the marker. Pre-fix the client
    body contained both 'RuntimeError' and the marker."""
    import logging

    from forgemind.api.server import _ROUTES

    def _boom(_body):
        raise RuntimeError("secret-marker")

    monkeypatch.setitem(_ROUTES, ("GET", "/api/health"), (_boom, False))
    with caplog.at_level(logging.ERROR, logger="forgemind.api.server"):
        status, payload = _get(api_base, "/api/health")
    assert status == 500
    assert payload == {"error": "internal error"}
    assert "secret-marker" not in json.dumps(payload)
    assert "RuntimeError" not in json.dumps(payload)
    assert "secret-marker" in caplog.text


def test_serve_wrapper_returns_and_prints(monkeypatch, capsys):
    """L4: the thin serve() wrapper (run_server + warning + print +
    serve_forever + close) was the only untested part of the server
    bootstrap — run_server itself is live-tested above. serve_forever
    is monkeypatched to a no-op so this never blocks, binds, or races
    on a port."""
    from http.server import ThreadingHTTPServer

    from forgemind.api.server import serve
    monkeypatch.setattr(ThreadingHTTPServer, "serve_forever",
                        lambda self: None)
    serve(host="127.0.0.1", port=0)
    out = capsys.readouterr().out
    assert "ForgeMind API listening on http://127.0.0.1:" in out
