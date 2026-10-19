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
    status, payload = _post(api_base, "/api/benchmark", {"problem": "stats"})
    assert status == 200
    assert payload["comparison"]["delta_success"] is not None
    assert payload["forgemind"]["success_rate"] is not None
    assert payload["baseline"]["success_rate"] is not None


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
