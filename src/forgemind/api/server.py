"""ForgeMind HTTP API server (stdlib-only).

Exposes the same operations as the CLI over HTTP:

    GET  /api/health    -> {"status": "ok", "version": ...}
    GET  /api/problems  -> {"problems": ["stats", ...]}
    POST /api/run       -> run ForgeMind search on one problem
    POST /api/trace     -> run search and return the full event trace
    POST /api/benchmark -> baseline vs ForgeMind comparison

Request bodies are JSON, e.g.::

    {"problem": "stats", "seed": 0, "beam_width": 3, "max_depth": 6}
"""

from __future__ import annotations

import dataclasses
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

import forgemind
from forgemind.benchmark.experiment import ExperimentRunner
from forgemind.benchmark.metrics import record_from_search
from forgemind.core.config import ForgeMindConfig
from forgemind.core.history import History
from forgemind.recovery.failure import RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee


def list_problems() -> list[str]:
    """Problem ids exposed by the API. Mirrors forgemind.cli's registry."""
    from forgemind.cli import PROBLEMS

    return sorted(PROBLEMS)


def _problem_spec(name: str):
    from forgemind.cli import problem_spec

    return problem_spec(name)


def _make_engine(spec, config: ForgeMindConfig) -> SearchEngine:
    env = SubprocessExecutionEnvironment(config.resources)
    # Test sources come from spec.test_sources (H2); no provider wiring
    # is needed here — the referee reads them from the spec.
    referee = Referee(spec, env, config)
    return SearchEngine(
        spec, config, referee=referee,
        recovery=RecoveryPolicy(), history=History(),
    )


def api_run(problem: str = "stats", seed: int = 0,
            beam_width: int = 3, max_depth: int = 6) -> dict[str, Any]:
    spec = _problem_spec(problem)
    config = ForgeMindConfig(random_seed=seed, beam_width=beam_width,
                             max_depth=max_depth)
    engine = _make_engine(spec, config)
    result = engine.run()
    record = dataclasses.asdict(record_from_search(spec.problem_id, result))
    node = result.accepted_node
    accepted_files: dict[str, str] = {}
    if node is not None and node.state.files:
        for path in sorted(node.state.files):
            accepted_files[path] = node.state.files[path]
    return {
        "record": record,
        "summary": result.summary(),
        "accepted_files": accepted_files,
    }


def api_trace(problem: str = "stats", seed: int = 0,
              beam_width: int = 3, max_depth: int = 6) -> dict[str, Any]:
    spec = _problem_spec(problem)
    config = ForgeMindConfig(random_seed=seed, beam_width=beam_width,
                             max_depth=max_depth)
    engine = _make_engine(spec, config)
    result = engine.run()
    record = dataclasses.asdict(record_from_search(spec.problem_id, result))
    return {
        "record": record,
        "trace": engine.history.render_trace(),
        "summary": result.summary(),
    }


def api_benchmark(problems: list[str] | None = None, seed: int = 0) -> dict[str, Any]:
    from forgemind.benchmark.problems import list_problem_names

    names = problems or list_problem_names()
    runner = ExperimentRunner(ForgeMindConfig(random_seed=seed))
    suite = runner.run_suite(names)
    return suite.to_json_dict()


# --------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------

def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8")) if raw.strip() else {}


_ROUTES: dict[tuple[str, str], Callable[[dict[str, Any]], Any]] = {}


def route(method: str, path: str):
    def decorator(fn):
        _ROUTES[(method.upper(), path)] = fn
        return fn
    return decorator


@route("GET", "/api/health")
def _health(_body: dict[str, Any]) -> dict[str, Any]:
    return {"status": "ok", "version": forgemind.__version__}


@route("GET", "/api/problems")
def _problems(_body: dict[str, Any]) -> dict[str, Any]:
    return {"problems": list_problems()}


@route("POST", "/api/run")
def _run(body: dict[str, Any]) -> dict[str, Any]:
    return api_run(
        problem=body.get("problem", "stats"),
        seed=int(body.get("seed", 0)),
        beam_width=int(body.get("beam_width", 3)),
        max_depth=int(body.get("max_depth", 6)),
    )


@route("POST", "/api/trace")
def _trace(body: dict[str, Any]) -> dict[str, Any]:
    return api_trace(
        problem=body.get("problem", "stats"),
        seed=int(body.get("seed", 0)),
        beam_width=int(body.get("beam_width", 3)),
        max_depth=int(body.get("max_depth", 6)),
    )


@route("POST", "/api/benchmark")
def _benchmark(body: dict[str, Any]) -> dict[str, Any]:
    problems = body.get("problems")
    if problems is not None and not isinstance(problems, list):
        raise ValueError("'problems' must be a list of problem ids")
    return api_benchmark(
        problems=problems,
        seed=int(body.get("seed", 0)),
    )


class ForgeMindHandler(BaseHTTPRequestHandler):
    server_version = f"ForgeMindAPI/{forgemind.__version__}"

    def _send(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _dispatch(self, method: str) -> None:
        path = urlparse(self.path).path
        handler = _ROUTES.get((method, path))
        if handler is None:
            self._send(404, {"error": f"unknown endpoint: {method} {path}"})
            return
        try:
            body = _read_json(self) if method == "POST" else {}
        except (json.JSONDecodeError, ValueError):
            self._send(400, {"error": "invalid JSON request body"})
            return
        try:
            self._send(200, handler(body))
        except KeyError as exc:
            self._send(404, {"error": f"unknown problem: {exc}"})
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        pass


def run_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """Create (but do not start) the ForgeMind API server."""
    return ThreadingHTTPServer((host, port), ForgeMindHandler)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = run_server(host, port)
    print(f"ForgeMind API listening on http://{host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
