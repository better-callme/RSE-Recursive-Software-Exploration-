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
import ipaddress
import json
import sys
import threading
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


def _make_engine(spec, config: ForgeMindConfig, kit=None) -> SearchEngine:
    env = SubprocessExecutionEnvironment(config.resources)
    # Test sources come from spec.test_sources (H2); no provider wiring
    # is needed here — the referee reads them from the spec.
    referee = Referee(spec, env, config)
    # H5: when the caller resolved a kit, pass its agents explicitly;
    # otherwise the engine's legacy defaults apply (stats family).
    agents = {}
    if kit is not None:
        agents = {"architect": kit.architect(), "builder": kit.builder(),
                  "critic": kit.critic(), "optimizer": kit.optimizer()}
    return SearchEngine(
        spec, config, referee=referee,
        recovery=RecoveryPolicy(), history=History(), **agents,
    )


def api_run(problem: str = "stats", seed: int = 0,
            beam_width: int = 3, max_depth: int = 6) -> dict[str, Any]:
    spec = _problem_spec(problem)
    config = ForgeMindConfig(random_seed=seed, beam_width=beam_width,
                             max_depth=max_depth)
    from forgemind.benchmark.kits import mock_kit_for

    engine = _make_engine(spec, config, kit=mock_kit_for(problem))
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

# H6: API/resource-abuse hardening (NOT a secure sandbox — see README).
# Request bodies are capped, enforced incrementally while reading,
# never by reading everything and rejecting afterwards.
MAX_REQUEST_BYTES = 1024 * 1024
_READ_CHUNK = 64 * 1024


class RequestTooLargeError(Exception):
    """Request body exceeds MAX_REQUEST_BYTES (→ HTTP 413)."""


def _read_exact(rfile, nbytes: int) -> bytes:
    """Read exactly nbytes in bounded chunks (no single huge alloc)."""
    chunks: list[bytes] = []
    remaining = nbytes
    while remaining > 0:
        chunk = rfile.read(min(_READ_CHUNK, remaining))
        if not chunk:
            break  # client closed early; caller parses what arrived
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_crlf(rfile) -> None:
    """Consume one chunked-body line ending (CRLF, or bare LF)."""
    first = rfile.read(1)
    if first == b"\r":
        rfile.read(1)


def _read_chunked(rfile, cap: int) -> bytes:
    """Incrementally de-chunk a Transfer-Encoding: chunked body,
    enforcing cap as bytes arrive. Raises RequestTooLargeError the
    moment the cap would be exceeded, without consuming the rest."""
    chunks: list[bytes] = []
    total = 0
    while True:
        line = rfile.readline(1024)
        if not line:
            break
        try:
            size = int(line.split(b";")[0].strip(), 16)
        except ValueError:
            raise ValueError("malformed chunk size") from None
        if size == 0:
            # Consume trailers up to the blank line.
            while True:
                trailer = rfile.readline(1024)
                if trailer in (b"\r\n", b"\n", b""):
                    break
            break
        if total + size > cap:
            raise RequestTooLargeError(
                f"chunked body exceeds {cap} bytes")
        chunks.append(_read_exact(rfile, size))
        total += size
        _read_crlf(rfile)
    return b"".join(chunks)


def _read_body(handler: BaseHTTPRequestHandler,
               cap: int = MAX_REQUEST_BYTES) -> bytes:
    """Read one request body with the cap enforced during consumption.

    - Content-Length above the cap → 413 before reading a byte.
    - Content-Length within the cap → exactly that many bytes, in
      bounded chunks.
    - Transfer-Encoding: chunked → de-chunked incrementally, cap
      enforced per chunk.
    - Neither → body is NOT consumed. Length-less bodies are
      unsupported: reading to EOF would hang on keep-alive
      connections. Callers see an empty body.
    """
    transfer = (handler.headers.get("Transfer-Encoding") or "").lower()
    if "chunked" in transfer:
        return _read_chunked(handler.rfile, cap)
    raw_length = handler.headers.get("Content-Length")
    if raw_length is None:
        return b""
    try:
        length = int(raw_length)
    except ValueError:
        raise ValueError("malformed Content-Length") from None
    if length < 0:
        raise ValueError("negative Content-Length")
    if length > cap:
        raise RequestTooLargeError(
            f"Content-Length {length} exceeds {cap} bytes")
    return _read_exact(handler.rfile, length)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    # Cap resolved from the module global at call time (tests monkeypatch
    # MAX_REQUEST_BYTES for small-cap live-server checks).
    raw = _read_body(handler, cap=MAX_REQUEST_BYTES)
    return json.loads(raw.decode("utf-8")) if raw.strip() else {}


_ROUTES: dict[tuple[str, str], tuple[Callable, bool]] = {}


def route(method: str, path: str, *, gated: bool = False):
    """Register a route. gated=True marks an execution route (run /
    trace / benchmark) subject to the concurrency semaphore; health
    and problem listing stay ungated and always responsive. The flag
    is explicit at registration — never inferred from endpoint names
    in the dispatcher."""
    def decorator(fn):
        _ROUTES[(method.upper(), path)] = (fn, gated)
        return fn
    return decorator


@route("GET", "/api/health")
def _health(_body: dict[str, Any]) -> dict[str, Any]:
    return {"status": "ok", "version": forgemind.__version__}


@route("GET", "/api/problems")
def _problems(_body: dict[str, Any]) -> dict[str, Any]:
    return {"problems": list_problems()}


@route("POST", "/api/run", gated=True)
def _run(body: dict[str, Any]) -> dict[str, Any]:
    return api_run(
        problem=body.get("problem", "stats"),
        seed=int(body.get("seed", 0)),
        beam_width=int(body.get("beam_width", 3)),
        max_depth=int(body.get("max_depth", 6)),
    )


@route("POST", "/api/trace", gated=True)
def _trace(body: dict[str, Any]) -> dict[str, Any]:
    return api_trace(
        problem=body.get("problem", "stats"),
        seed=int(body.get("seed", 0)),
        beam_width=int(body.get("beam_width", 3)),
        max_depth=int(body.get("max_depth", 6)),
    )


@route("POST", "/api/benchmark", gated=True)
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
        entry = _ROUTES.get((method, path))
        if entry is None:
            self._send(404, {"error": f"unknown endpoint: {method} {path}"})
            return
        handler, gated = entry
        try:
            body = _read_json(self) if method == "POST" else {}
        except RequestTooLargeError:
            self._send(413, {"error": "request body too large"})
            return
        except (json.JSONDecodeError, ValueError):
            self._send(400, {"error": "invalid JSON request body"})
            return
        # H6: bound concurrent execution work. Four is an intentionally
        # conservative application-level concurrency cap for the current
        # research workload; it is not a guarantee of OS-level resource
        # containment (H4 isolates engine state, not OS resources).
        # No queueing: a run takes minutes, so a queue would convert
        # CPU/process exhaustion into memory exhaustion.
        semaphore = getattr(self.server, "execution_semaphore", None)
        acquired = False
        if gated and semaphore is not None:
            acquired = semaphore.acquire(blocking=False)
            if not acquired:
                self._send(503, {"error": "server busy"})
                return
        try:
            self._send(200, handler(body))
        except KeyError as exc:
            self._send(404, {"error": f"unknown problem: {exc}"})
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
        finally:
            if acquired:
                semaphore.release()

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        pass


def is_loopback_host(host: str) -> bool:
    """Proper loopback classification (H6): never string equality
    against "127.0.0.1". Literal "localhost" counts as loopback;
    anything unparseable fails closed to non-loopback (warn)."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def warn_if_non_localhost(host: str) -> bool:
    """Print the explicit non-loopback warning; True if warned. The
    warning is an operator safeguard, NOT a security boundary: the API
    provides no authentication and non-loopback binding does not
    constitute a secure remote sandbox."""
    if is_loopback_host(host):
        return False
    print(
        "WARNING: RSE API is binding to a non-loopback interface. "
        "Requests can execute untrusted code on this machine. "
        "Do not expose this service to untrusted networks.",
        file=sys.stderr,
    )
    return True


def run_server(host: str = "127.0.0.1", port: int = 8000,
               *, max_concurrent: int = 4) -> ThreadingHTTPServer:
    """Create (but do not start) the ForgeMind API server."""
    server = ThreadingHTTPServer((host, port), ForgeMindHandler)
    server.execution_semaphore = threading.BoundedSemaphore(max_concurrent)
    return server


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = run_server(host, port)
    warn_if_non_localhost(host)
    print(f"ForgeMind API listening on http://{host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
