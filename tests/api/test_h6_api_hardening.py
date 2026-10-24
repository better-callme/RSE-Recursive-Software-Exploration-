"""H6: API/resource-abuse hardening tests.

Request cap (declared, chunked, malformed), concurrency semaphore
(unit + 5-simultaneous integration), loopback warning, and the
process_limit removal. H6 is hardening, not a secure sandbox — these
tests bound the front door; they do not claim candidate isolation.
"""

import http.client
import io
import json
import threading
import types
import urllib.error
import urllib.request
from collections import Counter

import pytest

from forgemind.api.server import (
    MAX_REQUEST_BYTES,
    RequestTooLargeError,
    _read_body,
    _read_json,
    run_server,
    warn_if_non_localhost,
)
from forgemind.core.config import ResourceLimits


# ----------------------------------------------------------------------
# Body reading (unit level, fake handler)
# ----------------------------------------------------------------------

def _handler(headers, body: bytes):
    return types.SimpleNamespace(headers=headers, rfile=io.BytesIO(body))


def _chunked(*chunks: bytes) -> bytes:
    out = b""
    for chunk in chunks:
        out += f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n"
    return out + b"0\r\n\r\n"


def test_body_exact_cap_accepted():
    raw = _read_body(_handler({"Content-Length": "10"}, b"0123456789"), cap=10)
    assert raw == b"0123456789"


def test_body_over_declared_cap_rejected_before_reading():
    handler = _handler({"Content-Length": "11"}, b"0123456789A")
    with pytest.raises(RequestTooLargeError):
        _read_body(handler, cap=10)
    assert handler.rfile.tell() == 0, "oversized body must not be read"


def test_body_malformed_and_negative_length_400():
    with pytest.raises(ValueError):
        _read_body(_handler({"Content-Length": "not-a-number"}, b""), cap=10)
    with pytest.raises(ValueError):
        _read_body(_handler({"Content-Length": "-5"}, b""), cap=10)


def test_body_chunked_small_parses():
    raw = _read_body(_handler({"Transfer-Encoding": "chunked"},
                              _chunked(b'{"a":', b" 1}")), cap=1024)
    assert json.loads(raw) == {"a": 1}


def test_body_chunked_over_cap_rejected_mid_stream():
    handler = _handler({"Transfer-Encoding": "chunked"},
                       _chunked(b"AAAA", b"BBBBBB"))
    with pytest.raises(RequestTooLargeError):
        _read_body(handler, cap=8)
    # Second chunk's data was never consumed: the stream still holds it.
    assert handler.rfile.read(), "oversized chunk must stop consumption"


def test_body_without_length_not_consumed():
    handler = _handler({}, b"should-not-be-read")
    assert _read_body(handler, cap=10) == b""
    assert handler.rfile.tell() == 0


def test_read_json_small_body():
    body = json.dumps({"problem": "stats"}).encode()
    assert _read_json(_handler({"Content-Length": str(len(body))}, body)) == {
        "problem": "stats"}


# ----------------------------------------------------------------------
# Live server
# ----------------------------------------------------------------------

@pytest.fixture()
def live():
    server = run_server("127.0.0.1", 0)  # ephemeral port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _url(server):
    return f"http://127.0.0.1:{server.server_port}"


def _post(url, data: bytes):
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def test_live_oversized_declared_body_413(live, monkeypatch):
    monkeypatch.setattr("forgemind.api.server.MAX_REQUEST_BYTES", 1024)
    status, _ = _post(_url(live) + "/api/run", b" " * 1025)
    assert status == 413


def test_live_exact_cap_body_accepted(live, monkeypatch):
    monkeypatch.setattr("forgemind.api.server.MAX_REQUEST_BYTES", 1024)
    body = json.dumps({"problem": "stats"}).encode()
    body = body + b" " * (1024 - len(body))  # trailing JSON whitespace
    assert len(body) == 1024
    status, _ = _post(_url(live) + "/api/run", body)
    assert status == 200


def test_live_chunked_oversized_body_413(live, monkeypatch):
    monkeypatch.setattr("forgemind.api.server.MAX_REQUEST_BYTES", 1024)
    conn = http.client.HTTPConnection("127.0.0.1", live.server_port,
                                      timeout=30)
    conn.putrequest("POST", "/api/run")
    conn.putheader("Transfer-Encoding", "chunked")
    conn.putheader("Content-Type", "application/json")
    conn.endheaders()
    conn.send(b"400\r\n" + b"x" * 1024 + b"\r\n")
    conn.send(b"400\r\n" + b"y" * 1024 + b"\r\n")
    conn.send(b"0\r\n\r\n")
    resp = conn.getresponse()
    assert resp.status == 413
    resp.read()
    conn.close()


def test_semaphore_saturated_503_and_health_200(live):
    sem = live.execution_semaphore
    for _ in range(4):
        assert sem.acquire(blocking=False)
    try:
        status, _ = _post(_url(live) + "/api/run",
                          json.dumps({"problem": "stats"}).encode())
        assert status == 503
        with urllib.request.urlopen(_url(live) + "/api/health") as resp:
            assert resp.status == 200
    finally:
        for _ in range(4):
            sem.release()


def test_five_simultaneous_runs_four_enter_one_503(live):
    barrier = threading.Barrier(5)
    results: list[int] = []

    def run_once():
        barrier.wait()
        status, _ = _post(_url(live) + "/api/run",
                          json.dumps({"problem": "stats"}).encode())
        results.append(status)

    threads = [threading.Thread(target=run_once) for _ in range(5)]
    for t in threads:
        t.start()
    # Health must stay responsive while the four runs are in flight.
    threading.Event().wait(0.3)
    with urllib.request.urlopen(_url(live) + "/api/health") as resp:
        assert resp.status == 200
    for t in threads:
        t.join()
    assert Counter(results) == Counter({200: 4, 503: 1})


# ----------------------------------------------------------------------
# Loopback warning + process_limit removal
# ----------------------------------------------------------------------

@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_hosts_no_warning(host, capsys):
    assert warn_if_non_localhost(host) is False
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10",
                                  "10.0.0.5", "example.com"])
def test_non_loopback_hosts_warn(host, capsys):
    assert warn_if_non_localhost(host) is True
    err = capsys.readouterr().err
    assert "non-loopback" in err
    assert "Do not expose this service to untrusted networks" in err


def test_process_limit_removed():
    assert not hasattr(ResourceLimits(), "process_limit")


def test_default_request_cap_is_1mib():
    assert MAX_REQUEST_BYTES == 1024 * 1024
