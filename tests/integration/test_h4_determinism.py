"""H4: runs are reproducible — per-run IDs, timeout inconclusiveness,
quarantine lifecycle, and seed semantics."""

import signal
from concurrent.futures import ThreadPoolExecutor

from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig
from forgemind.core.history import EventType
from forgemind.core.proposal import CodePatchProposal
from forgemind.recovery.failure import RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.search.node import NodeStatus
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee
from forgemind.verification.results import ExecutionResult


class FlakyEnvironment:
    """Test-only wrapper: force timed_out for the first `fail_count`
    execute() calls, then delegate to the real environment. This makes
    attempt-1 vs attempt-2 behavior deterministic without racing clocks."""

    def __init__(self, delegate, fail_count):
        self._delegate = delegate
        self._fail_count = fail_count
        self.calls = 0

    def materialize_state(self, state):
        return self._delegate.materialize_state(state)

    def cleanup(self, workdir):
        return self._delegate.cleanup(workdir)

    def execute(self, workdir, command, limits=None):
        self.calls += 1
        if self.calls <= self._fail_count:
            return ExecutionResult(
                exit_code=-signal.SIGKILL, stdout="", stderr="",
                timed_out=True, runtime_ms=0.0,
                killed_signal=int(signal.SIGKILL), limit_hit="timeout")
        return self._delegate.execute(workdir, command, limits)


def _engine(spec, seed, env, builder=None):
    cfg = ForgeMindConfig(random_seed=seed)
    referee = Referee(spec, env, cfg)
    referee.attach_test_provider(StatsProblem().test_files)
    kwargs = {} if builder is None else {"builder": builder}
    return SearchEngine(spec, cfg, referee=referee,
                        recovery=RecoveryPolicy(), **kwargs)


def _digest(result):
    cand_ids = tuple(n.candidate_id for n in result.graph.nodes.values()
                     if n.candidate_id)
    accepted_hash = (result.accepted_node.state.content_hash
                     if result.accepted_node else None)
    return (cand_ids, accepted_hash, result.termination.reason)


def _verify_starts(engine, node_id):
    return [e for e in engine.history.events
            if e.event_type is EventType.VERIFICATION_STARTED
            and e.metadata.get("node") == node_id]


def test_sequential_fresh_engines_same_seed_identical(spec):
    """Fresh engine per run, same seed, no manual reset: identical IDs,
    accepted artifact, and termination — back-to-back and across runs."""
    digests = []
    for _ in range(3):
        eng = _engine(spec, 7, SubprocessExecutionEnvironment())
        digests.append(_digest(eng.run()))
    assert digests[0] == digests[1] == digests[2]


def test_concurrent_engines_isolated_same_seed(spec):
    """A(seed=42) and B(seed=42) concurrently: identical IDs, accepted
    artifact, digest. C(seed=43): a different digest, via the
    seed-derived allocator start."""
    def run_with_seed(seed):
        eng = _engine(spec, seed, SubprocessExecutionEnvironment())
        return _digest(eng.run())

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(run_with_seed, [42, 42]))
    assert a == b
    assert a[0] == b[0]  # candidate ID sequences identical
    c = run_with_seed(43)
    assert c != a


def test_timeout_retry_then_accept(spec):
    """Good candidate slow on attempt 1 only: inconclusive, retry
    succeeds, run succeeds — and the retry is observable (exactly one
    node verified twice: the timed-out attempt plus its retry)."""
    flaky = FlakyEnvironment(SubprocessExecutionEnvironment(), fail_count=1)
    eng = _engine(spec, 7, flaky)
    res = eng.run()
    assert res.success
    assert not res.quarantined
    starts = {}
    for e in eng.history.events:
        if e.event_type is EventType.VERIFICATION_STARTED:
            starts[e.metadata["node"]] = starts.get(e.metadata["node"], 0) + 1
    assert max(starts.values()) == 2


BAD_MEAN = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t / len(data) + 1}\n"
)


class SingleBadBuilder:
    role = "builder"

    def propose(self, projection, request_context):
        return (CodePatchProposal(
            agent_role=self.role, intent="bad",
            rationale="timeout-fail scenario", estimated_cost=1.0,
            created_files={"solution.py": BAD_MEAN}),)


def test_timeout_retry_then_fail(spec):
    """Bad candidate: timeout, retry, then the real verdict stands — the
    retry must not magically turn a timeout into acceptance."""
    flaky = FlakyEnvironment(SubprocessExecutionEnvironment(), fail_count=1)
    eng = _engine(spec, 7, flaky, builder=SingleBadBuilder())
    res = eng.run()
    assert not res.success
    assert not res.quarantined
    assert not any(e.event_type is EventType.QUARANTINED
                   for e in eng.history.events)
    failures = [e for e in eng.history.events
                if e.event_type is EventType.FAILURE_RECORDED]
    assert any(e.metadata.get("type") == "TEST_FAILURE" for e in failures)
    bad_id = next(n.node_id for n in res.graph.nodes.values()
                  if n.candidate_id and n.depth == 1)
    assert len(_verify_starts(eng, bad_id)) == 2


def test_timeout_twice_quarantined_not_rejected(spec):
    """Every child slow on both attempts: quarantined, never accepted,
    never recorded as a failure — and the reason is honest."""
    flaky = FlakyEnvironment(SubprocessExecutionEnvironment(), fail_count=99)
    eng = _engine(spec, 7, flaky)
    res = eng.run()
    assert not res.success
    assert res.termination.reason == "quarantined"
    children = [n for n in res.graph.nodes.values() if n.depth == 1]
    assert children and len(res.quarantined) == len(children)
    assert all(r.attempts == 2 for r in res.quarantined)
    assert not any(e.event_type is EventType.FAILURE_RECORDED
                   for e in eng.history.events)
    assert any(e.event_type is EventType.QUARANTINED
               for e in eng.history.events)
    assert len(eng.frontier) == 0
    for record in res.quarantined:
        assert res.graph.nodes[record.node_id].status is NodeStatus.QUARANTINED


def test_quarantined_state_readmitted_in_new_instance(spec):
    """Quarantine is not a blacklist: a fresh engine instance with a
    normal environment accepts the same problem."""
    flaky = FlakyEnvironment(SubprocessExecutionEnvironment(), fail_count=99)
    eng1 = _engine(spec, 7, flaky)
    res1 = eng1.run()
    assert not res1.success and res1.termination.reason == "quarantined"
    eng2 = _engine(spec, 7, SubprocessExecutionEnvironment())
    res2 = eng2.run()
    assert res2.success
