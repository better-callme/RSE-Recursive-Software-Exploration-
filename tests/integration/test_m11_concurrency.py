"""M11: concurrent engines must not interfere with each other.

The audit's thread-safety complaint named two shared mechanisms: the
process-global ID counter (H4) and process-lifetime rusage (H3). Both
were made per-run by those fixes; this test proves the resulting
isolation at engine level and guards the closure. It changes no
production code — that is the point (see README "Concurrency").

Isolation asserted here, per Kashi's approval requirement:
- concurrent run's candidate-ID sequence == its solo sequence, exactly
  and in order (the universal property: concurrency must not perturb
  a run). This FAILS on the old mechanism: the process-global legacy
  allocator, used from two threads with no per-run allocator installed,
  shifts each thread's sequence (validated by a scratch run, not by
  this test — see the design doc).
- cross-run disjointness, with a caveat stated plainly: disjointness
  is NOT a general system property. Same-seed runs produce identical
  ID sequences by design (deterministic, seed-derived starts). The
  test earns disjointness by choosing far-apart seeds (0 vs
  50,000,000, with a few dozen IDs per run).
- both verdicts == solo verdicts.
- every reported per-node verification peak > 0. Engine-level peak
  isolation is deliberately loose: exact RSS equality across runs is
  jittery, and per-child peak correctness under concurrency is pinned
  by H3's environment-level test (4 simultaneous execute() calls,
  distinct markers/peaks). This test proves ID + verdict isolation;
  H3's test proves peak isolation.

The concurrent pair runs 5 times: the old bug class is
scheduling-dependent, so one pair proves little.
"""

import threading

from forgemind.benchmark.kits import mock_kit_for
from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig
from forgemind.recovery.failure import RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee

_SEED_A = 0
_SEED_B = 50_000_000
_REPETITIONS = 5


def _run_stats(seed):
    """One full stats run. Returns (candidate IDs in creation order,
    success, per-node verification peaks)."""
    problem = StatsProblem()
    spec = problem.spec
    kit = mock_kit_for("stats")
    env = SubprocessExecutionEnvironment()
    cfg = ForgeMindConfig(random_seed=seed)
    referee = Referee(spec, env, cfg)
    referee.attach_test_provider(problem.test_files)
    engine = SearchEngine(
        spec, cfg,
        architect=kit.architect(), builder=kit.builder(),
        critic=kit.critic(), optimizer=kit.optimizer(),
        referee=referee, recovery=RecoveryPolicy(),
    )
    result = engine.run()
    cand_ids = [
        node.candidate_id for node in result.graph.nodes.values()
        if node.candidate_id is not None
    ]
    peaks = [
        node.verification.peak_memory_mb
        for node in result.graph.nodes.values()
        if node.verification is not None
    ]
    return cand_ids, result.success, peaks


def test_concurrent_engines_are_isolated():
    solo_a = _run_stats(_SEED_A)
    solo_b = _run_stats(_SEED_B)
    assert solo_a[1] and solo_b[1]
    assert set(solo_a[0]).isdisjoint(solo_b[0])

    for _ in range(_REPETITIONS):
        results = {}
        errors = {}

        def _worker(name, seed):
            try:
                results[name] = _run_stats(seed)
            except Exception as exc:  # re-raised on the test thread
                errors[name] = exc

        threads = [
            threading.Thread(target=_worker, args=("a", _SEED_A)),
            threading.Thread(target=_worker, args=("b", _SEED_B)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        conc_a, conc_b = results["a"], results["b"]
        # (a) isolation: concurrency must not perturb a run's IDs.
        assert conc_a[0] == solo_a[0]
        assert conc_b[0] == solo_b[0]
        # (b) disjointness under far-apart seeds.
        assert set(conc_a[0]).isdisjoint(conc_b[0])
        # (c) verdicts unchanged by concurrency.
        assert conc_a[1] == solo_a[1] and conc_b[1] == solo_b[1]
        # (d) peaks present (per-child correctness: H3's env test).
        assert conc_a[2] and all(p > 0 for p in conc_a[2])
        assert conc_b[2] and all(p > 0 for p in conc_b[2])
