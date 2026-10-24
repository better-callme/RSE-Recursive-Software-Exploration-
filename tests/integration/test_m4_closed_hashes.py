"""M4: close hashes only on definitive, complete verdicts.

seen = currently-live/claimed hashes; closed = hashes with a
definitive state verdict. Quarantine and depth-exhaustion release
seen without closing; reverify_duplicates bypasses closed only.
"""

import os
import signal

from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig
from forgemind.core.history import EventType
from forgemind.core.proposal import (
    AttackTestProposal,
    CodePatchProposal,
    CritiqueProposal,
)
from forgemind.core.state import CodebaseState
from forgemind.recovery.failure import RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.search.node import SearchFrontier, SearchNode
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee
from forgemind.verification.results import ExecutionResult

_HEADER = (
    "import struct\n\n\n"
    "def _f32(x):\n"
    "    return struct.unpack('f', struct.pack('f', x))[0]\n\n\n"
)

GOOD = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t / len(data)}\n"
)

_X_BODY = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': _f32(t / len(data))}\n"
)


def _x_code(marker):
    return f"# variant {marker}\n" + _HEADER + _X_BODY


# X body with no marker: identical from every parent.
_X_PLAIN = _HEADER + _X_BODY

# Deterministic acceptance failure, complete evidence.
_BAD_MEAN = GOOD.replace("t / len(data)", "t / len(data) + 1")

# Def present (passes the transition's interface check) but kills
# its own process at import: the tier subprocess dies mid-test, so
# no evidence file is written and the gate failure rests on
# incomplete evidence only -> reopen, not close. (A plain top-level
# raise does NOT crash the tier: tests import solution inside
# functions, so the raise fails each test individually with
# complete evidence. os._exit is uncatchable and unforbidden.)
_CRASH = (
    "import os\n"
    "def compute_stats(data):\n"
    "    return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "os._exit(1)\n"
)

# Forbidden construct with otherwise-correct code: static fails on
# security with COMPLETE evidence -> definitive close. (Syntax-broken
# and def-less states never reach verification: the transition
# layer rejects them first, so they cannot test closure.)
_FORBIDDEN = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    x = eval('1')\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t / len(data)}\n"
)

_SLOW = "# slow-variant\n" + GOOD


class MarkerBuilder:
    """Root -> one X variant per marker; any marked parent -> child.

    With child_map given, root proposes root_markers variants and
    each mapped parent marker proposes its mapped child (used by
    the depth test, where B's child P must differ from A's child).
    """

    role = "builder"

    def __init__(self, markers, child_src, child_map=None):
        self._markers = markers
        self._child = child_src
        self._child_map = child_map or {}

    def propose(self, projection, request_context):
        current = projection.target_files.get("solution.py", "")
        mapped = [m for m in self._child_map
                  if f"# variant {m}" in current]
        if mapped:
            srcs = [self._child_map[mapped[0]]]
        elif any(f"# variant {m}" in current for m in self._markers):
            srcs = [self._child]
        elif current:
            srcs = [GOOD]
        else:
            srcs = [_x_code(m) for m in self._markers]
        if "solution.py" in projection.target_files:
            return tuple(
                CodePatchProposal(
                    agent_role=self.role, intent="m4 variant",
                    rationale="m4", estimated_cost=1.0,
                    modified_files={"solution.py": s})
                for s in srcs
            )
        return tuple(
            CodePatchProposal(
                agent_role=self.role, intent="m4 variant",
                rationale="m4", estimated_cost=1.0,
                created_files={"solution.py": s})
            for s in srcs
        )


class PrecisionCritic:
    """One attack: float32 mean precision. X fails it, GOOD passes."""

    role = "critic"

    def propose(self, projection, request_context):
        attack = AttackTestProposal(
            agent_role=self.role,
            test_name="attack_mean_precision",
            hypothesis="float32 mean breaks precision",
            test_code=(
                "def attack_mean_precision():\n"
                "    from solution import compute_stats\n"
                "    r = compute_stats([-3, -3, 5, 5, 5])\n"
                "    assert abs(r['mean'] - 1.8) < 1e-9\n"
            ),
        )
        critique = CritiqueProposal(
            agent_role=self.role, intent="adversarial review", issues=(),
            suggested_repair_directions=(), confidence=0.8)
        return (critique,), (attack,)


class CrashingCritic:
    """Attack module that fails to import: every adversarial tier
    crashes (incomplete evidence), so nodes are admitted, not closed."""

    role = "critic"

    def propose(self, projection, request_context):
        attack = AttackTestProposal(
            agent_role=self.role, test_name="attack_broken",
            hypothesis="h",
            test_code=(
                "import definitely_missing_module_xyz\n"
                "def attack_broken():\n    assert True\n"
            ),
        )
        critique = CritiqueProposal(
            agent_role=self.role, intent="adversarial review", issues=(),
            suggested_repair_directions=(), confidence=0.8)
        return (critique,), (attack,)


class SlowContentEnvironment:
    """Time out executes whose workdir solution.py contains MARKER,
    for the first `fail_count` such executes only, then delegate."""

    def __init__(self, delegate, marker, fail_count):
        self._delegate = delegate
        self._marker = marker
        self._fail_count = fail_count
        self._slow_calls = 0

    def materialize_state(self, state):
        return self._delegate.materialize_state(state)

    def cleanup(self, workdir):
        return self._delegate.cleanup(workdir)

    def execute(self, workdir, command, limits=None):
        try:
            with open(os.path.join(workdir, "solution.py")) as fh:
                content = fh.read()
        except OSError:
            content = ""
        if self._marker in content and self._slow_calls < self._fail_count:
            self._slow_calls += 1
            return ExecutionResult(
                exit_code=-signal.SIGKILL, stdout="", stderr="",
                timed_out=True, runtime_ms=0.0,
                killed_signal=int(signal.SIGKILL), limit_hit="timeout")
        return self._delegate.execute(workdir, command, limits)


def _engine(spec, env, builder, critic, **cfg_kwargs):
    cfg = ForgeMindConfig(random_seed=7, **cfg_kwargs)
    referee = Referee(spec, env, cfg)
    referee.attach_test_provider(StatsProblem().test_files)
    return SearchEngine(spec, cfg, builder=builder, critic=critic,
                        referee=referee, recovery=RecoveryPolicy())


def _nodes_with(eng, content):
    return [n for n in eng.graph.nodes.values()
            if n.state.files.get("solution.py") == content]


def _duplicates(eng):
    return [e for e in eng.history.events
            if e.event_type is EventType.DUPLICATE_DETECTED]


def test_quarantine_reopens_hash(spec):
    """Quarantined hash is in NEITHER seen nor closed (Kashi's
    membership assertion)."""
    env = SlowContentEnvironment(SubprocessExecutionEnvironment(),
                                 "# slow-variant", 99)
    eng = _engine(spec, env, MarkerBuilder(["M1"], _SLOW),
                  PrecisionCritic())
    res = eng.run()
    assert not res.success
    assert res.termination.reason == "quarantined"
    assert res.quarantined
    slow = _nodes_with(eng, _SLOW)
    assert slow
    h = slow[0].state.content_hash
    assert h not in eng.frontier.closed_hashes
    assert h not in eng.frontier.seen_hashes


def test_quarantined_state_reverified_to_success(spec):
    """Discriminator: transient spike quarantines S from M1; M2's
    re-proposal verifies fast and is accepted. Pre-M4 the second S
    is killed as a DUPLICATE."""
    env = SlowContentEnvironment(SubprocessExecutionEnvironment(),
                                 "# slow-variant", 4)
    eng = _engine(spec, env, MarkerBuilder(["M1", "M2"], _SLOW),
                  PrecisionCritic())
    res = eng.run()
    assert res.success
    assert res.accepted_node is not None
    assert "# slow-variant" in res.accepted_node.state.files["solution.py"]
    # Kashi's assertion: after success, the hash IS closed.
    assert res.accepted_node.state.content_hash in eng.frontier.closed_hashes
    assert not _duplicates(eng)


def test_depth_exhausted_reopens_hash(spec):
    """State selected at max_depth is depth-exhausted; its hash must
    end in NEITHER seen nor closed (depth is a path outcome, not a
    state verdict). Crashing attacks make the node admitted with
    INCOMPLETE evidence — the only way a depth-exhausted node holds
    a seen, unclosed hash (a complete verdict would already have
    closed). NOTE: full re-proposal after depth-exhaustion is
    unobservable under beam search — level order plus the max-depth
    gate mean no expandable node is ever selected after a
    depth-exhausted one (the strategy is hardcoded beam) — so this
    test pins the membership invariant at the site, and
    re-proposal-after-reopen is covered end-to-end by the
    quarantine test plus frontier units."""
    env = SubprocessExecutionEnvironment()
    eng = _engine(spec, env, MarkerBuilder(["A"], _X_PLAIN),
                  CrashingCritic(), max_depth=2)
    res = eng.run()
    assert not res.success
    assert res.termination.reason == "frontier_empty"
    deep = _nodes_with(eng, _X_PLAIN)
    assert len(deep) == 1
    assert any(e.event_type is EventType.FAILURE_RECORDED
               and e.metadata.get("type") == "DEPTH_EXHAUSTED"
               and e.metadata.get("node") == deep[0].node_id
               for e in eng.history.events)
    h = deep[0].state.content_hash
    assert h not in eng.frontier.closed_hashes
    assert h not in eng.frontier.seen_hashes


def test_incomplete_evidence_reopens_hash(spec):
    """Tier-subprocess death (candidate kills its own process at
    import) leaves no evidence: the gate failure rests on
    incomplete evidence only, so re-proposals are verified, not
    skipped as duplicates, and the hash ends in neither set.
    (C2 may re-expand a parent, so the count is >= 2, not exact.)"""
    eng = _engine(spec, SubprocessExecutionEnvironment(),
                  MarkerBuilder(["M1", "M2"], _CRASH), PrecisionCritic())
    eng.run()
    assert len(_nodes_with(eng, _CRASH)) >= 2
    assert not _duplicates(eng)
    h = _nodes_with(eng, _CRASH)[0].state.content_hash
    assert h not in eng.frontier.closed_hashes
    assert h not in eng.frontier.seen_hashes


def test_reverify_duplicates_switch(spec):
    """Gate-failed state F (complete evidence) proposed by two
    parents. Switch off: second F is DUPLICATE. Switch on: second
    F node created + verified (fails again)."""
    off = _engine(spec, SubprocessExecutionEnvironment(),
                  MarkerBuilder(["M1", "M2"], _BAD_MEAN), PrecisionCritic(),
                  reverify_duplicates=False)
    off.run()
    assert len(_nodes_with(off, _BAD_MEAN)) == 1
    assert _duplicates(off)
    f_hash = _nodes_with(off, _BAD_MEAN)[0].state.content_hash
    assert f_hash in off.frontier.closed_hashes

    # C2 re-expands M2 after its duplicate/failure, so the switch-on
    # run creates MORE than one extra F node; the switch's property
    # is "no duplicate skips + re-verification happens", not an
    # exact count.
    on = _engine(spec, SubprocessExecutionEnvironment(),
                 MarkerBuilder(["M1", "M2"], _BAD_MEAN), PrecisionCritic(),
                 reverify_duplicates=True)
    on.run()
    assert len(_nodes_with(on, _BAD_MEAN)) >= 2
    assert len(_nodes_with(on, _BAD_MEAN)) > len(_nodes_with(off, _BAD_MEAN))
    assert not _duplicates(on)


def test_forbidden_construct_closes_hash(spec):
    """Guard: a definitive failure still closes. Forbidden-construct
    state (complete static evidence) proposed twice -> second is
    DUPLICATE, hash closed. (Passes pre-fix too — pins that close
    was not removed. Syntax-broken states cannot test this: the
    transition layer rejects them before verification.)"""
    eng = _engine(spec, SubprocessExecutionEnvironment(),
                  MarkerBuilder(["M1", "M2"], _FORBIDDEN), PrecisionCritic())
    eng.run()
    assert len(_nodes_with(eng, _FORBIDDEN)) == 1
    assert _duplicates(eng)
    h = _nodes_with(eng, _FORBIDDEN)[0].state.content_hash
    assert h in eng.frontier.closed_hashes


def _mknode(node_id, files, depth=0, idx=0):
    state = CodebaseState(spec_hash="a" * 64, files=files)
    return SearchNode(node_id=node_id, state=state,
                      parent_node_id=None if depth == 0 else "virtual-root",
                      candidate_id=None if depth == 0 else "c" + node_id,
                      depth=depth, creation_index=idx)


def test_frontier_seen_closed_split():
    f = SearchFrontier()
    h = _mknode("n1", {"a.py": "x"}).state.content_hash
    assert f.remember_state(h) is True
    assert f.remember_state(h) is False  # seen blocks
    assert not f.is_closed(h)
    assert not f.contains_state(h)
    f.close(h)
    assert f.is_closed(h)
    assert f.contains_state(h)
    assert h not in f.seen_hashes
    assert f.remember_state(h) is False  # closed blocks
    f.reopen(h)
    assert f.is_closed(h)  # reopen never un-closes


def test_frontier_reopen_releases_seen():
    f = SearchFrontier()
    h = _mknode("n1", {"a.py": "x"}).state.content_hash
    assert f.remember_state(h) is True
    f.reopen(h)
    assert h not in f.seen_hashes
    assert f.remember_state(h) is True  # re-proposable


def test_frontier_prune_releases_seen_keeps_closed():
    f = SearchFrontier(capacity=1)
    n1 = _mknode("n1", {"a.py": "x"}, depth=1, idx=1)
    n2 = _mknode("n2", {"a.py": "y"}, depth=1, idx=2)
    h1 = n1.state.content_hash
    assert f.remember_state(h1)
    assert f.admit(n1, guidance=0.5)
    assert f.remember_state(n2.state.content_hash)
    assert f.admit(n2, guidance=0.9)  # prunes n1
    assert h1 not in f.seen_hashes  # eviction released the seen claim

    f2 = SearchFrontier(capacity=1)
    assert f2.remember_state(h1)
    assert f2.admit(n1, guidance=0.5)
    f2.close(h1)
    assert f2.remember_state(n2.state.content_hash)
    assert f2.admit(n2, guidance=0.9)  # prunes n1
    assert f2.is_closed(h1)  # pruning never un-closes
