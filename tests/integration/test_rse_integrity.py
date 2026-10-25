"""RSE Integrity Challenge / Integrity Suite.

Validation experiment, NOT a feature demo. Question under test:

    "Can RSE correctly explore, verify, reject, recover, converge,
     quarantine, and select software states without any LLM intelligence?"

Hard rules for this file:
- ZERO LLM calls, ZERO network access, ZERO wall-clock dependence for
  correctness, ZERO machine-specific memory limits for correctness.
- No production code is weakened to make a test pass. Every engine-level
  test runs the real SearchEngine / TransitionEngine / Referee / frontier.
- The optimizer is OFF in suite engine configs (documented choice): the
  optimizer is orthogonal and already covered by the M5 tests, and winner
  identity must be exact for I5 / I15 / I16. Gates, frontier, verification
  and recovery all still run.
- Critics are explicit everywhere (NoAttackCritic / PrecisionCritic /
  CrashCritic), never the default critic, so attack-set behavior is
  deterministic.
- Counts are asserted only where they are contractual (exactly 1 ACCEPTED
  / 0 ACCEPTED / duplicates >= 1) or derived from the test's own
  deterministic landscape (I16: 7 + 2 + 3 = 12 proposals). Everything else
  in the report helper is information, not assertion.

KNOWN HONEST GAP (see I12): the engine exposes no deterministic
memory-limit hook (MEMORY_LIMIT was deleted in L1), so memory-pressure
containment cannot be tested at engine level and is not faked here.
"""

from __future__ import annotations

import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig
from forgemind.core.history import EventType
from forgemind.core.problem import AcceptanceTest
from forgemind.core.proposal import (
    AttackTestProposal,
    CodePatchProposal,
    CritiqueProposal,
)
from forgemind.core.state import CodebaseState
from forgemind.core.transition import TransitionEngine
from forgemind.recovery.failure import FailureType, RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.search.node import NodeStatus, SearchFrontier
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee
from forgemind.verification.results import ExecutionResult

SPEC = StatsProblem().spec
TEST_FILES = StatsProblem().test_files


# --------------------------------------------------------------------------
# Code fixtures. GOOD is the proven-correct variant (same semantics as
# MockBuilder's correct variant). F32 passes all acceptance tests but fails
# the 1.8 precision attack (float32 rounding) -- the "partial" variant.
# --------------------------------------------------------------------------

GOOD = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t / len(data)}\n"
)

_F32_HEADER = (
    "import struct\n\n\n"
    "def _f32(x):\n"
    "    return struct.unpack('f', struct.pack('f', x))[0]\n\n\n"
)

_F32_BODY = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': _f32(t / len(data))}\n"
)


def _f32_code(marker: str) -> str:
    return f"# {marker}\n" + _F32_HEADER + _F32_BODY


WRONG_EMPTY = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        raise ValueError('empty input')\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t / len(data)}\n"
)

WRONG_MEAN = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t // len(data)}\n"
)

# Distinct failing variant (wrong count, correct mean): needed in I16 so
# the dead end's second child is not a duplicate of the root's WRONG_MEAN.
WRONG_COUNT = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data) + 1, 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t / len(data)}\n"
)

SYNTAX_BROKEN = "def compute_stats(:\n  pass\n"

# A solution that forges success on the untrusted channel: it prints a fake
# HARNESS_JSON marker at import time, while actually computing a WRONG mean.
FORGE = (
    "print('HARNESS_JSON: {\"passed\": true}')\n"
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t // len(data)}\n"
)

FLOOD_STDOUT = (
    "for _i in range(3000):\n"
    "    print('FLOOD ' * 40)\n"
    + WRONG_MEAN
)

FLOOD_STDERR = (
    "import sys\n"
    "for _i in range(3000):\n"
    "    sys.stderr.write('FLOODERR ' * 40 + '\\n')\n"
    + WRONG_MEAN
)

ABORT_SOLUTION = "import os\nos.abort()\n" + GOOD


# --------------------------------------------------------------------------
# Deterministic agents
# --------------------------------------------------------------------------

class NoAttackCritic:
    """Critic that never adds attacks: adversarial tier is vacuous."""

    role = "critic"

    def propose(self, projection, request_context):
        return (), ()


class PrecisionCritic:
    """The 1.8 precision attack: float32 mean fails, true division passes."""

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


class CrashCritic:
    """Attack module that raises at import: the adversarial tier crashes
    and produces no valid evidence."""

    role = "critic"

    def propose(self, projection, request_context):
        attack = AttackTestProposal(
            agent_role=self.role,
            test_name="attack_crashes",
            hypothesis="adversarial tier import failure",
            test_code="raise RuntimeError('adversarial tier boom')\n",
        )
        critique = CritiqueProposal(
            agent_role=self.role, intent="adversarial review", issues=(),
            suggested_repair_directions=(), confidence=0.8)
        return (critique,), (attack,)


class MarkerBuilder:
    """Deterministic tree builder: proposals keyed on a marker comment in
    the parent's solution.py. The "" key matches only the root (no
    solution yet). Unknown parents propose nothing, so branches end."""

    role = "builder"

    def __init__(self, mapping):
        self._mapping = dict(mapping)

    def propose(self, projection, request_context):
        current = projection.target_files.get("solution.py", "")
        codes = None
        for marker, cs in self._mapping.items():
            if marker and marker in current:
                codes = cs
                break
        if codes is None and not current:
            codes = self._mapping.get("", ())
        if not codes:
            return ()
        out = []
        for code in codes:
            if "solution.py" in projection.target_files:
                out.append(CodePatchProposal(
                    agent_role=self.role, intent="integrity variant",
                    rationale="integrity", estimated_cost=1.0,
                    modified_files={"solution.py": code}))
            else:
                out.append(CodePatchProposal(
                    agent_role=self.role, intent="integrity variant",
                    rationale="integrity", estimated_cost=1.0,
                    created_files={"solution.py": code}))
        return tuple(out)


class SingleCodeBuilder:
    """Always proposes exactly one fixed solution (create at root, modify
    below) -- for calibration-sensitive tests (I8)."""

    role = "builder"

    def __init__(self, code):
        self._code = code

    def propose(self, projection, request_context):
        if "solution.py" in projection.target_files:
            return (CodePatchProposal(
                agent_role=self.role, intent="single", rationale="single",
                estimated_cost=1.0,
                modified_files={"solution.py": self._code}),)
        return (CodePatchProposal(
            agent_role=self.role, intent="single", rationale="single",
            estimated_cost=1.0,
            created_files={"solution.py": self._code}),)


# --------------------------------------------------------------------------
# Deterministic verification environments (H4 pattern, extended)
# --------------------------------------------------------------------------

def _timed_out_result():
    return ExecutionResult(
        exit_code=-signal.SIGKILL, stdout="", stderr="",
        timed_out=True, runtime_ms=0.0,
        killed_signal=int(signal.SIGKILL), limit_hit="timeout")


class FlakyEnvironment:
    """Force timed_out for the first `fail_first_n` execute() calls, then
    delegate. Deterministic retry semantics without racing clocks."""

    def __init__(self, delegate, fail_first_n):
        self._delegate = delegate
        self._fail_first_n = fail_first_n
        self.calls = 0

    def materialize_state(self, state):
        return self._delegate.materialize_state(state)

    def cleanup(self, workdir):
        return self._delegate.cleanup(workdir)

    def execute(self, workdir, command, limits=None):
        self.calls += 1
        if self.calls <= self._fail_first_n:
            return _timed_out_result()
        return self._delegate.execute(workdir, command, limits)


class MarkerTimeoutEnvironment:
    """Per-state deterministic timeouts, keyed on marker comments in the
    candidate's solution.py (tracked by content hash, not workdir, because
    M3 materializes a fresh workdir per tier):
    - '# QUARANTINE_ME' -> every execute() for that state times out.
    - '# RECOVER_ME'   -> only the FIRST execute() for that state times
      out; the retry's executes delegate (recoverable timeout)."""

    def __init__(self, delegate):
        self._delegate = delegate
        self._markers_by_hash = {}
        self._hash_by_workdir = {}
        self._calls_by_hash = {}

    def materialize_state(self, state):
        workdir = self._delegate.materialize_state(state)
        code = state.files.get("solution.py", "")
        markers = {m for m in ("# QUARANTINE_ME", "# RECOVER_ME") if m in code}
        self._markers_by_hash[state.content_hash] = markers
        self._hash_by_workdir[workdir] = state.content_hash
        return workdir

    def cleanup(self, workdir):
        self._hash_by_workdir.pop(workdir, None)
        return self._delegate.cleanup(workdir)

    def execute(self, workdir, command, limits=None):
        state_hash = self._hash_by_workdir.get(workdir)
        markers = self._markers_by_hash.get(state_hash, set()) if state_hash else set()
        if state_hash is not None:
            self._calls_by_hash[state_hash] = self._calls_by_hash.get(state_hash, 0) + 1
            n = self._calls_by_hash[state_hash]
        else:
            n = 0
        if "# QUARANTINE_ME" in markers:
            return _timed_out_result()
        if "# RECOVER_ME" in markers and n == 1:
            return _timed_out_result()
        return self._delegate.execute(workdir, command, limits)


# --------------------------------------------------------------------------
# Engine construction + trace/report helpers
# --------------------------------------------------------------------------

def _engine(*, seed=0, env=None, builder=None, critic=None, **cfg_overrides):
    cfg = ForgeMindConfig(
        random_seed=seed, enable_optimizer=False, **cfg_overrides)
    env = env if env is not None else SubprocessExecutionEnvironment()
    referee = Referee(SPEC, env, cfg)
    referee.attach_test_provider(TEST_FILES)
    kwargs = {}
    if builder is not None:
        kwargs["builder"] = builder
    if critic is not None:
        kwargs["critic"] = critic
    return SearchEngine(SPEC, cfg, referee=referee,
                        recovery=RecoveryPolicy(), **kwargs)


def _digest(result):
    """Determinism digest: candidate IDs, accepted artifact hash, and
    termination reason. Deliberately EXCLUDES elapsed_seconds, runtime_ms
    and peak_memory_mb: those are environmental, not contractual."""
    cand_ids = tuple(n.candidate_id for n in result.graph.nodes.values()
                     if n.candidate_id)
    accepted_hash = (result.accepted_node.state.content_hash
                     if result.accepted_node else None)
    return (cand_ids, accepted_hash, result.termination.reason)


def _nodes_with(engine, result, code):
    return [n for n in result.graph.nodes.values()
            if n.state.files.get("solution.py") == code]


def _events(engine, event_type):
    return [e for e in engine.history.events if e.event_type is event_type]


def _verify_starts_for(engine, node_id):
    return [e for e in _events(engine, EventType.VERIFICATION_STARTED)
            if e.metadata.get("node") == node_id]


def _records(engine, failure_type=None):
    recs = list(engine._all_records)
    if failure_type is not None:
        recs = [r for r in recs if r.failure_type is failure_type]
    return recs


def _accepted_nodes(result):
    return [n for n in result.graph.nodes.values()
            if n.status is NodeStatus.ACCEPTED]


def _exec_solution(code):
    """Independently execute a delivered solution and check its semantics.
    This is the test's own oracle, not the engine's verdict."""
    ns = {}
    exec(code, ns)
    fn = ns["compute_stats"]
    assert fn([]) == {"count": 0, "sum": 0, "min": None, "max": None, "mean": None}
    assert fn([1, 2, 3, 4]) == {"count": 4, "sum": 10, "min": 1, "max": 4, "mean": 2.5}
    assert abs(fn([-3, -3, 5, 5, 5])["mean"] - 1.8) < 1e-9
    assert fn([7])["mean"] == 7


def summarize_integrity(result, engine):
    """Integrity report for a run: counts + the semantic facts that matter.
    Counts here are INFORMATION; the tests assert semantic invariants, not
    arbitrary totals (except where the landscape itself derives them)."""
    nodes = list(result.graph.nodes.values())
    verify_starts = _events(engine, EventType.VERIFICATION_STARTED)
    retries = {}
    for e in verify_starts:
        retries[e.metadata.get("node")] = retries.get(e.metadata.get("node"), 0) + 1
    return {
        "candidates_created": engine.metrics.candidates_created,
        "states_discovered": len(nodes),
        "states_rejected": sum(1 for n in nodes if n.status is NodeStatus.REJECTED),
        "duplicates": engine.metrics.duplicates,
        "verifications_started": len(verify_starts),
        "verifications_counted": engine.metrics.verifications,
        "retries": {k: v for k, v in retries.items() if v > 1},
        "quarantines": len(result.quarantined),
        "adversarial_failures": len(_records(engine, FailureType.ADVERSARIAL_FAILURE)),
        "test_failures": len(_records(engine, FailureType.TEST_FAILURE)),
        "final_winner": (result.accepted_node.state.content_hash
                         if result.accepted_node else None),
        "acceptance_count": len(_accepted_nodes(result)),
        "termination": result.termination.reason,
        "success": result.success,
    }

# ==========================================================================
# I1 -- STATE IDENTITY
# ==========================================================================

def _state(**overrides):
    base = dict(spec_hash=SPEC.spec_hash, files={"solution.py": GOOD})
    base.update(overrides)
    return CodebaseState(**base)


def test_i1_different_states_different_identity():
    """Every verification-relevant input the current hashing contract
    claims to cover must actually change the identity when it changes."""
    base = _state()
    variants = {
        "file content": _state(files={"solution.py": WRONG_MEAN}),
        "added file": _state(files={"solution.py": GOOD, "extra.py": "x = 1\n"}),
        "removed file": _state(files={}),
        "dependency": _state(dependencies={"numpy": "2.0"}),
        "configuration": _state(configuration={"strict": "true"}),
        "runtime_spec": _state(runtime_spec="python3.12"),
        "spec_hash": _state(spec_hash="0" * 64),
    }
    hashes = {base.content_hash}
    for name, st in variants.items():
        assert st.content_hash != base.content_hash, name
        hashes.add(st.content_hash)
    assert len(hashes) == len(variants) + 1  # no accidental collisions


def test_i1_equivalent_canonical_states_converge():
    """Same canonical content -> same identity, regardless of dict order
    or non-canonical path spelling (normalization is canonicalization)."""
    a = _state(files={"solution.py": GOOD, "util.py": "x = 1\n"})
    b = _state(files={"util.py": "x = 1\n", "solution.py": GOOD})
    assert a.content_hash == b.content_hash
    c = _state(files={"./solution.py": GOOD, "util.py": "x = 1\n"})
    assert c.content_hash == a.content_hash
    d = _state(files={"a//b.py": "x = 1\n", "solution.py": GOOD})
    e = _state(files={"a/b.py": "x = 1\n", "solution.py": GOOD})
    assert d.content_hash == e.content_hash


def test_i1_spec_identity_covers_tests():
    """H2: the spec's identity must move when its verification inputs
    move -- acceptance test IDs and test sources alike."""
    base_hash = SPEC.spec_hash
    by_id = replace(SPEC, acceptance_tests=(
        AcceptanceTest(name="other", test_id="test_other",
                       description="other"),))
    assert by_id.spec_hash != base_hash
    by_source = replace(SPEC, test_sources={
        k: v + "\n# changed" for k, v in SPEC.test_sources.items()})
    assert by_source.spec_hash != base_hash
    by_desc = replace(SPEC, description=SPEC.description + " (amended)")
    assert by_desc.spec_hash != base_hash
    assert replace(SPEC).spec_hash == base_hash  # identical spec, same hash


def test_i1_no_caller_supplied_hash():
    """M9: the hash is always computed, never supplied -- a caller cannot
    stamp an arbitrary identity onto a state."""
    with pytest.raises(TypeError):
        CodebaseState(spec_hash=SPEC.spec_hash, files={"solution.py": GOOD},
                      content_hash="0" * 64)


# ==========================================================================
# I2 -- PATH NORMALIZATION / COLLISION INTEGRITY (L2 + L3)
# ==========================================================================

def _patch(**kwargs):
    return CodePatchProposal(agent_role="builder", intent="i2",
                             rationale="i2", estimated_cost=1.0, **kwargs)


def test_i2_normalized_collisions_rejected():
    """Destructive collisions after normalization are errors, in both
    create and modify positions, and the parent state is untouched."""
    engine = TransitionEngine(SPEC)
    parent = _state()
    for kwargs in (
        {"created_files": {"new.py": "x = 1\n", "./new.py": "x = 2\n"}},
        {"created_files": {"a//b.py": "x = 1\n", "a/b.py": "x = 2\n"}},
        {"modified_files": {"solution.py": GOOD, "./solution.py": WRONG_MEAN}},
    ):
        result = engine.apply(parent, _patch(**kwargs))
        assert not result.ok
        assert result.violation == "INVALID_PATH"
        assert result.new_state is None
        assert parent.files == {"solution.py": GOOD}
        assert parent.content_hash == _state().content_hash


def test_i2_single_noncanonical_spelling_works():
    """A lone non-canonical spelling is normalization, not a collision:
    it must apply, and land on the canonical path. (The parent carries
    the required entry function where the new file does not provide it,
    so the interface check is satisfied.)"""
    engine = TransitionEngine(SPEC)
    parent = _state(files={})
    result = engine.apply(parent, _patch(created_files={"./solution.py": GOOD}))
    assert result.ok
    assert result.new_state.files == {"solution.py": GOOD}
    result2 = engine.apply(_state(), _patch(created_files={"a//b.py": "x = 1\n"}))
    assert result2.ok
    assert result2.new_state.files["a/b.py"] == "x = 1\n"
    assert result2.new_state.files["solution.py"] == GOOD


def test_i2_deleting_equivalent_spellings_idempotent():
    """Deleting two spellings of the same canonical path removes it once
    and succeeds -- deletes carry no competing data, so the collision
    rule does not apply to them (L3). The boundary is explicit: within
    one patch, repeated spellings are idempotent; deleting a path that
    does not exist at all is a DELETE_NONEXISTENT error, not a silent
    no-op -- also an integrity property."""
    engine = TransitionEngine(SPEC)
    parent = _state(files={"solution.py": GOOD, "a/b.py": "x = 1\n"})
    result = engine.apply(
        parent, _patch(deleted_files=("a//b.py", "a/b.py")))
    assert result.ok
    assert "a/b.py" not in result.new_state.files
    assert result.new_state.files["solution.py"] == GOOD
    absent = engine.apply(result.new_state, _patch(deleted_files=("ghost.py",)))
    assert not absent.ok
    assert absent.violation == "DELETE_NONEXISTENT"


def test_i2_spec_path_policy():
    """L2 policy, exactly as specified. Substring lookalikes are legal;
    exact and trailing-component spec paths are forbidden."""
    engine = TransitionEngine(SPEC)
    # The parent carries the required entry function, so legal additions
    # are judged on the spec-path policy alone, not the interface check.
    parent = _state()
    legal = ("my_problem_spec.json", "problem_spec.json.bak",
             "docs/problem_spec.json.bak")
    for path in legal:
        result = engine.apply(parent, _patch(created_files={path: "{}\n"}))
        assert result.ok, path
    forbidden = ("problem_spec.json", "sub/problem_spec.json",
                 ".forgemind/spec.json", "x/.forgemind/spec.json")
    for path in forbidden:
        result = engine.apply(parent, _patch(created_files={path: "{}\n"}))
        assert not result.ok, path
        assert result.violation == "PROBLEM_SPEC_MUTATION"
        assert result.new_state is None


# ==========================================================================
# I3 -- TRANSITION ATOMICITY
# ==========================================================================

def test_i3_rejected_transitions_never_partially_mutate():
    """A proposal mixing a valid change with ANY invalid one is rejected
    whole: no new state, parent bytes and hash unchanged, and the parent
    still accepts a clean proposal afterwards."""
    engine = TransitionEngine(SPEC)
    empty = _state(files={})
    full = _state()
    bad_cases = [
        # valid file + forbidden spec path
        (empty, _patch(created_files={"solution.py": GOOD, "problem_spec.json": "{}\n"}),
         "PROBLEM_SPEC_MUTATION"),
        # valid file + normalized collision
        (empty, _patch(created_files={"solution.py": GOOD, "./solution.py": GOOD}),
         "INVALID_PATH"),
        # valid file + syntax error in another file
        (empty, _patch(created_files={"solution.py": GOOD, "broken.py": SYNTAX_BROKEN}),
         "SYNTAX"),
        # valid file + dependency violation (spec allows no dependencies)
        (empty, _patch(created_files={"solution.py": GOOD},
                       dependency_changes={"numpy": "2.0"}),
         "DEPENDENCY_POLICY"),
        # no compute_stats anywhere -> interface violation
        (empty, _patch(created_files={"other.py": "x = 1\n"}),
         "INTERFACE_CONSTRAINT"),
        # create of a file that already exists (modify position required)
        (full, _patch(created_files={"solution.py": GOOD}), "CREATE_EXISTING"),
    ]
    for parent, proposal, violation in bad_cases:
        before_files = dict(parent.files)
        before_hash = parent.content_hash
        result = engine.apply(parent, proposal)
        assert not result.ok, proposal
        assert result.violation == violation
        assert result.new_state is None
        assert dict(parent.files) == before_files
        assert parent.content_hash == before_hash
    # The parent is uncorrupted: a clean proposal still applies.
    clean = engine.apply(_state(files={}),
                         _patch(created_files={"solution.py": GOOD}))
    assert clean.ok and clean.new_state.files["solution.py"] == GOOD


def test_i3_parent_state_is_immutable():
    """Parent states cannot be mutated through their mappings -- a
    rejected (or accepted) transition has no aliasing path back in."""
    parent = _state()
    with pytest.raises(TypeError):
        parent.files["solution.py"] = WRONG_MEAN
    engine = TransitionEngine(SPEC)
    result = engine.apply(parent, _patch(modified_files={"solution.py": WRONG_MEAN}))
    assert result.ok
    assert parent.files["solution.py"] == GOOD  # original untouched
    assert result.new_state.files["solution.py"] == WRONG_MEAN

# ==========================================================================
# I4 -- SEARCH BRANCHING (real engine, real trace)
# ==========================================================================

def test_i4_branching_landscape():
    """ROOT -> A(SYNTAX reject) / B(TEST_FAILURE) / C(adv-fail, admitted)
    -> C expands -> D(TEST_FAILURE) / E(PASS -> ACCEPTED). The engine must
    actually create and evaluate all five candidates through its real
    machinery; C is 'partial' (acceptance-clean, attack-weak), so it is
    expanded rather than accepted, and E is the winner."""
    f32 = _f32_code("# MARK-C")
    builder = MarkerBuilder({
        "": [SYNTAX_BROKEN, WRONG_EMPTY, f32],
        "# MARK-C": [WRONG_MEAN, GOOD],
    })
    eng = _engine(builder=builder, critic=PrecisionCritic())
    result = eng.run()

    assert result.success
    assert result.termination.reason == "goal_accepted"
    accepted = _accepted_nodes(result)
    assert len(accepted) == 1
    assert accepted[0].state.files["solution.py"] == GOOD
    _exec_solution(accepted[0].state.files["solution.py"])

    # The trace proves real exploration, in beam order: root expanded,
    # then the admitted partial branch C, and nothing else selected.
    selected = [e.metadata.get("node") for e in _events(eng, EventType.NODE_SELECTED)]
    c_nodes = _nodes_with(eng, result, f32)
    assert len(c_nodes) == 1
    root = [n for n in result.graph.nodes.values() if n.parent_node_id is None][0]
    assert selected == [root.node_id, c_nodes[0].node_id]
    assert eng.metrics.nodes_explored == 2
    assert eng.metrics.candidates_created == 5

    # Each failure mode left its own record; C stays GATE_PASSED, never
    # promoted, and no failed candidate is ACCEPTED.
    assert _records(eng, FailureType.TEST_FAILURE)  # B and D
    assert any(r.stage == "adversarial"
               for r in _records(eng, FailureType.ADVERSARIAL_FAILURE))  # C
    assert c_nodes[0].status is NodeStatus.GATE_PASSED
    failed_ids = {r.candidate_id for r in eng._all_records if r.candidate_id}
    assert accepted[0].candidate_id not in failed_ids
    # A never became a node: the transition rejected it before creation.
    assert not any(n.state.files.get("solution.py") == SYNTAX_BROKEN
                   for n in result.graph.nodes.values())


# ==========================================================================
# I5 -- REQUIRED RECOVERY (wrong-first, engine level)
# ==========================================================================

class WrongThenCorrectBuilder:
    """Deterministic two-shot builder: the first proposal is wrong, every
    later proposal is correct. With branching_factor=1 the engine can
    only reach the correct candidate via C2 recovery -- re-expanding the
    parent after the first failure -- so this is genuine recovery, not
    mere candidate ordering. (Wrong-first ordering across one expansion
    is already covered by the L4 rewritten test with
    MockBuilder(correct_first=False).)"""

    role = "builder"

    def __init__(self):
        self.calls = 0

    def propose(self, projection, request_context):
        self.calls += 1
        code = WRONG_MEAN if self.calls == 1 else GOOD
        if "solution.py" in projection.target_files:
            return (CodePatchProposal(
                agent_role=self.role, intent="recovery variant",
                rationale="recovery", estimated_cost=1.0,
                modified_files={"solution.py": code}),)
        return (CodePatchProposal(
            agent_role=self.role, intent="recovery variant",
            rationale="recovery", estimated_cost=1.0,
            created_files={"solution.py": code}),)


def test_i5_wrong_first_recovery():
    """The first candidate MUST fail, the engine MUST recover (backtrack
    and re-expand the parent -- visible as a BACKTRACK event), the later
    candidate MUST be selected, and exactly it MUST be ACCEPTED."""
    eng = _engine(builder=WrongThenCorrectBuilder(), critic=NoAttackCritic(),
                  branching_factor=1)
    result = eng.run()

    assert result.success
    test_failures = _records(eng, FailureType.TEST_FAILURE)
    assert test_failures, "the wrong candidate must leave a failure record"
    assert eng.metrics.candidates_created == 2, \
        "exactly two candidates: the failure, then the recovery"
    assert _events(eng, EventType.BACKTRACK), \
        "recovery must reinsert the parent for re-expansion"
    accepted = _accepted_nodes(result)
    assert len(accepted) == 1
    assert accepted[0].state.files["solution.py"] == GOOD
    _exec_solution(accepted[0].state.files["solution.py"])
    failed_ids = {r.candidate_id for r in eng._all_records if r.candidate_id}
    assert accepted[0].candidate_id not in failed_ids
    assert result.accepted_node is accepted[0]


# ==========================================================================
# I6 -- DUPLICATE STATE CONVERGENCE
# ==========================================================================

def test_i6_duplicate_state_not_reverified():
    """Two paths to one canonical state: root -> X directly, and
    root -> B -> C -> X again. X must be verified exactly once, the
    second arrival must be detected as a duplicate, and the run must not
    redundantly execute X merely because it arrived by another path."""
    x_code = _f32_code("# MARK-X")
    b_code = _f32_code("# MARK-B")
    c_code = _f32_code("# MARK-C")
    builder = MarkerBuilder({
        "": [x_code, b_code],
        "# MARK-B": [c_code],
        "# MARK-C": [x_code],
    })
    eng = _engine(builder=builder, critic=PrecisionCritic())
    result = eng.run()

    assert not result.success  # every variant is attack-weak: no winner
    assert len(_accepted_nodes(result)) == 0
    x_nodes = _nodes_with(eng, result, x_code)
    assert len(x_nodes) == 1, "one canonical state, one node"
    assert len(_verify_starts_for(eng, x_nodes[0].node_id)) == 1, \
        "the duplicate arrival must not trigger a second verification"
    assert eng.metrics.duplicates >= 1
    assert _events(eng, EventType.DUPLICATE_DETECTED)
    dup_records = _records(eng, FailureType.DUPLICATE_STATE)
    assert dup_records and dup_records[0].state_hash == x_nodes[0].state.content_hash


def test_i6_seen_only_duplicate_still_skipped():
    """Do NOT assume 'duplicate' means 'closed': a hash that is only
    SEEN (live, admitted after an incomplete-evidence adversarial
    verdict -- the crashed-tier case) must still be duplicate-detected
    when it arrives again, without being verified a second time, and it
    must remain unclosed."""
    marked_good = "# MARK-G\n" + GOOD
    builder = MarkerBuilder({"": [marked_good], "# MARK-G": [marked_good]})
    eng = _engine(builder=builder, critic=CrashCritic())
    result = eng.run()
    nodes = _nodes_with(eng, result, marked_good)
    assert len(nodes) == 1, "the re-proposed state must not create a node"
    assert len(_verify_starts_for(eng, nodes[0].node_id)) == 1
    assert eng.metrics.duplicates >= 1
    assert not eng.frontier.is_closed(nodes[0].state.content_hash)
    assert nodes[0].state.content_hash in eng.frontier.seen_hashes


# ==========================================================================
# I7 -- SEEN vs CLOSED (M4)
# ==========================================================================

def test_i7a_definitive_pass_closes():
    """A definitive PASS (acceptance) closes the hash: the delivered
    winner's state is decided for the run."""
    eng = _engine(builder=SingleCodeBuilder(GOOD), critic=NoAttackCritic())
    result = eng.run()
    assert result.success
    winner_hash = result.accepted_node.state.content_hash
    assert eng.frontier.is_closed(winner_hash)
    assert winner_hash not in eng.frontier.seen_hashes


def test_i7b_timeout_does_not_close_and_is_retryable():
    """Timeout is inconclusive, not a verdict: the quarantined hash is
    NOT closed, and a fresh engine verifies and accepts the same state."""
    env = FlakyEnvironment(SubprocessExecutionEnvironment(), 10 ** 9)
    eng = _engine(env=env, builder=SingleCodeBuilder(GOOD), critic=NoAttackCritic())
    result = eng.run()
    assert not result.success
    assert len(result.quarantined) == 1
    q = result.quarantined[0]
    assert q.attempts == 2
    assert not eng.frontier.is_closed(q.state_hash)

    fresh = _engine(builder=SingleCodeBuilder(GOOD), critic=NoAttackCritic())
    fresh_result = fresh.run()
    assert fresh_result.success
    assert fresh_result.accepted_node.state.content_hash == q.state_hash


def test_i7c_depth_exhaustion_does_not_close_and_is_retryable():
    """Depth exhaustion is a path/budget outcome, not a state verdict:
    the exhausted hash is NOT closed, and a deeper engine expands it.
    The node must be one whose hash is genuinely unclosed at exhaustion
    time: an incomplete-evidence adversarial verdict (crashed tier) does
    not close, whereas a complete adversarial verdict would have closed
    it at gate time -- so this test uses the crashed-tier critic."""
    marked_good = "# MARK-D\n" + GOOD
    builder = MarkerBuilder({"": [marked_good], "# MARK-D": [GOOD]})
    shallow = _engine(builder=builder, critic=CrashCritic(), max_depth=1)
    result = shallow.run()
    depth_records = _records(shallow, FailureType.DEPTH_EXHAUSTED)
    assert depth_records, "the depth-1 admitted node must exhaust on selection"
    exhausted_hash = depth_records[0].state_hash
    assert not shallow.frontier.is_closed(exhausted_hash)

    deep = _engine(builder=builder, critic=CrashCritic(), max_depth=6)
    deep_result = deep.run()
    deep_marked = _nodes_with(deep, deep_result, marked_good)
    assert len(deep_marked) == 1
    children = [n for n in deep_result.graph.nodes.values()
                if n.parent_node_id == deep_marked[0].node_id]
    assert children, "the deeper engine must actually expand the node"
    assert not _records(deep, FailureType.DEPTH_EXHAUSTED)


def test_i7d_reverify_duplicates_bypasses_closed_only():
    """A definitively failed (closed) state, re-proposed from another
    admitted branch: with reverify_duplicates=False it is skipped
    (duplicate, no second verification); with True it is re-verified.
    The engine's reverify branch -- and only that branch -- bypasses a
    closed hash."""
    f32 = _f32_code("# MARK-R")
    builder = MarkerBuilder({"": [WRONG_MEAN, f32], "# MARK-R": [WRONG_MEAN]})

    eng = _engine(builder=builder, critic=PrecisionCritic(),
                  reverify_duplicates=False)
    result = eng.run()
    wrong_nodes = _nodes_with(eng, result, WRONG_MEAN)
    assert len(wrong_nodes) == 1
    assert len(_verify_starts_for(eng, wrong_nodes[0].node_id)) == 1
    assert eng.metrics.duplicates >= 1
    assert eng.frontier.is_closed(wrong_nodes[0].state.content_hash)

    eng2 = _engine(builder=builder, critic=PrecisionCritic(),
                   reverify_duplicates=True)
    result2 = eng2.run()
    wrong_nodes2 = _nodes_with(eng2, result2, WRONG_MEAN)
    # Re-verification re-admits the branch for recovery, so the failed
    # state may be re-proposed up to the per-node repair cap: assert the
    # semantic fact (verified more than once, once per node), not a
    # brittle exact count.
    assert len(wrong_nodes2) >= 2, "reverify must create additional nodes"
    starts = sum(len(_verify_starts_for(eng2, n.node_id)) for n in wrong_nodes2)
    assert starts == len(wrong_nodes2) >= 2, \
        "the closed state must be verified again on each re-proposal"

# ==========================================================================
# I8 -- TIMEOUT / RETRY / QUARANTINE (deterministic, no real sleeps)
# ==========================================================================

def test_i8a_timeout_then_pass_recovers():
    """Attempt 1 TIMEOUT, attempt 2 PASS: the same candidate succeeds.
    The first execute() call is forced to time out; the retry delegates
    to the real environment. No sleep, no scheduler dependence."""
    env = FlakyEnvironment(SubprocessExecutionEnvironment(), fail_first_n=1)
    eng = _engine(env=env, builder=SingleCodeBuilder(GOOD), critic=NoAttackCritic())
    result = eng.run()
    assert result.success
    assert len(result.quarantined) == 0
    child = _nodes_with(eng, result, GOOD)[0]
    assert len(_verify_starts_for(eng, child.node_id)) == 2, \
        "the timed-out attempt and the retry are both visible in the trace"
    assert result.accepted_node is child


def test_i8b_repeated_timeout_quarantines_not_rejects():
    """Attempt 1 TIMEOUT, attempt 2 TIMEOUT: the candidate is quarantined
    for slowness -- NOT semantically rejected. No FailureRecord, no gate
    verdict, hash not closed, termination names quarantine."""
    env = FlakyEnvironment(SubprocessExecutionEnvironment(), 10 ** 9)
    eng = _engine(env=env, builder=SingleCodeBuilder(GOOD), critic=NoAttackCritic())
    result = eng.run()
    assert not result.success
    assert result.termination.reason == "quarantined"
    assert len(result.quarantined) == 1
    q = result.quarantined[0]
    assert q.attempts == 2
    assert len(_accepted_nodes(result)) == 0
    assert not any(r.candidate_id and r.state_hash == q.state_hash
                   for r in eng._all_records), \
        "quarantine must not manufacture a failure record"
    q_nodes = [n for n in result.graph.nodes.values() if n.node_id == q.node_id]
    assert len(q_nodes) == 1 and q_nodes[0].status is NodeStatus.QUARANTINED
    assert not eng.frontier.is_closed(q.state_hash)


# ==========================================================================
# I9 -- ADVERSARIAL FAILURE (M5 / M7)
# ==========================================================================

def test_i9_adversarial_failure_never_accepted():
    """syntax PASS, contract PASS, required PASS, security PASS,
    regression PASS (vacuous), adversarial FAIL -> NOT ACCEPTED. The node
    may remain a GATE_PASSED search node, but it must never be the
    delivered solution."""
    f32 = _f32_code("# MARK-ADV")
    builder = MarkerBuilder({"": [f32]})
    eng = _engine(builder=builder, critic=PrecisionCritic())
    result = eng.run()
    assert not result.success
    assert len(_accepted_nodes(result)) == 0
    nodes = _nodes_with(eng, result, f32)
    assert len(nodes) == 1
    node = nodes[0]
    assert node.status is NodeStatus.GATE_PASSED
    # Engine level: the gate's verdict is authoritative -- the node was
    # adversarially failed (record, stage adversarial) and not accepted.
    # NOTE: node.verification is the FIRST verification, taken when no
    # attacks existed yet, so its adversarial fields are vacuous; the
    # gate's re-verification (which actually ran the attack) is what
    # failed the node. The tier verdict itself is proven below, directly
    # through the Referee, so this test never trusts the stale fields.
    v = node.verification
    assert v is not None
    assert v.static_gate_passed and v.contract_passed
    assert v.acceptance_passed and v.acceptance_passes == 6
    assert v.regression_passed  # no regression tier: vacuous pass
    assert not v.forbidden_constructs and v.static_analysis_complete
    adv_records = _records(eng, FailureType.ADVERSARIAL_FAILURE)
    assert adv_records and adv_records[0].stage == "adversarial"
    assert adv_records[0].state_hash == node.state.content_hash
    # Referee level: the same state against the attack -- acceptance
    # passes 6/6, the adversarial tier runs and fails exactly there.
    _, attacks = PrecisionCritic().propose(None, {})
    referee = Referee(SPEC, SubprocessExecutionEnvironment(),
                      ForgeMindConfig(random_seed=0))
    referee.attach_test_provider(TEST_FILES)
    state = CodebaseState(spec_hash=SPEC.spec_hash, files={"solution.py": f32})
    verdict = referee.verify(state, attack_tests=attacks)
    assert verdict.acceptance_passed and verdict.acceptance_passes == 6
    assert not verdict.adversarial_passed
    assert verdict.adversarial_total == 1 and verdict.adversarial_failures == 1


# ==========================================================================
# I10 -- REFEREE EVIDENCE AUTHENTICITY (C1)
# ==========================================================================

def test_i10_forged_stdout_marker_ignored():
    """A wrong candidate that prints a fake HARNESS_JSON success marker at
    import time must still be rejected on its real test results. The
    marker IS present in the captured stdout (proving the forgery reached
    the untrusted channel) and is ignored by the verdict."""
    eng = _engine(builder=SingleCodeBuilder(FORGE), critic=NoAttackCritic())
    result = eng.run()
    assert not result.success
    assert len(_accepted_nodes(result)) == 0
    nodes = _nodes_with(eng, result, FORGE)
    assert len(nodes) == 1
    v = nodes[0].verification
    assert v is not None
    assert "HARNESS_JSON" in v.stdout
    assert not v.acceptance_passed  # integer-division mean fails test_normal
    assert v.acceptance_failures >= 1


def test_i10_malformed_and_missing_evidence_fail_closed():
    """The evidence validator is fail-closed: missing file, corrupt JSON,
    wrong nonce, and wrong section shape all yield no evidence; only a
    payload with this run's nonce and a dict of sections is accepted."""
    import tempfile
    import os
    referee = Referee(SPEC, SubprocessExecutionEnvironment(),
                      ForgeMindConfig(random_seed=0))
    assert Referee._read_evidence("/nonexistent/result.json", "abc") is None
    with tempfile.TemporaryDirectory() as tmp:
        corrupt = os.path.join(tmp, "corrupt.json")
        with open(corrupt, "w") as fh:
            fh.write("{not json")
        assert Referee._read_evidence(corrupt, "abc") is None
        wrong_nonce = os.path.join(tmp, "wrong.json")
        with open(wrong_nonce, "w") as fh:
            fh.write('{"nonce": "other", "sections": {"acceptance": {}}}')
        assert Referee._read_evidence(wrong_nonce, "abc") is None
        bad_shape = os.path.join(tmp, "shape.json")
        with open(bad_shape, "w") as fh:
            fh.write('{"nonce": "abc", "sections": ["not", "a", "dict"]}')
        assert Referee._read_evidence(bad_shape, "abc") is None
        good = os.path.join(tmp, "good.json")
        with open(good, "w") as fh:
            fh.write('{"nonce": "abc", "sections": {"acceptance": {"total": 1}}}')
        assert Referee._read_evidence(good, "abc") == {"acceptance": {"total": 1}}


def test_i10_missing_evidence_from_crashed_tier():
    """A solution that kills its own tier process (os.abort at import)
    produces no valid structured evidence: the acceptance tier fails
    with no evidence file, the verdict is not manufactured, and the
    candidate is not accepted. (Observed harness behavior, kept honest:
    a plain import-time exception is instead reported as failed tests
    WITH evidence -- that path is covered by the acceptance-failure
    assertions elsewhere, not by this missing-evidence case.)"""
    eng = _engine(builder=SingleCodeBuilder(ABORT_SOLUTION), critic=NoAttackCritic())
    result = eng.run()
    assert not result.success
    assert len(_accepted_nodes(result)) == 0
    nodes = _nodes_with(eng, result, ABORT_SOLUTION)
    assert len(nodes) == 1
    v = nodes[0].verification
    assert v is not None
    assert not v.acceptance_passed
    assert not v.evidence_complete
    assert any("no valid structured result" in x for x in v.violations)


# ==========================================================================
# I11 -- VERIFICATION TIER ISOLATION (M3)
# ==========================================================================

def test_i11_tier_failure_is_contained():
    """A crashing adversarial tier must not fabricate or erase the
    acceptance verdict: acceptance stays 6/6 with rate 1.0, adversarial
    reports 0 totals and fails, the node stays GATE_PASSED (not ACCEPTED),
    and the incomplete-evidence hash is NOT closed."""
    eng = _engine(builder=SingleCodeBuilder(GOOD), critic=CrashCritic())
    result = eng.run()
    assert not result.success
    assert len(_accepted_nodes(result)) == 0
    nodes = _nodes_with(eng, result, GOOD)
    assert len(nodes) == 1
    node = nodes[0]
    assert node.status is NodeStatus.GATE_PASSED
    # Engine level: acceptance intact (6/6, rate 1.0), gate failed the
    # node adversarially (record), hash seen but NOT closed (incomplete
    # evidence never closes). As in I9, node.verification is the first
    # verification (pre-attack, vacuous adversarial fields); the crashed
    # tier verdict is proven directly through the Referee below.
    v = node.verification
    assert v.acceptance_passed and v.acceptance_passes == 6
    assert v.acceptance_pass_rate == 1.0
    adv_records = _records(eng, FailureType.ADVERSARIAL_FAILURE)
    assert adv_records and adv_records[0].state_hash == node.state.content_hash
    assert not eng.frontier.is_closed(node.state.content_hash)
    assert node.state.content_hash in eng.frontier.seen_hashes
    # Referee level: the crashing attack tier contributes zero totals,
    # fails, and reports no valid structured result -- while acceptance
    # still passes 6/6 on the same state.
    _, attacks = CrashCritic().propose(None, {})
    referee = Referee(SPEC, SubprocessExecutionEnvironment(),
                      ForgeMindConfig(random_seed=0))
    referee.attach_test_provider(TEST_FILES)
    state = CodebaseState(spec_hash=SPEC.spec_hash, files={"solution.py": GOOD})
    verdict = referee.verify(state, attack_tests=attacks)
    assert verdict.acceptance_passed and verdict.acceptance_passes == 6
    assert not verdict.adversarial_passed
    assert verdict.adversarial_total == 0
    assert any("adversarial tier produced no valid structured result"
               in x for x in verdict.violations)


def test_i11_workdir_forgery_cannot_become_evidence():
    """A wrong candidate that writes a fake evidence file into its own
    workdir must still be rejected: evidence lives in the referee-only
    result directory, never in the candidate's workdir."""
    forge_file = (
        "import json, os\n"
        "open(os.path.join(os.getcwd(), 'result_acceptance.json'), 'w').write(\n"
        "    json.dumps({'nonce': 'forged', 'sections': {'acceptance':\n"
        "        {'total': 6, 'passed': 6, 'failed': 0, 'details': []}}}))\n"
        + WRONG_MEAN
    )
    eng = _engine(builder=SingleCodeBuilder(forge_file), critic=NoAttackCritic())
    result = eng.run()
    assert not result.success
    assert len(_accepted_nodes(result)) == 0
    nodes = _nodes_with(eng, result, forge_file)
    assert len(nodes) == 1
    assert not nodes[0].verification.acceptance_passed


# ==========================================================================
# I12 -- RESOURCE FAILURE CONTAINMENT
# ==========================================================================

def test_i12_bad_candidates_do_not_corrupt_the_engine():
    """Stdout flooding, stderr flooding, and a subprocess that aborts are
    each rejected without corrupting the engine, the frontier, or the
    next candidate's evaluation: GOOD is accepted afterwards, and its
    verification output is clean of the floods."""
    builder = MarkerBuilder({
        "": [FLOOD_STDOUT, FLOOD_STDERR, ABORT_SOLUTION, GOOD],
    })
    eng = _engine(builder=builder, critic=NoAttackCritic(), branching_factor=4)
    result = eng.run()
    assert result.success
    accepted = _accepted_nodes(result)
    assert len(accepted) == 1
    assert accepted[0].state.files["solution.py"] == GOOD
    _exec_solution(accepted[0].state.files["solution.py"])
    v = accepted[0].verification
    assert "FLOOD" not in v.stdout and "FLOODERR" not in v.stderr
    # Every bad candidate was evaluated and rejected, none accepted.
    for code in (FLOOD_STDOUT, FLOOD_STDERR, ABORT_SOLUTION):
        nodes = _nodes_with(eng, result, code)
        assert len(nodes) == 1
        assert nodes[0].status is NodeStatus.REJECTED
    assert eng.metrics.candidates_created == 4

# ==========================================================================
# I13 -- DETERMINISM (actual contract, not byte-for-byte fantasy)
# ==========================================================================

def _landscape_run(seed):
    f32 = _f32_code("# MARK-C")
    builder = MarkerBuilder({
        "": [SYNTAX_BROKEN, WRONG_EMPTY, f32],
        "# MARK-C": [WRONG_MEAN, GOOD],
    })
    eng = _engine(builder=builder, critic=PrecisionCritic(), seed=seed)
    return eng, eng.run()


def test_i13_same_seed_same_result_repeatedly():
    """Three sequential runs, same seed, no manual reset anywhere: same
    candidate IDs, same winner hash, same termination. The digest
    deliberately excludes elapsed time and resource measurements, which
    are environmental and NOT part of the determinism contract."""
    digests = [_digest(_landscape_run(7)[1]) for _ in range(3)]
    assert digests[0] == digests[1] == digests[2]
    assert digests[0][1] is not None  # a real winner, not a shared None


def test_i13_independent_instances_do_not_leak():
    """Two fresh engine instances, same seed, run back to back with no
    reset calls: identical digests (no global ID or frontier leakage).
    A different seed changes IDs but not the winner or the termination."""
    _, r1 = _landscape_run(7)
    _, r2 = _landscape_run(7)
    assert _digest(r1) == _digest(r2)
    _, r3 = _landscape_run(8)
    d1, d3 = _digest(r1), _digest(r3)
    assert d1[0] != d3[0]  # IDs are seed-derived
    assert d1[1] == d3[1] and d1[2] == d3[2]  # semantics are not


# ==========================================================================
# I14 -- CONCURRENT ENGINE ISOLATION (M11 contract)
# ==========================================================================

def test_i14_concurrent_engines_isolated():
    """One engine instance per thread (instances are NOT shared, per the
    M11 contract). Concurrent runs reproduce their solo digests exactly,
    and far-apart seeds keep ID sets disjoint: no cross-run contamination
    of IDs, verdicts, or frontiers."""
    def landscape(seed):
        return _digest(_landscape_run(seed)[1])

    def good(seed):
        eng = _engine(builder=SingleCodeBuilder(GOOD),
                      critic=NoAttackCritic(), seed=seed)
        return _digest(eng.run())

    solo_land = landscape(0)
    solo_good = good(50_000_000)
    assert solo_land[1] is not None and solo_good[1] is not None

    def run_pair(_):
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_land = pool.submit(landscape, 0)
            fut_good = pool.submit(good, 50_000_000)
            return fut_land.result(), fut_good.result()

    for _ in range(3):
        conc_land, conc_good = run_pair(None)
        assert conc_land == solo_land
        assert conc_good == solo_good
    # Disjoint IDs under far-apart seeds (the M11 universal property is
    # solo == concurrent; disjointness holds here because the seed
    # starts are far apart and each run mints only a handful of IDs).
    assert not (set(solo_land[0]) & set(solo_good[0]))


# ==========================================================================
# I15 -- ACCEPTANCE STATUS INTEGRITY (M5)
# ==========================================================================

def test_i15_acceptance_status_integrity():
    """GATE_PASSED != ACCEPTED, as enum and as behavior. A successful run
    has exactly one ACCEPTED node, and it IS the delivered winner. An
    unsuccessful run has zero ACCEPTED nodes, and adversarial-failed,
    timed-out, quarantined, and rejected nodes never become ACCEPTED."""
    assert NodeStatus.GATE_PASSED != NodeStatus.ACCEPTED

    # Successful run: exactly one ACCEPTED == delivered node.
    land_eng, good_result = _landscape_run(7)
    accepted = _accepted_nodes(good_result)
    assert len(accepted) == 1
    assert good_result.accepted_node is accepted[0]
    assert good_result.success

    # Unsuccessful run (adversarial failure): zero ACCEPTED, the
    # attack-weak node stays GATE_PASSED.
    f32 = _f32_code("# MARK-ADV")
    eng = _engine(builder=MarkerBuilder({"": [f32]}), critic=PrecisionCritic())
    bad_result = eng.run()
    assert not bad_result.success
    assert len(_accepted_nodes(bad_result)) == 0
    assert _nodes_with(eng, bad_result, f32)[0].status is NodeStatus.GATE_PASSED

    # Quarantined run: zero ACCEPTED, quarantined node is QUARANTINED.
    env = FlakyEnvironment(SubprocessExecutionEnvironment(), 10 ** 9)
    eng2 = _engine(env=env, builder=SingleCodeBuilder(GOOD), critic=NoAttackCritic())
    q_result = eng2.run()
    assert len(_accepted_nodes(q_result)) == 0
    assert q_result.quarantined[0].node_id in {
        n.node_id for n in q_result.graph.nodes.values()
        if n.status is NodeStatus.QUARANTINED}

    # No node anywhere in any of these runs is both ACCEPTED and failed.
    for result, e in ((good_result, land_eng), (bad_result, eng),
                      (q_result, eng2)):
        failed_states = {r.state_hash for r in e._all_records}
        for node in _accepted_nodes(result):
            assert node.state.content_hash not in failed_states


# ==========================================================================
# I16 -- FULL INTEGRITY CHALLENGE
# ==========================================================================

# I16 landscape (see module plan for the derivation):
#   ROOT -> SYN (syntax) / WRONG (acceptance fail) /
#           F32 (adv-fail, admitted) / DEEP (adv-fail, admitted) /
#           QUAR (always-timeout -> quarantined) / SPEC (spec mutation) /
#           COLL (normalization attack)
#   F32  -> D1, D2 (both fail: dead end)
#   DEEP -> REC (timeout -> retry -> adv-fail, admitted) /
#           DUP (F32's exact content again: duplicate) /
#           E (GOOD: accepted)
# The winner is reached only after failures, a dead end, a duplicate, a
# recoverable timeout, and a quarantine -- and only because it verifies.
#
# NOTE: the spec-mutation proposal routes through the full engine here
# since F1 (previously it crashed the engine with AttributeError -- the
# FailureType member did not exist; see test_f1_spec_mutation.py).

_I16_F32 = _f32_code("# MARK-F32")
_I16_DEEP = _f32_code("# MARK-DEEP")
_I16_REC = _f32_code("# MARK-F32-REC") + "# RECOVER_ME\n"
_I16_QUAR = GOOD + "# QUARANTINE_ME\n"
_I16_COLL = CodePatchProposal(
    agent_role="builder", intent="normalization attack",
    rationale="integrity", estimated_cost=1.0,
    created_files={"solution.py": GOOD, "./solution.py": GOOD})
_I16_SPEC = CodePatchProposal(
    agent_role="builder", intent="spec mutation",
    rationale="integrity", estimated_cost=1.0,
    created_files={"problem_spec.json": "{}\n"})


class _I16Builder:
    """Root proposals are fixed (including two hand-built attack
    proposals); deeper levels are marker-keyed like MarkerBuilder."""

    role = "builder"

    def propose(self, projection, request_context):
        current = projection.target_files.get("solution.py", "")
        if not current:
            out = []
            for code in (SYNTAX_BROKEN, WRONG_MEAN, _I16_F32, _I16_DEEP,
                         _I16_QUAR):
                out.append(CodePatchProposal(
                    agent_role=self.role, intent="integrity variant",
                    rationale="integrity", estimated_cost=1.0,
                    created_files={"solution.py": code}))
            out.append(_I16_SPEC)
            out.append(_I16_COLL)
            return tuple(out)
        if "# MARK-F32" in current and "# MARK-F32-REC" not in current:
            codes = [WRONG_EMPTY, WRONG_COUNT]
        elif "# MARK-DEEP" in current:
            codes = [_I16_REC, _I16_F32, GOOD]
        else:
            return ()
        out = []
        for code in codes:
            out.append(CodePatchProposal(
                agent_role=self.role, intent="integrity variant",
                rationale="integrity", estimated_cost=1.0,
                modified_files={"solution.py": code}))
        return tuple(out)


def test_i16_full_integrity_challenge():
    """The whole deterministic core, at once: explore, reject, recover,
    backtrack/continue, deduplicate, quarantine, verify, select -- with
    no LLM. The winner must be GOOD, reached only after the gauntlet."""
    eng = _engine(builder=_I16Builder(), critic=PrecisionCritic(),
                  env=MarkerTimeoutEnvironment(SubprocessExecutionEnvironment()),
                  branching_factor=8)
    result = eng.run()
    report = summarize_integrity(result, eng)

    # --- the headline invariants --------------------------------------
    assert result.success
    assert result.termination.reason == "goal_accepted"
    accepted = _accepted_nodes(result)
    assert len(accepted) == 1
    assert accepted[0].state.files["solution.py"] == GOOD
    _exec_solution(accepted[0].state.files["solution.py"])
    assert result.accepted_node is accepted[0]
    assert report["acceptance_count"] == 1

    # --- every failure mode left its trace ----------------------------
    # Transition rejections: syntax, spec mutation, normalization attack.
    rejected = _events(eng, EventType.TRANSITION_REJECTED)
    assert len(rejected) == 3
    # Acceptance failures: WRONG at root, D1 + D2 in the dead end.
    assert report["test_failures"] >= 3
    # Adversarial failures: F32, DEEP, REC.
    assert report["adversarial_failures"] >= 3
    # Duplicate convergence: DEEP's second proposal is F32's exact state.
    assert report["duplicates"] >= 1
    assert _events(eng, EventType.DUPLICATE_DETECTED)
    # Quarantine: QUAR timed out twice and was set aside, not rejected.
    assert report["quarantines"] == 1
    q = result.quarantined[0]
    assert q.attempts == 2
    assert not any(r.state_hash == q.state_hash for r in eng._all_records)
    # Recoverable timeout: REC's first attempt timed out, its retry ran.
    rec_nodes = _nodes_with(eng, result, _I16_REC)
    assert len(rec_nodes) == 1
    assert len(_verify_starts_for(eng, rec_nodes[0].node_id)) == 2
    assert rec_nodes[0].status is NodeStatus.GATE_PASSED  # admitted, not won
    # Dead end: F32's branch produced no admitted children.
    f32_nodes = _nodes_with(eng, result, _I16_F32)
    assert len(f32_nodes) == 1
    assert not any(n.parent_node_id == f32_nodes[0].node_id
                   for n in result.graph.nodes.values()
                   if n.status is NodeStatus.GATE_PASSED)
    # Misleading branch: F32 and DEEP both looked perfect on acceptance
    # (rate 1.0) yet neither was accepted -- the adversarial gate, not
    # the score, decided.
    for node in f32_nodes + _nodes_with(eng, result, _I16_DEEP):
        assert node.verification.acceptance_pass_rate == 1.0
        assert node.status is NodeStatus.GATE_PASSED

    # --- the run really evaluated everything (derived, not arbitrary) --
    # 7 root proposals + 2 in the dead end + 3 under DEEP = 12 candidates.
    assert report["candidates_created"] == 12
    # Verifications started: root 5 (WRONG 1, F32 1, DEEP 1, QUAR 2) +
    # dead end 2 + DEEP 3 (REC 2, E 1); DUP skipped before verification,
    # and adversarial-gate re-verifications emit no VERIFICATION_STARTED.
    assert report["verifications_started"] == 10
