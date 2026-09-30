import pytest

from forgemind.core.proposal import CodePatchProposal
from forgemind.core.state import CodebaseState
from forgemind.core.transition import make_root_state
from forgemind.recovery.failure import FailureType, RecoveryAction, RecoveryPolicy
from forgemind.search.node import (
    BeamSearchStrategy,
    FrontierEntry,
    NodeStatus,
    SearchFrontier,
    SearchGraph,
    SearchNode,
)
from forgemind.search.stagnation import StagnationDetector


def mknode(node_id, files, parent=None, depth=0, idx=0):
    if parent is None and depth != 0:
        parent = "virtual-root"  # node construction only; no graph registered
    state = CodebaseState(spec_hash="a" * 64, files=files)
    return SearchNode(node_id=node_id, state=state, parent_node_id=parent,
                      candidate_id="c" + node_id if parent else None,
                      depth=depth, creation_index=idx)


# ---------------- frontier ----------------

def test_frontier_ordering_deterministic():
    f = SearchFrontier()
    n_low = mknode("n1", {"a.py": "1"})
    n_high = mknode("n2", {"a.py": "2"})
    assert f.insert(n_high, guidance=0.9)
    assert f.insert(n_low, guidance=0.2)
    best = f.peek_best()
    assert best is not None and best.node_id == "n2"


def test_frontier_tie_break_by_depth_then_index():
    f = SearchFrontier()
    a = mknode("na", {"a.py": "1"}, parent="root", depth=3, idx=5)
    b = mknode("nb", {"b.py": "1"}, parent="root", depth=1, idx=9)
    f.insert(a, 0.5)
    f.insert(b, 0.5)
    best = f.peek_best()
    assert best is not None and best.node_id == "nb"  # shallower wins ties


def test_frontier_capacity_prunes_worst():
    f = SearchFrontier(capacity=2)
    for i in range(4):
        f.insert(mknode(f"n{i}", {f"f{i}.py": "x"}), guidance=i / 10)
    entries = f.top_k(10)
    assert len(entries) == 2
    assert [e.guidance for e in entries] == [0.3, 0.2]


def test_frontier_duplicate_state_rejected():
    f = SearchFrontier()
    s = {"a.py": "same"}
    assert f.insert(mknode("n1", s), 0.5)
    assert not f.insert(mknode("n2", s), 0.7)  # identical content hash
    assert len(f) == 1


# ---------------- graph ----------------

def test_graph_ancestry_and_valid_parent():
    g = SearchGraph()
    root = mknode("r", {})
    g.add(root)
    child = mknode("c", {"a.py": "1"}, parent=root.node_id, depth=1)
    g.add(child)
    chain = g.ancestry(child.node_id)
    assert [n.node_id for n in chain] == ["r", "c"]
    with pytest.raises(ValueError):
        g.add(mknode("orphan", {"x.py": "y"}, parent="missing", depth=1))


def test_graph_allows_shared_content_states():
    """Two nodes may point at equivalent states (diamond convergence)."""
    g = SearchGraph()
    root = mknode("r", {})
    g.add(root)
    d1 = mknode("d1", {"x.py": "v"}, parent="r", depth=1, idx=1)
    d2 = mknode("d2", {"x.py": "v"}, parent="r", depth=1, idx=2)
    g.add(d1)
    g.add(d2)
    assert d1.state == d2.state
    assert d1.node_id != d2.node_id


# ---------------- beam strategy ----------------

def test_beam_selects_best_and_removes():
    f = SearchFrontier()
    g = SearchGraph()
    for i, guid in enumerate([0.1, 0.9, 0.4]):
        n = mknode(f"n{i}", {f"f{i}.py": "x"})
        g.add(n)
        f.insert(n, guid)
    strat = BeamSearchStrategy(beam_width=2)
    node = strat.select_next(f, g)
    assert node is not None and node.node_id == "n1"
    assert len(f) == 2


def test_beam_empty_frontier_returns_none():
    assert BeamSearchStrategy().select_next(SearchFrontier(), SearchGraph()) is None


# ---------------- stagnation ----------------

def test_stagnation_flat_guidance():
    d = StagnationDetector(window=3, epsilon=1e-3)
    assert not d.record(guidance=0.50, content_hash="h1")
    assert d.record(guidance=0.5001, content_hash="h2")  # delta < epsilon


def test_stagnation_progress_is_not_stagnant():
    d = StagnationDetector(window=3, epsilon=1e-3)
    assert not d.record(guidance=0.30, content_hash="h1")
    assert not d.record(guidance=0.80, content_hash="h2")


def test_stagnation_oscillation_detected():
    d = StagnationDetector(window=6, epsilon=1e-9)
    for h in ["A", "B", "A", "B"]:
        detected = d.record(guidance=0.5 if h == "A" else 0.50005, content_hash=h)
    assert detected


# ---------------- recovery ----------------

def test_recovery_mapping_defaults():
    p = RecoveryPolicy()
    assert p.action_for(FailureType.SYNTAX_ERROR) is RecoveryAction.BUILDER_RETRY
    assert p.action_for(FailureType.TEST_FAILURE) is RecoveryAction.CRITIC_ANALYSIS
    assert p.action_for(FailureType.DUPLICATE_STATE) is RecoveryAction.PRUNE
    assert p.action_for(FailureType.DEPTH_EXHAUSTED) is RecoveryAction.BACKTRACK


def test_recovery_overrides():
    p = RecoveryPolicy(overrides={FailureType.TEST_FAILURE: RecoveryAction.BACKTRACK})
    assert p.action_for(FailureType.TEST_FAILURE) is RecoveryAction.BACKTRACK


def test_failure_record_creation():
    p = RecoveryPolicy()
    rec = p.record(state_hash="h", candidate_id="c",
                   failure_type=FailureType.TIMEOUT, stage="dynamic",
                   evidence="wall clock exceeded")
    assert rec.failure_type is FailureType.TIMEOUT
    assert rec.recovery is RecoveryAction.ALTERNATIVE_IMPLEMENTATION
