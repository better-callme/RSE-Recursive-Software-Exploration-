"""UCTSearchStrategy: UCB1 ordering within the shallowest level.

Same level discipline as the beam (no deeper node is ever picked while
a shallower one waits); the ONLY difference is the within-level order:
UCB1 instead of raw guidance. Visit counts are owned by the strategy
instance, so every test below is deterministic.

The negative control is test_uct_differs_from_beam_on_revisited_node:
the same revisited-node scenario picks the first-created node under
beam ordering and the unvisited node under UCT. If UCT were reverted
to beam ordering, that test fails.
"""

import math

import pytest

from forgemind.core.config import ForgeMindConfig
from forgemind.core.state import CodebaseState
from forgemind.search.node import (
    BeamSearchStrategy,
    SearchFrontier,
    SearchGraph,
    SearchNode,
    UCTSearchStrategy,
)


def _graph_with(spec_hash="a" * 64):
    graph = SearchGraph()
    root = SearchNode(
        node_id=graph.next_node_id(),
        state=CodebaseState(spec_hash=spec_hash, files={}),
        parent_node_id=None, candidate_id=None, depth=0,
        creation_index=0,
    )
    graph.add(root)
    return graph, root


def _child(graph, parent, tag, depth):
    node = SearchNode(
        node_id=graph.next_node_id(),
        state=CodebaseState(spec_hash="a" * 64, files={"f.py": tag}),
        parent_node_id=parent.node_id, candidate_id="c-" + tag,
        depth=depth, creation_index=len(graph),
    )
    graph.add(node)
    return node


def _frontier_with(entries):
    """entries: list of (node, guidance). Returns (graph, frontier)."""
    graph, root = _graph_with()
    frontier = SearchFrontier()
    for node, guidance in entries:
        frontier.insert(node, guidance=guidance)
    return graph, frontier


def test_uct_first_round_matches_beam_order():
    # total_selections == 0 -> ln(1) == 0 -> the exploration term is
    # zero and UCB == guidance, so the first round is beam order.
    graph, root = _graph_with()
    a = _child(graph, root, "a", depth=1)
    b = _child(graph, root, "b", depth=1)
    c = _child(graph, a, "c", depth=2)
    frontier = SearchFrontier()
    frontier.insert(a, guidance=0.9)
    frontier.insert(b, guidance=0.8)
    frontier.insert(c, guidance=0.99)  # highest guidance, but deeper

    beam = UCTSearchStrategy(beam_width=2).select_beam(frontier, graph)
    assert [n.node_id for n in beam] == [a.node_id, b.node_id]
    assert len(frontier) == 1
    assert frontier.peek_best().node_id == c.node_id


def test_uct_favors_unvisited_over_visited_equal_guidance():
    # A and B tie on guidance. A is selected once, then reinserted:
    # its exploration bonus shrinks (visits 1 vs 0) while B's stays
    # full, so the next round picks B. With c = sqrt(2) and total = 1:
    #   B: 0.5 + c*sqrt(ln 2 / 1) = 1.677 > A: 0.5 + c*sqrt(ln 2 / 2) = 1.333
    graph, root = _graph_with()
    a = _child(graph, root, "a", depth=1)
    b = _child(graph, root, "b", depth=1)
    frontier = SearchFrontier()
    frontier.insert(a, guidance=0.5)
    frontier.insert(b, guidance=0.5)

    strategy = UCTSearchStrategy(beam_width=1)
    first = strategy.select_beam(frontier, graph)
    assert [n.node_id for n in first] == [a.node_id]
    assert strategy.visits(a.node_id) == 1
    assert strategy.total_selections == 1

    assert frontier.reinsert(a, guidance=0.5)
    second = strategy.select_beam(frontier, graph)
    assert [n.node_id for n in second] == [b.node_id]


def test_uct_differs_from_beam_on_revisited_node():
    # Negative control: the identical revisited-node scenario under
    # beam ordering picks the FIRST-CREATED node (A) again — beam has
    # no memory of the visit. UCT picks the unvisited node (B).
    # If UCT were reverted to beam ordering, this test fails.
    def run(strategy):
        graph, root = _graph_with()
        a = _child(graph, root, "a", depth=1)
        b = _child(graph, root, "b", depth=1)
        frontier = SearchFrontier()
        frontier.insert(a, guidance=0.5)
        frontier.insert(b, guidance=0.5)
        strategy.select_beam(frontier, graph)
        assert frontier.reinsert(a, guidance=0.5)
        picked = strategy.select_beam(frontier, graph)
        return [n.node_id for n in picked], a.node_id, b.node_id

    beam_pick, a_id, b_id = run(BeamSearchStrategy(beam_width=1))
    assert beam_pick == [a_id]
    uct_pick, a_id, b_id = run(UCTSearchStrategy(beam_width=1))
    assert uct_pick == [b_id]


def test_uct_exploitation_wins_when_guidance_gap_is_large():
    # H (guidance 0.9, visited once) vs L (guidance 0.1, unvisited)
    # with total = 1: H's UCB is 0.9 + c*sqrt(ln 2 / 2) = 1.733, L's
    # is 0.1 + c*sqrt(ln 2 / 1) = 1.277. The guidance gap (0.8) beats
    # the exploration gap (0.344), so exploitation wins.
    graph, root = _graph_with()
    high = _child(graph, root, "high", depth=1)
    low = _child(graph, root, "low", depth=1)
    frontier = SearchFrontier()
    frontier.insert(high, guidance=0.9)
    frontier.insert(low, guidance=0.1)

    strategy = UCTSearchStrategy(beam_width=1)
    first = strategy.select_beam(frontier, graph)
    assert [n.node_id for n in first] == [high.node_id]

    assert frontier.reinsert(high, guidance=0.9)
    second = strategy.select_beam(frontier, graph)
    assert [n.node_id for n in second] == [high.node_id]
    assert strategy.visits(high.node_id) == 2


def test_uct_level_discipline_preserved():
    # A deeper node with a MUCH higher UCB is never picked while a
    # shallower node waits — level discipline is the same as beam.
    graph, root = _graph_with()
    shallow = _child(graph, root, "shallow", depth=1)
    deep = _child(graph, shallow, "deep", depth=2)
    frontier = SearchFrontier()
    frontier.insert(shallow, guidance=0.0)
    frontier.insert(deep, guidance=1.0)

    strategy = UCTSearchStrategy(beam_width=2)
    first = strategy.select_beam(frontier, graph)
    assert [n.node_id for n in first] == [shallow.node_id]
    second = strategy.select_beam(frontier, graph)
    assert [n.node_id for n in second] == [deep.node_id]


def test_uct_select_next_keeps_level_discipline():
    graph, root = _graph_with()
    shallow = _child(graph, root, "shallow", depth=1)
    deep = _child(graph, shallow, "deep", depth=2)
    frontier = SearchFrontier()
    frontier.insert(shallow, guidance=0.0)
    frontier.insert(deep, guidance=1.0)

    strategy = UCTSearchStrategy(beam_width=2)
    assert strategy.select_next(frontier, graph).node_id == shallow.node_id
    assert strategy.select_next(frontier, graph).node_id == deep.node_id
    assert strategy.select_next(frontier, graph) is None


def test_uct_beam_width_cap_per_round():
    graph, root = _graph_with()
    a = _child(graph, root, "a", depth=1)
    b = _child(graph, root, "b", depth=1)
    c = _child(graph, root, "c", depth=1)
    frontier = SearchFrontier()
    frontier.insert(a, guidance=0.9)
    frontier.insert(b, guidance=0.8)
    frontier.insert(c, guidance=0.7)

    beam = UCTSearchStrategy(beam_width=2).select_beam(frontier, graph)
    assert [n.node_id for n in beam] == [a.node_id, b.node_id]
    assert len(frontier) == 1


def test_uct_select_next_removes_node_and_counts_visit():
    graph, root = _graph_with()
    a = _child(graph, root, "a", depth=1)
    b = _child(graph, root, "b", depth=1)
    frontier = SearchFrontier()
    frontier.insert(a, guidance=0.5)
    frontier.insert(b, guidance=0.5)

    strategy = UCTSearchStrategy(beam_width=1)
    picked = strategy.select_next(frontier, graph)
    assert picked.node_id == a.node_id
    assert len(frontier) == 1  # A is out of the frontier
    assert strategy.visits(a.node_id) == 1
    assert strategy.total_selections == 1

    # Reinserting A does not reset its visit count: the next
    # select_next favors the unvisited B.
    assert frontier.reinsert(a, guidance=0.5)
    assert strategy.select_next(frontier, graph).node_id == b.node_id


def test_uct_empty_frontier():
    graph, _ = _graph_with()
    frontier = SearchFrontier()
    strategy = UCTSearchStrategy(beam_width=3)
    assert strategy.select_next(frontier, graph) is None
    assert strategy.select_beam(frontier, graph) == ()
    assert strategy.total_selections == 0


def test_uct_deterministic_across_runs():
    # The same scenario, played twice with fresh strategies, must
    # produce the identical selection sequence (no randomness, no
    # wall-clock, no global state).
    def play():
        graph, root = _graph_with()
        a = _child(graph, root, "a", depth=1)
        b = _child(graph, root, "b", depth=1)
        c = _child(graph, a, "c", depth=2)
        frontier = SearchFrontier()
        frontier.insert(a, guidance=0.7)
        frontier.insert(b, guidance=0.7)
        frontier.insert(c, guidance=0.9)
        strategy = UCTSearchStrategy(beam_width=1)
        picks = []
        for _ in range(3):
            node = strategy.select_next(frontier, graph)
            if node is None:
                break
            picks.append(node.node_id)
            if node.node_id == a.node_id:
                # Reinsert A once so the second round must choose
                # between a visited A and an unvisited B.
                frontier.reinsert(node, guidance=0.7)
        return picks

    assert play() == play()


def test_uct_config_validation():
    assert ForgeMindConfig().frontier_strategy == "beam"
    assert ForgeMindConfig().uct_exploration == pytest.approx(math.sqrt(2))
    assert ForgeMindConfig(frontier_strategy="uct").frontier_strategy == "uct"
    with pytest.raises(ValueError):
        ForgeMindConfig(frontier_strategy="dfs")
    with pytest.raises(ValueError):
        ForgeMindConfig(frontier_strategy="")
    with pytest.raises(ValueError):
        ForgeMindConfig(uct_exploration=0)
    with pytest.raises(ValueError):
        ForgeMindConfig(uct_exploration=-1.0)
