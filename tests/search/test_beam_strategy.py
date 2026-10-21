"""H1: BeamSearchStrategy must run a real beam — level-synchronous
rounds of up to beam_width nodes at the shallowest frontier depth."""

from forgemind.core.state import CodebaseState
from forgemind.search.node import (
    BeamSearchStrategy,
    SearchFrontier,
    SearchGraph,
    SearchNode,
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


def test_beam_takes_shallowest_depth_top_k_in_sort_order():
    graph, root = _graph_with()
    a = _child(graph, root, "a", depth=1)
    b = _child(graph, root, "b", depth=1)
    c = _child(graph, a, "c", depth=2)
    frontier = SearchFrontier()
    frontier.insert(a, guidance=0.9)
    frontier.insert(b, guidance=0.8)
    frontier.insert(c, guidance=0.99)  # highest guidance, but deeper

    strategy = BeamSearchStrategy(beam_width=2)
    beam = strategy.select_beam(frontier, graph)

    assert [n.node_id for n in beam] == [a.node_id, b.node_id]
    # The deeper node was NOT taken despite outscoring both, and the
    # beam members are out of the frontier.
    assert len(frontier) == 1
    assert frontier.peek_best().node_id == c.node_id


def test_beam_width_one_takes_single_best_shallow_node():
    graph, root = _graph_with()
    a = _child(graph, root, "a", depth=1)
    b = _child(graph, root, "b", depth=1)
    frontier = SearchFrontier()
    frontier.insert(a, guidance=0.5)
    frontier.insert(b, guidance=0.9)
    beam = BeamSearchStrategy(beam_width=1).select_beam(frontier, graph)
    assert [n.node_id for n in beam] == [b.node_id]
    assert len(frontier) == 1


def test_beam_smaller_than_width_when_level_is_small():
    graph, root = _graph_with()
    a = _child(graph, root, "a", depth=1)
    frontier = SearchFrontier()
    frontier.insert(a, guidance=0.5)
    beam = BeamSearchStrategy(beam_width=3).select_beam(frontier, graph)
    assert [n.node_id for n in beam] == [a.node_id]
    assert len(frontier) == 0


def test_beam_empty_frontier_returns_empty():
    graph, _ = _graph_with()
    frontier = SearchFrontier()
    assert BeamSearchStrategy(beam_width=3).select_beam(frontier, graph) == ()


def test_beam_next_round_serves_the_deeper_level():
    graph, root = _graph_with()
    a = _child(graph, root, "a", depth=1)
    c = _child(graph, a, "c", depth=2)
    frontier = SearchFrontier()
    frontier.insert(a, guidance=0.9)
    frontier.insert(c, guidance=0.99)
    strategy = BeamSearchStrategy(beam_width=2)
    first = strategy.select_beam(frontier, graph)
    second = strategy.select_beam(frontier, graph)
    assert [n.node_id for n in first] == [a.node_id]
    assert [n.node_id for n in second] == [c.node_id]
