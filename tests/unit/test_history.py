from forgemind.core.history import EventType, History


def test_history_append_only_and_ordered():
    h = History()
    h.append(EventType.SEARCH_STARTED, problem="p")
    h.append(EventType.NODE_SELECTED, node="n1")
    events = h.events
    assert [e.seq for e in events] == [0, 1]
    assert events[0].event_type is EventType.SEARCH_STARTED


def test_history_counts():
    h = History()
    for _ in range(3):
        h.append(EventType.AGENT_CALLED, role="builder")
    assert h.count(EventType.AGENT_CALLED) == 3
    assert len(h.of_type(EventType.AGENT_CALLED)) == 3


def test_history_metadata_preserved():
    h = History()
    h.append(EventType.TRANSITION_REJECTED, reason="SYNTAX", candidate="c1")
    e = h.events[-1]
    assert e.metadata["reason"] == "SYNTAX" and e.metadata["candidate"] == "c1"


def test_trace_rendering_contains_events():
    h = History()
    h.append(EventType.STATE_ACCEPTED, node="n9")
    assert "STATE_ACCEPTED" in h.render_trace()
    assert "n9" in h.render_trace()
