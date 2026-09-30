import pytest

from forgemind.core.identity import state_content_hash, normalize_path, file_tree_hash
from forgemind.core.state import CodebaseState


SPEC = "a" * 64


def test_state_immutable_by_attribute():
    s = CodebaseState(spec_hash=SPEC, files={"a.py": "x = 1"})
    with pytest.raises(AttributeError):
        s.files = {}


def test_state_files_view_is_frozen():
    s = CodebaseState(spec_hash=SPEC, files={"a.py": "x = 1"})
    try:
        s.files["b.py"] = "y"  # type: ignore[index]
        mutated = True
    except TypeError:
        mutated = False
    assert not mutated
    assert "b.py" not in s.files


def test_same_state_same_hash_regardless_of_insertion_order():
    a = CodebaseState(spec_hash=SPEC,
                      files={"a.py": "1", "b.py": "2", "c.py": "3"},
                      dependencies={"p": "1.0", "q": "2.0"},
                      configuration={"k": "v"})
    b = CodebaseState(spec_hash=SPEC,
                      files={"c.py": "3", "a.py": "1", "b.py": "2"},
                      dependencies={"q": "2.0", "p": "1.0"},
                      configuration={"k": "v"})
    assert a.content_hash == b.content_hash


def test_hash_changes_on_file_modification():
    a = CodebaseState(spec_hash=SPEC, files={"a.py": "1"})
    b = CodebaseState(spec_hash=SPEC, files={"a.py": "2"})
    assert a.content_hash != b.content_hash


def test_hash_changes_on_dependency_modification():
    a = CodebaseState(spec_hash=SPEC, files={}, dependencies={"p": "1"})
    b = CodebaseState(spec_hash=SPEC, files={}, dependencies={"p": "2"})
    assert a.content_hash != b.content_hash


def test_hash_changes_on_runtime_change():
    a = CodebaseState(spec_hash=SPEC, files={}, runtime_spec="python3")
    b = CodebaseState(spec_hash=SPEC, files={}, runtime_spec="python3.13")
    assert a.content_hash != b.content_hash


def test_hash_includes_spec():
    a = CodebaseState(spec_hash="1" * 64, files={})
    b = CodebaseState(spec_hash="2" * 64, files={})
    assert a.content_hash != b.content_hash


def test_normalize_path():
    assert normalize_path("./a//b/../c") == "a/c"
    assert normalize_path("a/b/") == "a/b"


def test_node_vs_content_identity_distinct():
    from forgemind.search.node import SearchNode, SearchGraph
    s = CodebaseState(spec_hash=SPEC, files={"a.py": "1"})
    n1 = SearchNode(node_id="n1", state=s, parent_node_id=None, candidate_id=None,
                    depth=0, creation_index=0)
    n2 = SearchNode(node_id="n2", state=s, parent_node_id="n1",
                    candidate_id="c1", depth=1, creation_index=1)
    assert n1.node_id != n2.node_id          # distinct history occurrences
    assert n1.state == n2.state              # same semantic content
    assert n1.state.content_hash == n2.state.content_hash


def test_non_root_requires_ancestry():
    from forgemind.search.node import SearchNode
    s = CodebaseState(spec_hash=SPEC, files={})
    with pytest.raises(ValueError):
        SearchNode(node_id="n", state=s, parent_node_id="", candidate_id=None,
                   depth=3, creation_index=0)
