import pytest

from forgemind.core.proposal import CodePatchProposal
from forgemind.core.state import CodebaseState
from forgemind.core.transition import TransitionEngine, make_root_state


@pytest.fixture()
def engine(spec):
    return TransitionEngine(spec)


GOOD = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t / len(data)}\n"
)

BAD_SYNTAX = "def compute_stats(:\n  pass\n"


def test_root_state(engine):
    s = make_root_state(engine.spec)
    assert s.files == {} and s.spec_hash == engine.spec.spec_hash


def test_apply_creates_state(engine):
    root = make_root_state(engine.spec)
    r = engine.apply(root, CodePatchProposal(created_files={"solution.py": GOOD}))
    assert r.ok and r.new_state is not None
    assert "solution.py" in r.new_state.files


def test_transition_deterministic(engine):
    root = make_root_state(engine.spec)
    proposal = CodePatchProposal(created_files={"solution.py": GOOD})
    a = engine.apply(root, proposal)
    b = engine.apply(root, proposal)
    assert a.ok and b.ok
    assert a.new_state.content_hash == b.new_state.content_hash


def test_problem_spec_mutation_rejected(engine):
    root = make_root_state(engine.spec)
    r = engine.apply(root, CodePatchProposal(
        created_files={"problem_spec.json": '{"requirements": []}'}))
    assert not r.ok and r.violation == "PROBLEM_SPEC_MUTATION"


def test_syntax_error_rejected_at_transition(engine):
    root = make_root_state(engine.spec)
    r = engine.apply(root, CodePatchProposal(created_files={"solution.py": BAD_SYNTAX}))
    assert not r.ok and r.violation == "SYNTAX"


def test_interface_constraint_enforced(engine):
    root = make_root_state(engine.spec)
    wrong_name = GOOD.replace("compute_stats", "compute_statistics")
    r = engine.apply(root, CodePatchProposal(created_files={"solution.py": wrong_name}))
    assert not r.ok and r.violation == "INTERFACE_CONSTRAINT"


def test_delete_nonexistent_rejected(engine):
    root = make_root_state(engine.spec)
    r = engine.apply(root, CodePatchProposal(deleted_files=("ghost.py",)))
    assert not r.ok


def test_dependency_policy(engine):
    root = make_root_state(engine.spec)
    r = engine.apply(root, CodePatchProposal(
        created_files={"a.py": "pass"},
        dependency_changes={"requests": "2.0"},  # not in allowlist
    ))
    assert not r.ok
