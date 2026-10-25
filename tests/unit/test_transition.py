import dataclasses

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


def test_innocent_spec_like_names_allowed_l2(engine):
    # L2: the old substring clause rejected these; exact/suffix matching
    # allows them. The GOOD solution file matters: without it the patch
    # fails the interface constraint regardless, which would hide the
    # policy under test (the old dependency-test trap).
    for innocent in ("my_problem_spec.json", "docs/problem_spec.json.bak"):
        root = make_root_state(engine.spec)
        r = engine.apply(root, CodePatchProposal(
            created_files={"solution.py": GOOD, innocent: "pass"}))
        assert r.ok, (innocent, r.violation)


def test_normalized_collision_rejected_l3(engine):
    # L3: two spellings, one canonical path — the second value used to
    # overwrite the first silently. The GOOD content keeps the interface
    # constraint satisfied so the collision is the deciding rejection.
    root = make_root_state(engine.spec)
    r = engine.apply(root, CodePatchProposal(
        created_files={"solution.py": GOOD, "./solution.py": GOOD}))
    assert not r.ok and r.violation == "INVALID_PATH"


def test_single_noncanonical_spelling_ok_l3(engine):
    # Guard: normalization itself is intended — one spelling alone works.
    root = make_root_state(engine.spec)
    r = engine.apply(root, CodePatchProposal(
        created_files={"./solution.py": GOOD}))
    assert r.ok


def test_state_construction_collision_rejected_l3(engine):
    # L3: CodebaseState was the second silent last-wins entry point.
    from forgemind.core.state import CodebaseState
    with pytest.raises(ValueError):
        CodebaseState(spec_hash=engine.spec.spec_hash,
                      files={"a//b.py": "x", "a/b.py": "y"})


def test_nested_spec_path_still_forbidden_l2(engine):
    for forbidden in ("sub/problem_spec.json", ".forgemind/spec.json",
                      "x/.forgemind/spec.json"):
        root = make_root_state(engine.spec)
        r = engine.apply(root, CodePatchProposal(
            created_files={"solution.py": GOOD, forbidden: "{}"}))
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


def test_empty_allowlist_denies_new_deps_m12(engine):
    # M12: () means NO new dependencies. Pre-fix the truthiness guard
    # skipped the check for () and this patch was allowed (it satisfies
    # the interface constraint, so dependency policy is the deciding
    # rejection — unlike test_dependency_policy above, whose patch is
    # rejected by the interface constraint regardless of policy).
    spec = dataclasses.replace(engine.spec, dependency_constraints=())
    eng = TransitionEngine(spec)
    root = make_root_state(spec)
    r = eng.apply(root, CodePatchProposal(
        created_files={"solution.py": GOOD},
        dependency_changes={"requests": "2.0"},
    ))
    assert not r.ok and r.violation == "DEPENDENCY_POLICY"


def test_allowlisted_dependency_permitted_m12(engine):
    spec = dataclasses.replace(engine.spec, dependency_constraints=("requests",))
    eng = TransitionEngine(spec)
    root = make_root_state(spec)
    r = eng.apply(root, CodePatchProposal(
        created_files={"solution.py": GOOD},
        dependency_changes={"requests": "2.0"},
    ))
    assert r.ok


def test_empty_allowlist_without_dep_changes_ok_m12(engine):
    # Guard: () forbids new dependencies, not dependency-free patches.
    spec = dataclasses.replace(engine.spec, dependency_constraints=())
    eng = TransitionEngine(spec)
    root = make_root_state(spec)
    r = eng.apply(root, CodePatchProposal(created_files={"solution.py": GOOD}))
    assert r.ok
