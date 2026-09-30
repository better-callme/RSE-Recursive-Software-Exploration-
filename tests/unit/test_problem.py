import dataclasses

import pytest

from forgemind.core.problem import AcceptanceTest, ProblemSpec


@pytest.fixture()
def base_kwargs():
    return dict(
        problem_id="p1",
        description="d",
        requirements=("r",),
        acceptance_tests=(AcceptanceTest("t1", "test_t1"),),
        interface_constraints={"f": "function: f(x)"},
    )


def test_problem_spec_immutable(base_kwargs):
    spec = ProblemSpec(**base_kwargs)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.problem_id = "other"
    # Normal mutation paths raise. NOTE (documented limitation): CPython's
    # object.__setattr__() is a documented escape hatch around *any* frozen
    # dataclass; genuine tamper-resistance comes from the engine never calling
    # it plus content hashing detecting any drift.
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.description = "mutated"
    assert spec.description == "d"


def test_spec_hash_order_independent():
    kw = dict(
        problem_id="p1",
        description="d",
        requirements=("r",),
        acceptance_tests=(AcceptanceTest("t1", "test_t1"),),
    )
    a = ProblemSpec(**kw, interface_constraints={"x": "a", "y": "b"})
    b = ProblemSpec(**kw, interface_constraints={"y": "b", "x": "a"})
    assert a.spec_hash == b.spec_hash


def test_spec_hash_changes_with_requirements(base_kwargs):
    a = ProblemSpec(**base_kwargs)
    b = ProblemSpec(**{**base_kwargs, "requirements": ("changed",)})
    assert a.spec_hash != b.spec_hash


def test_spec_requires_acceptance_tests(base_kwargs):
    kw = dict(base_kwargs)
    kw["acceptance_tests"] = ()
    with pytest.raises(ValueError):
        ProblemSpec(**kw)
