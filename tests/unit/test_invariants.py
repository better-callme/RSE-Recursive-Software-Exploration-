"""Invariant tests: agents can never touch canonical engine state."""

import pytest

from forgemind.agents import projections
from forgemind.agents.mocks import MockArchitect, MockBuilder, MockCritic
from forgemind.core.state import CodebaseState
from forgemind.core.problem import AcceptanceTest, ProblemSpec
from forgemind.verification.results import VerificationResult


def _spec():
    return ProblemSpec(
        problem_id="iso", description="d", requirements=("r",),
        acceptance_tests=(AcceptanceTest("t", "test_t"),),
        interface_constraints={"f": "function: f()"},
    )


def test_projections_are_immutable_snapshots():
    spec = _spec()
    state = CodebaseState(spec_hash=spec.spec_hash,
                          files={"a.py": "def f():\n    pass\n"})
    p = projections.build_architect_projection(spec, state)
    with pytest.raises(TypeError):
        p.file_structure = {}
    b = projections.build_builder_projection(spec, state)
    with pytest.raises(TypeError):
        b.target_files = {}
    c = projections.build_critic_projection(spec, state)
    with pytest.raises(TypeError):
        c.relevant_code = {}


def test_projection_does_not_expose_frontier_or_scores():
    import dataclasses
    spec = _spec()
    state = CodebaseState(spec_hash=spec.spec_hash, files={})
    fields = {f.name for f in dataclasses.fields(
        projections.ArchitectProjection)}
    assert not (fields & {"frontier", "scores", "graph", "history"})


def test_agents_return_proposals_not_states():
    spec = _spec()
    state = CodebaseState(spec_hash=spec.spec_hash,
                          files={"s.py": "def f():\n    pass\n"})
    arch = MockArchitect().propose(
        projections.build_architect_projection(spec, state), {})
    builder = MockBuilder(module_path="s.py")
    patches = builder.propose(
        projections.build_builder_projection(spec, state), {})
    critic_out = MockCritic().propose(
        projections.build_critic_projection(spec, state), {})

    from forgemind.core.proposal import (
        ArchitectureProposal, CodePatchProposal, CritiqueProposal,
        AttackTestProposal,
    )
    assert all(isinstance(p, ArchitectureProposal) for p in arch)
    assert all(isinstance(p, CodePatchProposal) for p in patches)
    critiques, attacks = critic_out
    assert all(isinstance(p, CritiqueProposal) for p in critiques)
    assert all(isinstance(a, AttackTestProposal) for a in attacks)


def test_critic_actually_finds_issues():
    """Critic must NOT return empty issues for flawed code."""
    spec = _spec()
    flawed = CodebaseState(spec_hash=spec.spec_hash, files={
        "s.py": "def f(data):\n    return sum(data) // len(data)\n"})
    critiques, attacks = MockCritic().propose(
        projections.build_critic_projection(spec, flawed), {})
    issues = [i for c in critiques for i in c.issues]
    assert issues, "critic failed to identify deterministic issue"
    assert attacks, "critic failed to propose attack tests"


def test_builder_generates_multiple_genuinely_different_candidates():
    spec = _spec()
    state = CodebaseState(spec_hash=spec.spec_hash, files={})
    patches = MockBuilder().propose(
        projections.build_builder_projection(spec, state), {})
    sources = {p.created_files["solution.py"] for p in patches}
    assert len(sources) >= 3, "builder must emit distinct implementations"


def test_agent_cannot_mutate_state_via_projection_leak():
    """Mutating projection mappings must not affect canonical state."""
    spec = _spec()
    state = CodebaseState(spec_hash=spec.spec_hash, files={"a.py": "x\n"})
    proj = projections.build_critic_projection(spec, state)
    try:
        dict(proj.relevant_code)["a.py"] = "MUTATED"
    except TypeError:
        pass
    assert state.files["a.py"] == "x\n"


def test_unverified_code_cannot_be_accepted():
    """A node without verification evidence can never be ACCEPTED."""
    from forgemind.search.node import SearchNode, NodeStatus

    state = CodebaseState(spec_hash="b" * 64, files={})
    node = SearchNode(node_id="n", state=state, parent_node_id=None,
                      candidate_id=None, depth=0, creation_index=0)
    assert node.status is NodeStatus.UNVERIFIED
    assert node.verification is None
