"""H5 integration: the agent pipeline derives identity from the spec.

Rename discriminator: the stats problem with names swapped
(``describe_numbers`` in ``numbers.py``) must run end to end with no
hardcoded ``solution`` / ``compute_stats`` literal in any generated
artifact. Depth-2 test: a mock-builder expansion below the root must
genuinely modify an existing file (the H1-exposed create-only gap).
Fail-closed tests: ``resolve_entry_point`` is the one canonical
function, and it refuses empty / ambiguous specs.
"""

import dataclasses
import hashlib

import pytest

from forgemind.agents.mocks import (
    MockArchitect,
    MockBuilder,
    MockCritic,
    MockKit,
    MockOptimizer,
)
from forgemind.agents.projections import (
    build_architect_projection,
    build_builder_projection,
    build_critic_projection,
)
from forgemind.benchmark.kits import mock_kit_for
from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig
from forgemind.core.history import History
from forgemind.core.problem import ProblemSpec, resolve_entry_point
from forgemind.core.proposal import (
    AttackTestProposal,
    CritiqueProposal,
)
from forgemind.recovery.failure import RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------

class _State:
    """Minimal stand-in for projection builders (they only read .files)."""

    def __init__(self, files=None):
        self.files = files or {}


def rename_spec() -> ProblemSpec:
    """Stats acceptance tests with names swapped: describe_numbers in
    numbers.py. Same semantics, different identity — the test proves
    naming is projection-driven end to end."""
    stats = StatsProblem().spec
    src = stats.test_sources["__fm_acceptance_tests.py"]
    renamed = src.replace(
        "from solution import compute_stats",
        "from numbers import describe_numbers",
    ).replace("compute_stats", "describe_numbers")
    assert "solution" not in renamed and "compute_stats" not in renamed
    return ProblemSpec(
        problem_id="rename",
        description="stats problem under different names",
        requirements=stats.requirements,
        acceptance_tests=stats.acceptance_tests,
        interface_constraints={"describe_numbers": "function: describe_numbers(data)"},
        test_sources={"__fm_acceptance_tests.py": renamed},
    )


def rename_kit() -> MockKit:
    return MockKit(module_path="numbers.py", module="numbers")


class RecordingBuilder(MockBuilder):
    """MockBuilder that records every propose/repair call."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls: list[dict] = []

    def propose(self, projection, request_context):
        proposals = super().propose(projection, request_context)
        self.calls.append({
            "target_files": dict(projection.target_files),
            "proposals": proposals,
        })
        return proposals

    def repair(self, projection, failure_evidence):
        proposals = super().repair(projection, failure_evidence)
        self.calls.append({
            "target_files": dict(projection.target_files),
            "proposals": proposals,
        })
        return proposals


class ParentMarkerBuilder(RecordingBuilder):
    """MockBuilder with parent-sensitive content (H1 pattern).

    Mock variants are parent-independent: expanding any node proposes the
    same three contents, so a re-expansion is always a duplicate by
    content hash and depth-2 nodes could never appear. Prefixing a marker
    derived from the parent state's files makes each level's variants
    distinct, while the placement logic under test stays MockBuilder's
    (``_place``: create vs modify). Root calls are unmarked, so root
    children are the canonical mock variants.
    """

    def propose(self, projection, request_context):
        proposals = super().propose(projection, request_context)
        existing = projection.target_files
        if not existing:
            return proposals
        marker = "# parent:%s\n" % hashlib.sha256(
            "\n".join(f"{k}={v}" for k, v in sorted(existing.items()))
            .encode("utf-8")
        ).hexdigest()[:8]
        marked = []
        for p in proposals:
            marked.append(dataclasses.replace(
                p,
                created_files={k: marker + v for k, v in p.created_files.items()},
                modified_files={k: marker + v for k, v in p.modified_files.items()},
            ))
        # Keep the recording consistent with what the engine received.
        self.calls[-1]["proposals"] = tuple(marked)
        return tuple(marked)


class AlwaysFailCritic:
    """Test-only critic: one attack that always fails, forcing the
    gate-pass / adversarial-fail branch (admission to the frontier)
    without any mock-variant semantics."""

    role = "critic"

    def propose(self, projection, request_context):
        attack = AttackTestProposal(
            agent_role=self.role,
            test_name="attack_always",
            hypothesis="forced adversarial failure",
            test_code=(
                "def attack_always():\n"
                "    from solution import compute_stats\n"
                "    assert False, 'forced'\n"
            ),
        )
        critique = CritiqueProposal(
            agent_role=self.role,
            intent="forced failure",
            issues=(),
            suggested_repair_directions=(),
            confidence=1.0,
        )
        return (critique,), (attack,)


def _engine(spec, *, builder, critic, optimizer, config=None):
    config = config or ForgeMindConfig()
    env = SubprocessExecutionEnvironment(config.resources)
    referee = Referee(spec, env, config)
    return SearchEngine(
        spec, config,
        architect=MockArchitect(), builder=builder, critic=critic,
        optimizer=optimizer, referee=referee,
        recovery=RecoveryPolicy(), history=History(),
    )


# ----------------------------------------------------------------------
# Fail-closed entry-point resolution
# ----------------------------------------------------------------------

def test_resolve_entry_point_fail_closed():
    base = StatsProblem().spec
    empty = ProblemSpec(
        problem_id="empty", description="no constraints",
        requirements=base.requirements, acceptance_tests=base.acceptance_tests,
        interface_constraints={},
    )
    with pytest.raises(ValueError, match="no entry point"):
        resolve_entry_point(empty)

    ambiguous = ProblemSpec(
        problem_id="amb", description="two constraints",
        requirements=base.requirements, acceptance_tests=base.acceptance_tests,
        interface_constraints={"aaa": "function: aaa(data)",
                               "bbb": "function: bbb(data)"},
    )
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_entry_point(ambiguous)

    # A single key with an UNKNOWN name is valid — any name must work.
    odd = ProblemSpec(
        problem_id="odd", description="arbitrary name",
        requirements=base.requirements, acceptance_tests=base.acceptance_tests,
        interface_constraints={"frobnicate": "function: frobnicate(data)"},
    )
    assert resolve_entry_point(odd) == "frobnicate"


def test_architect_uses_spec_entry_point_not_hardcoded():
    base = StatsProblem().spec
    spec = ProblemSpec(
        problem_id="odd", description="arbitrary name",
        requirements=base.requirements, acceptance_tests=base.acceptance_tests,
        interface_constraints={"frobnicate": "function: frobnicate(data)"},
    )
    proj = build_architect_projection(spec, _State())
    (proposal,) = MockArchitect().propose(proj, {})
    assert set(proposal.interfaces) == {"frobnicate"}
    blob = " ".join([*proposal.components, *proposal.implementation_steps,
                     *proposal.interfaces.values()])
    assert "compute_stats" not in blob


def test_critic_uses_spec_entry_point_and_kit_module():
    spec = rename_spec()
    code = MockBuilder(module_path="numbers.py")._correct("describe_numbers")
    proj = build_critic_projection(spec, _State({"numbers.py": code}), None)
    (_, attacks) = MockCritic(module="numbers").propose(proj, {})
    assert attacks, "good code must still draw the duplicates/negatives probe"
    for attack in attacks:
        assert "from numbers import describe_numbers" in attack.test_code
        assert "from solution import" not in attack.test_code


def test_builder_create_vs_modify_unit():
    builder = MockBuilder()
    spec = StatsProblem().spec
    # Root: no files yet -> create.
    (created,) = builder.propose(
        build_builder_projection(spec, _State()), {})[:1]
    assert created.created_files == {"solution.py": created.created_files["solution.py"]}
    assert created.modified_files == {}
    # Below root: module exists -> modify, never create.
    existing = {"solution.py": "old content"}
    (modified,) = builder.propose(
        build_builder_projection(spec, _State(existing)), {})[:1]
    assert modified.created_files == {}
    assert set(modified.modified_files) == {"solution.py"}
    (repaired,) = builder.repair(
        build_builder_projection(spec, _State(existing)), "evidence")
    assert repaired.created_files == {}
    assert set(repaired.modified_files) == {"solution.py"}


def test_registry_kits():
    assert mock_kit_for("stats").builder().correct_first is True
    assert mock_kit_for("stats-trap").builder().correct_first is False
    with pytest.raises(KeyError):
        mock_kit_for("nope")


# ----------------------------------------------------------------------
# Rename discriminator (engine run)
# ----------------------------------------------------------------------

def test_rename_discriminator_end_to_end():
    spec = rename_spec()
    kit = rename_kit()
    builder = RecordingBuilder(module_path="numbers.py")
    engine = _engine(spec, builder=builder, critic=kit.critic(),
                     optimizer=kit.optimizer())
    result = engine.run()

    assert result.success, result.summary()
    node = result.accepted_node
    assert node is not None
    assert set(node.state.files) == {"numbers.py"}
    assert "def describe_numbers" in node.state.files["numbers.py"]

    # The actual candidate move created numbers.py with describe_numbers.
    created = [
        p for call in builder.calls for p in call["proposals"]
        if "numbers.py" in p.created_files
        and "def describe_numbers" in p.created_files["numbers.py"]
    ]
    assert created, "no recorded proposal created numbers.py"

    # Generated verification artifacts use the rename identity only.
    assert engine._attack_tests, "critic must have produced attacks"
    for attack in engine._attack_tests:
        assert "from numbers import describe_numbers" in attack.test_code
        assert "from solution import" not in attack.test_code
    assert "from solution import" not in node.state.files["numbers.py"]


# ----------------------------------------------------------------------
# Depth-2: mock builder genuinely modifies below the root
# ----------------------------------------------------------------------

def test_depth2_mock_builder_modifies_existing_file():
    spec = StatsProblem().spec
    builder = ParentMarkerBuilder()  # default module_path solution.py
    engine = _engine(spec, builder=builder, critic=AlwaysFailCritic(),
                     optimizer=MockOptimizer())
    result = engine.run()

    nodes = engine.graph.nodes
    depth2 = [n for n in nodes.values() if n.depth == 2]
    assert depth2, "no depth-2 node: mock search never went below the root"
    depth1 = {n.node_id: n for n in nodes.values() if n.depth == 1}
    assert depth1, "no depth-1 node admitted to the frontier"

    # Genuine-transition assertions (Revision 2): parent link, changed
    # content, same file set — not just node existence.
    chain = None
    for s2 in depth2:
        s1 = depth1.get(s2.parent_node_id)
        if s1 is None:
            continue
        if (s2.state.files != s1.state.files
                and set(s2.state.files) == set(s1.state.files) == {"solution.py"}):
            chain = (s1, s2)
            break
    assert chain is not None, "no depth-2 node is a genuine modify of its parent"
    s1, s2 = chain

    # The move that created S2 modified the existing file (recorded
    # proposals), and S2's files match one such modify exactly.
    modify_calls = [
        call for call in builder.calls
        if "solution.py" in call["target_files"]
    ]
    assert modify_calls, "builder never proposed against an existing solution.py"
    for call in modify_calls:
        for p in call["proposals"]:
            assert p.created_files == {}, "builder created an existing file"
            assert set(p.modified_files) == {"solution.py"}
    assert any(
        dict(s2.state.files) == dict(p.modified_files)
        for call in modify_calls for p in call["proposals"]
    ), "S2's files do not match any recorded modify proposal"
