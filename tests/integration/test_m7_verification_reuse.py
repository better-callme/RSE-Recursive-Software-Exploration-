"""M7: the adversarial gate reuses a stored verdict only when the attack
set is identical to the set it was verified against.

Staging modeled on M6's helpers: X variants fail the float32-precision
attack and are admitted; GOOD passes it and is accepted.
"""

from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig
from forgemind.core.history import EventType
from forgemind.core.proposal import (
    AttackTestProposal,
    CodePatchProposal,
    CritiqueProposal,
)
from forgemind.core.state import CodebaseState
from forgemind.recovery.failure import FailureType, RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.search.node import SearchNode
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee

_HEADER = (
    "import struct\n\n\n"
    "def _f32(x):\n"
    "    return struct.unpack('f', struct.pack('f', x))[0]\n\n\n"
)

GOOD = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t / len(data)}\n"
)

_X_BODY = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': _f32(t / len(data))}\n"
)


def _x_code(marker):
    return f"# variant {marker}\n" + _HEADER + _X_BODY


def _proposals(srcs, target_files):
    if "solution.py" in target_files:
        return tuple(
            CodePatchProposal(
                agent_role="builder", intent="m7 variant",
                rationale="m7", estimated_cost=1.0,
                modified_files={"solution.py": s})
            for s in srcs
        )
    return tuple(
        CodePatchProposal(
            agent_role="builder", intent="m7 variant",
            rationale="m7", estimated_cost=1.0,
            created_files={"solution.py": s})
        for s in srcs
    )


class PrecisionCritic:
    """One attack: float32 mean precision. X fails it, GOOD passes."""

    role = "critic"

    def propose(self, projection, request_context):
        attack = AttackTestProposal(
            agent_role=self.role,
            test_name="attack_mean_precision",
            hypothesis="float32 mean breaks precision",
            test_code=(
                "def attack_mean_precision():\n"
                "    from solution import compute_stats\n"
                "    r = compute_stats([-3, -3, 5, 5, 5])\n"
                "    assert abs(r['mean'] - 1.8) < 1e-9\n"
            ),
        )
        critique = CritiqueProposal(
            agent_role=self.role, intent="adversarial review", issues=(),
            suggested_repair_directions=(), confidence=0.8)
        return (critique,), (attack,)


class EarlyAcceptBuilder:
    """Root -> [A, B] (admitted); any marked parent -> GOOD (accepted)."""

    role = "builder"

    def propose(self, projection, request_context):
        current = projection.target_files.get("solution.py", "")
        if not current:
            return _proposals([_x_code("A"), _x_code("B")],
                              projection.target_files)
        return _proposals([GOOD], projection.target_files)


def _engine(spec, env, builder, critic, **cfg_kwargs):
    cfg = ForgeMindConfig(random_seed=7, **cfg_kwargs)
    referee = Referee(spec, env, cfg)
    referee.attach_test_provider(StatsProblem().test_files)
    return SearchEngine(spec, cfg, builder=builder, critic=critic,
                        referee=referee, recovery=RecoveryPolicy())


def test_gate_reuses_stored_verdict_when_attacks_unchanged(spec):
    """M7 flow: A verifies without attacks -> gate runs critic -> set
    mismatch -> genuine re-verify -> admitted. B verifies WITH attacks
    -> gate reuses. GOOD verifies with attacks -> gate reuses ->
    accepted. Total 4 verifications; pre-fix every gate re-verifies
    (6) -> fails."""
    eng = _engine(spec, SubprocessExecutionEnvironment(),
                  EarlyAcceptBuilder(), PrecisionCritic())
    result = eng.run()
    assert result.success
    # Note: the gate's re-verify emits no VERIFICATION_STARTED event
    # (pre-existing behavior — it calls referee.verify directly), so
    # the verifications metric, not the event count, discriminates.
    assert eng.metrics.verifications == 4
    adv_records = [r for r in eng._all_records
                   if r.failure_type is FailureType.ADVERSARIAL_FAILURE]
    assert len(adv_records) == 2  # A (re-verify path) and B (reuse path)


def _direct_engine(spec):
    """Engine + X node, no run: for direct _verify/_adversarial_gate
    sequences that pin the set comparison."""
    eng = _engine(spec, SubprocessExecutionEnvironment(),
                  EarlyAcceptBuilder(), PrecisionCritic())
    state = CodebaseState(spec_hash=spec.spec_hash,
                          files={"solution.py": _x_code("T")})
    node = SearchNode(
        node_id=eng.graph.next_node_id(), state=state,
        parent_node_id="node-000000", candidate_id=None,
        depth=1, creation_index=0,
    )
    return eng, node


def _seed_attacks(eng, node):
    """Give the engine the critic's attack, then verify the node
    against it. Returns the node with its verification attached."""
    verification = eng._verify(node)
    eng._run_critic(node, verification)
    assert eng._attack_tests, "critic must supply an attack"
    verification = eng._verify(node)
    return node.with_updates(verification=verification)


def _passing_attack(name):
    return AttackTestProposal(
        agent_role="critic", test_name=name, hypothesis="passes",
        test_code=f"def {name}():\n    assert True\n",
    )


def test_gate_reuse_unchanged_set(spec):
    eng, node = _direct_engine(spec)
    node = _seed_attacks(eng, node)
    before = eng.metrics.verifications
    assert eng._adversarial_gate(node) is False  # X fails the attack
    assert eng.metrics.verifications == before  # reused, no re-verify


def test_gate_reverifies_when_attack_added(spec):
    eng, node = _direct_engine(spec)
    node = _seed_attacks(eng, node)
    eng._attack_tests.append(_passing_attack("attack_extra"))
    before = eng.metrics.verifications
    assert eng._adversarial_gate(node) is False  # precision attack fails
    assert eng.metrics.verifications == before + 1


def test_gate_reverifies_when_attack_removed(spec):
    eng, node = _direct_engine(spec)
    node = _seed_attacks(eng, node)
    eng._attack_tests.append(_passing_attack("attack_extra"))
    node = node.with_updates(verification=eng._verify(node))
    del eng._attack_tests[-1]  # attacks still non-empty: no critic run
    before = eng.metrics.verifications
    assert eng._adversarial_gate(node) is False
    assert eng.metrics.verifications == before + 1


def test_gate_reverifies_when_attack_code_modified(spec):
    eng, node = _direct_engine(spec)
    node = _seed_attacks(eng, node)
    # Same name, passing code: reuse would return the OLD verdict
    # (False); re-verification returns the NEW verdict (True).
    eng._attack_tests[0] = _passing_attack("attack_mean_precision")
    before = eng.metrics.verifications
    assert eng._adversarial_gate(node) is True
    assert eng.metrics.verifications == before + 1
