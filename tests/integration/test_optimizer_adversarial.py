"""C3 regression: the optimizer path must not bypass the adversarial gate.

An optimized state that passes acceptance but fails the critic's attack
tests must be rejected, and the run must deliver the originally verified
artifact instead.
"""

from forgemind.agents.mocks import MockBuilder, MockCritic
from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig
from forgemind.core.proposal import CodePatchProposal, OptimizationProposal
from forgemind.recovery.failure import RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.search.node import NodeStatus
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee


class Float32MeanOptimizer:
    """'Optimizes' the mean by narrowing it to float32 precision.

    Passes every acceptance test — their expected means (2.5, 7, -5, 3)
    are exactly representable in float32 — but fails the critic's
    duplicates/negatives attack, which requires mean([-3, -3, 5, 5, 5])
    == 1.8 within 1e-9 (float32 gives 1.80000004768...).
    """

    role = "optimizer"

    def propose(self, projection, request_context):
        src = projection.verified_files["solution.py"]
        broken = src.replace("'mean': total / n}", "'mean': _f32(total / n)}")
        assert broken != src
        broken = (
            "import struct\n\n\n"
            "def _f32(x):\n"
            "    return struct.unpack('f', struct.pack('f', x))[0]\n\n\n"
        ) + broken
        patch = CodePatchProposal(
            agent_role=self.role,
            intent="float32 mean",
            modified_files={"solution.py": broken},
            rationale="narrower float arithmetic for speed",
        )
        return (OptimizationProposal(agent_role=self.role, patch=patch,
                                     target_metric="runtime"),)


def test_optimizer_cannot_bypass_adversarial_gate(spec):
    config = ForgeMindConfig()
    referee = Referee(spec, SubprocessExecutionEnvironment(config.resources),
                      config)
    referee.attach_test_provider(StatsProblem().test_files)
    engine = SearchEngine(
        spec, config,
        builder=MockBuilder(), critic=MockCritic(),
        optimizer=Float32MeanOptimizer(), referee=referee,
        recovery=RecoveryPolicy(),
    )
    result = engine.run()
    assert result.success
    accepted = result.accepted_node
    assert accepted is not None
    # The delivered artifact is the ORIGINAL verified code, not the
    # adversarially broken "optimization".
    assert "_f32" not in accepted.state.files["solution.py"]
    # The optimized node really was proposed, verified, and rejected.
    optimized = [n for n in result.graph.nodes.values()
                 if "_f32" in n.state.files.get("solution.py", "")]
    assert optimized
    assert all(n.status is NodeStatus.REJECTED for n in optimized)
    assert all(not n.verification.adversarial_passed for n in optimized)
