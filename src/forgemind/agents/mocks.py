"""Deterministic mock agents for v0.1 experiments.

These stand in for LLM-backed adapters. They are stateless, deterministic,
and communicate exclusively through projections -> proposals.
"""

from __future__ import annotations

from typing import Mapping

from forgemind.agents.projections import (
    ArchitectProjection,
    BuilderProjection,
    CriticProjection,
    OptimizerProjection,
)
from forgemind.core.proposal import (
    ArchitectureProposal,
    AttackTestProposal,
    CodePatchProposal,
    CritiqueProposal,
    Issue,
)


class MockArchitect:
    role = "architect"

    def propose(
        self, projection: ArchitectProjection, request_context: Mapping[str, str]
    ) -> tuple[ArchitectureProposal, ...]:
        entry_fn = "compute_stats"
        return (
            ArchitectureProposal(
                agent_role=self.role,
                intent="single-module implementation of the required interface",
                components=(f"stats_module:{entry_fn}",),
                interfaces={entry_fn: projection.interface_constraints.get(entry_fn,
                               "function: compute_stats(data) -> dict")},
                dependency_graph={"stats_module": ()},
                implementation_steps=(
                    f"create stats.py exposing {entry_fn}(data)",
                    "handle empty input explicitly",
                    "compute count/sum/min/max/mean",
                ),
                rationale="deterministic single-file design keeps search space small",
            ),
        )


class MockBuilder:
    """Emits MULTIPLE genuinely different candidates per call.

    Candidate A: correct implementation.
    Candidate B: wrong empty-list handling (raises on empty).
    Candidate C: wrong mean (integer division truncation).
    """

    role = "builder"

    def __init__(
        self,
        *,
        module_path: str = "solution.py",
        function_name: str = "compute_stats",
        correct_first: bool = True,
    ) -> None:
        self.module_path = module_path
        self.function_name = function_name
        self.correct_first = correct_first

    def _correct(self) -> str:
        fn = self.function_name
        return (
            f"def {fn}(data):\n"
            "    if not data:\n"
            "        return {'count': 0, 'sum': 0, 'min': None, 'max': None,\n"
            "                'mean': None}\n"
            "    total = sum(data)\n"
            "    n = len(data)\n"
            "    return {'count': n, 'sum': total, 'min': min(data),\n"
            "            'max': max(data), 'mean': total / n}\n"
        )

    def _bad_empty(self) -> str:
        fn = self.function_name
        return (
            f"def {fn}(data):\n"
            "    total = sum(data)\n"          # raises on empty
            "    n = len(data)\n"
            "    return {'count': n, 'sum': total, 'min': min(data),\n"
            "            'max': max(data), 'mean': total / n}\n"
        )

    def _bad_mean(self) -> str:
        fn = self.function_name
        return (
            f"def {fn}(data):\n"
            "    if not data:\n"
            "        return {'count': 0, 'sum': 0, 'min': None, 'max': None,\n"
            "                'mean': None}\n"
            "    total = sum(data)\n"
            "    n = len(data)\n"
            "    return {'count': n, 'sum': total, 'min': min(data),\n"
            f"            'max': max(data), 'mean': total // n}}\n"  # truncates
        )

    def _syntax_broken(self) -> str:
        fn = self.function_name
        return f"def {fn}(data:\n    return {{}}\n"

    def propose(
        self, projection: BuilderProjection, request_context: Mapping[str, str]
    ) -> tuple[CodePatchProposal, ...]:
        # After failures recorded, prefer repairing toward the correct variant.
        variants = [self._correct(), self._bad_empty(), self._bad_mean()]
        if not self.correct_first:
            variants.reverse()
        proposals = []
        for i, src in enumerate(variants):
            proposals.append(
                CodePatchProposal(
                    agent_role=self.role,
                    source_node_id=projection.spec.problem_id and request_context.get(
                        "source_node_id", ""),
                    intent=f"implementation variant v{i}",
                    rationale=("variant %d of deterministic builder" % i),
                    estimated_cost=1.0,
                    created_files={self.module_path: src},
                )
            )
        return tuple(proposals)

    def repair(
        self, projection: BuilderProjection, failure_evidence: str
    ) -> tuple[CodePatchProposal, ...]:
        """Deterministic repair: always converge on the correct implementation."""
        return (
            CodePatchProposal(
                agent_role=self.role,
                intent="repair after failure evidence",
                rationale=f"repairing based on evidence: {failure_evidence[:120]}",
                estimated_cost=2.0,
                created_files={self.module_path: self._correct()},
            ),
        )


class MockCritic:
    """Actually analyzes the code; issues are converted to attack tests by the
    engine, and the Referee decides truth."""

    role = "critic"

    def propose(
        self, projection: CriticProjection, request_context: Mapping[str, str]
    ) -> tuple[tuple[CritiqueProposal, ...], tuple[AttackTestProposal, ...]]:
        issues: list[Issue] = []
        attacks: list[AttackTestProposal] = []

        code = "\n".join(projection.relevant_code.values())
        fn = next((name for name in projection.contract_info), "compute_stats")

        if "//" in code and "/" in code:
            issues.append(Issue(
                location=fn,
                severity="high",
                description="possible integer division truncating the mean",
                evidence="'//' operator used where float division expected",
            ))
            attacks.append(AttackTestProposal(
                agent_role=self.role,
                test_name="attack_mean_precision",
                hypothesis="mean is truncated by integer division",
                test_code=(
                    "def attack_mean_precision():\n"
                    f"    from solution import {fn}\n"
                    f"    r = {fn}([1, 2])\n"
                    "    assert r['mean'] == 1.5, r['mean']\n"
                ),
            ))

        if "if not data" not in code and "len(data) == 0" not in code:
            issues.append(Issue(
                location=fn,
                severity="critical",
                description="empty-input handling appears missing",
                evidence="no explicit empty-list branch detected",
            ))
            attacks.append(AttackTestProposal(
                agent_role=self.role,
                test_name="attack_empty_input",
                hypothesis="empty input raises instead of returning stats",
                test_code=(
                    "def attack_empty_input():\n"
                    f"    from solution import {fn}\n"
                    f"    r = {fn}([])\n"
                    "    assert r['count'] == 0\n"
                ),
            ))
        else:
            # Even good code gets an adversarial probe (duplicates/negatives).
            attacks.append(AttackTestProposal(
                agent_role=self.role,
                test_name="attack_duplicates_and_negatives",
                hypothesis="duplicates or negatives break min/max/mean",
                test_code=(
                    "def attack_duplicates_and_negatives():\n"
                    f"    from solution import {fn}\n"
                    f"    r = {fn}([-3, -3, 5, 5, 5])\n"
                    "    assert r['min'] == -3 and r['max'] == 5\n"
                    "    assert abs(r['mean'] - 1.8) < 1e-9\n"
                    "    assert r['count'] == 5\n"
                ),
            ))

        critique = CritiqueProposal(
            agent_role=self.role,
            intent="adversarial review",
            issues=tuple(issues),
            suggested_repair_directions=tuple(
                i.description for i in issues),
            confidence=0.8,
        )
        return (critique,), tuple(attacks)


class MockOptimizer:
    """Proposes a micro-optimization on verified states; must re-verify."""

    role = "optimizer"

    def __init__(self, *, module_path: str = "solution.py") -> None:
        self.module_path = module_path

    def propose(
        self, projection: OptimizerProjection, request_context: Mapping[str, str]
    ) -> tuple:
        # Deterministic no-op-ish optimization: cache len() into a local var.
        optimized_sources: list[str] = []
        for path, content in sorted(projection.verified_files.items()):
            if path.endswith(".py") and "len(data)" in content:
                new = content.replace("n = len(data)",
                                      "n = len(data)  # cached length")
                if new != content:
                    optimized_sources.append(new)
        if not optimized_sources:
            return ()
        from forgemind.core.proposal import OptimizationProposal

        patch = CodePatchProposal(
            agent_role=self.role,
            intent="micro-optimization",
            modified_files={self.module_path: optimized_sources[0]},
            rationale="cache repeated builtin calls",
        )
        return (OptimizationProposal(agent_role=self.role, patch=patch,
                                     target_metric="runtime"),)
