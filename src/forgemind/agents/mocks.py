"""Deterministic mock agents for v0.1 experiments.

These stand in for LLM-backed adapters. They are stateless, deterministic,
and communicate exclusively through projections -> proposals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from forgemind.agents.projections import (
    ArchitectProjection,
    BuilderProjection,
    CriticProjection,
    OptimizerProjection,
)
from forgemind.core.problem import resolve_entry_point
from forgemind.core.proposal import (
    ArchitectureProposal,
    AttackTestProposal,
    CodePatchProposal,
    CritiqueProposal,
    Issue,
)


class MockArchitect:
    """Fully spec-driven (H5): the entry function comes from the spec via
    resolve_entry_point, never from a hardcoded name."""

    role = "architect"

    def propose(
        self, projection: ArchitectProjection, request_context: Mapping[str, str]
    ) -> tuple[ArchitectureProposal, ...]:
        entry_fn = resolve_entry_point(projection.spec)
        return (
            ArchitectureProposal(
                agent_role=self.role,
                intent="single-module implementation of the required interface",
                components=(f"module:{entry_fn}",),
                interfaces={entry_fn: projection.interface_constraints[entry_fn]},
                dependency_graph={"module": ()},
                implementation_steps=(
                    f"implement {entry_fn} per its interface constraint",
                    "satisfy the acceptance tests",
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
        correct_first: bool = True,
    ) -> None:
        self.module_path = module_path
        self.correct_first = correct_first

    def _correct(self, fn: str) -> str:
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

    def _bad_empty(self, fn: str) -> str:
        return (
            f"def {fn}(data):\n"
            "    total = sum(data)\n"          # raises on empty
            "    n = len(data)\n"
            "    return {'count': n, 'sum': total, 'min': min(data),\n"
            "            'max': max(data), 'mean': total / n}\n"
        )

    def _bad_mean(self, fn: str) -> str:
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

    def _place(self, src: str, target_files: Mapping[str, str]
               ) -> tuple[dict[str, str], dict[str, str]]:
        """Create vs modify (H5): a file that already exists in the state
        must be modified, never created — otherwise the transition
        contract rejects the proposal as CREATE_EXISTING and no mock
        search can ever go below the root."""
        if self.module_path in target_files:
            return {}, {self.module_path: src}
        return {self.module_path: src}, {}

    def propose(
        self, projection: BuilderProjection, request_context: Mapping[str, str]
    ) -> tuple[CodePatchProposal, ...]:
        # H5: the function name comes from the spec at propose time, never
        # from construction — the same builder serves any problem name.
        fn = resolve_entry_point(projection.spec)
        # After failures recorded, prefer repairing toward the correct variant.
        variants = [self._correct(fn), self._bad_empty(fn), self._bad_mean(fn)]
        if not self.correct_first:
            variants.reverse()
        proposals = []
        for i, src in enumerate(variants):
            created, modified = self._place(src, projection.target_files)
            proposals.append(
                CodePatchProposal(
                    agent_role=self.role,
                    source_node_id=projection.spec.problem_id and request_context.get(
                        "source_node_id", ""),
                    intent=f"implementation variant v{i}",
                    rationale=("variant %d of deterministic builder" % i),
                    estimated_cost=1.0,
                    created_files=created,
                    modified_files=modified,
                )
            )
        return tuple(proposals)

    def repair(
        self, projection: BuilderProjection, failure_evidence: str
    ) -> tuple[CodePatchProposal, ...]:
        """Deterministic repair: always converge on the correct implementation."""
        fn = resolve_entry_point(projection.spec)
        created, modified = self._place(self._correct(fn), projection.target_files)
        return (
            CodePatchProposal(
                agent_role=self.role,
                intent="repair after failure evidence",
                rationale=f"repairing based on evidence: {failure_evidence[:120]}",
                estimated_cost=2.0,
                created_files=created,
                modified_files=modified,
            ),
        )


class MockCritic:
    """Actually analyzes the code; issues are converted to attack tests by the
    engine, and the Referee decides truth.

    H5: the import module is construction config (kit-provided) and the
    function name comes from the spec via resolve_entry_point — no
    hardcoded ``solution`` / ``compute_stats`` literals remain. The attack
    assertions themselves assume the stats dict contract; this is the
    stats-family critic, explicitly selected by a kit — a future problem
    family supplies its own critic through its own kit.
    """

    role = "critic"

    def __init__(self, *, module: str = "solution") -> None:
        self.module = module

    def propose(
        self, projection: CriticProjection, request_context: Mapping[str, str]
    ) -> tuple[tuple[CritiqueProposal, ...], tuple[AttackTestProposal, ...]]:
        issues: list[Issue] = []
        attacks: list[AttackTestProposal] = []

        code = "\n".join(projection.relevant_code.values())
        fn = resolve_entry_point(projection.spec)

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
                    f"    from {self.module} import {fn}\n"
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
                    f"    from {self.module} import {fn}\n"
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
                    f"    from {self.module} import {fn}\n"
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


@dataclass(frozen=True)
class MockKit:
    """A per-problem bundle of mock agents (H5).

    The stats-ness that used to be smeared through shared code — module
    path, import module, variant order — is now explicit configuration
    chosen once, here. The agents themselves stay generic: the function
    name always comes from the spec at propose time via
    ``resolve_entry_point``. A kit constructs agents; it is never a
    dependency of the engine — the registry that maps problem names to
    kits lives in the benchmark layer (``benchmark/kits.py``).
    """

    module_path: str = "solution.py"
    module: str = "solution"
    correct_first: bool = True

    def architect(self) -> MockArchitect:
        return MockArchitect()

    def builder(self) -> MockBuilder:
        return MockBuilder(
            module_path=self.module_path, correct_first=self.correct_first
        )

    def critic(self) -> MockCritic:
        return MockCritic(module=self.module)

    def optimizer(self) -> MockOptimizer:
        return MockOptimizer(module_path=self.module_path)
