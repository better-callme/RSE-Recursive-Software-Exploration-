"""TransitionEngine: the ONLY path from state + proposal to a new state."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Mapping

from forgemind.core.identity import normalize_path, state_content_hash
from forgemind.core.problem import ProblemSpec
from forgemind.core.proposal import CodePatchProposal, OptimizationProposal, Proposal
from forgemind.core.state import CodebaseState


@dataclass(frozen=True)
class TransitionResult:
    ok: bool
    new_state: CodebaseState | None
    reason: str  # "" when ok
    violation: str  # e.g. "PROBLEM_SPEC_MUTATION", "SYNTAX", ""


_FORBIDDEN_SPEC_PATHS = ("problem_spec.json", ".forgemind/spec.json")


def _check_python_syntax(path: str, content: str) -> str | None:
    """Return an error message or None."""
    if not path.endswith(".py"):
        return None
    try:
        ast.parse(content)
    except SyntaxError as exc:
        return f"{path}: syntax error line {exc.lineno}: {exc.msg}"
    return None


_INTERFACE_RE = re.compile(r"^def\s+(\w+)\s*\(")


def _public_symbols(path: str, content: str) -> set[str]:
    """Top-level public functions AND classes (contract check)."""
    if not path.endswith(".py"):
        return set()
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                names.add(node.name)
    return names


class TransitionEngine:
    """Deterministic T(S,P): validate, apply, hash. Never invokes an agent."""

    def __init__(self, spec: ProblemSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> ProblemSpec:
        return self._spec

    def apply(self, state: CodebaseState, proposal: Proposal) -> TransitionResult:
        patch: CodePatchProposal | None
        if isinstance(proposal, OptimizationProposal):
            patch = proposal.patch
        elif isinstance(proposal, CodePatchProposal):
            patch = proposal
        else:
            return TransitionResult(
                False, None,
                f"proposal type {type(proposal).__name__} cannot transition state",
                "INVALID_PROPOSAL_TYPE",
            )

        # --- validation -----------------------------------------------------
        if state.spec_hash != self._spec.spec_hash:
            return TransitionResult(False, None, "state belongs to another problem",
                                    "SPEC_MISMATCH")

        created: dict[str, str] = {}
        modified: dict[str, str] = {}
        deleted: set[str] = set()

        for raw_path in patch.deleted_files:
            path = normalize_path(raw_path)
            if any(path == p or path.endswith("/" + p) or p in path
                   for p in _FORBIDDEN_SPEC_PATHS):
                return TransitionResult(
                    False, None, f"cannot delete protected file {path}",
                    "PROBLEM_SPEC_MUTATION")
            if path not in state.files and path not in created:
                return TransitionResult(False, None,
                                        f"deleted file does not exist: {path}",
                                        "DELETE_NONEXISTENT")
            deleted.add(path)

        for mapping, target in ((patch.created_files, created),
                                (patch.modified_files, modified)):
            for raw_path, content in mapping.items():
                path = normalize_path(raw_path)
                if not path:
                    return TransitionResult(False, None, "empty file path",
                                            "INVALID_PATH")
                if any(path == p or path.endswith("/" + p) or p in path
                       for p in _FORBIDDEN_SPEC_PATHS):
                    return TransitionResult(
                        False, None,
                        f"attempt to modify problem specification via {path}",
                        "PROBLEM_SPEC_MUTATION")
                target[path] = content

        overlap = created.keys() & state.files.keys()
        if overlap:
            return TransitionResult(False, None,
                                    f"creating existing files: {sorted(overlap)}",
                                    "CREATE_EXISTING")
        overlap = created.keys() & modified.keys()
        if overlap:
            return TransitionResult(False, None,
                                    f"files both created and modified: {sorted(overlap)}",
                                    "AMBIGUOUS_PATCH")
        overlap = (created.keys() | modified.keys()) & deleted
        if overlap:
            return TransitionResult(False, None,
                                    f"files written and deleted: {sorted(overlap)}",
                                    "AMBIGUOUS_PATCH")

        # Interface constraints: required public symbols must exist somewhere.
        required_funcs = {
            name for name, kind in self._spec.interface_constraints.items()
            if kind.startswith("function:")
        }
        final_files = self._compose_files(state, created, modified, deleted)

        # Dependency policy: only allowed deps may be added.
        for dep, version in patch.dependency_changes.items():
            if self._spec.dependency_constraints and dep not in self._spec.dependency_constraints:
                return TransitionResult(False, None,
                                        f"dependency not allowed by policy: {dep}",
                                        "DEPENDENCY_POLICY")

        # Static syntax gate on every resulting python file (Tier-1 pre-check).
        for path, content in final_files.items():
            err = _check_python_syntax(path, content)
            if err:
                return TransitionResult(False, None, err, "SYNTAX")

        present: set[str] = set()
        for path, content in final_files.items():
            present |= _public_symbols(path, content)
        missing = required_funcs - present
        if missing:
            return TransitionResult(
                False, None,
                f"interface constraint violated; missing functions: {sorted(missing)}",
                "INTERFACE_CONSTRAINT")

        # --- construct immutable successor -----------------------------------
        new_state = CodebaseState(
            spec_hash=state.spec_hash,
            files=final_files,
            dependencies=self._compose_deps(state, patch.dependency_changes),
            runtime_spec=state.runtime_spec,
            configuration=state.configuration,
        )
        return TransitionResult(True, new_state, "", "")

    @staticmethod
    def _compose_files(
        state: CodebaseState,
        created: Mapping[str, str],
        modified: Mapping[str, str],
        deleted: set[str],
    ) -> dict[str, str]:
        files = dict(state.files)
        for path in deleted:
            files.pop(path, None)
        files.update(created)
        files.update(modified)
        return files

    @staticmethod
    def _compose_deps(state: CodebaseState, changes: Mapping[str, str]) -> dict[str, str]:
        deps = dict(state.dependencies)
        deps.update(changes)
        return deps


def make_root_state(spec: ProblemSpec) -> CodebaseState:
    """Empty initial state bound to the problem."""
    return CodebaseState(spec_hash=spec.spec_hash, files={})
