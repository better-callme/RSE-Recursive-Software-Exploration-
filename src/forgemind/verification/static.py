"""Tier-1 static verifier: AST parsing and forbidden-construct checks.

Defense-in-depth only — this is NOT the security boundary.
"""

from __future__ import annotations

import ast

from forgemind.verification.results import StaticGateResult

_FORBIDDEN_NODES: dict[str, str] = {
    "Exec": "exec statement",
}

_FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "system", "popen"}
_FORBIDDEN_IMPORTS = {"ctypes", "subprocess", "socket", "resource"}


class StaticVerifier:
    def verify(self, files: dict[str, str], *, enforce_safety: bool = True) -> StaticGateResult:
        # M1: diagnostics (violations) and the safety verdict
        # (forbidden_constructs) are separate lists. A syntax error is
        # not a safety finding, and an unparseable file means safety
        # was NOT established — analysis_complete goes False so an
        # empty forbidden list can never masquerade as clean.
        violations: list[str] = []
        forbidden: list[str] = []
        syntax_ok = True
        analysis_complete = enforce_safety
        for path, content in sorted(files.items()):
            if not path.endswith(".py"):
                continue
            try:
                tree = ast.parse(content)
            except SyntaxError as exc:
                syntax_ok = False
                analysis_complete = False
                violations.append(f"{path}: syntax error line {exc.lineno}: {exc.msg}")
                continue
            if not enforce_safety:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn = node.func
                    name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                    if name in _FORBIDDEN_CALLS:
                        msg = f"{path}: forbidden call '{name}()' line {node.lineno}"
                        violations.append(msg)
                        forbidden.append(msg)
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in _FORBIDDEN_IMPORTS:
                            msg = (f"{path}: forbidden import '{alias.name}' "
                                   f"line {node.lineno}")
                            violations.append(msg)
                            forbidden.append(msg)
                if isinstance(node, ast.ImportFrom):
                    root = (node.module or "").split(".")[0]
                    if root in _FORBIDDEN_IMPORTS:
                        msg = (f"{path}: forbidden import from '{node.module}' "
                               f"line {node.lineno}")
                        violations.append(msg)
                        forbidden.append(msg)
        return StaticGateResult(
            syntax_valid=syntax_ok,
            forbidden_constructs=tuple(forbidden),
            violations=tuple(violations),
            analysis_complete=analysis_complete,
        )
