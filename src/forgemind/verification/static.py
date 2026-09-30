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
        violations: list[str] = []
        syntax_ok = True
        for path, content in sorted(files.items()):
            if not path.endswith(".py"):
                continue
            try:
                tree = ast.parse(content)
            except SyntaxError as exc:
                syntax_ok = False
                violations.append(f"{path}: syntax error line {exc.lineno}: {exc.msg}")
                continue
            if not enforce_safety:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn = node.func
                    name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                    if name in _FORBIDDEN_CALLS:
                        violations.append(
                            f"{path}: forbidden call '{name}()' line {node.lineno}")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in _FORBIDDEN_IMPORTS:
                            violations.append(
                                f"{path}: forbidden import '{alias.name}' line {node.lineno}")
                if isinstance(node, ast.ImportFrom):
                    root = (node.module or "").split(".")[0]
                    if root in _FORBIDDEN_IMPORTS:
                        violations.append(
                            f"{path}: forbidden import from '{node.module}' line {node.lineno}")
        return StaticGateResult(
            syntax_valid=syntax_ok,
            forbidden_constructs=tuple(violations),
            violations=tuple(violations),
        )
