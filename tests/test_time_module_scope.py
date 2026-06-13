"""Verify time module is imported at module scope, not inside request handlers."""

from __future__ import annotations

import ast
from pathlib import Path


def test_time_is_module_level_import():
    src = Path(__file__).resolve().parents[1] / "api" / "routes" / "notifications.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    function_scopes = []
    module_level_imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for child in ast.walk(node):
                if isinstance(child, ast.Import) and any(a.name == "time" for a in child.names):
                    function_scopes.append(node.name)
        if isinstance(node, ast.Import) and any(a.name == "time" for a in node.names):
            module_level_imports.append(node.lineno)

    assert function_scopes == [], f"time imported inside functions: {function_scopes}"
    assert len(module_level_imports) == 1, f"expected 1 module-level import, found {len(module_level_imports)}"
