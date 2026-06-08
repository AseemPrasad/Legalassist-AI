"""Priority in deadline responses is always computed from days_until_due."""

from __future__ import annotations

import ast
from pathlib import Path


def _get_source():
    return (Path(__file__).resolve().parents[1] / "api" / "routes" / "deadlines.py").read_text(encoding="utf-8")


def test_deadline_priority_function_exists():
    """_deadline_priority helper computes urgency from days_until_due."""
    src = _get_source()
    assert "_deadline_priority(days_until_due)" in src or "def _deadline_priority" in src


def test_priority_computed_in_all_endpoints():
    """Every endpoint response uses _deadline_priority(...) for its priority field."""
    src = _get_source()
    tree = ast.parse(src)

    # Find all DeadlineResponse instantiations
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "DeadlineResponse":
            found_computed = False
            for kw in node.keywords:
                if kw.arg == "priority" and isinstance(kw.value, ast.Call):
                    func = kw.value.func
                    if isinstance(func, ast.Name) and func.id == "_deadline_priority":
                        found_computed = True
            assert found_computed, f"DeadlineResponse at line {node.lineno} does not use _deadline_priority"


def test_create_has_no_priority_param():
    """create_deadline does not accept a priority parameter."""
    src = _get_source()
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "create_deadline":
            for arg in node.args.args:
                assert arg.arg != "priority", f"create_deadline should not accept priority (found at line {node.lineno})"
            return
    raise AssertionError("create_deadline function not found")


def test_update_has_no_priority_param():
    """update_deadline does not accept a priority parameter."""
    src = _get_source()
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "update_deadline":
            for arg in node.args.args:
                assert arg.arg != "priority", f"update_deadline should not accept priority (found at line {node.lineno})"
            return
    raise AssertionError("update_deadline function not found")


def test_deadline_priority_thresholds():
    """_deadline_priority returns correct values per threshold."""
    from api.routes.deadlines import _deadline_priority

    assert _deadline_priority(0) == "critical"
    assert _deadline_priority(1) == "critical"
    assert _deadline_priority(3) == "critical"
    assert _deadline_priority(4) == "high"
    assert _deadline_priority(10) == "high"
    assert _deadline_priority(11) == "medium"
    assert _deadline_priority(30) == "medium"
    assert _deadline_priority(31) == "low"
    assert _deadline_priority(365) == "low"
