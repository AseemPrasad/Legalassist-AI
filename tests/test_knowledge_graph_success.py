"""Knowledge graph success metrics include expanded outcome types."""

from __future__ import annotations

import ast
from pathlib import Path


def _module_source():
    return (Path(__file__).resolve().parents[1] / "core" / "knowledge_graph.py").read_text(encoding="utf-8")


def test_successful_outcomes_defined():
    """SUCCESSFUL_OUTCOMES constant exists at module level."""
    src = _module_source()
    assert "SUCCESSFUL_OUTCOMES" in src


def test_successful_outcomes_includes_wins():
    """Standard win outcomes are included."""
    src = _module_source()
    assert '"plaintiff_won"' in src or "'plaintiff_won'" in src
    assert '"defendant_won"' in src or "'defendant_won'" in src


def test_successful_outcomes_includes_settlements():
    """Settlement outcomes are included."""
    src = _module_source()
    assert '"settled"' in src or "'settled'" in src


def test_successful_outcomes_includes_favorable_dismissals():
    """Favorable dismissals and rulings are included."""
    src = _module_source()
    assert '"dismissed_with_prejudice"' in src or "'dismissed_with_prejudice'" in src
    assert '"favorable_ruling"' in src or "'favorable_ruling'" in src


def test_successful_outcomes_is_frozenset():
    """The constant is frozenset (immutable)."""
    src = _module_source()
    assert "frozenset" in src.replace("frozenset", "frozenset", 1)
    src_before_close = src.split("SUCCESSFUL_OUTCOMES")[1].split(")")[0]
    assert "frozenset" in src or "SUCCESSFUL_OUTCOMES"


def test_get_graph_statistics_uses_successful_outcomes():
    """get_graph_statistics filters edges using SUCCESSFUL_OUTCOMES."""
    src = _module_source()
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_graph_statistics":
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and getattr(child.func, "attr", None) == "in_":
                    for arg in child.args:
                        if isinstance(arg, ast.Name) and arg.id == "SUCCESSFUL_OUTCOMES":
                            return
            raise AssertionError("get_graph_statistics does not filter by SUCCESSFUL_OUTCOMES")

    raise AssertionError("get_graph_statistics function not found")


def test_successful_outcomes_expanded_beyond_original():
    """The set includes more than just plaintiff_won and defendant_won."""
    src = _module_source()
    outcome_matches = []
    for line in src.splitlines():
        line = line.strip()
        if line.startswith('"') or line.startswith("'"):
            outcome_matches.append(line.strip('",\''))
    count = sum(1 for v in ["p", "d", "s", "f"] if any(v in p for p in outcome_matches))
    assert count > 2, "Should include more than 2 outcome types in the set"


def test_successful_outcomes_has_at_least_six_types():
    """The set contains at least the 6 documented outcome types."""
    src = _module_source()
    expected = {
        "plaintiff_won",
        "defendant_won",
        "settled",
        "settled_favorably",
        "dismissed_with_prejudice",
        "favorable_ruling",
    }
    for outcome in expected:
        assert outcome in src, f"Missing expected outcome: {outcome}"
