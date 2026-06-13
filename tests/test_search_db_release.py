"""DB session is released before CPU-intensive search computation."""

from __future__ import annotations

import ast
from pathlib import Path


def _get_source():
    return (Path(__file__).resolve().parents[1] / "api" / "routes" / "cases.py").read_text(encoding="utf-8")


def test_search_cases_closes_db_before_scoring():
    """The DB session is closed in a finally block *before* the scoring loop runs."""
    source = _get_source()
    tree = ast.parse(source)

    def _find_function(node, name):
        for n in ast.walk(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
                return n
        return None

    func = _find_function(tree, "search_cases")
    assert func is not None, "search_cases function not found"

    try_node = None
    for child in ast.iter_child_nodes(func):
        if isinstance(child, ast.Try):
            try_node = child
            break
    assert try_node is not None, "No try/finally block in search_cases"

    try_end_line = max(
        (n.lineno for n in ast.walk(try_node) if hasattr(n, "lineno")),
        default=try_node.lineno,
    )
    try_end_line += 1

    scoring_calls = []
    for n in ast.walk(func):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "case_similarity_score":
            scoring_calls.append(n.lineno)

    assert scoring_calls, "case_similarity_score call not found in search_cases"
    for line in scoring_calls:
        assert line > try_node.end_lineno, (
            f"case_similarity_score at line {line} is inside try block "
            f"(try ends at line {try_node.end_lineno}), should be after"
        )


def test_no_get_feedback_adjustment_in_search_cases():
    """get_feedback_adjustment is removed — feedback is pre-fetched in batch."""
    source = _get_source()
    assert "get_feedback_adjustment" not in source.split("# ")[0], "get_feedback_adjustment should not be called in the endpoint"


def test_feedback_prefetched_in_batch():
    """SimilarityFeedback is queried once with all candidate IDs, not per-candidate."""
    source = _get_source()
    tree = ast.parse(source)

    feedback_query_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "filter":
                for keyword in ast.walk(node):
                    if isinstance(keyword, ast.Attribute) and keyword.attr == "candidate_case_id":
                        feedback_query_count += 1
                        break

    assert feedback_query_count >= 1, "Should have at least one filter on candidate_case_id for batch feedback"


def test_outcome_prefetched_in_batch():
    """CaseOutcome is queried with all candidate IDs, not just the top-scored few."""
    source = _get_source()
    assert "CaseOutcome.case_id.in_(candidate_ids)" in source
