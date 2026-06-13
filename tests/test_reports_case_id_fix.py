"""Regression test for issue #2070: case_id_int undefined in generate_report."""
import ast
from pathlib import Path

REPORTS_PATH = Path(__file__).resolve().parents[1] / "api" / "routes" / "reports.py"


def _load_source() -> str:
    return REPORTS_PATH.read_text(encoding="utf-8")


def test_case_id_int_is_defined_before_use():
    source = _load_source()
    assert "case_id_int = int(request.case_id)" in source, (
        "Expected 'case_id_int = int(request.case_id)' to be present "
        "(fix for issue #2070)."
    )


def test_generate_report_function_is_syntactically_valid():
    source = _load_source()
    ast.parse(source)


def test_no_undefined_case_id_int_in_delay_call():
    source = _load_source()
    lines = source.splitlines()

    assignment_line = None
    for i, line in enumerate(lines, start=1):
        if "case_id_int = int(request.case_id)" in line:
            assignment_line = i
            break

    assert assignment_line is not None, "case_id_int assignment not found"
