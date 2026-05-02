import pytest
from core.app_utils import parse_summary_bullets

def test_parse_standard_bullets():
    raw_text = """
    Here is your summary:
    - Point one about the case.
    - Point two about the outcome.
    - Point three about what to do next.
    """
    parsed = parse_summary_bullets(raw_text)
    assert parsed.count("- ") == 3
    assert "Point one" in parsed
    assert "Point two" in parsed
    assert "Point three" in parsed
    assert "Here is your summary" not in parsed

def test_parse_numbered_bullets():
    raw_text = """
    Judgment Analysis:
    1. The defendant is guilty.
    2. A fine was imposed.
    3. The case is closed.
    """
    parsed = parse_summary_bullets(raw_text)
    assert parsed.count("- ") == 3
    assert "defendant is guilty" in parsed
    assert "Judgment Analysis" not in parsed

def test_parse_mixed_markers():
    raw_text = """
    - First point
    * Second point
    • Third point
    Extra point that should be ignored.
    """
    parsed = parse_summary_bullets(raw_text)
    assert parsed.count("- ") == 3
    assert "First point" in parsed
    assert "Second point" in parsed
    assert "Third point" in parsed
    assert "Extra point" not in parsed

def test_parse_no_markers_with_intro():
    raw_text = """
    Here is what happened:
    The court ruled in favor of the plaintiff.
    The defendant must pay damages.
    The appeal is allowed within 30 days.
    """
    parsed = parse_summary_bullets(raw_text)
    assert parsed.count("- ") == 3
    assert "ruled in favor" in parsed
    assert "pay damages" in parsed
    assert "appeal is allowed" in parsed
    assert "Here is what happened" not in parsed

def test_parse_too_many_bullets():
    raw_text = """
    - One
    - Two
    - Three
    - Four
    - Five
    """
    parsed = parse_summary_bullets(raw_text)
    assert parsed.count("- ") == 3
    assert "Four" not in parsed

def test_parse_empty():
    assert parse_summary_bullets("") == ""
    assert parse_summary_bullets(None) == ""

def test_parse_unstructured():
    raw_text = "This is a single line that doesn't look like bullets at all."
    parsed = parse_summary_bullets(raw_text)
    # Should fallback to raw or take it as a bullet if long enough
    assert "This is a single line" in parsed
