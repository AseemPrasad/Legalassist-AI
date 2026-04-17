import logging

import pytest

from core import parse_remedies_response


def test_parse_well_formatted_response():
    response = """
1. What happened?
Plaintiff won the case.
2. Can the loser appeal?
Yes, they can appeal.
3. Appeal timeline
30 days
4. Appeal court
High Court
5. Cost estimate
5000-15000
6. First action
Apply for certified copy.
7. Important deadline
Within 30 days from judgment.
"""
    remedies = parse_remedies_response(response)

    assert remedies is not None
    assert remedies["what_happened"] == "Plaintiff won the case."
    assert remedies["can_appeal"] == "yes"
    assert remedies["appeal_days"] == "30"
    assert remedies["appeal_court"] == "High Court"
    assert remedies["cost_estimate"] == "5000-15000"
    assert remedies["first_action"] == "Apply for certified copy."
    assert remedies["deadline"] == "Within 30 days from judgment."


def test_parse_with_extra_spaces_and_newlines():
    response = """
1.   What happened?

   Defendant was acquitted.   

2. Can the loser appeal?  
No, not in this stage.

3. Appeal timeline
 Not applicable 

4. Appeal court
District Court
"""
    remedies = parse_remedies_response(response)

    assert remedies is not None
    assert remedies["what_happened"] == "Defendant was acquitted."
    assert remedies["can_appeal"] == "no"
    assert remedies["appeal_days"] is None
    assert remedies["appeal_court"] == "District Court"


def test_parse_missing_sections_gracefully():
    response = """
1. What happened?
Defendant won.
2. Can the loser appeal?
Yes.
"""
    remedies = parse_remedies_response(response)

    assert remedies is not None
    assert remedies["what_happened"] == "Defendant won."
    assert remedies["can_appeal"] == "yes"
    assert remedies["appeal_days"] is None
    assert remedies["appeal_court"] is None
    assert remedies["cost_estimate"] is None
    assert remedies["first_action"] is None
    assert remedies["deadline"] is None


@pytest.mark.parametrize("marker", [".", ")", ":", "-"])
def test_parse_numbering_formats(marker):
    response = f"""
1{marker} What happened?
Plaintiff won.
2{marker} Can the loser appeal?
Yes.
3{marker} Appeal timeline
45 days
4{marker} Appeal court
Supreme Court
"""
    remedies = parse_remedies_response(response)

    assert remedies is not None
    assert remedies["what_happened"] == "Plaintiff won."
    assert remedies["can_appeal"] == "yes"
    assert remedies["appeal_days"] == "45"
    assert remedies["appeal_court"] == "Supreme Court"


def test_parse_returns_none_for_unstructured_text(caplog):
    with caplog.at_level(logging.WARNING):
        remedies = parse_remedies_response("This answer has no numbering and cannot be parsed")

    assert remedies is None
    assert "no numbered sections found" in caplog.text


def test_parse_returns_none_for_empty_response(caplog):
    with caplog.at_level(logging.WARNING):
        remedies = parse_remedies_response("   ")

    assert remedies is None
    assert "empty response text" in caplog.text


def test_invalid_can_appeal_is_set_to_none(caplog):
    response = """
1. What happened?
Case dismissed.
2. Can the loser appeal?
Maybe, depends.
"""
    with caplog.at_level(logging.WARNING):
        remedies = parse_remedies_response(response)

    assert remedies is not None
    assert remedies["can_appeal"] is None
    assert "invalid can_appeal" in caplog.text


def test_appeal_days_extracts_number_from_text():
    response = """
1. What happened?
Conviction sustained.
2. Can the loser appeal?
Yes.
3. Appeal timeline
They must file within 90 days from certified copy.
"""
    remedies = parse_remedies_response(response)

    assert remedies is not None
    assert remedies["appeal_days"] == "90"


def test_invalid_appeal_days_logs_warning(caplog):
    response = """
1. What happened?
Conviction sustained.
2. Can the loser appeal?
Yes.
3. Appeal timeline
As soon as possible.
"""
    with caplog.at_level(logging.WARNING):
        remedies = parse_remedies_response(response)

    assert remedies is not None
    assert remedies["appeal_days"] is None
    assert "invalid appeal_days" in caplog.text


def test_unknown_court_is_rejected(caplog):
    response = """
1. What happened?
Plaintiff lost.
2. Can the loser appeal?
Yes.
3. Appeal timeline
30
4. Appeal court
Village Council Bench
"""
    with caplog.at_level(logging.WARNING):
        remedies = parse_remedies_response(response)

    assert remedies is not None
    assert remedies["appeal_court"] is None
    assert "unknown appeal_court" in caplog.text


def test_known_court_is_accepted_when_extended_name():
    response = """
1. What happened?
Plaintiff lost.
2. Can the loser appeal?
Yes.
3. Appeal timeline
30
4. Appeal court
High Court of Delhi
"""
    remedies = parse_remedies_response(response)

    assert remedies is not None
    assert remedies["appeal_court"] == "High Court of Delhi"


def test_malformed_text_does_not_crash():
    response = """
1: ???
... ???
2) !!
YES!!! maybe yes
3- timeline >>> 60 ???
4. court = Sessions Court
"""
    remedies = parse_remedies_response(response)

    assert remedies is not None
    assert remedies["can_appeal"] == "yes"
    assert remedies["appeal_days"] == "60"
    assert remedies["appeal_court"] == "court = Sessions Court"
