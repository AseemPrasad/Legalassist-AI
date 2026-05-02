import logging

import pytest

from core import parse_remedies_response


def test_parse_well_formatted_response_7section():
    """Test 7-section format (new)"""
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
    assert "Plaintiff won" in remedies["what_happened"]
    # 7-section format normalizes to "yes"/"no"
    assert remedies["can_appeal"] == "yes"
    assert remedies["appeal_days"] == "30"
    assert "High Court" in remedies["appeal_court"]
    assert remedies["_is_partial"] is False


def test_parse_with_extra_spaces_and_newlines_5section():
    """Test 5-section format (old) which stores raw content"""
    response = """
1. What happened?
Defendant was acquitted.
2. Can the loser appeal?
No, not in this stage.
3. Appeal details
Not applicable
4. First action
Nothing
5. Timeline
None
"""
    remedies = parse_remedies_response(response)

    assert remedies is not None
    assert "acquitted" in remedies["what_happened"].lower()
    # Now normalized to lowercase
    assert "no" in remedies["can_appeal"].lower()
    assert remedies["_is_partial"] is True


def test_parse_missing_sections_gracefully_5section():
    """Test handling of fewer sections"""
    response = """
1. What happened?
Defendant won.
2. Can the loser appeal?
Yes.
3. Appeal details
Some details here.
"""
    remedies = parse_remedies_response(response)

    assert remedies is not None
    assert "Defendant won" in remedies["what_happened"]
    # Now normalized to lowercase
    assert "yes" in remedies["can_appeal"].lower()
    assert remedies["_is_partial"] is True


@pytest.mark.parametrize("marker", [".", ")", ":", "-"])
def test_parse_numbering_formats_7section(marker):
    """Test different numbering markers work with 7-section format"""
    response = f"""
1{marker} What happened?
Plaintiff won.
2{marker} Can the loser appeal?
Yes.
3{marker} Appeal timeline
45 days
4{marker} Appeal court
Supreme Court
5{marker} Cost estimate
5000-10000
6{marker} First action
Appeal
7{marker} Deadline
30 days
"""
    remedies = parse_remedies_response(response)

    assert remedies is not None
    assert "Plaintiff won" in remedies["what_happened"]
    assert remedies["can_appeal"] == "yes"
    assert remedies["appeal_days"] == "45"
    assert "Supreme" in remedies["appeal_court"]


def test_parse_3section_format():
    """Test 3-section format is treated as old format"""
    remedies = parse_remedies_response("This answer has no numbering and cannot be parsed")
    assert remedies is None


def test_parse_empty_response():
    remedies = parse_remedies_response("   ")
    assert remedies is None


def test_parse_only_unmapped_sections_returns_none():
    """Test that numbered input with no valid sections returns None"""
    response = """
8. Extra section
Not part of the supported schema.
9. Another extra section
Still unsupported.
"""
    remedies = parse_remedies_response(response)

    assert remedies is None


def test_can_appeal_7section_yes_no():
    """Test can_appeal normalization only happens with 7+ sections"""
    # With 7 sections, yes/no is normalized
    response_yes = """
1. What happened?
Case decided.
2. Can the loser appeal?
Yes
3. Appeal timeline
30
4. Appeal court
Court
5. Cost
1000
6. First action
Act
7. Deadline
30 days
"""
    response_no = """
1. What happened?
Case decided.
2. Can the loser appeal?
No
3. Appeal timeline
0
4. Appeal court
None
5. Cost
0
6. First action
Nothing
7. Deadline
Never
"""
    remedies_yes = parse_remedies_response(response_yes)
    remedies_no = parse_remedies_response(response_no)
    
    assert remedies_yes["can_appeal"] == "yes"
    assert remedies_no["can_appeal"] == "no"


def test_appeal_days_extraction_7section():
    """Test appeal_days extraction from section 3 with 7+ sections"""
    response = """
1. What happened?
Conviction sustained.
2. Can the loser appeal?
Yes.
3. Appeal timeline
They must file within 90 days from certified copy.
4. Appeal court
High Court
5. Cost
5000
6. First action
File
7. Deadline
90 days
"""
    remedies = parse_remedies_response(response)

    assert remedies is not None
    assert remedies["appeal_days"] == "90"


def test_appeal_days_simple_number_7section():
    """Test simple number extraction for appeal_days"""
    response = """
1. What happened?
Conviction sustained.
2. Can the loser appeal?
Yes.
3. Appeal timeline
60
4. Appeal court
Court
5. Cost
1000
6. First action
Act
7. Deadline
60 days
"""
    remedies = parse_remedies_response(response)

    assert remedies is not None
    assert remedies["appeal_days"] == "60"


def test_known_court_names_7section():
    """Test court extraction with 7+ sections"""
    response = """
1. What happened?
Plaintiff lost.
2. Can the loser appeal?
Yes.
3. Appeal timeline
30
4. Appeal court
High Court of Delhi
5. Cost
5000
6. First action
Appeal
7. Deadline
30 days
"""
    remedies = parse_remedies_response(response)

    assert remedies is not None
    assert "High Court" in remedies["appeal_court"]


def test_malformed_text_does_not_crash():
    """Ensure parser doesn't crash on malformed input"""
    # Malformed text 1
    malformed = """
   1. What happened
   )))
   2 Can the loser appeal
   Yes Yes Yes
   3)) Timeline))
   """
    remedies = parse_remedies_response(malformed)
    assert remedies is not None

    # Malformed text 2 - with special characters
    response = """
1: Some text here
2) With strange content!!!
YES!!! maybe yes
3- More malformed data >>> 60 ???
4. And more stuff
5. Final section
6. More
7. Data
"""
    remedies = parse_remedies_response(response)
    assert remedies is not None
    # Just verify it parses without crashing
    assert isinstance(remedies, dict)
