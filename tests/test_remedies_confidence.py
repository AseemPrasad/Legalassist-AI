import pytest

from core.app_utils import parse_remedies_response


@pytest.mark.parametrize(
    "response_text, expected_min, expected_max, expected_has_evidence",
    [
        (
            """
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
""",
            0.75,
            1.01,
            True,
        ),
        (
            """
1. What happened?
Plaintiff won the case.
2. Can the loser appeal?
Unknown
""",
            0.0,
            0.4,
            True,
        ),
        (
            """
1. What happened?
Defendant was acquitted.
2. Can the loser appeal?
No, not in this stage.
3. Appeal details
Not applicable
4. First action
Nothing
""",
            0.35,
            0.8,
            True,
        ),
    ],
)
def test_parse_remedies_response_includes_confidence_and_evidence(
    response_text, expected_min, expected_max, expected_has_evidence
):
    remedies = parse_remedies_response(response_text)
    assert isinstance(remedies, dict)

    assert "confidence_score" in remedies
    assert 0.0 <= remedies["confidence_score"] <= 1.0
    assert expected_min <= remedies["confidence_score"] <= expected_max

    assert "evidence_spans" in remedies
    assert isinstance(remedies["evidence_spans"], list)
    if expected_has_evidence:
        assert remedies["evidence_spans"]
        assert all("field" in span and "span_text" in span for span in remedies["evidence_spans"])


def test_confidence_low_when_essential_fields_missing():
    response_text = """
1. What happened?
Plaintiff won.
2. Can the loser appeal?
No
"""
    remedies = parse_remedies_response(response_text)
    assert remedies["can_appeal"] != ""
    # appeal_days/court/deadline are missing => should be low
    assert remedies["confidence_score"] <= 0.35

