import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.case_comparison import CaseComparison
from database import Base, Case, CaseDocument, CaseIssue, CaseArgument, CaseStatus, User

@pytest.fixture()
def db_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = Session()

    user = User(email="testuser@example.com")
    db.add(user)
    db.commit()
    db.refresh(user)

    # User Case
    user_case = Case(
        user_id=user.id,
        case_number="UC001",
        case_type="Civil",
        jurisdiction="Delhi High Court",
        status=CaseStatus.ACTIVE,
        title="User Property Dispute"
    )
    # Precedent Case
    precedent_case = Case(
        user_id=user.id,
        case_number="PC001",
        case_type="Civil",
        jurisdiction="Delhi High Court",
        status=CaseStatus.CLOSED,
        title="Precedent Property Dispute"
    )
    db.add_all([user_case, precedent_case])
    db.commit()
    db.refresh(user_case)
    db.refresh(precedent_case)

    # Documents
    user_doc = CaseDocument(
        case_id=user_case.id,
        document_type="Judgment",
        summary="This is a summary of the user case."
    )
    precedent_doc = CaseDocument(
        case_id=precedent_case.id,
        document_type="Judgment",
        summary="This is a summary of the precedent case."
    )
    db.add_all([user_doc, precedent_doc])

    # Issues
    issue_name = "Property Easement Rights"
    user_issue = CaseIssue(case_id=user_case.id, issue_name=issue_name, confidence_score="0.9")
    precedent_issue = CaseIssue(case_id=precedent_case.id, issue_name=issue_name, confidence_score="0.95")
    db.add_all([user_issue, precedent_issue])
    db.commit()
    db.refresh(user_issue)
    db.refresh(precedent_issue)

    # Arguments
    user_arg = CaseArgument(
        case_id=user_case.id,
        issue_id=user_issue.id,
        argument_text="The pathway has been used continuously for over 20 years without interruption.",
        argument_type="plaintiff"
    )
    precedent_arg = CaseArgument(
        case_id=precedent_case.id,
        issue_id=precedent_issue.id,
        argument_text="The pathway has been used continuously for over 20 years without interruption by the easement claimant.",
        argument_type="plaintiff",
        argument_succeeded=True,
        supporting_evidence="Documentary evidence showing continuous access and witness testimony."
    )
    db.add_all([user_arg, precedent_arg])
    db.commit()

    return db, user_case.id, precedent_case.id


def test_calculate_length_penalized_similarity():
    # Identical strings
    t1 = "Continuous easement usage for twenty years."
    t2 = "Continuous easement usage for twenty years."
    sim1 = CaseComparison.calculate_length_penalized_similarity(t1, t2)
    assert sim1 == 1.0

    # Empty inputs
    assert CaseComparison.calculate_length_penalized_similarity("", t2) == 0.0
    assert CaseComparison.calculate_length_penalized_similarity(t1, None) == 0.0

    # Extreme length mismatch should have heavy penalty
    short_text = "Easement path."
    long_text = "Easement path. " + " ".join(["extra word"] * 50) + " representing long legal filings."
    sim2 = CaseComparison.calculate_length_penalized_similarity(short_text, long_text)
    # The penalty should significantly reduce the match ratio
    assert sim2 < 0.5


def test_compare_cases(db_session):
    db, user_case_id, precedent_case_id = db_session
    comparison = CaseComparison.compare_cases(db, user_case_id, precedent_case_id)

    assert comparison != {}
    assert comparison["user_case"]["number"] == "UC001"
    assert comparison["precedent_case"]["number"] == "PC001"
    assert comparison["similarities"]["shared_issues"] == ["Property Easement Rights"]
    assert len(comparison["similarities"]["similar_arguments"]) > 0
    assert comparison["similarities"]["similar_arguments"][0]["precedent_succeeded"] is True


def test_suggest_arguments(db_session):
    db, user_case_id, precedent_case_id = db_session
    suggestions = CaseComparison.suggest_arguments(db, user_case_id, precedent_case_id)

    assert len(suggestions) > 0
    assert suggestions[0]["issue"] == "Property Easement Rights"
    assert "UC001" not in suggestions[0]["reason"]  # should mention precedent case
    assert suggestions[0]["precedent_case_number"] == "PC001"


def test_highlight_differences(db_session):
    db, user_case_id, precedent_case_id = db_session
    highlights = CaseComparison.highlight_differences(db, user_case_id, precedent_case_id)

    # Since they are both Civil type and Delhi High Court, no warnings for case type/jurisdiction difference
    assert "warning_different_case_type" not in highlights
    assert "warning_different_jurisdiction" not in highlights
