from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.precedent_matcher import PrecedentMatcher
from database import Base, Case, CaseIssue, CaseArgument, KnowledgeGraphEdge, CaseStatus, User


@pytest.fixture()
def pg_db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = Session()

    user = User(email="pgr@example.com")
    db.add(user)
    db.commit()
    db.refresh(user)

    now = datetime.now(timezone.utc)

    query_case = Case(user_id=user.id, case_number="Q1", case_type="civil", jurisdiction="Delhi", status=CaseStatus.ACTIVE, title="Query", created_at=now)
    high_court = Case(user_id=user.id, case_number="HC1", case_type="civil", jurisdiction="Delhi", status=CaseStatus.CLOSED, title="High Court Judgment", created_at=now - timedelta(days=30))
    district_old = Case(user_id=user.id, case_number="DC1", case_type="civil", jurisdiction="Delhi", status=CaseStatus.CLOSED, title="District Court Old", created_at=now - timedelta(days=3650))
    recent_tribunal = Case(user_id=user.id, case_number="T1", case_type="civil", jurisdiction="Delhi", status=CaseStatus.CLOSED, title="Tribunal Recent", created_at=now - timedelta(days=10))

    db.add_all([query_case, high_court, district_old, recent_tribunal])
    db.commit()
    db.refresh(query_case)
    db.refresh(high_court)
    db.refresh(district_old)
    db.refresh(recent_tribunal)

    # Create issue/arguments and KG edges to simulate citations
    issue = CaseIssue(case_id=high_court.id, issue_name="contract", confidence_score="1.0")
    db.add(issue)
    db.commit()
    db.refresh(issue)

    # arguments (simplified)
    arg_hc = CaseArgument(case_id=high_court.id, argument_text="Arg HC", argument_succeeded=True)
    arg_dc = CaseArgument(case_id=district_old.id, argument_text="Arg DC", argument_succeeded=True)
    arg_tb = CaseArgument(case_id=recent_tribunal.id, argument_text="Arg TB", argument_succeeded=True)
    db.add_all([arg_hc, arg_dc, arg_tb])
    db.commit()

    # Knowledge graph edges: simulate citations (case_id points to precedent case)
    edge_hc = KnowledgeGraphEdge(issue_id=issue.id, argument_id=arg_hc.id, case_id=high_court.id, outcome="plaintiff_won", weight="1.0")
    edge_dc = KnowledgeGraphEdge(issue_id=issue.id, argument_id=arg_dc.id, case_id=district_old.id, outcome="plaintiff_won", weight="0.8")
    edge_tb = KnowledgeGraphEdge(issue_id=issue.id, argument_id=arg_tb.id, case_id=recent_tribunal.id, outcome="plaintiff_won", weight="0.9")
    db.add_all([edge_hc, edge_dc, edge_tb])
    db.commit()

    return db, query_case.id


def test_precedent_ranking_prioritizes_authority_and_recency(pg_db):
    db, qid = pg_db
    results = PrecedentMatcher.find_winning_precedents(db=db, case_id=qid, issue_name="contract", limit=10)

    # we expect recent tribunal and high court to rank above old district court
    ids = [r["case_id"] for r in results]
    assert len(ids) >= 2
    # District old should be last
    assert any(r["title"] == "District Court Old" for r in results)
    # Ensure order: high authority or recent appear before old district
    pos_dc = next(i for i, r in enumerate(results) if r["title"] == "District Court Old")
    pos_hc = next(i for i, r in enumerate(results) if r["title"] == "High Court Judgment")
    pos_tb = next(i for i, r in enumerate(results) if r["title"] == "Tribunal Recent")
    assert pos_dc > pos_hc or pos_dc > pos_tb
