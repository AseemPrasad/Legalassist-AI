from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import api.routes.cases as cases_route
from api.auth import CurrentUser, get_current_user
from database import Base, CaseOutcome, CaseRecord, SimilarityFeedback


@pytest.fixture()
def test_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        bind=engine,
    )
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture()
def client(test_db, monkeypatch):
    app = FastAPI()
    app.include_router(cases_route.router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser("42", "tester@example.com", "user")
    app.dependency_overrides[cases_route.get_db] = lambda: test_db
    app.dependency_overrides[cases_route.get_db_rls] = lambda: test_db
    monkeypatch.setattr(cases_route, "get_db", lambda: test_db)
    yield TestClient(app)


def _seed_similarity_cases(test_db):
    now = datetime.now(timezone.utc)

    reference = CaseRecord(
        hashed_case_id="ref-case",
        case_type="civil",
        jurisdiction="Delhi",
        court_name="High Court",
        judge_name="Judge Alpha",
        plaintiff_type="individual",
        defendant_type="company",
        case_value="1-5L",
        outcome="plaintiff_won",
        judgment_summary="Reference case",
        created_at=now,
    )
    candidate_one = CaseRecord(
        hashed_case_id="cand-one",
        case_type="civil",
        jurisdiction="Delhi",
        court_name="High Court",
        judge_name="Judge Beta",
        plaintiff_type="individual",
        defendant_type="company",
        case_value="5-10L",
        outcome="defendant_won",
        judgment_summary="Candidate one",
        created_at=now - timedelta(days=1),
    )
    candidate_two = CaseRecord(
        hashed_case_id="cand-two",
        case_type="civil",
        jurisdiction="Delhi",
        court_name="High Court",
        judge_name="Judge Gamma",
        plaintiff_type="individual",
        defendant_type="company",
        case_value="5-10L",
        outcome="plaintiff_won",
        judgment_summary="Candidate two",
        created_at=now - timedelta(days=2),
    )
    excluded = CaseRecord(
        hashed_case_id="excluded",
        case_type="civil",
        jurisdiction="Delhi",
        court_name="District Court",
        judge_name="Judge Delta",
        plaintiff_type="individual",
        defendant_type="company",
        case_value="5-10L",
        outcome="plaintiff_won",
        judgment_summary="Excluded case",
        created_at=now - timedelta(days=3),
    )

    test_db.add_all([reference, candidate_one, candidate_two, excluded])
    test_db.commit()
    test_db.refresh(reference)
    test_db.refresh(candidate_one)
    test_db.refresh(candidate_two)
    test_db.refresh(excluded)

    test_db.add_all(
        [
            CaseOutcome(case_id=candidate_one.id, appeal_filed=True, appeal_success=True),
            CaseOutcome(case_id=candidate_two.id, appeal_filed=True, appeal_success=False),
        ]
    )
    test_db.commit()

    return reference, candidate_one, candidate_two, excluded


def test_similarity_search_filters_threshold_and_appeal_rate(client, test_db):
    reference, candidate_one, candidate_two, excluded = _seed_similarity_cases(test_db)

    response = client.post(
        "/api/v1/cases/search",
        json={
            "jurisdiction": "Delhi",
            "case_type": "civil",
            "court_name": "High Court",
            "relevance_threshold": 0.7,
            "limit": 5,
            "query_signature": "jurisdiction=Delhi|case_type=civil|court_name=High Court|judge_name=|year_from=|year_to=",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["total_results"] <= 5
    assert len(payload["results"]) == 2
    assert payload["appealed_cases"] == 2
    assert payload["appeal_successful_cases"] == 1
    assert payload["appeal_success_rate"] == 0.5
    assert all(item["relevance_score"] > 0.7 for item in payload["results"])

    returned_ids = {item["case_id"] for item in payload["results"]}
    assert str(excluded.id) not in returned_ids


def test_similarity_feedback_persists_and_adjusts_ranking(client, test_db):
    reference, candidate_one, candidate_two, _ = _seed_similarity_cases(test_db)
    query_signature = "jurisdiction=Delhi|case_type=civil|court_name=High Court|judge_name=|year_from=|year_to="

    feedback_response = client.post(
        "/api/v1/cases/similarity-feedback",
        json={
            "candidate_case_id": candidate_two.id,
            "query_signature": query_signature,
            "relevance": True,
        },
    )

    assert feedback_response.status_code == 200
    feedback_payload = feedback_response.json()
    assert feedback_payload["success"] is True

    feedback_rows = test_db.query(SimilarityFeedback).all()
    assert len(feedback_rows) == 1
    assert feedback_rows[0].candidate_case_id == candidate_two.id
    assert feedback_rows[0].relevance is True

    response = client.post(
        "/api/v1/cases/search",
        json={
            "jurisdiction": "Delhi",
            "case_type": "civil",
            "court_name": "High Court",
            "relevance_threshold": 0.7,
            "limit": 5,
            "query_signature": query_signature,
        },
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["case_id"] == str(candidate_two.id)


def test_similarity_search_query_count_is_bounded(client, test_db):
    from sqlalchemy import event
    reference, candidate_one, candidate_two, _ = _seed_similarity_cases(test_db)
    query_signature = "jurisdiction=Delhi|case_type=civil|court_name=High Court|judge_name=|year_from=|year_to="

    # Seed feedback for candidates to ensure there are feedback records to query
    test_db.add_all([
        SimilarityFeedback(
            user_id="42",
            candidate_case_id=candidate_one.id,
            query_signature=query_signature,
            relevance=True,
        ),
        SimilarityFeedback(
            user_id="42",
            candidate_case_id=candidate_two.id,
            query_signature=query_signature,
            relevance=False,
        ),
    ])
    test_db.commit()

    # Track executed queries using before_cursor_execute
    queries = []

    @event.listens_for(test_db.bind, "before_cursor_execute")
    def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        queries.append(statement)

    try:
        response = client.post(
            "/api/v1/cases/search",
            json={
                "jurisdiction": "Delhi",
                "case_type": "civil",
                "court_name": "High Court",
                "relevance_threshold": 0.7,
                "limit": 5,
                "query_signature": query_signature,
            },
        )
    finally:
        event.remove(test_db.bind, "before_cursor_execute", receive_before_cursor_execute)

    assert response.status_code == 200

    # Find queries related to similarity_feedback (checking table name)
    feedback_queries = [q for q in queries if "similarity_feedback" in q.lower()]

    # Assert that there is exactly 1 query targeting the similarity_feedback table
    assert len(feedback_queries) == 1, f"Expected exactly 1 query targeting similarity_feedback table, got {len(feedback_queries)}"