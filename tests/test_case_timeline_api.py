import os
from datetime import datetime, timedelta, timezone

import pytest

# api.auth -> api.config loads settings at import-time. Ensure required env vars exist first.
os.environ["JWT_SECRET_KEY"] = "test-secret-key-value-12345"
os.environ["APP_ALLOWED_HOSTS"] = "localhost"

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import api.routes.cases as cases_route
from api.auth import CurrentUser, get_current_user
from database import Base, Case, CaseStatus, CaseTimeline


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
def client(test_db):
    app = FastAPI()
    app.include_router(cases_route.router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser("42", "tester@example.com", "user")
    app.dependency_overrides[cases_route.get_db] = lambda: test_db
    app.dependency_overrides[cases_route.get_db_rls] = lambda: test_db
    return TestClient(app)


def _seed_case_with_timeline(test_db, user_id: int = 42):
    created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    case = Case(
        user_id=user_id,
        case_number="2023-CV-00001",
        case_type="civil",
        jurisdiction="Delhi",
        status=CaseStatus.CLOSED,
        title="Example Case",
        created_at=created_at,
        updated_at=created_at + timedelta(days=365),
    )
    test_db.add(case)
    test_db.commit()
    test_db.refresh(case)

    test_db.add_all(
        [
            CaseTimeline(
                case_id=case.id,
                event_type="filing",
                event_date=created_at,
                description="Case filed",
                event_metadata={
                    "court": "District Court",
                    "location": "New York, NY",
                    "documents": ["complaint.pdf"],
                },
            ),
            CaseTimeline(
                case_id=case.id,
                event_type="hearing",
                event_date=created_at + timedelta(days=30),
                description="Initial hearing",
                event_metadata={
                    "court": "District Court",
                    "judge": "Judge Smith",
                    "location": "New York, NY",
                },
            ),
            CaseTimeline(
                case_id=case.id,
                event_type="decision",
                event_date=created_at + timedelta(days=365),
                description="Court decision rendered",
                event_metadata={
                    "court": "District Court",
                    "judge": "Judge Smith",
                    "location": "New York, NY",
                    "documents": ["decision.pdf"],
                },
            ),
        ]
    )
    test_db.commit()

    return case


def test_case_timeline_response_matches_model(client, test_db):
    case = _seed_case_with_timeline(test_db)

    response = client.get(f"/api/v1/cases/{case.id}/timeline")

    assert response.status_code == 200
    payload = response.json()

    assert payload["case_id"] == str(case.id)
    assert payload["case_number"] == case.case_number
    assert payload["title"] == "Example Case"
    assert payload["status"] == "closed"
    assert payload["total_events"] == 3
    assert payload["duration_years"] == 1.0
    assert len(payload["events"]) == 3

    filing = next(event for event in payload["events"] if event["event_type"] == "filing")
    hearing = next(event for event in payload["events"] if event["event_type"] == "hearing")
    decision = next(event for event in payload["events"] if event["event_type"] == "decision")

    assert filing["court"] == "District Court"
    assert filing["location"] == "New York, NY"
    assert filing["documents"] == ["complaint.pdf"]
    assert hearing["judge"] == "Judge Smith"
    assert decision["documents"] == ["decision.pdf"]


def test_case_timeline_forbidden_for_other_user(client, test_db):
    case = _seed_case_with_timeline(test_db, user_id=99)

    response = client.get(f"/api/v1/cases/{case.id}/timeline")

    assert response.status_code == 404
    assert response.json()["detail"] == "Case not found"
