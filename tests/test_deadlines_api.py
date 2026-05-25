import os
from datetime import datetime, timedelta, timezone

os.environ["JWT_SECRET_KEY"] = "test-secret-key-value-12345"
os.environ["APP_ALLOWED_HOSTS"] = "localhost"

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import api.routes.deadlines as deadlines_route
from api.auth import CurrentUser, get_current_user
from database import Base, Case, CaseDeadline, CaseStatus


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
    app.include_router(deadlines_route.router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser("42", "tester@example.com", "user")
    app.dependency_overrides[deadlines_route.get_db] = lambda: test_db
    return TestClient(app)


def _seed_case_and_deadline(test_db, *, user_id: int = 42):
    case = Case(
        user_id=user_id,
        case_number=f"CASE-{user_id}",
        case_type="civil",
        jurisdiction="Delhi",
        status=CaseStatus.ACTIVE,
        title="Example Case",
    )
    test_db.add(case)
    test_db.commit()
    test_db.refresh(case)

    deadline = CaseDeadline(
        user_id=user_id,
        case_id=case.id,
        case_title=case.title,
        deadline_date=datetime.now(timezone.utc) + timedelta(days=10),
        deadline_type="appeal",
        description="Appeal deadline",
        is_completed=False,
    )
    test_db.add(deadline)
    test_db.commit()
    test_db.refresh(deadline)
    return case, deadline


def _seed_upcoming_deadlines(test_db):
    now = datetime.now(timezone.utc)

    owned_case_one = Case(
        user_id=42,
        case_number="CASE-42-1",
        case_type="civil",
        jurisdiction="Delhi",
        status=CaseStatus.ACTIVE,
        title="Owned Case One",
    )
    owned_case_two = Case(
        user_id=42,
        case_number="CASE-42-2",
        case_type="civil",
        jurisdiction="Mumbai",
        status=CaseStatus.ACTIVE,
        title="Owned Case Two",
    )
    other_user_case = Case(
        user_id=99,
        case_number="CASE-99-1",
        case_type="civil",
        jurisdiction="Chennai",
        status=CaseStatus.ACTIVE,
        title="Other User Case",
    )
    test_db.add_all([owned_case_one, owned_case_two, other_user_case])
    test_db.commit()
    test_db.refresh(owned_case_one)
    test_db.refresh(owned_case_two)
    test_db.refresh(other_user_case)

    upcoming_critical = CaseDeadline(
        user_id=42,
        case_id=owned_case_one.id,
        case_title="Stale Stored Title",
        deadline_date=now + timedelta(days=3),
        deadline_type="appeal",
        description="Due soon",
        is_completed=False,
    )
    upcoming_high = CaseDeadline(
        user_id=42,
        case_id=owned_case_two.id,
        case_title="Stale Stored Title 2",
        deadline_date=now + timedelta(days=12),
        deadline_type="filing",
        description="Due later",
        is_completed=False,
    )
    out_of_range = CaseDeadline(
        user_id=42,
        case_id=owned_case_one.id,
        case_title="Out of Range Stored Title",
        deadline_date=now + timedelta(days=45),
        deadline_type="hearing",
        description="Too far out",
        is_completed=False,
    )
    completed_deadline = CaseDeadline(
        user_id=42,
        case_id=owned_case_one.id,
        case_title="Completed Stored Title",
        deadline_date=now + timedelta(days=7),
        deadline_type="motion",
        description="Already done",
        is_completed=True,
    )
    other_user_deadline = CaseDeadline(
        user_id=99,
        case_id=other_user_case.id,
        case_title="Other Stored Title",
        deadline_date=now + timedelta(days=5),
        deadline_type="brief",
        description="Not owned",
        is_completed=False,
    )

    test_db.add_all([
        upcoming_critical,
        upcoming_high,
        out_of_range,
        completed_deadline,
        other_user_deadline,
    ])
    test_db.commit()
    test_db.refresh(upcoming_critical)
    test_db.refresh(upcoming_high)
    test_db.refresh(out_of_range)
    test_db.refresh(completed_deadline)
    test_db.refresh(other_user_deadline)

    return {
        "owned_case_one": owned_case_one,
        "owned_case_two": owned_case_two,
        "other_user_case": other_user_case,
        "upcoming_critical": upcoming_critical,
        "upcoming_high": upcoming_high,
        "out_of_range": out_of_range,
        "completed_deadline": completed_deadline,
        "other_user_deadline": other_user_deadline,
    }


def test_get_deadline_details_forbidden_for_other_user(client, test_db):
    _, deadline = _seed_case_and_deadline(test_db, user_id=99)

    response = client.get(f"/api/v1/deadlines/{deadline.id}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Deadline not found"


def test_create_deadline_forbidden_for_other_users_case(client, test_db):
    case, _ = _seed_case_and_deadline(test_db, user_id=99)
    due_date = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat()

    response = client.post(
        "/api/v1/deadlines",
        params={
            "title": "Cross-user deadline",
            "due_date": due_date,
            "case_id": str(case.id),
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Case not found"


def test_update_deadline_forbidden_for_other_user(client, test_db):
    _, deadline = _seed_case_and_deadline(test_db, user_id=99)

    response = client.put(
        f"/api/v1/deadlines/{deadline.id}",
        params={"title": "Updated title"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Deadline not found"


def test_get_upcoming_deadlines_returns_db_results(client, test_db):
    seeded = _seed_upcoming_deadlines(test_db)

    response = client.get("/api/v1/deadlines/upcoming?days=30")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "user_id",
        "total_deadlines",
        "critical_count",
        "high_count",
        "medium_count",
        "low_count",
        "deadlines",
        "generated_at",
    }
    assert payload["user_id"] == "42"
    assert payload["total_deadlines"] == 2
    assert payload["critical_count"] == 1
    assert payload["high_count"] == 1
    assert payload["medium_count"] == 0
    assert payload["low_count"] == 0
    assert len(payload["deadlines"]) == 2

    deadlines = payload["deadlines"]
    assert [item["title"] for item in deadlines] == ["Owned Case One", "Owned Case Two"]
    assert [item["case_id"] for item in deadlines] == [
        str(seeded["upcoming_critical"].case_id),
        str(seeded["upcoming_high"].case_id),
    ]
    assert deadlines[0]["deadline_id"] == str(seeded["upcoming_critical"].id)
    assert deadlines[0]["description"] == "Due soon"
    assert deadlines[0]["priority"] == "critical"
    assert deadlines[0]["status"] == "pending"
    assert deadlines[1]["deadline_id"] == str(seeded["upcoming_high"].id)
    assert deadlines[1]["description"] == "Due later"
    assert deadlines[1]["priority"] == "high"
    assert deadlines[1]["status"] == "pending"
    assert deadlines[0]["due_date"] < deadlines[1]["due_date"]
    assert deadlines[0]["title"] == seeded["owned_case_one"].title
    assert deadlines[1]["title"] == seeded["owned_case_two"].title