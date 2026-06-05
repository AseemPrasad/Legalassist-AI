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
from sqlalchemy import text

import api.routes.deadlines as deadlines_route
from api.auth import CurrentUser, get_current_user
from database import Base


@pytest.fixture()
def test_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                email VARCHAR(255) NOT NULL
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE cases (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                case_number VARCHAR(255) NOT NULL,
                case_type VARCHAR(255) NOT NULL,
                jurisdiction VARCHAR(255) NOT NULL,
                status VARCHAR(50) NOT NULL,
                title VARCHAR(255),
                version INTEGER NOT NULL DEFAULT 1,
                created_at DATETIME,
                updated_at DATETIME
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE case_deadlines (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                case_id INTEGER NOT NULL,
                case_title VARCHAR(255) NOT NULL,
                deadline_date DATETIME NOT NULL,
                deadline_type VARCHAR(255) NOT NULL,
                description TEXT,
                created_at DATETIME,
                updated_at DATETIME,
                is_completed BOOLEAN DEFAULT 0 NOT NULL
            )
            """
        )
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
    app.dependency_overrides[deadlines_route.get_db_rls] = lambda: test_db
    return TestClient(app)


def _seed_case(test_db, *, user_id: int = 42, case_id: int = 1, title: str = "Example Case"):
    test_db.execute(
        text(
            """
            INSERT INTO cases (id, user_id, case_number, case_type, jurisdiction, status, title, version)
            VALUES (:id, :user_id, :case_number, :case_type, :jurisdiction, :status, :title, :version)
            """
        ),
        {
            "id": case_id,
            "user_id": user_id,
            "case_number": f"CASE-{user_id}",
            "case_type": "civil",
            "jurisdiction": "Delhi",
            "status": "active",
            "title": title,
            "version": 1,
        },
    )
    test_db.commit()
    return {"id": case_id, "title": title, "user_id": user_id}


def _seed_case_and_deadline(test_db, *, user_id: int = 42):
    case = _seed_case(test_db, user_id=user_id)
    deadline_id = 1
    test_db.execute(
        text(
            """
            INSERT INTO case_deadlines (
                id, user_id, case_id, case_title, deadline_date, deadline_type,
                description, created_at, updated_at, is_completed
            ) VALUES (
                :id, :user_id, :case_id, :case_title, :deadline_date, :deadline_type,
                :description, :created_at, :updated_at, :is_completed
            )
            """
        ),
        {
            "id": deadline_id,
            "user_id": user_id,
            "case_id": case["id"],
            "case_title": case["title"],
            "deadline_date": (datetime.now(timezone.utc) + timedelta(days=10)).isoformat(),
            "deadline_type": "appeal",
            "description": "Appeal deadline",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "is_completed": 0,
        },
    )
    test_db.commit()
    return case, {"id": deadline_id, "case_id": case["id"], "case_title": case["title"]}


def _seed_upcoming_deadlines(test_db):
    now = datetime.now(timezone.utc)

    owned_case_one = _seed_case(test_db, user_id=42, case_id=1, title="Owned Case One")
    owned_case_two = _seed_case(test_db, user_id=42, case_id=2, title="Owned Case Two")
    other_user_case = _seed_case(test_db, user_id=99, case_id=3, title="Other User Case")

    deadlines = [
        (1, 42, owned_case_one["id"], "Stale Stored Title", now + timedelta(days=3), "appeal", "Due soon", 0),
        (2, 42, owned_case_two["id"], "Stale Stored Title 2", now + timedelta(days=12), "filing", "Due later", 0),
        (3, 42, owned_case_one["id"], "Out of Range Stored Title", now + timedelta(days=45), "hearing", "Too far out", 0),
        (4, 42, owned_case_one["id"], "Completed Stored Title", now + timedelta(days=7), "motion", "Already done", 1),
        (5, 99, other_user_case["id"], "Other Stored Title", now + timedelta(days=5), "brief", "Not owned", 0),
    ]
    for deadline_id, user_id, case_id, case_title, deadline_date, deadline_type, description, is_completed in deadlines:
        test_db.execute(
            text(
                """
                INSERT INTO case_deadlines (
                    id, user_id, case_id, case_title, deadline_date, deadline_type,
                    description, created_at, updated_at, is_completed
                ) VALUES (
                    :id, :user_id, :case_id, :case_title, :deadline_date, :deadline_type,
                    :description, :created_at, :updated_at, :is_completed
                )
                """
            ),
            {
                "id": deadline_id,
                "user_id": user_id,
                "case_id": case_id,
                "case_title": case_title,
                "deadline_date": deadline_date.isoformat(),
                "deadline_type": deadline_type,
                "description": description,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "is_completed": is_completed,
            },
        )
    test_db.commit()

    return {
        "owned_case_one": owned_case_one,
        "owned_case_two": owned_case_two,
        "other_user_case": other_user_case,
        "upcoming_critical": {"id": 1, "case_id": owned_case_one["id"]},
        "upcoming_high": {"id": 2, "case_id": owned_case_two["id"]},
        "out_of_range": {"id": 3, "case_id": owned_case_one["id"]},
        "completed_deadline": {"id": 4, "case_id": owned_case_one["id"]},
        "other_user_deadline": {"id": 5, "case_id": other_user_case["id"]},
    }


def test_get_deadline_details_forbidden_for_other_user(client, test_db):
    _, deadline = _seed_case_and_deadline(test_db, user_id=99)

    response = client.get(f"/api/v1/deadlines/{deadline['id']}")

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
            "case_id": str(case["id"]),
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Case not found"


def test_create_deadline_persists_and_can_be_retrieved(client, test_db):
    case = _seed_case(test_db, user_id=42)
    due_date = datetime.now(timezone.utc) + timedelta(days=14, hours=2)

    response = client.post(
        "/api/v1/deadlines",
        params={
            "title": "Appeal filing",
            "due_date": due_date.isoformat(),
            "description": "File the appeal",
            "priority": "high",
            "case_id": str(case["id"]),
            "reminder_days": 10,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["deadline_id"] != "dl_new"

    created = test_db.execute(
        text("SELECT id, case_id, case_title, description, deadline_date FROM case_deadlines WHERE id = :id"),
        {"id": int(payload["deadline_id"])},
    ).mappings().one()
    assert created["case_id"] == case["id"]
    assert created["case_title"] == case["title"]
    assert created["description"] == "File the appeal"
    assert created["deadline_date"] is not None

    detail_response = client.get(f"/api/v1/deadlines/{payload['deadline_id']}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["deadline_id"] == payload["deadline_id"]
    assert detail["case_id"] == str(case["id"])
    assert detail["title"] == case["title"]
    assert detail["description"] == "File the appeal"
    assert detail["priority"] == "medium"


def test_update_deadline_forbidden_for_other_user(client, test_db):
    _, deadline = _seed_case_and_deadline(test_db, user_id=99)

    response = client.put(
        f"/api/v1/deadlines/{deadline['id']}",
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
        "limit",
        "offset",
        "critical_count",
        "high_count",
        "medium_count",
        "low_count",
        "deadlines",
        "generated_at",
    }
    assert payload["user_id"] == "42"
    assert payload["total_deadlines"] == 2
    assert payload["limit"] == 50
    assert payload["offset"] == 0
    assert payload["critical_count"] == 1
    assert payload["high_count"] == 0
    assert payload["medium_count"] == 1
    assert payload["low_count"] == 0
    assert len(payload["deadlines"]) == 2

    deadlines = payload["deadlines"]
    assert [item["title"] for item in deadlines] == ["Owned Case One", "Owned Case Two"]
    assert [item["case_id"] for item in deadlines] == [
        str(seeded["upcoming_critical"]["case_id"]),
        str(seeded["upcoming_high"]["case_id"]),
    ]
    assert deadlines[0]["deadline_id"] == str(seeded["upcoming_critical"]["id"])
    assert deadlines[0]["description"] == "Due soon"
    assert deadlines[0]["priority"] == "critical"
    assert deadlines[0]["status"] == "pending"
    assert deadlines[1]["deadline_id"] == str(seeded["upcoming_high"]["id"])
    assert deadlines[1]["description"] == "Due later"
    assert deadlines[1]["priority"] == "medium"
    assert deadlines[1]["status"] == "pending"
    assert deadlines[0]["due_date"] < deadlines[1]["due_date"]
    assert deadlines[0]["title"] == seeded["owned_case_one"]["title"]
    assert deadlines[1]["title"] == seeded["owned_case_two"]["title"]


def test_get_upcoming_deadlines_uses_stable_ordering_with_pagination(client, test_db):
    now = datetime.now(timezone.utc)
    case = _seed_case(test_db, user_id=42, case_id=10, title="Paged Case")

    same_due_date = now + timedelta(days=9)
    for deadline_id in [30, 20, 10]:
        test_db.execute(
            text(
                """
                INSERT INTO case_deadlines (
                    id, user_id, case_id, case_title, deadline_date, deadline_type,
                    description, created_at, updated_at, is_completed
                ) VALUES (
                    :id, :user_id, :case_id, :case_title, :deadline_date, :deadline_type,
                    :description, :created_at, :updated_at, :is_completed
                )
                """
            ),
            {
                "id": deadline_id,
                "user_id": 42,
                "case_id": case["id"],
                "case_title": case["title"],
                "deadline_date": same_due_date.isoformat(),
                "deadline_type": "appeal",
                "description": f"Deadline {deadline_id}",
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "is_completed": 0,
            },
        )
    test_db.commit()

    response = client.get("/api/v1/deadlines/upcoming?days=30&limit=2&offset=1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_deadlines"] == 3
    assert payload["limit"] == 2
    assert payload["offset"] == 1
    assert [deadline["deadline_id"] for deadline in payload["deadlines"]] == ["20", "30"]
    assert [deadline["description"] for deadline in payload["deadlines"]] == ["Deadline 20", "Deadline 30"]