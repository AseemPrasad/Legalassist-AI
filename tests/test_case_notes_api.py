import os
os.environ["JWT_SECRET_KEY"] = "test-secret-key-value-12345"
os.environ["APP_ALLOWED_HOSTS"] = "localhost"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"

from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import api.routes.cases as cases_route
from api.auth import CurrentUser, get_current_user
from database import Base, Case, CaseStatus, create_user, create_case, save_case_note_draft, publish_case_note, get_case_note_history, CaseNoteVersion


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
    yield TestClient(app)


def _seed_case(test_db):
    user = create_user(test_db, "notes@example.com")
    case = create_case(test_db, user.id, "CASE-NOTES-001", "civil", "Delhi", title="Notes Case")
    return user, case


def test_case_note_draft_publish_and_history_api(client, test_db):
    user, case = _seed_case(test_db)
    test_db.query(Case).filter(Case.id == case.id).update({"user_id": 42}, synchronize_session=False)
    test_db.commit()

    draft_response = client.post(
        f"/api/v1/cases/{case.id}/notes/draft",
        json={"note_text": "Initial draft for the client meeting."},
    )
    assert draft_response.status_code == 200
    assert draft_response.json()["draft_text"] == "Initial draft for the client meeting."

    publish_response = client.post(
        f"/api/v1/cases/{case.id}/notes/publish",
        json={"note_text": "Initial draft for the client meeting."},
    )
    assert publish_response.status_code == 200
    assert publish_response.json()["version_number"] == 1

    history_response = client.get(f"/api/v1/cases/{case.id}/notes/history")
    assert history_response.status_code == 200
    history = history_response.json()
    assert history["total_versions"] == 1
    assert history["versions"][0]["note_text"] == "Initial draft for the client meeting."
    assert history["versions"][0]["changed_by_email"] == "tester@example.com"


def test_published_versions_are_immutable_after_draft_updates(test_db):
    user, case = _seed_case(test_db)

    save_case_note_draft(test_db, case.id, user.id, "Draft v1", changed_by_email="tester@example.com")
    version_one = publish_case_note(test_db, case.id, user.id, note_text="Draft v1", changed_by_email="tester@example.com")

    save_case_note_draft(test_db, case.id, user.id, "Draft v2", changed_by_email="tester@example.com")
    version_two = publish_case_note(test_db, case.id, user.id, note_text="Draft v2", changed_by_email="tester@example.com")

    stored_versions = test_db.query(CaseNoteVersion).filter(CaseNoteVersion.case_id == case.id).order_by(CaseNoteVersion.version_number).all()
    assert len(stored_versions) == 2
    assert stored_versions[0].note_text == "Draft v1"
    assert stored_versions[1].note_text == "Draft v2"
    assert version_one.note_text == "Draft v1"
    assert version_two.note_text == "Draft v2"

    history = get_case_note_history(test_db, case.id, user.id)
    assert [version.version_number for version in history] == [2, 1]
    assert history[1].note_text == "Draft v1"
