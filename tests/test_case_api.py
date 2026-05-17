import os
os.environ["JWT_SECRET_KEY"] = "test-secret-key-value-12345"
os.environ["APP_ALLOWED_HOSTS"] = "localhost"

from datetime import datetime, timezone
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import api.routes.cases as cases_route
from api.auth import CurrentUser, get_current_user
from database import Base, Case, CaseStatus, CaseDocument, DocumentType


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
    # Mock current_user as user_id 42
    app.dependency_overrides[get_current_user] = lambda: CurrentUser("42", "tester@example.com", "user")
    app.dependency_overrides[cases_route.get_db] = lambda: test_db
    yield TestClient(app)


def _seed_cases(test_db):
    now = datetime.now(timezone.utc)
    
    # Cases for user 42 (current_user)
    case_one = Case(
        user_id=42,
        case_number="2023-CV-00001",
        title="Owned Case One",
        case_type="civil",
        jurisdiction="Delhi",
        status=CaseStatus.ACTIVE,
        created_at=now,
    )
    case_two = Case(
        user_id=42,
        case_number="2023-CV-00002",
        title="Owned Case Two",
        case_type="labor",
        jurisdiction="Mumbai",
        status=CaseStatus.CLOSED,
        created_at=now,
    )
    
    # Case for user 99 (other user)
    case_three = Case(
        user_id=99,
        case_number="2023-CV-00003",
        title="Other User Case",
        case_type="criminal",
        jurisdiction="Chennai",
        status=CaseStatus.ACTIVE,
        created_at=now,
    )
    
    test_db.add_all([case_one, case_two, case_three])
    test_db.commit()
    test_db.refresh(case_one)
    test_db.refresh(case_two)
    test_db.refresh(case_three)
    
    # Document for Case One to test summary
    doc = CaseDocument(
        case_id=case_one.id,
        document_type=DocumentType.JUDGMENT,
        document_content="This is the plaint content.",
        summary="Summary of Case One Plaint",
    )
    test_db.add(doc)
    test_db.commit()
    
    return case_one, case_two, case_three


def test_list_cases_empty(client):
    response = client.get("/api/v1/cases")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 0
    assert len(payload["cases"]) == 0


def test_list_cases_success(client, test_db):
    case_one, case_two, case_three = _seed_cases(test_db)
    
    response = client.get("/api/v1/cases?limit=10&offset=0")
    assert response.status_code == 200
    payload = response.json()
    
    assert payload["total"] == 2
    assert len(payload["cases"]) == 2
    
    # Check that cases belong to user 42 and details are correct
    case_ids = {c["case_id"] for c in payload["cases"]}
    assert str(case_one.id) in case_ids
    assert str(case_two.id) in case_ids
    assert str(case_three.id) not in case_ids
    
    # Verify latest document summary mapping works
    for c in payload["cases"]:
        if c["case_id"] == str(case_one.id):
            assert c["summary"] == "Summary of Case One Plaint"
        else:
            assert c["summary"] == ""


def test_get_case_details_success(client, test_db):
    case_one, _, _ = _seed_cases(test_db)
    
    response = client.get(f"/api/v1/cases/{case_one.id}")
    assert response.status_code == 200
    payload = response.json()
    
    assert payload["case_id"] == str(case_one.id)
    assert payload["case_number"] == case_one.case_number
    assert payload["title"] == case_one.title
    assert payload["summary"] == "Summary of Case One Plaint"
    assert payload["status"] == "active"


def test_get_case_details_forbidden(client, test_db):
    _, _, case_three = _seed_cases(test_db)
    
    response = client.get(f"/api/v1/cases/{case_three.id}")
    assert response.status_code == 403
    assert response.json()["detail"] == "Forbidden: You do not own this case"


def test_get_case_details_not_found(client, test_db):
    _seed_cases(test_db)
    
    response = client.get("/api/v1/cases/999999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Case not found"


def test_get_case_details_invalid_id(client):
    response = client.get("/api/v1/cases/abc")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid case ID format"
