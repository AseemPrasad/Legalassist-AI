import os
os.environ["JWT_SECRET_KEY"] = "test-secret-key-value-12345"
os.environ["APP_ALLOWED_HOSTS"] = "localhost"

from datetime import datetime, timedelta, timezone
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import api.routes.cases as cases_route
from api.auth import CurrentUser, get_current_user
from database import Base, Case, CaseStatus, CaseDocument, DocumentType, AnonymizedShareToken


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
    
    # Documents for Case One to test latest-document selection
    older_doc = CaseDocument(
        case_id=case_one.id,
        document_type=DocumentType.JUDGMENT,
        document_content="This is the plaint content.",
        summary="Older summary",
        uploaded_at=now - timedelta(days=2),
    )
    newer_doc = CaseDocument(
        case_id=case_one.id,
        document_type=DocumentType.ORDER,
        document_content="This is the later order content.",
        summary="Latest summary",
        uploaded_at=now - timedelta(days=1),
    )
    test_db.add_all([older_doc, newer_doc])
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
    assert all(set(case_payload) == {
        "case_id",
        "case_number",
        "title",
        "parties",
        "jurisdiction",
        "status",
        "summary",
    } for case_payload in payload["cases"])
    
    # Check that cases belong to user 42 and details are correct
    case_ids = {c["case_id"] for c in payload["cases"]}
    assert str(case_one.id) in case_ids
    assert str(case_two.id) in case_ids
    assert str(case_three.id) not in case_ids
    
    # Verify latest document summary mapping works
    for c in payload["cases"]:
        if c["case_id"] == str(case_one.id):
            assert c["summary"] == "Latest summary"
        else:
            assert c["summary"] == ""


def test_list_cases_trailing_slash_schema(client, test_db):
    _seed_cases(test_db)

    response = client.get("/api/v1/cases/")
    assert response.status_code == 200

    payload = response.json()
    assert set(payload) == {"total", "limit", "offset", "cases"}
    assert isinstance(payload["cases"], list)
    assert all(set(case_payload) == {
        "case_id",
        "case_number",
        "title",
        "parties",
        "jurisdiction",
        "status",
        "summary",
    } for case_payload in payload["cases"])


def test_get_case_details_success(client, test_db):
    case_one, _, _ = _seed_cases(test_db)
    
    response = client.get(f"/api/v1/cases/{case_one.id}")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "case_id",
        "case_number",
        "title",
        "parties",
        "jurisdiction",
        "status",
        "summary",
    }
    
    assert payload["case_id"] == str(case_one.id)
    assert payload["case_number"] == case_one.case_number
    assert payload["title"] == case_one.title
    assert payload["summary"] == "Latest summary"
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


def test_create_anonymized_share_link(client, test_db):
    case_one, _, _ = _seed_cases(test_db)

    response = client.post(
        f"/api/v1/cases/{case_one.id}/share-anonymized",
        json={"scope": "full_party_removal", "expires_in_hours": 24},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["scope"] == "full_party_removal"
    assert payload["token"]
    assert payload["share_url"].endswith(f"/api/v1/cases/share/{payload['token']}")

    share_row = test_db.query(AnonymizedShareToken).filter(AnonymizedShareToken.token == payload["token"]).first()
    assert share_row is not None
    assert share_row.case_id == case_one.id
    assert share_row.used_at is None


def test_share_token_invalid_returns_404(client):
    response = client.get("/api/v1/cases/share/not-a-real-token")
    assert response.status_code == 404
    assert response.json()["detail"] == "Share token not found"


def test_share_token_expired_returns_410(client, test_db):
    case_one, _, _ = _seed_cases(test_db)
    create_response = client.post(
        f"/api/v1/cases/{case_one.id}/share-anonymized",
        json={"scope": "personal_identifiers", "expires_in_hours": 24},
    )
    token = create_response.json()["token"]

    share_row = test_db.query(AnonymizedShareToken).filter(AnonymizedShareToken.token == token).first()
    assert share_row is not None
    share_row.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    test_db.commit()

    response = client.get(f"/api/v1/cases/share/{token}")
    assert response.status_code == 410
    assert response.json()["detail"] == "Share token expired"


def test_share_token_json_excludes_pii(client, test_db):
    case_one, _, _ = _seed_cases(test_db)
    sensitive_doc = CaseDocument(
        case_id=case_one.id,
        document_type=DocumentType.ORDER,
        document_content="Order for Alice Johnson",
        summary="Alice Johnson can be reached at alice@example.com or 555-0100.",
        uploaded_at=datetime.now(timezone.utc),
    )
    test_db.add(sensitive_doc)
    test_db.commit()

    create_response = client.post(
        f"/api/v1/cases/{case_one.id}/share-anonymized",
        json={"scope": "full_party_removal", "expires_in_hours": 24},
    )
    token = create_response.json()["token"]

    response = client.get(f"/api/v1/cases/share/{token}")
    assert response.status_code == 200
    payload = response.json()

    assert payload["scope"] == "full_party_removal"
    assert payload["case_type"] == case_one.case_type
    assert payload["documents"]
    assert all(doc["summary"] is None for doc in payload["documents"])
    assert all(event["description"] is None or event["description"] == "" for event in payload["timeline"])
    raw_text = response.text
    assert "alice@example.com" not in raw_text
    assert "555-0100" not in raw_text
    assert case_one.case_number not in raw_text
