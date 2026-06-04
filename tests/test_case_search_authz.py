import os

os.environ["JWT_SECRET_KEY"] = "test-secret-key-value-12345"
os.environ["APP_ALLOWED_HOSTS"] = "localhost"

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import api.routes.case_search as case_search_route
from api.auth import CurrentUser, get_current_user
from database import Base, Case, CaseStatus


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
    app.include_router(case_search_route.router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser("42", "tester@example.com", "user")
    app.dependency_overrides[case_search_route.get_db] = lambda: test_db
    return TestClient(app)


def _seed_case(test_db, *, user_id: int):
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
    return case


def test_search_similar_rejects_other_users_case(client, test_db, monkeypatch):
    case = _seed_case(test_db, user_id=99)

    monkeypatch.setattr(
        case_search_route.SemanticCaseSearch,
        "search_similar_cases",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("search should not run for unauthorized case")),
    )

    response = client.get(f"/api/cases/{case.id}/search-similar")

    assert response.status_code == 404
    assert response.json()["detail"] == "Case not found"