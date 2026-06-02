from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import api.routes.models as models_route
from api.auth import CurrentUser, get_current_user
from database import Base, ModelFeedback


@pytest.fixture()
def test_db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    yield db
    db.close()


@pytest.fixture()
def client(test_db, monkeypatch):
    app = FastAPI()
    app.include_router(models_route.router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(1, "u1@example.com", "user")
    monkeypatch.setattr(models_route, "get_db", lambda: test_db)
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_submit_model_feedback_and_aggregate(client, test_db):
    # Submit two feedback rows for two models
    payload1 = {
        "model_name": "model-A",
        "task": "summary",
        "case_id": None,
        "is_accurate": True,
        "corrected_text": None,
    }
    payload2 = {
        "model_name": "model-B",
        "task": "summary",
        "case_id": None,
        "is_accurate": False,
        "corrected_text": "Better wording",
    }

    r1 = client.post("/api/v1/models/feedback", json=payload1)
    r2 = client.post("/api/v1/models/feedback", json=payload2)
    assert r1.status_code == 200 and r2.status_code == 200

    perf = client.get("/api/v1/models/performance").json()
    items = perf.get("items", [])
    assert isinstance(items, list)
    # We have two model entries recorded
    names = {it["model_name"] for it in items}
    assert "model-A" in names and "model-B" in names
