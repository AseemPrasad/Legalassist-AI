from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.routes.cases as cases_route
from api.auth import CurrentUser, get_current_user


def test_case_timeline_response_matches_model(monkeypatch):
    app = FastAPI()
    app.include_router(cases_route.router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser("42", "tester@example.com", "user")

    monkeypatch.setattr(cases_route, "get_db", lambda: None)

    client = TestClient(app)
    response = client.get("/api/v1/cases/CASE-123/timeline")

    assert response.status_code == 200
    payload = response.json()

    assert payload["case_id"] == "CASE-123"
    assert payload["case_number"] == "2023-CV-00001"
    assert payload["title"] == "Example Case"
    assert payload["status"] == "closed"
    assert payload["total_events"] == 5
    assert payload["duration_years"] == 1.0
    assert len(payload["events"]) == 5
    assert all("date" in event and "event_type" in event and "description" in event for event in payload["events"])