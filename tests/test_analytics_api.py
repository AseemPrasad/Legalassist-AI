import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.analytics import router as analytics_router
from api.auth import CurrentUser, get_current_user


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(analytics_router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(
        user_id="test-user-123",
        email="test@example.com",
        role="user"
    )
    yield TestClient(app)


def test_get_cost_breakdown_correctness(client):
    """
    Test that the cost breakdown does not double count document processing costs.
    Specifically, total_cost should be exactly equal to the sum of llm_api_cost,
    document_processing_cost, and storage_cost.
    """
    response = client.get("/api/v1/analytics/costs?period=monthly")
    assert response.status_code == 200
    
    payload = response.json()
    assert payload["user_id"] == "test-user-123"
    
    cost_breakdown = payload["cost_breakdown"]
    assert cost_breakdown["period"] == "monthly"
    
    llm_api_cost = cost_breakdown["llm_api_cost"]
    doc_proc_cost = cost_breakdown["document_processing_cost"]
    storage_cost = cost_breakdown["storage_cost"]
    total_cost = cost_breakdown["total_cost"]
    
    # Assert specific non-overlapping mock values
    assert llm_api_cost == 39.50
    assert doc_proc_cost == 35.50
    assert storage_cost == 15.00
    
    # Assert that the total cost is mathematically correct and does not double-count
    expected_total = llm_api_cost + doc_proc_cost + storage_cost
    assert total_cost == expected_total
    assert total_cost == 90.00
