import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.base import Base
from db.models.analytics import ModelRoutingRule, ModelPerformance
from core.model_router import ModelRouter
from config import Config

@pytest.fixture()
def test_db(monkeypatch):
    """Set up an in-memory SQLite database for test runs."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    
    # Patch SessionLocal to act as the sessionmaker class
    monkeypatch.setattr("core.model_router.SessionLocal", Session)
    
    db = Session()
    yield db
    db.close()

def test_model_router_default_route(test_db):
    """Test that model router defaults to DEFAULT_MODEL when no rules exist."""
    router = ModelRouter()
    model, provider = router.route_task("summary")
    assert model == Config.DEFAULT_MODEL
    assert provider == "primary"

def test_model_router_with_custom_rule(test_db):
    """Test that model router picks custom rules from the database."""
    # Insert custom rule
    rule = ModelRoutingRule(
        name="Llama-Delhi-Civil",
        task="remedies",
        case_type="civil",
        jurisdiction="delhi",
        preferred_model="meta-llama/custom-llama-3",
        approved=True
    )
    test_db.add(rule)
    test_db.commit()

    router = ModelRouter()
    
    # Specific query matching filters
    model, provider = router.route_task("remedies", case_type="civil", jurisdiction="delhi")
    assert model == "meta-llama/custom-llama-3"
    assert provider == "primary"

    # Query not matching filters falls back
    model_fallback, _ = router.route_task("remedies", case_type="criminal", jurisdiction="mumbai")
    assert model_fallback == Config.DEFAULT_MODEL

def test_model_router_performance_logging(test_db):
    """Test that model performance is successfully written to database."""
    router = ModelRouter()
    router.log_performance("test-model", "summary", 250, 100)

    perf = test_db.query(ModelPerformance).filter(
        ModelPerformance.model_name == "test-model",
        ModelPerformance.task == "summary"
    ).first()

    assert perf is not None
    assert perf.samples == 1
    assert perf.average_latency_ms == 250

    # Add second sample to calculate average
    router.log_performance("test-model", "summary", 350, 150)
    test_db.refresh(perf)
    assert perf.samples == 2
    assert perf.average_latency_ms == 300

@patch("core.model_router.OpenAI")
def test_execute_call_fallback_flow(mock_openai_class, test_db):
    """Test that if the primary LLM call fails, the execution falls back to the secondary client."""
    # Setup mock clients
    mock_primary_client = MagicMock()
    mock_secondary_client = MagicMock()

    # Configure mock classes
    mock_openai_class.side_effect = [mock_primary_client, mock_secondary_client]

    # Primary client raises exception
    mock_primary_client.chat.completions.create.side_effect = Exception("OpenRouter API Down")
    
    # Secondary client returns successfully
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Fallback Summary Result"
    mock_response.usage = MagicMock()
    mock_response.usage.total_tokens = 50
    mock_secondary_client.chat.completions.create.return_value = mock_response

    router = ModelRouter()
    router.primary_client = mock_primary_client
    router.secondary_client = mock_secondary_client

    content, error = router.execute_call(
        task="summary",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        temperature=0.1,
        timeout=10.0
    )

    # Verify fallback executed and succeeded
    assert error is None
    assert content == "Fallback Summary Result"
    mock_primary_client.chat.completions.create.assert_called_once()
    mock_secondary_client.chat.completions.create.assert_called_once()
