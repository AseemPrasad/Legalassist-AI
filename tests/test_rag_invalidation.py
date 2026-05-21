"""
Unit tests for RAG invalidation and state reset logic.
"""

import pytest
from unittest.mock import MagicMock, patch

from core.rag_engine import LegalRAG, get_judgment_hash


@pytest.fixture
def mock_embeddings():
    """Mock HuggingFaceEmbeddings to avoid model download / loading."""
    with patch("core.rag_engine.HuggingFaceEmbeddings") as mock_hf:
        mock_hf.return_value = MagicMock()
        yield mock_hf


def test_get_judgment_hash_valid():
    """Test get_judgment_hash with valid string inputs."""
    text1 = "This is a sample judgment text."
    text2 = "This is a different judgment text."
    
    hash1 = get_judgment_hash(text1)
    hash2 = get_judgment_hash(text2)
    
    assert len(hash1) == 32
    assert len(hash2) == 32
    assert hash1 != hash2
    # Ensure same input yields same hash
    assert get_judgment_hash(text1) == hash1


def test_get_judgment_hash_empty_and_none():
    """Test get_judgment_hash with empty or None input."""
    assert get_judgment_hash("") == ""
    assert get_judgment_hash(None) == ""


def test_legal_rag_reset(mock_embeddings):
    """Test that LegalRAG reset method clears the vector store."""
    rag_engine = LegalRAG()
    
    # Manually assign a mock vector store
    mock_vs = MagicMock()
    rag_engine.vector_store = mock_vs
    assert rag_engine.vector_store is not None
    
    # Reset
    rag_engine.reset()
    assert rag_engine.vector_store is None


def test_invalidation_state_logic():
    """Test the simulated session state invalidation logic flow."""
    # Simulation of st.session_state
    session_state = {
        "judgment_raw_text": "Initial Document Text",
        "chat_history": [{"role": "user", "content": "hello"}],
        "rag_initialized": True,
        "last_judgment_hash": get_judgment_hash("Initial Document Text")
    }
    
    # 1. Simulate new document upload
    session_state["judgment_raw_text"] = "New Document Text"
    
    # 2. Run invalidation helper logic (simulating pages/4_Chat.py logic)
    current_hash = get_judgment_hash(session_state["judgment_raw_text"])
    if session_state.get("last_judgment_hash") != current_hash:
        session_state["chat_history"] = []
        session_state["rag_initialized"] = False
        session_state["last_judgment_hash"] = current_hash
        
    # 3. Assert states are correctly reset/invalidated
    assert session_state["chat_history"] == []
    assert session_state["rag_initialized"] is False
    assert session_state["last_judgment_hash"] == current_hash
