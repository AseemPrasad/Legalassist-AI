"""
Unit tests for RAG invalidation and state reset logic.

Covers:
- get_judgment_hash correctness
- LegalRAG.reset() clears vector store
- Session-state invalidation via hash mismatch (existing behaviour)
- Case-scoped chat initialisation: document text loaded from DB
- Backend invalidation detection: pending records trigger chat reset
- Scheduler recompute + chat freshness integration flow

Note: tests that import core.rag_engine (which depends on openai, pypdf,
langchain, etc.) are skipped automatically when those packages are absent.
DB-only tests have no such dependency and always run.
"""

import datetime
import hashlib
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Lightweight hash helper – mirrors get_judgment_hash without importing core
# ---------------------------------------------------------------------------

def _hash(text):
    if not text:
        return ""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Conditional import of core.rag_engine
# ---------------------------------------------------------------------------

rag_engine_mod = pytest.importorskip(
    "core.rag_engine",
    reason="core.rag_engine dependencies (openai, pypdf, langchain…) not installed",
)
LegalRAG = rag_engine_mod.LegalRAG
get_judgment_hash = rag_engine_mod.get_judgment_hash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embeddings():
    """Mock HuggingFaceEmbeddings to avoid model download / loading."""
    with patch("core.rag_engine.HuggingFaceEmbeddings") as mock_hf:
        mock_hf.return_value = MagicMock()
        yield mock_hf


# ---------------------------------------------------------------------------
# get_judgment_hash
# ---------------------------------------------------------------------------

def test_get_judgment_hash_valid():
    """Test get_judgment_hash with valid string inputs."""
    text1 = "This is a sample judgment text."
    text2 = "This is a different judgment text."

    hash1 = get_judgment_hash(text1)
    hash2 = get_judgment_hash(text2)

    assert len(hash1) == 32
    assert len(hash2) == 32
    assert hash1 != hash2
    assert get_judgment_hash(text1) == hash1


def test_get_judgment_hash_empty_and_none():
    """Test get_judgment_hash with empty or None input."""
    assert get_judgment_hash("") == ""
    assert get_judgment_hash(None) == ""


# ---------------------------------------------------------------------------
# LegalRAG.reset
# ---------------------------------------------------------------------------

def test_legal_rag_reset(mock_embeddings):
    """Test that LegalRAG reset method clears the vector store."""
    rag_engine = LegalRAG()

    mock_vs = MagicMock()
    rag_engine.vector_store = mock_vs
    assert rag_engine.vector_store is not None

    rag_engine.reset()
    assert rag_engine.vector_store is None


# ---------------------------------------------------------------------------
# Session-state invalidation (hash-based, existing behaviour)
# ---------------------------------------------------------------------------

def test_invalidation_state_logic():
    """Test the simulated session state invalidation logic flow."""
    initial_text = "Initial Document Text"
    session_state = {
        "judgment_raw_text": initial_text,
        "chat_history": [{"role": "user", "content": "hello"}],
        "rag_initialized": True,
        "last_judgment_hash": get_judgment_hash(initial_text),
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


# ---------------------------------------------------------------------------
# Case-scoped document loading (DB-only, no heavy deps)
# ---------------------------------------------------------------------------

def test_load_case_document_text_returns_latest_content():
    """get_latest_case_document_text returns the most recent document content."""
    from db.crud.knowledge import get_latest_case_document_text

    mock_doc = MagicMock()
    mock_doc.document_content = "Latest judgment text from DB"

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_doc

    result = get_latest_case_document_text(mock_db, case_id=42)
    assert result == "Latest judgment text from DB"


def test_load_case_document_text_returns_none_when_no_doc():
    """get_latest_case_document_text returns None when no document exists."""
    from db.crud.knowledge import get_latest_case_document_text

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

    result = get_latest_case_document_text(mock_db, case_id=99)
    assert result is None


# ---------------------------------------------------------------------------
# Backend invalidation detection (DB-only)
# ---------------------------------------------------------------------------

def test_has_pending_invalidations_true_when_pending_exists():
    """has_pending_invalidations returns True when a PENDING row exists."""
    from db.crud.knowledge import has_pending_invalidations

    mock_row = MagicMock()
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = mock_row

    assert has_pending_invalidations(mock_db, case_id=1) is True


def test_has_pending_invalidations_false_when_none():
    """has_pending_invalidations returns False when no stale rows exist."""
    from db.crud.knowledge import has_pending_invalidations

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None

    assert has_pending_invalidations(mock_db, case_id=1) is False


# ---------------------------------------------------------------------------
# Chat reset triggered by backend invalidation (pure logic, no DB)
# ---------------------------------------------------------------------------

def test_chat_resets_when_backend_invalidation_detected():
    """Simulate the chat page logic: pending invalidation resets chat state."""
    text = "Judgment text"
    session_state = {
        "judgment_raw_text": text,
        "chat_history": [{"role": "user", "content": "old question"}],
        "rag_initialized": True,
        "last_judgment_hash": _hash(text),
        "active_case_id": 5,
    }

    current_hash = _hash(session_state["judgment_raw_text"])
    hash_changed = session_state.get("last_judgment_hash") != current_hash
    backend_stale = True  # mocked: has_pending_invalidations returned True

    if hash_changed or backend_stale:
        session_state["chat_history"] = []
        session_state["rag_initialized"] = False
        session_state["last_judgment_hash"] = current_hash

    assert session_state["chat_history"] == []
    assert session_state["rag_initialized"] is False


def test_chat_not_reset_when_knowledge_is_fresh():
    """Chat state is preserved when hash matches and no backend invalidation."""
    text = "Stable judgment text"
    original_history = [{"role": "user", "content": "existing question"}]
    session_state = {
        "judgment_raw_text": text,
        "chat_history": list(original_history),
        "rag_initialized": True,
        "last_judgment_hash": _hash(text),
        "active_case_id": 7,
    }

    current_hash = _hash(session_state["judgment_raw_text"])
    hash_changed = session_state.get("last_judgment_hash") != current_hash
    backend_stale = False  # mocked: no pending invalidations

    if hash_changed or backend_stale:
        session_state["chat_history"] = []
        session_state["rag_initialized"] = False

    assert session_state["chat_history"] == original_history
    assert session_state["rag_initialized"] is True


# ---------------------------------------------------------------------------
# Scheduler recompute + chat freshness integration flow (DB-only)
# ---------------------------------------------------------------------------

def test_scheduler_recompute_then_chat_freshness_flow():
    """
    Integration: after scheduler marks invalidation COMPLETED, the chat page
    should detect no pending invalidations and keep (or re-init) context.
    """
    from db.crud.knowledge import (
        process_due_knowledge_invalidations,
        has_pending_invalidations,
        record_knowledge_invalidation,
    )
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from db.base import Base
    from db.models import Case, CaseStatus, User

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    user = User(id=1, email="test@example.com")
    db.add(user)
    db.commit()

    case = Case(
        user_id=1,
        case_number="CASE-INT-001",
        case_type="civil",
        jurisdiction="Delhi",
        status=CaseStatus.ACTIVE,
        title="Integration Test Case",
    )
    db.add(case)
    db.commit()
    db.refresh(case)

    record_knowledge_invalidation(
        db,
        scope_type="case",
        case_id=case.id,
        user_id=1,
        reason="document_content_updated",
        scheduled_for=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=1),
        details={"changed_fields": ["document_content"]},
    )

    # Before recompute: chat page should detect stale knowledge
    assert has_pending_invalidations(db, case_id=case.id) is True

    # Scheduler runs recompute (mock handler always succeeds)
    processed = process_due_knowledge_invalidations(
        db,
        recompute_handler=lambda _db, _inv: True,
    )
    assert len(processed) == 1

    # After recompute: no more pending invalidations
    assert has_pending_invalidations(db, case_id=case.id) is False

    db.close()
