"""Tests for knowledge invalidation persistence and recompute tracking."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.base import Base
from db.models import Case, CaseStatus, DocumentType, KnowledgeInvalidationStatus, User
from db.case_service import create_case_document, update_case_document
from db.crud.knowledge import (
    list_knowledge_invalidations,
    process_due_knowledge_invalidations,
    record_knowledge_invalidation,
)


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()


def _create_user_and_case(db):
    user = User(id=1, email="user@example.com")
    db.add(user)
    db.commit()

    case = Case(
        user_id=1,
        case_number="CASE-100",
        case_type="civil",
        jurisdiction="Delhi",
        status=CaseStatus.ACTIVE,
        title="Test Case",
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return user, case


def test_create_case_document_records_document_created_invalidation(test_db):
    _user, case = _create_user_and_case(test_db)

    doc = create_case_document(
        test_db,
        case_id=case.id,
        document_type=DocumentType.JUDGMENT,
        user_id=1,
        document_content="Initial judgment text",
        summary="Initial summary",
    )

    rows = list_knowledge_invalidations(test_db, case_id=case.id)
    assert len(rows) == 1

    invalidation = rows[0]
    assert invalidation.case_id == case.id
    assert invalidation.document_id == doc.id
    assert invalidation.reason == "document_created"
    assert invalidation.scope_type == "case"
    assert invalidation.scope_value == f"case:{case.id}"
    assert invalidation.status == KnowledgeInvalidationStatus.PENDING.value
    assert invalidation.details["document_type"] == DocumentType.JUDGMENT.value
    assert "changed_fields" in invalidation.details


def test_update_case_document_tracks_reason_and_changed_fields(test_db):
    _user, case = _create_user_and_case(test_db)

    doc = create_case_document(
        test_db,
        case_id=case.id,
        document_type=DocumentType.JUDGMENT,
        user_id=1,
        document_content="Initial judgment text",
        summary="Initial summary",
    )

    updated = update_case_document(
        test_db,
        document_id=doc.id,
        summary="Updated summary",
        ocr_used=True,
    )

    assert updated is not None

    rows = list_knowledge_invalidations(test_db, case_id=case.id)
    assert len(rows) == 2

    # Find the update invalidation regardless of ordering (SQLite may return
    # both rows with the same invalidated_at timestamp in tests).
    update_row = next((r for r in rows if r.reason == "summary_updated"), None)
    assert update_row is not None, "Expected an invalidation with reason='summary_updated'"
    assert update_row.details["changed_fields"] == ["summary", "ocr_used"]
    assert update_row.document_id == doc.id
    assert update_row.status == KnowledgeInvalidationStatus.PENDING.value


def test_due_knowledge_invalidations_are_processed(test_db):
    _user, case = _create_user_and_case(test_db)

    record_knowledge_invalidation(
        test_db,
        scope_type="case",
        case_id=case.id,
        user_id=1,
        reason="manual_refresh",
        scheduled_for=datetime.now(timezone.utc) - timedelta(minutes=5),
        details={"changed_fields": ["summary"]},
    )

    processed = process_due_knowledge_invalidations(
        test_db,
        recompute_handler=lambda db, invalidation: True,
    )

    assert len(processed) == 1
    refreshed = list_knowledge_invalidations(test_db, case_id=case.id)[0]
    assert refreshed.status == KnowledgeInvalidationStatus.COMPLETED.value
    assert refreshed.recompute_attempts == 1
    assert refreshed.recompute_started_at is not None
    assert refreshed.recompute_completed_at is not None

def test_has_pending_invalidations_true_after_document_created(test_db):
    """has_pending_invalidations returns True immediately after a document is created."""
    from db.crud.knowledge import has_pending_invalidations

    _user, case = _create_user_and_case(test_db)

    create_case_document(
        test_db,
        case_id=case.id,
        document_type=DocumentType.JUDGMENT,
        user_id=1,
        document_content="Judgment text",
        summary="Summary",
    )

    assert has_pending_invalidations(test_db, case_id=case.id) is True


def test_has_pending_invalidations_false_after_recompute(test_db):
    """has_pending_invalidations returns False once all invalidations are COMPLETED."""
    from db.crud.knowledge import has_pending_invalidations

    _user, case = _create_user_and_case(test_db)

    record_knowledge_invalidation(
        test_db,
        scope_type="case",
        case_id=case.id,
        user_id=1,
        reason="document_content_updated",
        scheduled_for=datetime.now(timezone.utc) - timedelta(minutes=1),
        details={"changed_fields": ["document_content"]},
    )

    # Before recompute: stale
    assert has_pending_invalidations(test_db, case_id=case.id) is True

    # Scheduler processes the invalidation
    process_due_knowledge_invalidations(
        test_db,
        recompute_handler=lambda db, inv: True,
    )

    # After recompute: fresh
    assert has_pending_invalidations(test_db, case_id=case.id) is False


def test_get_latest_case_document_text_returns_most_recent(test_db):
    """get_latest_case_document_text returns the content of the most recently uploaded doc."""
    from db.crud.knowledge import get_latest_case_document_text

    _user, case = _create_user_and_case(test_db)

    create_case_document(
        test_db,
        case_id=case.id,
        document_type=DocumentType.JUDGMENT,
        user_id=1,
        document_content="First judgment text",
        summary="First summary",
    )

    create_case_document(
        test_db,
        case_id=case.id,
        document_type=DocumentType.JUDGMENT,
        user_id=1,
        document_content="Second judgment text",
        summary="Second summary",
    )

    text = get_latest_case_document_text(test_db, case_id=case.id)
    assert text == "Second judgment text"


def test_get_latest_case_document_text_returns_none_for_unknown_case(test_db):
    """get_latest_case_document_text returns None when the case has no documents."""
    from db.crud.knowledge import get_latest_case_document_text

    _user, _case = _create_user_and_case(test_db)

    text = get_latest_case_document_text(test_db, case_id=9999)
    assert text is None


def test_scheduler_recompute_clears_stale_flag_for_chat(test_db):
    """
    Full integration: document update → pending invalidation → scheduler recompute
    → no more pending invalidations → chat page sees fresh knowledge.
    """
    from db.crud.knowledge import has_pending_invalidations

    _user, case = _create_user_and_case(test_db)

    # Step 1: create document (triggers invalidation)
    doc = create_case_document(
        test_db,
        case_id=case.id,
        document_type=DocumentType.JUDGMENT,
        user_id=1,
        document_content="Original judgment",
        summary="Original summary",
    )

    # Step 2: update document (triggers another invalidation)
    update_case_document(
        test_db,
        document_id=doc.id,
        document_content="Updated judgment content",
    )

    # Chat page: knowledge is stale
    assert has_pending_invalidations(test_db, case_id=case.id) is True

    # Step 3: scheduler processes all due invalidations
    processed = process_due_knowledge_invalidations(
        test_db,
        recompute_handler=lambda db, inv: True,
    )
    assert len(processed) == 2

    # Chat page: knowledge is now fresh
    assert has_pending_invalidations(test_db, case_id=case.id) is False
