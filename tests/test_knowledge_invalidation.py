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

    latest = rows[0]
    assert latest.reason == "summary_updated"
    assert latest.details["changed_fields"] == ["summary", "ocr_used"]
    assert latest.document_id == doc.id
    assert latest.status == KnowledgeInvalidationStatus.PENDING.value


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
