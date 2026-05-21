import datetime as dt
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from db.models import (
    Case,
    CaseDocument,
    KnowledgeInvalidation,
    KnowledgeInvalidationStatus,
)


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _scope_value(scope_type: str, case_id: Optional[int], document_id: Optional[int], explicit_scope_value: Optional[str]) -> str:
    if explicit_scope_value:
        return explicit_scope_value
    if scope_type == "document" and document_id is not None:
        return f"document:{document_id}"
    if case_id is not None:
        return f"case:{case_id}"
    return scope_type


def record_knowledge_invalidation(
    db: Session,
    *,
    scope_type: str,
    reason: str,
    case_id: Optional[int] = None,
    document_id: Optional[int] = None,
    user_id: Optional[int] = None,
    scope_value: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    scheduled_for: Optional[dt.datetime] = None,
) -> KnowledgeInvalidation:
    invalidation = KnowledgeInvalidation(
        user_id=user_id,
        case_id=case_id,
        document_id=document_id,
        scope_type=scope_type,
        scope_value=_scope_value(scope_type, case_id, document_id, scope_value),
        reason=reason,
        details=details or {},
        status=KnowledgeInvalidationStatus.PENDING.value,
        invalidated_at=_utcnow(),
        scheduled_for=scheduled_for or _utcnow(),
    )
    db.add(invalidation)
    db.commit()
    db.refresh(invalidation)
    return invalidation


def list_knowledge_invalidations(
    db: Session,
    *,
    user_id: Optional[int] = None,
    case_id: Optional[int] = None,
    document_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> List[KnowledgeInvalidation]:
    query = db.query(KnowledgeInvalidation)

    if user_id is not None:
        query = query.filter(KnowledgeInvalidation.user_id == user_id)
    if case_id is not None:
        query = query.filter(KnowledgeInvalidation.case_id == case_id)
    if document_id is not None:
        query = query.filter(KnowledgeInvalidation.document_id == document_id)
    if status is not None:
        query = query.filter(KnowledgeInvalidation.status == status)

    return (
        query.order_by(KnowledgeInvalidation.invalidated_at.desc())
        .limit(limit)
        .all()
    )


def get_knowledge_freshness_summary(
    db: Session,
    *,
    user_id: Optional[int] = None,
    case_id: Optional[int] = None,
) -> Dict[str, Any]:
    query = db.query(KnowledgeInvalidation)
    if user_id is not None:
        query = query.filter(KnowledgeInvalidation.user_id == user_id)
    if case_id is not None:
        query = query.filter(KnowledgeInvalidation.case_id == case_id)

    rows = query.all()
    stale_rows = [row for row in rows if row.status != KnowledgeInvalidationStatus.COMPLETED.value]
    next_recompute_at = None
    if stale_rows:
        scheduled_times = [row.scheduled_for for row in stale_rows if row.scheduled_for]
        if scheduled_times:
            next_recompute_at = min(scheduled_times)

    latest = max(rows, key=lambda row: row.invalidated_at, default=None)
    return {
        "total": len(rows),
        "stale": len(stale_rows),
        "fresh": len(rows) - len(stale_rows),
        "next_recompute_at": next_recompute_at,
        "latest": latest,
    }


def _recompute_case_scope(db: Session, invalidation: KnowledgeInvalidation) -> bool:
    case_id = invalidation.case_id
    document_id = invalidation.document_id

    if case_id is None and document_id is None and invalidation.scope_value.startswith("case:"):
        try:
            case_id = int(invalidation.scope_value.split(":", 1)[1])
        except (TypeError, ValueError):
            case_id = None

    if document_id is not None and case_id is None:
        document = db.query(CaseDocument).filter(CaseDocument.id == document_id).first()
        if document:
            case_id = document.case_id

    if case_id is None:
        return True

    from core.embedding_engine import EmbeddingEngine

    engine = EmbeddingEngine()
    if document_id is not None:
        return engine.embed_case(db, case_id=case_id, document_id=document_id, force_regenerate=True) is not None

    return engine.embed_case(db, case_id=case_id, force_regenerate=True) is not None


def process_due_knowledge_invalidations(
    db: Session,
    *,
    now: Optional[dt.datetime] = None,
    limit: int = 20,
    recompute_handler: Optional[Callable[[Session, KnowledgeInvalidation], bool]] = None,
) -> List[KnowledgeInvalidation]:
    now = now or _utcnow()
    recompute_handler = recompute_handler or _recompute_case_scope

    pending = (
        db.query(KnowledgeInvalidation)
        .filter(
            KnowledgeInvalidation.status.in_(
                [
                    KnowledgeInvalidationStatus.PENDING.value,
                    KnowledgeInvalidationStatus.FAILED.value,
                ]
            ),
            KnowledgeInvalidation.scheduled_for <= now,
        )
        .order_by(KnowledgeInvalidation.scheduled_for.asc(), KnowledgeInvalidation.invalidated_at.asc())
        .limit(limit)
        .all()
    )

    processed: List[KnowledgeInvalidation] = []
    for invalidation in pending:
        invalidation.status = KnowledgeInvalidationStatus.PROCESSING.value
        invalidation.recompute_attempts += 1
        invalidation.recompute_started_at = now
        db.commit()
        db.refresh(invalidation)

        try:
            recomputed = recompute_handler(db, invalidation)
            if not recomputed:
                raise RuntimeError("Recompute handler returned no result")

            invalidation.status = KnowledgeInvalidationStatus.COMPLETED.value
            invalidation.recompute_completed_at = _utcnow()
            invalidation.error_message = None
        except Exception as exc:
            invalidation.status = KnowledgeInvalidationStatus.FAILED.value
            invalidation.error_message = str(exc)
        finally:
            db.commit()
            db.refresh(invalidation)
            processed.append(invalidation)

    return processed
