"""Batched query helpers for case data.

These functions perform efficient batch queries using subqueries and CTEs
to avoid N+1 problems when fetching related data for multiple cases.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

from sqlalchemy import func, and_
from sqlalchemy.orm import Session

from db.models import Case, CaseDocument, CaseDeadline, CaseTimeline, Attachment


def fetch_latest_documents_per_case(db: Session, case_ids: List[int]) -> Dict[int, CaseDocument]:
    """
    Fetch the latest document for each case in batch.
    
    Returns a dict mapping case_id -> latest CaseDocument (or None if no documents).
    
    Uses a subquery to avoid N+1 queries.
    """
    if not case_ids:
        return {}
    
    # Rank documents per case in SQL so the latest row can be selected without
    # Python-side sorting. Tie-break on id for deterministic results.
    ranked_docs = (
        db.query(
            CaseDocument.id.label("doc_id"),
            CaseDocument.case_id.label("case_id"),
            func.row_number()
            .over(
                partition_by=CaseDocument.case_id,
                order_by=(CaseDocument.uploaded_at.desc(), CaseDocument.id.desc()),
            )
            .label("row_number"),
        )
        .filter(CaseDocument.case_id.in_(case_ids))
        .subquery()
    )
    
    latest_docs = (
        db.query(CaseDocument)
        .join(ranked_docs, CaseDocument.id == ranked_docs.c.doc_id)
        .filter(ranked_docs.c.row_number == 1)
        .all()
    )
    
    # Map to dict
    result: Dict[int, CaseDocument] = {}
    for doc in latest_docs:
        result[doc.case_id] = doc
    
    return result


def fetch_next_deadlines_per_case(db: Session, case_ids: List[int]) -> Dict[int, CaseDeadline]:
    """
    Fetch the next upcoming deadline for each case in batch.
    
    Returns a dict mapping case_id -> next CaseDeadline (or None if no upcoming deadlines).
    
    Uses a subquery to find the minimum deadline_date per case that hasn't passed and isn't completed.
    """
    if not case_ids:
        return {}
    
    now = datetime.now(timezone.utc)
    
    # Rank deadlines per case in SQL so the next upcoming row can be selected without
    # Python-side sorting or duplicate join matches on date ties. Tie-break on id.
    ranked_deadlines = (
        db.query(
            CaseDeadline.id.label("deadline_id"),
            CaseDeadline.case_id.label("case_id"),
            func.row_number()
            .over(
                partition_by=CaseDeadline.case_id,
                order_by=(CaseDeadline.deadline_date.asc(), CaseDeadline.id.asc()),
            )
            .label("row_number"),
        )
        .filter(
            CaseDeadline.case_id.in_(case_ids),
            CaseDeadline.is_completed.is_(False),
            CaseDeadline.deadline_date > now,
        )
        .subquery()
    )
    
    # Join to get the actual deadline records
    next_deadlines = (
        db.query(CaseDeadline)
        .join(ranked_deadlines, CaseDeadline.id == ranked_deadlines.c.deadline_id)
        .filter(ranked_deadlines.c.row_number == 1)
        .all()
    )
    
    # Map to dict
    result: Dict[int, CaseDeadline] = {}
    for deadline in next_deadlines:
        result[deadline.case_id] = deadline
    
    return result


def fetch_document_counts_per_case(db: Session, case_ids: List[int]) -> Dict[int, int]:
    """
    Fetch document count for each case in batch.
    
    Returns a dict mapping case_id -> document count.
    
    Uses a single query with aggregation to avoid N+1 queries.
    """
    if not case_ids:
        return {}
    
    counts = (
        db.query(
            CaseDocument.case_id,
            func.count(CaseDocument.id).label("doc_count"),
        )
        .filter(CaseDocument.case_id.in_(case_ids))
        .group_by(CaseDocument.case_id)
        .all()
    )
    
    result: Dict[int, int] = {}
    for case_id, count in counts:
        result[case_id] = count
    
    # Fill in zeros for cases with no documents
    for case_id in case_ids:
        if case_id not in result:
            result[case_id] = 0
    
    return result


def fetch_all_documents_per_case(db: Session, case_ids: List[int]) -> Dict[int, List[CaseDocument]]:
    """
    Fetch all documents for each case in batch.
    
    Returns a dict mapping case_id -> list of CaseDocuments (sorted by uploaded_at desc).
    
    Uses a single query instead of N queries.
    """
    if not case_ids:
        return {}
    
    docs = (
        db.query(CaseDocument)
        .filter(CaseDocument.case_id.in_(case_ids))
        .order_by(CaseDocument.case_id, CaseDocument.uploaded_at.desc())
        .all()
    )
    
    result: Dict[int, List[CaseDocument]] = {case_id: [] for case_id in case_ids}
    for doc in docs:
        result[doc.case_id].append(doc)
    
    return result


def fetch_all_deadlines_per_case(db: Session, case_ids: List[int]) -> Dict[int, List[CaseDeadline]]:
    """
    Fetch all deadlines for each case in batch.
    
    Returns a dict mapping case_id -> list of CaseDeadlines (sorted by deadline_date).
    
    Uses a single query instead of N queries.
    """
    if not case_ids:
        return {}
    
    deadlines = (
        db.query(CaseDeadline)
        .filter(CaseDeadline.case_id.in_(case_ids))
        .order_by(CaseDeadline.case_id, CaseDeadline.deadline_date)
        .all()
    )
    
    result: Dict[int, List[CaseDeadline]] = {case_id: [] for case_id in case_ids}
    for deadline in deadlines:
        result[deadline.case_id].append(deadline)
    
    return result


def fetch_all_timeline_per_case(db: Session, case_ids: List[int]) -> Dict[int, List[CaseTimeline]]:
    """
    Fetch all timeline events for each case in batch.
    
    Returns a dict mapping case_id -> list of CaseTimelines (sorted by event_date).
    
    Uses a single query instead of N queries.
    """
    if not case_ids:
        return {}
    
    events = (
        db.query(CaseTimeline)
        .filter(CaseTimeline.case_id.in_(case_ids))
        .order_by(CaseTimeline.case_id, CaseTimeline.event_date)
        .all()
    )
    
    result: Dict[int, List[CaseTimeline]] = {case_id: [] for case_id in case_ids}
    for event in events:
        result[event.case_id].append(event)
    
    return result


def fetch_all_attachments_per_case(db: Session, case_ids: List[int]) -> Dict[int, List[Attachment]]:
    """
    Fetch all attachments for each case in batch.
    
    Returns a dict mapping case_id -> list of Attachments.
    
    Uses a single query instead of N queries.
    """
    if not case_ids:
        return {}
    
    attachments = (
        db.query(Attachment)
        .filter(Attachment.case_id.in_(case_ids))
        .order_by(Attachment.case_id)
        .all()
    )
    
    result: Dict[int, List[Attachment]] = {case_id: [] for case_id in case_ids}
    for attachment in attachments:
        result[attachment.case_id].append(attachment)
    
    return result


def fetch_case_summary_data_batch(
    db: Session,
    case_ids: List[int],
) -> Tuple[Dict[int, CaseDocument], Dict[int, CaseDeadline], Dict[int, int]]:
    """
    Fetch all data needed for case summaries in batch.
    
    Returns:
        (latest_docs_per_case, next_deadlines_per_case, doc_counts_per_case)
    
    This is a convenience function that combines the three most common queries
    for building case summary lists. Performs 3 queries instead of 3*N queries.
    """
    latest_docs = fetch_latest_documents_per_case(db, case_ids)
    next_deadlines = fetch_next_deadlines_per_case(db, case_ids)
    doc_counts = fetch_document_counts_per_case(db, case_ids)
    
    return latest_docs, next_deadlines, doc_counts


def fetch_case_detail_data_batch(
    db: Session,
    case_id: int,
) -> Tuple[List[CaseDocument], List[CaseDeadline], List[CaseTimeline], List[Attachment]]:
    """
    Fetch all data needed for case detail view.
    
    Returns:
        (documents, deadlines, timeline, attachments)
    
    Performs 4 queries total for a single case.
    """
    documents = (
        db.query(CaseDocument)
        .filter(CaseDocument.case_id == case_id)
        .order_by(CaseDocument.uploaded_at.desc())
        .all()
    )
    
    deadlines = (
        db.query(CaseDeadline)
        .filter(CaseDeadline.case_id == case_id)
        .order_by(CaseDeadline.deadline_date)
        .all()
    )
    
    timeline = (
        db.query(CaseTimeline)
        .filter(CaseTimeline.case_id == case_id)
        .order_by(CaseTimeline.event_date)
        .all()
    )
    
    attachments = (
        db.query(Attachment)
        .filter(Attachment.case_id == case_id)
        .all()
    )
    
    return documents, deadlines, timeline, attachments
