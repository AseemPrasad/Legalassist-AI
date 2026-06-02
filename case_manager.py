"""
Case Management Service for LegalAssist AI.
CRUD operations for cases, documents, and timeline events.
"""

import re
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
import logging
import hashlib
import hmac
import os
from pathlib import Path


from sqlalchemy.orm import Session

from database import (
    SessionLocal,
    Case,
    CaseDocument,
    CaseTimeline,
    CaseDeadline,
    CaseStatus,
    DocumentType,
    UserPreference,
    create_case,
    get_user_cases,
    get_case_by_id,
    get_case_documents,
    get_case_timeline,
    create_case_document,
    create_timeline_event,
    update_case_status,
    create_attachment,
    get_attachments_for_case,
)

logger = logging.getLogger(__name__)


# ==================== Case Management ====================


def create_new_case(
    user_id: int,
    case_number: str,
    case_type: str,
    jurisdiction: str,
    title: Optional[str] = None,
) -> Optional[Case]:
    """
    Create a new case for a user.
    Returns the created Case object or None if failed.
    """
    db = SessionLocal()
    try:
        # Normalize inputs by trimming whitespace
        case_number = case_number.strip()
        jurisdiction = jurisdiction.strip()
        case_type = case_type.strip()
        if title:
            title = title.strip()
        
        # Check if case number already exists for this user
        existing = db.query(Case).filter(
            Case.user_id == user_id,
            Case.case_number == case_number,
        ).first()

        if existing:
            logger.warning(f"Case {case_number} already exists for user {user_id}")
            return existing

        case = create_case(
            db=db,
            user_id=user_id,
            case_number=case_number,
            case_type=case_type,
            jurisdiction=jurisdiction,
            title=title,
        )

        # Create timeline event for case creation
        create_timeline_event(
            db=db,
            case_id=case.id,
            event_type="case_created",
            description=f"Case {case_number} created",
            metadata={"case_type": case_type, "jurisdiction": jurisdiction},
        )

        db.refresh(case)
        logger.info(f"Created new case: {case_number} for user {user_id}")
        return case

    except Exception as e:
        logger.error(f"Error creating case: {str(e)}")
        return None
    finally:
        db.close()


def get_or_create_case_for_document(
    user_id: int,
    existing_case_id: Optional[int] = None,
    new_case_number: Optional[str] = None,
    new_case_type: Optional[str] = None,
    new_jurisdiction: Optional[str] = None,
    new_title: Optional[str] = None,
) -> Optional[Case]:
    """
    Get existing case or create new one for document upload.

    The returned Case object is expunged from the session before the session
    is closed, so all already-loaded scalar attributes remain accessible to
    the caller without raising DetachedInstanceError.
    """
    db = SessionLocal()
    try:
        if existing_case_id:
            case = get_case_by_id(db, existing_case_id)
            if case and case.user_id == user_id:
                db.expunge(case)
                return case

        # Create new case — create_new_case manages its own session internally,
        # so the object it returns is already detached. No expunge needed here.
        if new_case_number:
            case = create_new_case(
                user_id=user_id,
                case_number=new_case_number,
                case_type=new_case_type or "general",
                jurisdiction=new_jurisdiction or "Unknown",
                title=new_title,
            )
            return case

        return None

    finally:
        db.close()


def get_user_cases_summary(user_id: int, include_closed: bool = True) -> List[Dict[str, Any]]:
    """
    Get summary of all cases for a user.
    Returns list of case summaries with latest document info.
    """
    db = SessionLocal()
    try:
        cases = get_user_cases(db, user_id, include_closed=include_closed)
        summaries = []

        for case in cases:
            # Get latest document
            latest_doc = db.query(CaseDocument).filter(
                CaseDocument.case_id == case.id
            ).order_by(CaseDocument.uploaded_at.desc()).first()

            # Get next deadline
            next_deadline = db.query(CaseDeadline).filter(
                CaseDeadline.case_id == case.id,
                CaseDeadline.is_completed == False,
                CaseDeadline.deadline_date > datetime.now(timezone.utc),
            ).order_by(CaseDeadline.deadline_date).first()

            # Get document count
            doc_count = db.query(CaseDocument).filter(
                CaseDocument.case_id == case.id
            ).count()

            summaries.append({
                "id": case.id,
                "case_number": case.case_number,
                "title": case.title or case.case_number,
                "case_type": case.case_type,
                "jurisdiction": case.jurisdiction,
                "status": case.status.value,
                "created_at": case.created_at.isoformat(),
                "latest_document_type": latest_doc.document_type.value if latest_doc else None,
                "latest_document_date": latest_doc.uploaded_at.isoformat() if latest_doc else None,
                "next_deadline_date": next_deadline.deadline_date.isoformat() if next_deadline else None,
                "next_deadline_type": next_deadline.deadline_type if next_deadline else None,
                "days_until_deadline": next_deadline.days_until_deadline() if next_deadline else None,
                "document_count": doc_count,
            })

        return summaries

    except Exception as e:
        logger.error(f"Error getting user cases summary: {str(e)}")
        return []
    finally:
        db.close()


def get_case_detail(user_id: int, case_id: int) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific case.
    """
    db = SessionLocal()
    try:
        case = get_case_by_id(db, case_id)

        if not case or case.user_id != user_id:
            return None

        # Get all documents
        documents = get_case_documents(db, case_id)
        docs_list = [
            {
                "id": doc.id,
                "document_type": doc.document_type.value,
                "uploaded_at": doc.uploaded_at.isoformat(),
                "summary": doc.summary,
                "has_remedies": bool(doc.remedies),
            }
            for doc in documents
        ]

        # Get attachments
        attachments = get_attachments_for_case(db, case_id)
        attachments_list = [
            {
                "id": a.id,
                "original_filename": a.original_filename,
                "uploaded_at": a.uploaded_at.isoformat(),
                "size_bytes": a.size_bytes,
                "content_type": a.content_type,
            }
            for a in attachments
        ]

        # Get timeline
        timeline = get_case_timeline(db, case_id)
        timeline_list = [
            {
                "id": event.id,
                "event_type": event.event_type,
                "event_date": event.event_date.isoformat(),
                "description": event.description,
                "metadata": event.event_metadata,
            }
            for event in timeline
        ]

        # Get deadlines
        deadlines = db.query(CaseDeadline).filter(
            CaseDeadline.case_id == case_id
        ).order_by(CaseDeadline.deadline_date).all()

        deadlines_list = [
            {
                "id": d.id,
                "deadline_type": d.deadline_type,
                "deadline_date": d.deadline_date.isoformat(),
                "description": d.description,
                "is_completed": d.is_completed,
                "days_until": d.days_until_deadline(),
            }
            for d in deadlines
        ]

        # Get latest remedies from most recent document
        latest_doc = documents[-1] if documents else None
        remedies = latest_doc.remedies if latest_doc else None

        return {
            "case": {
                "id": case.id,
                "case_number": case.case_number,
                "title": case.title,
                "case_type": case.case_type,
                "jurisdiction": case.jurisdiction,
                "status": case.status.value,
                "created_at": case.created_at.isoformat(),
            },
            "documents": docs_list,
            "timeline": timeline_list,
            "deadlines": deadlines_list,
            "remedies": remedies,
            "attachments": attachments_list,
        }

    except Exception as e:
        logger.error(f"Error getting case detail: {str(e)}")
        return None
    finally:
        db.close()


# ==================== Document Management ====================


def upload_case_document(
    user_id: int,
    case_id: int,
    document_type: DocumentType,
    document_content: str,
    summary: Optional[str] = None,
    remedies: Optional[Dict] = None,
    file_path: Optional[str] = None,
) -> Optional[CaseDocument]:
    """
    Upload a document to an existing case.
    Creates timeline event automatically.
    """
    db = SessionLocal()
    try:
        # Verify case ownership
        case = get_case_by_id(db, case_id)
        if not case or case.user_id != user_id:
            logger.error(f"Case {case_id} not found or not owned by user {user_id}")
            return None

        # Create document
        doc = create_case_document(
            db=db,
            case_id=case_id,
            document_type=document_type,
            user_id=user_id,
            document_content=document_content,
            file_path=file_path,
            summary=summary,
            remedies=remedies,
        )

        # Create timeline event
        create_timeline_event(
            db=db,
            case_id=case_id,
            event_type="document_uploaded",
            description=f"{document_type.value} document uploaded",
            metadata={"document_id": doc.id},
        )

        # Auto-create deadline from remedies if available
        if remedies:
            _auto_create_deadlines_from_remedies(db, user_id, case_id, case.case_number, remedies, doc.id)

        db.refresh(doc)
        logger.info(f"Uploaded document to case {case_id}: {document_type.value}")
        return doc

    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        return None
    finally:
        db.close()


def upload_case_attachment(
    user_id: int,
    case_id: int,
    file_bytes: bytes,
    filename: str,
    content_type: Optional[str] = None,
    deadline_id: Optional[int] = None,
) -> Optional[dict]:
    """
    Save an attachment file to disk and register it in the DB.
    Returns attachment dict on success.
    """
    from core.storage import save_attachment

    db = SessionLocal()
    try:
        case = get_case_by_id(db, case_id)
        if not case or case.user_id != user_id:
            logger.error(f"Case {case_id} not found or not owned by user {user_id}")
            return None

        # Save file to storage
        stored_path, size = save_attachment(file_bytes, filename)

        att = create_attachment(
            db=db,
            user_id=user_id,
            original_filename=filename,
            stored_path=stored_path,
            content_type=content_type,
            size_bytes=size,
            case_id=case_id,
            deadline_id=deadline_id,
        )

        # Timeline event
        create_timeline_event(
            db=db,
            case_id=case_id,
            event_type="attachment_uploaded",
            description=f"Attachment uploaded: {filename}",
            metadata={"attachment_id": att.id, "deadline_id": deadline_id},
        )

        db.refresh(att)
        return {
            "id": att.id,
            "original_filename": att.original_filename,
            "stored_path": att.stored_path,
            "size_bytes": att.size_bytes,
            "uploaded_at": att.uploaded_at.isoformat(),
        }

    except Exception as e:
        logger.error(f"Error uploading attachment: {str(e)}", exc_info=True)
        return None
    finally:
        db.close()


def _extract_days_from_text(text: str) -> Optional[int]:
    """
    Extract the number of days from a natural language string.
    
    This function uses multiple regex patterns to handle various formats:
    - "30 days" (standard format)
    - "appeal within 15 days" (with prefix)
    - "file appeal in 7 days" (with prefix)
    - "30days" (no space)
    - "30 Days" (capitalized)
    
    Args:
        text: The text containing the days information
        
    Returns:
        The extracted number of days, or None if not found
        
    Examples:
        >>> _extract_days_from_text("30 days")
        30
        >>> _extract_days_from_text("appeal within 15 days")
        15
        >>> _extract_days_from_text("Cost is 500 Rs, appeal in 30 days")
        30
        >>> _extract_days_from_text("Invalid text")
        None
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    
    if text.isdigit():
        return int(text)
        
    # Primary pattern: digits followed by "day" or "days" (with optional space)
    # This is the most reliable pattern for our use case
    primary_match = re.search(r'(\d+)\s*days?\b', text, re.IGNORECASE)
    if primary_match:
        return int(primary_match.group(1))
    
    # Fallback pattern: digits that appear near "day" keywords
    # This handles cases like "in 30 days" or "within 15 days"
    fallback_match = re.search(r'(?:in|within|after)\s+(\d+)\s*days?', text, re.IGNORECASE)
    if fallback_match:
        return int(fallback_match.group(1))
    
    return None


def _validate_days_value(days: int) -> bool:
    """
    Validate that the extracted days value is within acceptable bounds.
    
    Args:
        days: The number of days to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Define reasonable bounds for appeal deadlines
    # Minimum: 1 day, Maximum: 365 days (1 year)
    # These bounds can be adjusted based on business requirements
    MIN_DAYS = 1
    MAX_DAYS = 365
    
    return MIN_DAYS <= days <= MAX_DAYS


def _auto_create_deadlines_from_remedies(
    db: Session,
    user_id: int,
    case_id: int,
    case_title: str,
    remedies: Dict,
    document_id: int,
):
    """
    Auto-create deadlines from remedies advice.
    
    This function extracts deadline information from LLM-generated remedies and creates
    appropriate deadline entries in the database. It includes robust parsing to handle
    various natural language formats and prevents duplicate deadline creation.
    
    Key improvements:
    - Uses helper functions for precise day extraction
    - Validates extracted values to prevent invalid deadline dates
    - Includes comprehensive logging for debugging and audit trails
    - Handles various natural language formats for days specification
    """
    try:
        # Retrieve appeal_days from remedies dictionary
        appeal_days = remedies.get("appeal_days")
        
        # Skip if no appeal_days information is present
        if not appeal_days:
            logger.debug(f"No appeal_days found in remedies for case {case_id}, skipping deadline creation")
            return
        
        # Convert to string for processing
        appeal_days_str = str(appeal_days).strip()
        
        # Extract the number of days using the helper function
        # This handles various formats like "30 days", "appeal in 15 days", etc.
        days = _extract_days_from_text(appeal_days_str)
        
        if days is None:
            logger.warning(
                f"Could not extract days from appeal_days value: '{appeal_days_str}' "
                f"for case {case_id}. Expected format: '30 days', '15 days', etc."
            )
            return
        
        # Validate the extracted days value
        if not _validate_days_value(days):
            logger.warning(
                f"Invalid appeal_days value ({days}) for case {case_id}. "
                f"Value must be between 1 and 365 days."
            )
            return
        
        # Calculate the deadline date as a timezone-aware UTC datetime.
        current_time = datetime.now(timezone.utc)
        deadline_date = current_time + timedelta(days=days)

        # Check for existing pending deadlines to prevent duplicates.
        # Both sides of the comparison use timezone-aware datetimes so the
        # ORM filter is consistent across SQLite and PostgreSQL.
        # A ±1 day tolerance handles minor variations from different processing times.
        existing_deadline = db.query(CaseDeadline).filter(
            CaseDeadline.case_id == case_id,
            CaseDeadline.deadline_type == "appeal",
            CaseDeadline.is_completed == False,
            CaseDeadline.deadline_date >= deadline_date - timedelta(days=1),
            CaseDeadline.deadline_date <= deadline_date + timedelta(days=1)
        ).first()
        
        if existing_deadline:
            logger.info(
                f"Skipped duplicate appeal deadline for case {case_id}. "
                f"Existing deadline ID: {existing_deadline.id}, "
                f"Date: {existing_deadline.deadline_date.strftime('%Y-%m-%d')}"
            )
            return
        
        # Create the new deadline entry
        deadline = CaseDeadline(
            user_id=user_id,
            case_id=case_id,
            case_title=case_title,
            deadline_date=deadline_date,
            deadline_type="appeal",
            description=f"Appeal deadline - {remedies.get('appeal_court', 'Unknown court')}",
        )
        db.add(deadline)
        db.flush()  # Flush to generate deadline.id before using it
        
        # Create a timeline event to document the automatic deadline creation
        create_timeline_event(
            db=db,
            case_id=case_id,
            event_type="deadline_created",
            description=f"Appeal deadline set for {deadline_date.strftime('%d %B %Y')} based on document analysis",
            metadata={
                "deadline_id": deadline.id,
                "document_id": document_id,
                "source_days": days,
                "original_text": appeal_days_str,
            },
        )
        
        logger.info(
            f"Auto-created appeal deadline for case {case_id}: "
            f"{deadline_date.strftime('%Y-%m-%d')} ({days} days from now). "
            f"Source: document {document_id}"
        )
        # Do not commit here — transaction boundaries are owned by the calling
        # function (upload_case_document).  All writes are staged via flush()
        # and will be committed atomically by the parent workflow.

    except Exception as e:
        # Log the full error with context for debugging and re-raise so the
        # parent session owner can decide whether to rollback.
        logger.error(
            f"Error auto-creating deadlines for case {case_id}: {str(e)}. "
            f"Remedies: {remedies}. Document ID: {document_id}",
            exc_info=True  # Include full traceback for debugging
        )
        raise


def get_document_content(document_id: int) -> Optional[str]:
    """Get full document content by ID"""
    db = SessionLocal()
    try:
        doc = db.query(CaseDocument).filter(CaseDocument.id == document_id).first()
        return doc.document_content if doc else None
    finally:
        db.close()


# ==================== Timeline Management ====================


def get_case_timeline_events(user_id: int, case_id: int) -> List[Dict[str, Any]]:
    """Get timeline events for a case"""
    db = SessionLocal()
    try:
        # Verify ownership
        case = get_case_by_id(db, case_id)
        if not case or case.user_id != user_id:
            return []

        events = get_case_timeline(db, case_id)
        return [
            {
                "id": e.id,
                "event_type": e.event_type,
                "event_date": e.event_date.isoformat(),
                "description": e.description,
                "metadata": e.event_metadata,
            }
            for e in events
        ]

    finally:
        db.close()


def get_case_full_timeline(user_id: int, case_id: int) -> List[Dict[str, Any]]:
    """
    Return a unified, ordered timeline for a case combining:
    - CaseTimeline events
    - CaseDocument uploads
    - CaseDeadline entries (created)
    - NotificationLog entries (reminders sent)

    Ensures user owns the case and performs batched queries to avoid N+1.
    """
    db = SessionLocal()
    try:
        case = get_case_by_id(db, case_id)
        if not case or case.user_id != user_id:
            return []

        # Batch fetch related items
        timelines = db.query(CaseTimeline).filter(CaseTimeline.case_id == case_id).all()
        documents = db.query(CaseDocument).filter(CaseDocument.case_id == case_id).all()
        deadlines = db.query(CaseDeadline).filter(CaseDeadline.case_id == case_id).all()

        notifications = []
        if deadlines:
            deadline_ids = [d.id for d in deadlines]
            NotificationLog = __import__("database").NotificationLog
            notifications = db.query(NotificationLog).filter(NotificationLog.deadline_id.in_(deadline_ids)).all()

        items: List[Dict[str, Any]] = []

        def _ensure_utc(dt_val):
            if dt_val is None:
                return None
            if dt_val.tzinfo is None:
                return dt_val.replace(tzinfo=timezone.utc)
            return dt_val.astimezone(timezone.utc)

        def _fmt_ts(dt_val):
            normalized = _ensure_utc(dt_val)
            return normalized.isoformat() if normalized is not None else ""

        for t in timelines:
            item = {
                "type": t.event_type,
                "timestamp": _fmt_ts(t.event_date),
                "description": t.description,
                "metadata": t.event_metadata or {},
                "source": "timeline",
            }
            # If a timeline event represents a reminder and includes a preview, expose it
            if t.event_type == "reminder" and t.event_metadata and isinstance(t.event_metadata, dict):
                mp = t.event_metadata.get("message_preview") or t.event_metadata.get("message")
                if mp:
                    item["message_preview"] = mp
            items.append(item)

        for d in documents:
            items.append({
                "type": "document_uploaded",
                "timestamp": _fmt_ts(d.uploaded_at),
                "description": f"{d.document_type.value} uploaded",
                "metadata": {"document_id": d.id},
                "source": "document",
            })

        for d in deadlines:
            ts = _fmt_ts(d.created_at) or _fmt_ts(d.deadline_date)
            items.append({
                "type": "deadline_created",
                "timestamp": ts,
                "description": f"{d.deadline_type} - {d.description or ''}",
                "metadata": {"deadline_id": d.id},
                "source": "deadline",
            })

        for n in notifications:
            items.append({
                "type": "reminder",
                "timestamp": _fmt_ts(n.created_at),
                "description": f"Reminder ({n.channel.value}) to {n.recipient} - {n.status.value}",
                "metadata": {"notification_id": n.id, "deadline_id": n.deadline_id, "days_before": n.days_before},
                "message_preview": n.message_preview,
                "source": "notification",
            })

        # Sort descending by normalized timestamp
        items_sorted = sorted(items, key=lambda x: x.get("timestamp") or "", reverse=True)
        return items_sorted
    finally:
        db.close()


def mark_deadline_completed(user_id: int, deadline_id: int) -> bool:
    """Mark a deadline as completed"""
    db = SessionLocal()
    try:
        deadline = db.query(CaseDeadline).filter(
            CaseDeadline.id == deadline_id,
            CaseDeadline.user_id == user_id,
        ).first()

        if not deadline:
            return False

        deadline.is_completed = True
        db.commit()

        # Create timeline event
        create_timeline_event(
            db=db,
            case_id=deadline.case_id,
            event_type="deadline_completed",
            description=f"Marked {deadline.deadline_type} deadline as completed",
            metadata={"deadline_id": deadline_id},
        )

        logger.info(f"Marked deadline {deadline_id} as completed")
        return True

    except Exception as e:
        logger.error(f"Error marking deadline completed: {str(e)}")
        db.rollback()
        return False
    finally:
        db.close()


def mark_deadline_incomplete(user_id: int, deadline_id: int) -> bool:
    """Mark a deadline as incomplete (undo completion)"""
    db = SessionLocal()
    try:
        deadline = db.query(CaseDeadline).filter(
            CaseDeadline.id == deadline_id,
            CaseDeadline.user_id == user_id,
        ).first()

        if not deadline:
            return False

        deadline.is_completed = False
        db.commit()

        logger.info(f"Marked deadline {deadline_id} as incomplete")
        return True

    except Exception as e:
        logger.error(f"Error marking deadline incomplete: {str(e)}")
        db.rollback()
        return False
    finally:
        db.close()


def add_manual_deadline(
    user_id: int,
    case_id: int,
    case_title: str,
    deadline_date: datetime,
    deadline_type: str,
    description: Optional[str] = None,
) -> Optional[CaseDeadline]:
    """Add a manual deadline to a case"""
    db = SessionLocal()
    try:
        # Verify case ownership
        case = get_case_by_id(db, case_id)
        if not case or case.user_id != user_id:
            return None

        deadline = CaseDeadline(
            user_id=user_id,
            case_id=case_id,
            case_title=case_title,
            deadline_date=deadline_date,
            deadline_type=deadline_type,
            description=description,
        )
        db.add(deadline)
        db.commit()
        db.refresh(deadline)

        # Create timeline event
        create_timeline_event(
            db=db,
            case_id=case_id,
            event_type="deadline_created",
            description=f"Manual deadline added: {deadline_type} on {deadline_date.strftime('%d %B %Y')}",
            metadata={"deadline_id": deadline.id},
        )

        db.refresh(deadline)
        logger.info(f"Added manual deadline to case {case_id}: {deadline_type} on {deadline_date}")
        return deadline

    except Exception as e:
        logger.error(f"Error adding manual deadline: {str(e)}")
        db.rollback()
        return None
    finally:
        db.close()


# ==================== Case Actions ====================


def mark_case_appealed(user_id: int, case_id: int) -> bool:
    """Mark a case as appealed"""
    return _update_case_status(user_id, case_id, CaseStatus.APPEALED)


def mark_case_closed(user_id: int, case_id: int) -> bool:
    """Mark a case as closed"""
    return _update_case_status(user_id, case_id, CaseStatus.CLOSED)


def mark_case_active(user_id: int, case_id: int) -> bool:
    """Mark a case as active"""
    return _update_case_status(user_id, case_id, CaseStatus.ACTIVE)


def _update_case_status(user_id: int, case_id: int, status: CaseStatus) -> bool:
    """Update case status with timeline event"""
    db = SessionLocal()
    try:
        case = get_case_by_id(db, case_id)
        if not case or case.user_id != user_id:
            return False

        update_case_status(db, case_id, status)

        # Create timeline event
        create_timeline_event(
            db=db,
            case_id=case_id,
            event_type="status_changed",
            description=f"Case status changed to {status.value}",
            metadata={"new_status": status.value},
        )

        logger.info(f"Updated case {case_id} status to {status.value}")
        return True

    except Exception as e:
        logger.error(f"Error updating case status: {str(e)}")
        return False
    finally:
        db.close()


# ==================== Export & Sharing ====================


def generate_case_summary_text(user_id: int, case_id: int) -> Optional[str]:
    """
    Generate a text summary of a case for export.
    """
    db = SessionLocal()
    try:
        case = get_case_by_id(db, case_id)
        if not case or case.user_id != user_id:
            return None

        documents = get_case_documents(db, case_id)
        timeline = get_case_timeline(db, case_id)
        deadlines = db.query(CaseDeadline).filter(
            CaseDeadline.case_id == case_id
        ).order_by(CaseDeadline.deadline_date).all()

        lines = [
            "=" * 60,
            f"CASE SUMMARY: {case.case_number}",
            "=" * 60,
            "",
            f"Title: {case.title or 'N/A'}",
            f"Type: {case.case_type}",
            f"Jurisdiction: {case.jurisdiction}",
            f"Status: {case.status.value}",
            f"Created: {case.created_at.strftime('%d %B %Y')}",
            "",
            "-" * 60,
            "DOCUMENTS",
            "-" * 60,
        ]

        for doc in documents:
            lines.append(f"\n[{doc.document_type.value}] - {doc.uploaded_at.strftime('%d %B %Y')}")
            if doc.summary:
                lines.append(f"Summary: {doc.summary}")

        lines.extend([
            "",
            "-" * 60,
            "TIMELINE",
            "-" * 60,
        ])

        for event in timeline:
            lines.append(f"[{event.event_date.strftime('%d %B %Y')}] {event.event_type}: {event.description}")

        lines.extend([
            "",
            "-" * 60,
            "DEADLINES",
            "-" * 60,
        ])

        for d in deadlines:
            status = "✓" if d.is_completed else "○"
            lines.append(f"[{status}] {d.deadline_type}: {d.deadline_date.strftime('%d %B %Y')} - {d.description or 'No description'}")

        lines.extend([
            "",
            "=" * 60,
            f"Generated: {datetime.now(timezone.utc).strftime('%d %B %Y %H:%M')}",
            "=" * 60,
        ])

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error generating case summary: {str(e)}")
        return None
    finally:
        db.close()


def _get_case_anonymization_secret() -> str:
    """Return secret used to generate anonymized case IDs.

    Primary source: CASE_ANONYMIZATION_SECRET env var.
    Fallback: contents of .jwt_secret (kept for local dev compatibility).

    Raises RuntimeError if no non-empty secret can be resolved.  An empty
    HMAC key produces deterministic, predictable outputs that undermine
    anonymization guarantees, so we fail loudly rather than silently
    degrading security.
    """
    secret = os.getenv("CASE_ANONYMIZATION_SECRET", "").strip()
    if secret:
        return secret

    # Fallback to repo-local .jwt_secret (best-effort; avoids breaking existing deployments).
    jwt_secret_path = Path(__file__).resolve().parents[0] / ".jwt_secret"
    if not jwt_secret_path.exists():
        # Also try one directory up (in case file layout differs)
        jwt_secret_path = Path(__file__).resolve().parents[1] / ".jwt_secret"

    if jwt_secret_path.exists():
        try:
            file_secret = jwt_secret_path.read_text(encoding="utf-8").strip()
            if file_secret:
                return file_secret
        except Exception:
            pass

    raise RuntimeError(
        "CASE_ANONYMIZATION_SECRET is not configured. "
        "Set the 'CASE_ANONYMIZATION_SECRET' environment variable to a strong, "
        "randomly generated value. Using an empty HMAC key produces predictable "
        "identifiers and must not be allowed."
    )


def _generate_anonymized_case_id(case_id: int, created_at: Any) -> str:
    created_at_str = getattr(created_at, "isoformat", None)
    created_at_str = created_at.isoformat() if callable(created_at_str) else str(created_at)
    secret = _get_case_anonymization_secret().encode("utf-8")
    msg = f"{case_id}-{created_at_str}".encode("utf-8")

    digest = hmac.new(secret, msg, hashlib.sha256).hexdigest()
    return digest[:12]


def generate_anonymized_case_data(case_id: int) -> Optional[Dict[str, Any]]:
    """
    Generate anonymized case data for sharing.
    Removes personal identifiers, hashes case ID.
    """
    db = SessionLocal()

    try:
        case = get_case_by_id(db, case_id)
        if not case:
            return None

        documents = get_case_documents(db, case_id)
        timeline = get_case_timeline(db, case_id)

        # Generate secret-based anonymized ID for anonymity
        anonymized_id = _generate_anonymized_case_id(case_id=case_id, created_at=case.created_at)


        return {
            "anonymized_id": anonymized_id,
            "case_type": case.case_type,
            "jurisdiction": case.jurisdiction,
            "status": case.status.value,
            "document_count": len(documents),
            "documents": [
                {
                    "type": doc.document_type.value,
                    "summary": doc.summary,
                    "remedies": doc.remedies,
                }
                for doc in documents
            ],
            "timeline": [
                {
                    "event_type": e.event_type,
                    "description": e.description,
                }
                for e in timeline
            ],
            "created_date": case.created_at.strftime("%B %Y"),
        }

    except Exception as e:
        logger.error(f"Error generating anonymized data: {str(e)}")
        return None
    finally:
        db.close()


# =============================================================================
# BULK OPERATIONS
# =============================================================================

def delete_user_cases(user_id: int, case_ids: List[int]) -> Dict[str, Any]:
    """
    Perform a bulk deletion of multiple cases belonging to a specific user.
    
    This function implements a high-performance bulk delete strategy to avoid the 
    N+1 query problem commonly associated with looping through individual 
    ORM delete statements. By using a single SQLAlchemy query with an 'IN' 
    clause, we drastically reduce database round-trips and transaction 
    overhead.
    
    Performance Optimizations:
    --------------------------
    1. Single Query Execution: All specified cases are deleted in one SQL 
       statement: DELETE FROM cases WHERE id IN (...) AND user_id = :user_id.
    2. synchronize_session=False: We bypass the expensive session state 
       synchronization logic since we are deleting records and don't need 
       to update in-memory objects.
    3. User ID Scoping: The query is strictly scoped to the user_id to 
       ensure that users can only delete their own data, preventing 
       unauthorized deletions.
    
    Args:
        user_id (int): The unique identifier of the user performing the deletion.
        case_ids (List[int]): A list of case IDs to be permanently removed.
        
    Returns:
        Dict[str, Any]: A result dictionary containing:
            - success (bool): True if the operation completed without error.
            - count (int): The number of case records actually deleted.
            - error (str, optional): Error message if the operation failed.
            
    Note:
        Due to the 'cascade="all, delete-orphan"' configuration in the Case 
        model, all related documents, timeline events, and deadlines will 
        be automatically purged from the database along with the cases.
    """
    
    # -------------------------------------------------------------------------
    # Initialization and Input Validation
    # -------------------------------------------------------------------------
    
    db = SessionLocal()
    result = {
        "success": False,
        "count": 0,
        "error": None
    }
    
    # Early exit if no case IDs are provided to save database resources
    if not case_ids:
        logger.warning(f"No case IDs provided for bulk deletion by user {user_id}")
        result["success"] = True
        return result
        
    try:
        # ---------------------------------------------------------------------
        # Bulk Deletion Execution
        # ---------------------------------------------------------------------
        
        logger.info(
            f"Initiating bulk deletion for user {user_id}. "
            f"Targeting {len(case_ids)} potential cases."
        )
        
        # We use session.query(Case).filter().delete() for maximum efficiency.
        # This translates to a single DELETE statement at the database level.
        #
        # Security Note: We MUST include the user_id in the filter to prevent 
        # ID-traversal attacks where a user could delete cases belonging 
        # to other users by guessing their IDs.
        
        query = db.query(Case).filter(
            Case.id.in_(case_ids),
            Case.user_id == user_id
        )
        
        # Execute the delete operation.
        # We set synchronize_session=False because we don't need to update 
        # any in-memory Case objects after they are deleted. This provides 
        # a slight performance boost by avoiding session state management.
        
        deleted_count = query.delete(synchronize_session=False)
        
        # Commit the transaction to persist the changes.
        db.commit()
        
        # ---------------------------------------------------------------------
        # Finalization and Audit Logging
        # ---------------------------------------------------------------------
        
        logger.info(
            f"Bulk deletion successful for user {user_id}. "
            f"Records removed: {deleted_count}."
        )
        
        result["success"] = True
        result["count"] = deleted_count
        
    except Exception as e:
        # ---------------------------------------------------------------------
        # Error Handling and Recovery
        # ---------------------------------------------------------------------
        
        # Roll back the transaction if any part of the deletion fails to 
        # maintain database consistency.
        db.rollback()
        
        error_msg = f"Failed to execute bulk deletion: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        result["success"] = False
        result["error"] = error_msg
        
    finally:
        # Always ensure the database session is closed to prevent 
        # connection pool exhaustion.
        db.close()
        
    return result


# =============================================================================
# END OF SERVICE
# =============================================================================
