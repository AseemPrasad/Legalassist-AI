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
from sqlalchemy.orm.exc import StaleDataError

# =============================================================================
# OPTIMISTIC CONCURRENCY CONTROL
# =============================================================================
# We have integrated optimistic concurrency control to prevent data loss 
# when multiple users are concurrently modifying the same legal case.
# By using a 'version' column with SQLAlchemy's `version_id_col` mapper argument,
# the database automatically checks the expected version during an UPDATE.
# If the version does not match, a StaleDataError is raised.
# This ensures data integrity and forces the second user to refresh their view
# and merge changes rather than blindly overwriting another user's data.
# =============================================================================

from database import (
    SessionLocal,
    Case,
    CaseDocument,
    CaseNote,
    CaseTimeline,
    CaseComment,
    CasePresence,
    CaseDeadline,
    CaseStatus,
    DocumentType,
    UserPreference,
    create_case,
    get_user_cases,
    get_case_by_id,
    get_case_documents,
    get_case_timeline,
    get_case_comments,
    get_case_presence,
    create_case_document,
    create_timeline_event,
    create_case_comment,
    upsert_case_presence,
    update_case_status,
    create_attachment,
    get_attachments_for_case,
    save_case_note_draft,
)
from core.deadline_engine import get_deadline_first_action
from db.case_service import get_case_note, publish_case_note, get_case_note_history
from services.timeline_service import timeline_service as _timeline_service
from services.deadlines_auto_creator import (
    _extract_days_from_text as _extract_days_from_text_service,
    _validate_days_value as _validate_days_value_service,
    auto_create_deadlines_from_remedies as _auto_create_deadlines_from_remedies_service,
)
from services.case_anonymization import (
    _get_case_anonymization_secret as _get_case_anonymization_secret_service,
    _generate_anonymized_case_id as _generate_anonymized_case_id_service,
    generate_anonymized_case_data as generate_anonymized_case_data_service,
)
from services.case_queries import (
    get_user_cases_summary as get_user_cases_summary_service,
    get_case_detail as get_case_detail_service,
    generate_case_summary_text as generate_case_summary_text_service,
)
from db.crud.audit import record_immutable_audit_event

logger = logging.getLogger(__name__)


# ==================== Case Management ====================


def create_new_case(
    user_id: int,
    case_number: str,
    case_type: str,
    jurisdiction: str,
    title: Optional[str] = None,
) -> tuple[Optional[Case], bool]:
    """
    Create a new case for a user.
    Returns (case, was_existing) tuple.
    was_existing=True indicates an existing case was returned without updates.
    """
    db = SessionLocal()
    try:
        case_type = case_type.strip()
        jurisdiction = jurisdiction.strip()
        if title:
            title = title.strip()
        
        # Check if case number already exists for this user
        normalized_number = case_number.strip()
        existing = db.query(Case).filter(
            Case.user_id == user_id,
            Case.case_number == normalized_number,
        ).first()

        if existing:
            # Check if metadata differs from existing case
            metadata_changed = (
                existing.case_type != case_type or
                existing.jurisdiction != jurisdiction or
                (title and existing.title != title)
            )
            
            if metadata_changed:
                logger.warning(
                    f"Case {case_number} exists but metadata differs. "
                    f"Expected: type={case_type}, jurisdiction={jurisdiction}, title={title}. "
                    f"Got: type={existing.case_type}, jurisdiction={existing.jurisdiction}, title={existing.title}. "
                    f"Returning existing case without updates."
                )
            else:
                logger.info(f"Case {case_number} already exists for user {user_id}")
            
            return existing, True

        case = create_case(
            db=db,
            user_id=user_id,
            case_number=case_number,
            case_type=case_type,
            jurisdiction=jurisdiction,
            title=title,
        )

        # Create timeline event for case creation
        _timeline_service.create_event(
            db=db,
            case_id=case.id,
            event_type="case_created",
            description=f"Case {case_number} created",
            metadata={"case_type": case_type, "jurisdiction": jurisdiction},
        )

        db.refresh(case)
        logger.info(f"Created new case: {case_number} for user {user_id}")
        return case, False

    except Exception as e:
        logger.error(f"Error creating case: {str(e)}")
        return None, False
    finally:
        db.close()


def update_case_details(
    user_id: int,
    case_id: int,
    expected_version: int,
    title: Optional[str] = None,
    case_type: Optional[str] = None,
    jurisdiction: Optional[str] = None,
) -> tuple[Optional[Case], Optional[str]]:
    """
    Update case details with optimistic concurrency control.
    Requires the client to provide the expected_version they last saw.
    Returns (Case, error_message). If error_message is not None, the update failed.
    """
    db = SessionLocal()
    try:
        case = get_case_by_id(db, case_id)
        if not case or case.user_id != user_id:
            return None, "Case not found or unauthorized."
        
        if case.version != expected_version:
            return None, f"Conflict: Case has been updated by another user. Please refresh."

        # If version matches, we can update fields
        updated = False
        if title is not None and case.title != title:
            case.title = title
            updated = True
        if case_type is not None and case.case_type != case_type:
            case.case_type = case_type
            updated = True
        if jurisdiction is not None and case.jurisdiction != jurisdiction:
            case.jurisdiction = jurisdiction
            updated = True
            
        if updated:
            try:
                db.commit()
                db.refresh(case)
                
                # Create timeline event
                _timeline_service.create_event(
                    db=db,
                    case_id=case_id,
                    event_type="case_updated",
                    description=f"Case details updated",
                    metadata={"version": case.version},
                )
                
                logger.info(f"Updated case {case_id} successfully (version {case.version})")
            except StaleDataError:
                db.rollback()
                return None, "Conflict: Case has been updated by another user. Please refresh."
        
        return case, None

    except Exception as e:
        logger.error(f"Error updating case: {str(e)}")
        db.rollback()
        return None, "Internal error updating case."
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
            case, was_existing = create_new_case(
                user_id=user_id,
                case_number=new_case_number,
                case_type=new_case_type or "general",
                jurisdiction=new_jurisdiction or "Unknown",
                title=new_title,
            )
            if was_existing:
                logger.info(f"Reusing existing case {new_case_number} for document processing")
            return case

        return None

    finally:
        db.close()


def get_user_cases_summary(user_id: int, include_closed: bool = True) -> List[Dict[str, Any]]:
    db = SessionLocal()
    try:
        return get_user_cases_summary_service(db, user_id, include_closed=include_closed)
    except Exception as e:
        logger.error(f"Error getting user cases summary: {str(e)}")
        return []
    finally:
        db.close()


def get_case_detail(user_id: int, case_id: int) -> Optional[Dict[str, Any]]:
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

        comments = get_case_comments(db, case_id, user_id)
        comments_list = [
            {
                "id": comment.id,
                "parent_comment_id": comment.parent_comment_id,
                "user_id": comment.user_id,
                "user_email": comment.user.email if comment.user else None,
                "comment_text": comment.comment_text,
                "is_resolved": comment.is_resolved,
                "created_at": comment.created_at.isoformat(),
                "updated_at": comment.updated_at.isoformat() if comment.updated_at else None,
            }
            for comment in comments
        ]

        presence = get_case_presence(db, case_id)
        presence_list = [
            {
                "id": item.id,
                "user_id": item.user_id,
                "user_email": item.user.email if item.user else None,
                "active_view": item.active_view,
                "cursor_anchor": item.cursor_anchor,
                "last_seen": item.last_seen.isoformat(),
            }
            for item in presence
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
            "comments": comments_list,
            "presence": presence_list,
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
        _timeline_service.create_event(
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


def add_case_comment(
    user_id: int,
    case_id: int,
    comment_text: str,
    parent_comment_id: Optional[int] = None,
    active_view: Optional[str] = None,
) -> Optional[CaseComment]:
    """Add a collaboration comment to a case."""
    db = SessionLocal()
    try:
        comment = create_case_comment(
            db=db,
            case_id=case_id,
            user_id=user_id,
            comment_text=comment_text,
            parent_comment_id=parent_comment_id,
        )
        upsert_case_presence(
            db=db,
            case_id=case_id,
            user_id=user_id,
            active_view=active_view or "collaboration",
        )
        return comment
    except Exception as e:
        logger.error(f"Error adding case comment: {str(e)}")
        return None
    finally:
        db.close()


def update_case_presence(
    user_id: int,
    case_id: int,
    active_view: Optional[str] = None,
    cursor_anchor: Optional[str] = None,
) -> Optional[CasePresence]:
    """Update a collaborator's presence for a case."""
    db = SessionLocal()
    try:
        return upsert_case_presence(
            db=db,
            case_id=case_id,
            user_id=user_id,
            active_view=active_view or "case_details",
            cursor_anchor=cursor_anchor,
        )
    except Exception as e:
        logger.error(f"Error updating case presence: {str(e)}")
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

        # Sanitize filename to prevent directory traversal
        safe_filename = os.path.basename(filename)

        # Save file to storage
        stored_path, size = save_attachment(file_bytes, safe_filename)

        att = create_attachment(
            db=db,
            user_id=user_id,
            original_filename=safe_filename,
            stored_path=stored_path,
            content_type=content_type,
            size_bytes=size,
            case_id=case_id,
            deadline_id=deadline_id,
        )

        # Timeline event
        _timeline_service.create_event(
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


def upload_case_document_file(
    user_id: int,
    case_id: int,
    file_bytes: bytes,
    filename: str,
    document_type: DocumentType = DocumentType.OTHER,
    content_type: Optional[str] = None,
) -> Optional[dict]:
    """Persist an uploaded case document and create its attachment link."""
    from core.storage import save_attachment

    db = SessionLocal()
    try:
        case = get_case_by_id(db, case_id)
        if not case or case.user_id != user_id:
            logger.error(f"Case {case_id} not found or not owned by user {user_id}")
            return None

        # Sanitize filename to prevent directory traversal
        safe_filename = os.path.basename(filename)

        stored_path, size = save_attachment(file_bytes, safe_filename)

        att = create_attachment(
            db=db,
            user_id=user_id,
            original_filename=safe_filename,
            stored_path=stored_path,
            content_type=content_type,
            size_bytes=size,
            case_id=case_id,
        )

        doc = create_case_document(
            db=db,
            case_id=case_id,
            document_type=document_type,
            user_id=user_id,
            file_path=stored_path,
            source_attachment_id=att.id,
            extraction_method="queued",
            ocr_used=False,
            extracted_metadata={"status": "queued"},
        )

        att.document_id = doc.id
        db.commit()

        _timeline_service.create_event(
            db=db,
            case_id=case_id,
            event_type="document_uploaded",
            description=f"{document_type.value} document uploaded",
            metadata={"attachment_id": att.id, "document_id": doc.id},
        )

        db.refresh(att)
        db.refresh(doc)
        return {
            "attachment": {
                "id": att.id,
                "original_filename": att.original_filename,
                "stored_path": att.stored_path,
                "size_bytes": att.size_bytes,
                "uploaded_at": att.uploaded_at.isoformat(),
                "content_type": att.content_type,
                "case_id": att.case_id,
                "document_id": getattr(att, "document_id", None),
            },
            "document": {
                "id": doc.id,
                "case_id": doc.case_id,
                "document_type": doc.document_type.value,
                "uploaded_at": doc.uploaded_at.isoformat(),
                "file_path": doc.file_path,
                "source_attachment_id": doc.source_attachment_id,
                "extraction_method": doc.extraction_method,
                "ocr_used": doc.ocr_used,
                "extracted_metadata": doc.extracted_metadata,
            },
        }

    except Exception as e:
        logger.error(f"Error uploading case document file: {str(e)}", exc_info=True)
        return None
    finally:
        db.close()


def _extract_days_from_text(text: str) -> Optional[int]:
    return _extract_days_from_text_service(text)


def _validate_days_value(days: int) -> bool:
    return _validate_days_value_service(days)


def _auto_create_deadlines_from_remedies(
    db: Session,
    user_id: int,
    case_id: int,
    case_title: str,
    remedies: Dict,
    document_id: int,
):
    return _auto_create_deadlines_from_remedies_service(db, user_id, case_id, case_title, remedies, document_id)


def get_document_content(document_id: int, user_id: int) -> Optional[str]:
    """Get full document content by ID, verifying user ownership."""
    db = SessionLocal()
    try:
        doc = db.query(CaseDocument).filter(CaseDocument.id == document_id).first()
        if doc is None:
            return None
        if doc.case.user_id != user_id:
            logger.warning("idor_document_access_denied", document_id=document_id, user_id=user_id, owner_id=doc.case.user_id)
            return None
        return doc.document_content
    finally:
        db.close()


# ==================== Timeline Management ====================


def get_case_timeline_events(user_id: int, case_id: int) -> List[Dict[str, Any]]:
    """Get timeline events for a case"""
    db = SessionLocal()
    try:
        case = get_case_by_id(db, case_id)
        if not case or case.user_id != user_id:
            return []

        return _timeline_service.get_case_timeline_events(db, case_id)

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

        return _timeline_service.get_case_full_timeline(db, case_id)
    finally:
        db.close()


def get_case_note_state(user_id: int, case_id: int):
    db = SessionLocal()
    try:
        case = get_case_by_id(db, case_id)
        if not case or case.user_id != user_id:
            return None
        return get_case_note(db, case_id, user_id)
    finally:
        db.close()


def save_case_note(user_id: int, case_id: int, note_text: str, changed_by_email: Optional[str] = None):
    db = SessionLocal()
    try:
        return save_case_note_draft(db, case_id, user_id, note_text, changed_by_email=changed_by_email)
    except Exception as e:
        logger.error(f"Error saving case note draft: {str(e)}")
        return None
    finally:
        db.close()


def publish_case_note_for_case(user_id: int, case_id: int, note_text: Optional[str] = None, changed_by_email: Optional[str] = None):
    db = SessionLocal()
    try:
        return publish_case_note(db, case_id, user_id, note_text=note_text, changed_by_email=changed_by_email)
    except Exception as e:
        logger.error(f"Error publishing case note: {str(e)}")
        return None
    finally:
        db.close()


def get_case_note_history_for_case(user_id: int, case_id: int):
    db = SessionLocal()
    try:
        return get_case_note_history(db, case_id, user_id)
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
        _timeline_service.create_event(
            db=db,
            case_id=deadline.case_id,
            event_type="deadline_completed",
            description=f"Marked {deadline.deadline_type} deadline as completed",
            metadata={"deadline_id": deadline_id},
        )

        record_immutable_audit_event(
            event_type="deadline.completed",
            action="completed",
            actor_user_id=user_id,
            resource_type="deadline",
            resource_id=str(deadline_id),
            outcome="success",
            case_id=deadline.case_id,
            metadata={
                "deadline_type": deadline.deadline_type,
            },
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
    court_name: Optional[str] = None,
) -> Optional[CaseDeadline]:
    """Add a manual deadline to a case"""
    if deadline_date.tzinfo is None:
        deadline_date = deadline_date.replace(tzinfo=timezone.utc)
    
    db = SessionLocal()
    try:
        # Validate deadline date is not in the past
        if deadline_date.tzinfo is None:
            deadline_date = deadline_date.replace(tzinfo=timezone.utc)
        if deadline_date < datetime.now(timezone.utc):
            return None

        # Verify case ownership
        case = get_case_by_id(db, case_id)
        if not case or case.user_id != user_id:
            return None

        deadline = CaseDeadline(
            user_id=user_id,
            case_id=case_id,
            case_title=case_title,
            court_name=court_name,
            deadline_date=deadline_date,
            deadline_type=deadline_type,
            first_action=get_deadline_first_action(deadline_type),
            description=description,
        )
        db.add(deadline)
        db.commit()
        db.refresh(deadline)

        # Create timeline event
        _timeline_service.create_event(
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
        _timeline_service.create_event(
            db=db,
            case_id=case_id,
            event_type="status_changed",
            description=f"Case status changed to {status.value}",
            metadata={"new_status": status.value},
        )

        record_immutable_audit_event(
            event_type="case.status_changed",
            action="status_change",
            actor_user_id=user_id,
            resource_type="case",
            resource_id=str(case_id),
            outcome="success",
            case_id=case_id,
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
    db = SessionLocal()
    try:
        return generate_case_summary_text_service(db, user_id, case_id)
    except Exception as e:
        logger.error(f"Error generating case summary: {str(e)}")
        return None
    finally:
        db.close()


def _get_case_anonymization_secret() -> str:
    return _get_case_anonymization_secret_service()


def _generate_anonymized_case_id(case_id: int, created_at: Any) -> str:
    return _generate_anonymized_case_id_service(case_id, created_at)


def generate_anonymized_case_data(case_id: int, profile_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    try:
        return generate_anonymized_case_data_service(case_id, profile_name=profile_name)
    except Exception as e:
        logger.error(f"Error generating anonymized data: {str(e)}")
        return None


# =============================================================================
# BULK OPERATIONS
# =============================================================================

def delete_user_cases(user_id: int, case_ids: List[int], confirm: bool = False) -> Dict[str, Any]:
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
    2. synchronize_session='fetch': We refresh the session state after deletion 
       to prevent ORM inconsistencies and silent data loss.
    3. User ID Scoping: The query is strictly scoped to the user_id to 
       ensure that users can only delete their own data, preventing 
       unauthorized deletions.
    
    Safety:
    -------
    This function REQUIRES confirm=True to execute. This prevents accidental 
    bulk data loss from unintended call paths. The caller (UI or API) should 
    present a confirmation dialog before passing confirm=True.
    
    Args:
        user_id (int): The unique identifier of the user performing the deletion.
        case_ids (List[int]): A list of case IDs to be permanently removed.
        confirm (bool): Must be True to execute. Defaults to False as a 
                        safety guard against accidental deletion.
        
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
    
    # Safety guard: require explicit confirmation to prevent accidental deletion
    if not confirm:
        logger.warning(
            f"Bulk deletion blocked for user {user_id}: "
            f"confirm=False. Target IDs: {case_ids}"
        )
        result["error"] = "Confirmation required. Call with confirm=True to proceed."
        return result
        
    try:
        # ---------------------------------------------------------------------
        # Pre-deletion audit — log target cases before deletion
        # ---------------------------------------------------------------------
        
        targets = db.query(Case.id, Case.case_number, Case.title).filter(
            Case.id.in_(case_ids),
            Case.user_id == user_id
        ).all()
        
        if not targets:
            logger.warning(f"No matching cases found for user {user_id} with IDs {case_ids}")
            result["success"] = True
            return result
        
        logger.info(
            "Bulk deletion requested for user %d: %d cases",
            user_id, len(targets),
            extra={"case_ids": [t.id for t in targets]},
        )
        
        # ---------------------------------------------------------------------
        # Bulk Deletion Execution
        # ---------------------------------------------------------------------
        
        query = db.query(Case).filter(
            Case.id.in_(case_ids),
            Case.user_id == user_id
        )
        
        # Execute the delete operation with synchronize_session='fetch' to 
        # ensure the ORM session stays consistent with the database after 
        # the bulk operation, preventing silent data loss.
        
        deleted_count = query.delete(synchronize_session='fetch')
        
        # Commit the transaction to persist the changes.
        db.commit()
        
        # ---------------------------------------------------------------------
        # Finalization and Audit Logging
        # ---------------------------------------------------------------------
        
        logger.info(
            "Bulk deletion successful for user %d. Records removed: %d.",
            user_id, deleted_count,
        )
        
        result["success"] = True
        result["count"] = deleted_count
        
    except Exception as e:
        # ---------------------------------------------------------------------
        # Error Handling and Recovery
        # ---------------------------------------------------------------------
        
        db.rollback()
        
        error_msg = f"Failed to execute bulk deletion: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        result["success"] = False
        result["error"] = error_msg
        
    finally:
        db.close()
        
    return result


# =============================================================================
# END OF SERVICE
# =============================================================================


def validate_case_transition(current_status: str, target_status: str) -> bool:
    """
    Validates if a transition from current case status to target case status 
    is permitted under standard case lifecycle rules.
    """
    allowed_transitions = {
        "pending": ["active", "dismissed"],
        "active": ["settled", "dismissed", "appealed"],
        "appealed": ["active", "dismissed"],
        "settled": [],
        "dismissed": []
    }
    return target_status in allowed_transitions.get(current_status, [])
