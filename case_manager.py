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


from sqlalchemy.orm import Session # type: ignore

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
    get_organization_cases,
    log_audit_action,
    get_user_role_in_org,
)

logger = logging.getLogger(__name__)


# ==================== Case Management ====================


def create_new_case(
    user_id: int,
    case_number: str,
    case_type: str,
    jurisdiction: str,
    organization_id: Optional[int] = None,
    title: Optional[str] = None,
) -> Optional[Case]:
    """
    Create a new case for a user.
    Returns the created Case object or None if failed.
    """
    db = SessionLocal()
    try:
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
            organization_id=organization_id,
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
    organization_id: Optional[int] = None,
) -> Optional[Case]:
    """
    Get existing case or create new one for document upload.
    """
    db = SessionLocal()
    try:
        if existing_case_id:
            case = get_case_by_id(db, existing_case_id)
            if case:
                # Allow access if it's user's case OR user is in the same organization
                if case.user_id == user_id:
                    return case
                if organization_id and case.organization_id == organization_id:
                    return case

        # Create new case
        if new_case_number:
            case = create_new_case(
                user_id=user_id,
                organization_id=organization_id,
                case_number=new_case_number,
                case_type=new_case_type or "general",
                jurisdiction=new_jurisdiction or "Unknown",
                title=new_title,
            )
            return case

        return None

    finally:
        db.close()


def get_user_cases_summary(user_id: int, organization_id: Optional[int] = None, include_closed: bool = True) -> List[Dict[str, Any]]:
    """
    Get summary of all cases for a user.
    Returns list of case summaries with latest document info.
    """
    db = SessionLocal()
    try:
        if organization_id:
            # Multi-tenant mode: show all cases for the organization
            cases = get_organization_cases(db, organization_id, include_closed=include_closed)
        else:
            # Single-user mode: show only user's cases
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


def get_case_detail(user_id: int, case_id: int, organization_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific case.
    """
    db = SessionLocal()
    try:
        case = get_case_by_id(db, case_id)

        if not case:
            return None
            
        # Permission Check: Must own case OR belong to the case's organization
        has_permission = (case.user_id == user_id) or (organization_id and case.organization_id == organization_id)
        if not has_permission:
            logger.warning(f"User {user_id} denied access to case {case_id}")
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
    organization_id: Optional[int] = None,
) -> Optional[CaseDocument]:
    """
    Upload a document to an existing case.
    Creates timeline event automatically.
    """
    db = SessionLocal()
    try:
        # Verify case ownership
        # Verify case ownership or org access
        case = get_case_by_id(db, case_id)
        if not case:
            return None
            
        has_permission = (case.user_id == user_id) or (organization_id and case.organization_id == organization_id)
        if not has_permission:
            logger.error(f"Case {case_id} not accessible for user {user_id} in org {organization_id}")
            return None

        # Create document
        doc = create_case_document(
            db=db,
            case_id=case_id,
            document_type=document_type,
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

        # Log the action for audit if organization context is present
        if organization_id or case.organization_id:
            log_audit_action(
                db,
                organization_id=organization_id or case.organization_id,
                user_id=user_id,
                action="upload_document",
                resource_type="document",
                resource_id=doc.id,
                details={"case_id": case_id, "document_type": document_type.value}
            )

        db.refresh(doc)
        logger.info(f"Uploaded document to case {case_id}: {document_type.value}")
        return doc

    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
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
        
        # Calculate the deadline date
        # Using timezone-aware datetime for consistency
        current_time = datetime.now(timezone.utc)
        deadline_date = current_time + timedelta(days=days)
        
        # Normalize to naive datetime for database storage
        # The database schema uses timezone-naive timestamps
        naive_deadline = deadline_date.replace(tzinfo=None)
        
        # Check for existing pending deadlines to prevent duplicates
        # We use a ±1 day tolerance to handle minor variations in deadline dates
        # that might result from different processing times or timezone conversions
        existing_deadline = db.query(CaseDeadline).filter(
            CaseDeadline.case_id == case_id,
            CaseDeadline.deadline_type == "appeal",
            CaseDeadline.is_completed == False,
            CaseDeadline.deadline_date >= naive_deadline - timedelta(days=1),
            CaseDeadline.deadline_date <= naive_deadline + timedelta(days=1)
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
        
        # Commit the transaction
        db.commit()
        
    except Exception as e:
        # Log the full error with context for debugging
        logger.error(
            f"Error auto-creating deadlines for case {case_id}: {str(e)}. "
            f"Remedies: {remedies}. Document ID: {document_id}",
            exc_info=True  # Include full traceback for debugging
        )
        # Rollback the transaction to prevent partial data writes
        db.rollback()


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


def mark_case_appealed(user_id: int, case_id: int, organization_id: Optional[int] = None) -> bool:
    """Mark a case as appealed"""
    return _update_case_status(user_id, case_id, CaseStatus.APPEALED, organization_id)


def mark_case_closed(user_id: int, case_id: int, organization_id: Optional[int] = None) -> bool:
    """Mark a case as closed"""
    return _update_case_status(user_id, case_id, CaseStatus.CLOSED, organization_id)


def mark_case_active(user_id: int, case_id: int, organization_id: Optional[int] = None) -> bool:
    """Mark a case as active"""
    return _update_case_status(user_id, case_id, CaseStatus.ACTIVE, organization_id)


def _update_case_status(user_id: int, case_id: int, status: CaseStatus, organization_id: Optional[int] = None) -> bool:
    """Update case status with timeline event"""
    db = SessionLocal()
    try:
        case = get_case_by_id(db, case_id)
        if not case:
            return False
            
        has_permission = (case.user_id == user_id) or (organization_id and case.organization_id == organization_id)
        if not has_permission:
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


def generate_case_summary_text(user_id: int, case_id: int, organization_id: Optional[int] = None) -> Optional[str]:
    """
    Generate a text summary of a case for export.
    """
    db = SessionLocal()
    try:
        case = get_case_by_id(db, case_id)
        if not case:
            return None
            
        has_permission = (case.user_id == user_id) or (organization_id and case.organization_id == organization_id)
        if not has_permission:
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
            return jwt_secret_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    # Last resort: no secret. This is insecure, but prevents runtime crashes.
    # Prefer setting CASE_ANONYMIZATION_SECRET.
    return ""


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
