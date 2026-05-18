from __future__ import annotations

from typing import List, Optional

from sqlalchemy.orm import Session

from db.models import Attachment


def create_attachment(
    db: Session,
    user_id: int,
    original_filename: str,
    stored_path: str,
    content_type: Optional[str] = None,
    size_bytes: Optional[int] = None,
    case_id: Optional[int] = None,
    deadline_id: Optional[int] = None,
) -> Attachment:
    """Create a new file attachment record"""
    att = Attachment(
        user_id=user_id,
        original_filename=original_filename,
        stored_path=stored_path,
        content_type=content_type,
        size_bytes=size_bytes,
        case_id=case_id,
        deadline_id=deadline_id,
    )
    db.add(att)
    db.commit()
    db.refresh(att)
    return att


def get_attachments_for_case(db: Session, case_id: int) -> List[Attachment]:
    """Get all attachments for a case"""
    return db.query(Attachment).filter(Attachment.case_id == case_id).all()
