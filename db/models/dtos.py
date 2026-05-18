"""Data Transfer Objects for case queries.

Standardized structures for returning case-related data from queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any

from db.models import CaseDocument, CaseDeadline, CaseTimeline, Case, Attachment


@dataclass
class DocumentDTO:
    """Transfer object for case documents."""
    id: int
    document_type: str
    uploaded_at: str  # ISO format
    summary: Optional[str]
    has_remedies: bool

    @staticmethod
    def from_entity(doc: CaseDocument) -> DocumentDTO:
        """Build from ORM entity."""
        return DocumentDTO(
            id=doc.id,
            document_type=doc.document_type.value,
            uploaded_at=doc.uploaded_at.isoformat(),
            summary=doc.summary,
            has_remedies=bool(doc.remedies),
        )


@dataclass
class DeadlineDTO:
    """Transfer object for case deadlines."""
    id: int
    deadline_type: str
    deadline_date: str  # ISO format
    description: Optional[str]
    is_completed: bool
    days_until: int

    @staticmethod
    def from_entity(deadline: CaseDeadline) -> DeadlineDTO:
        """Build from ORM entity."""
        return DeadlineDTO(
            id=deadline.id,
            deadline_type=deadline.deadline_type,
            deadline_date=deadline.deadline_date.isoformat(),
            description=deadline.description,
            is_completed=deadline.is_completed,
            days_until=deadline.days_until_deadline(),
        )


@dataclass
class TimelineDTO:
    """Transfer object for timeline events."""
    id: int
    event_date: str  # ISO format
    event_type: str
    description: str

    @staticmethod
    def from_entity(event: CaseTimeline) -> TimelineDTO:
        """Build from ORM entity."""
        return TimelineDTO(
            id=event.id,
            event_date=event.event_date.isoformat(),
            event_type=event.event_type,
            description=event.description,
        )


@dataclass
class AttachmentDTO:
    """Transfer object for attachments."""
    id: int
    original_filename: str
    uploaded_at: str  # ISO format
    size_bytes: int
    content_type: str

    @staticmethod
    def from_entity(attachment: Attachment) -> AttachmentDTO:
        """Build from ORM entity."""
        return AttachmentDTO(
            id=attachment.id,
            original_filename=attachment.original_filename,
            uploaded_at=attachment.uploaded_at.isoformat(),
            size_bytes=attachment.size_bytes,
            content_type=attachment.content_type,
        )


@dataclass
class CaseSummaryDTO:
    """Summary of a case for list views."""
    id: int
    case_number: str
    title: str
    case_type: str
    jurisdiction: str
    status: str
    created_at: str  # ISO format
    latest_document_type: Optional[str]
    latest_document_date: Optional[str]  # ISO format
    next_deadline_date: Optional[str]  # ISO format
    next_deadline_type: Optional[str]
    days_until_deadline: Optional[int]
    document_count: int

    @staticmethod
    def from_case_and_data(
        case: Case,
        latest_doc: Optional[CaseDocument],
        next_deadline: Optional[CaseDeadline],
        doc_count: int,
    ) -> CaseSummaryDTO:
        """Build from case and related data."""
        return CaseSummaryDTO(
            id=case.id,
            case_number=case.case_number,
            title=case.title or case.case_number,
            case_type=case.case_type,
            jurisdiction=case.jurisdiction,
            status=case.status.value,
            created_at=case.created_at.isoformat(),
            latest_document_type=latest_doc.document_type.value if latest_doc else None,
            latest_document_date=latest_doc.uploaded_at.isoformat() if latest_doc else None,
            next_deadline_date=next_deadline.deadline_date.isoformat() if next_deadline else None,
            next_deadline_type=next_deadline.deadline_type if next_deadline else None,
            days_until_deadline=next_deadline.days_until_deadline() if next_deadline else None,
            document_count=doc_count,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "case_number": self.case_number,
            "title": self.title,
            "case_type": self.case_type,
            "jurisdiction": self.jurisdiction,
            "status": self.status,
            "created_at": self.created_at,
            "latest_document_type": self.latest_document_type,
            "latest_document_date": self.latest_document_date,
            "next_deadline_date": self.next_deadline_date,
            "next_deadline_type": self.next_deadline_type,
            "days_until_deadline": self.days_until_deadline,
            "document_count": self.document_count,
        }


@dataclass
class CaseDetailDTO:
    """Complete case details for detail views."""
    case: Dict[str, Any]
    documents: List[DocumentDTO]
    deadlines: List[DeadlineDTO]
    attachments: List[AttachmentDTO]
    timeline: List[TimelineDTO]
    remedies: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "case": self.case,
            "documents": [d.__dict__ for d in self.documents],
            "deadlines": [d.__dict__ for d in self.deadlines],
            "attachments": [a.__dict__ for a in self.attachments],
            "timeline": [t.__dict__ for t in self.timeline],
            "remedies": self.remedies,
        }
