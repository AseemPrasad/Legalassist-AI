"""
GraphQL Schema for Legalassist-AI
Exposes relational legal entities with nested relationships
"""

import datetime as dt
from typing import List, Optional

import strawberry
from strawberry import relay
from strawberry.relay import ID

from db.case_service import (
    create_case_document,
    delete_case,
    get_case_by_id,
    get_user_cases,
    update_case_outcome,
    update_case_status,
)
from db.crud.notifications import get_upcoming_deadlines
from db.models.analytics import CaseOutcome, CaseRecord
from db.models.cases import CaseStatus, DocumentType


@strawberry.enum
class CaseStatusEnum(strawberry.Enum):
    ACTIVE = CaseStatus.ACTIVE
    APPEALED = CaseStatus.APPEALED
    CLOSED = CaseStatus.CLOSED
    PENDING = CaseStatus.PENDING


@strawberry.enum
class DocumentTypeEnum(strawberry.Enum):
    FIR = DocumentType.FIR
    CHARGESHEET = DocumentType.CHARGESHEET
    JUDGMENT = DocumentType.JUDGMENT
    APPEAL = DocumentType.APPEAL
    ORDER = DocumentType.ORDER
    OTHER = DocumentType.OTHER


@strawberry.type
class CaseDeadlineType:
    id: int
    case_id: int
    case_title: str
    deadline_date: dt.datetime
    deadline_type: str
    description: Optional[str]
    is_completed: bool
    days_until_deadline: int

    @strawberry.field
    def is_overdue(self) -> bool:
        return self.days_until_deadline == 0


@strawberry.type
class CaseDocumentType:
    id: int
    case_id: int
    document_type: DocumentTypeEnum
    file_path: Optional[str]
    uploaded_at: dt.datetime
    summary: Optional[str]
    remedies: Optional[strawberry.scalar(JSONScalar)]  # type: ignore[name-defined]

    @classmethod
    def from_db(cls, doc) -> "CaseDocumentType":
        return cls(
            id=doc.id,
            case_id=doc.case_id,
            document_type=DocumentTypeEnum(doc.document_type.value),
            file_path=doc.file_path,
            uploaded_at=doc.uploaded_at,
            summary=doc.summary,
            remedies=doc.remedies,
        )


@strawberry.type
class TimelineEventType:
    id: int
    case_id: int
    event_type: str
    event_date: dt.datetime
    description: str
    event_metadata: Optional[strawberry.scalar(JSONScalar)]  # type: ignore[name-defined]

    @classmethod
    def from_db(cls, event) -> "TimelineEventType":
        return cls(
            id=event.id,
            case_id=event.case_id,
            event_type=event.event_type,
            event_date=event.event_date,
            description=event.description,
            event_metadata=event.event_metadata,
        )


@strawberry.type
class CaseType:
    id: int
    user_id: int
    case_number: str
    case_type: str
    jurisdiction: str
    status: CaseStatusEnum
    title: Optional[str]
    created_at: dt.datetime
    updated_at: dt.datetime
    documents: List[CaseDocumentType]
    deadlines: List[CaseDeadlineType]
    timeline_events: List[TimelineEventType]

    @classmethod
    def from_db(cls, case, include_documents: bool = False, include_deadlines: bool = False, include_timeline: bool = False) -> "CaseType":
        docs = [CaseDocumentType.from_db(d) for d in case.documents] if include_documents else []
        deadlines = [
            CaseDeadlineType(
                id=d.id,
                case_id=d.case_id,
                case_title=d.case_title,
                deadline_date=d.deadline_date,
                deadline_type=d.deadline_type,
                description=d.description,
                is_completed=d.is_completed,
                days_until_deadline=d.days_until_deadline(),
            )
            for d in case.deadlines
        ] if include_deadlines else []
        events = [TimelineEventType.from_db(e) for e in case.timeline_events] if include_timeline else []
        return cls(
            id=case.id,
            user_id=case.user_id,
            case_number=case.case_number,
            case_type=case.case_type,
            jurisdiction=case.jurisdiction,
            status=CaseStatusEnum(case.status.value),
            title=case.title,
            created_at=case.created_at,
            updated_at=case.updated_at,
            documents=docs,
            deadlines=deadlines,
            timeline_events=events,
        )


@strawberry.type
class CaseOutcomeType:
    id: int
    case_id: int
    verdict: str
    appeal_filed: bool
    appeal_result: Optional[str]
    outcome_date: dt.datetime

    @classmethod
    def from_db(cls, outcome: CaseOutcome) -> "CaseOutcomeType":
        return cls(
            id=outcome.id,
            case_id=outcome.case_id,
            verdict=outcome.verdict,
            appeal_filed=outcome.appeal_filed,
            appeal_result=outcome.appeal_result,
            outcome_date=outcome.outcome_date,
        )


@strawberry.scalar
class JSONScalar:
    @staticmethod
    def serialize(value):
        return value


@strawberry.type
class CaseSearchResult:
    total: int
    cases: List[CaseType]


@strawberry.input
class CreateCaseInput:
    case_number: str
    case_type: str
    jurisdiction: str
    title: Optional[str] = None


@strawberry.input
class UpdateCaseStatusInput:
    case_id: int
    status: CaseStatusEnum


@strawberry.input
class CaseFiltersInput:
    status: Optional[CaseStatusEnum] = None
    case_type: Optional[str] = None
    jurisdiction: Optional[str] = None


@strawberry.type
class Query:
    @strawberry.field
    def case(
        self,
        case_id: int,
        include_documents: bool = False,
        include_deadlines: bool = False,
        include_timeline: bool = False,
    ) -> Optional[CaseType]:
        from db.session import get_db_context
        with get_db_context() as db:
            case = get_case_by_id(db, case_id)
            if case is None:
                return None
            return CaseType.from_db(case, include_documents, include_deadlines, include_timeline)

    @strawberry.field
    def cases(
        self,
        user_id: int,
        filters: Optional[CaseFiltersInput] = None,
        limit: int = 50,
        offset: int = 0,
        include_documents: bool = False,
        include_deadlines: bool = False,
        include_timeline: bool = False,
    ) -> List[CaseType]:
        from db.session import get_db_context
        with get_db_context() as db:
            cases = get_user_cases(db, user_id)
            if filters:
                if filters.status:
                    cases = [c for c in cases if c.status == CaseStatus(filters.status.value)]
                if filters.case_type:
                    cases = [c for c in cases if c.case_type == filters.case_type]
                if filters.jurisdiction:
                    cases = [c for c in cases if c.jurisdiction == filters.jurisdiction]
            return [
                CaseType.from_db(c, include_documents, include_deadlines, include_timeline)
                for c in cases[offset : offset + limit]
            ]

    @strawberry.field
    def upcoming_deadlines(
        self,
        user_id: int,
        days_ahead: int = 30,
        priority: Optional[str] = None,
    ) -> List[CaseDeadlineType]:
        from db.session import get_db_context
        with get_db_context() as db:
            deadlines = get_upcoming_deadlines(db, user_id, days_ahead)
            result = []
            for d in deadlines:
                item = CaseDeadlineType(
                    id=d["id"],
                    case_id=d["case_id"],
                    case_title=d["case_title"],
                    deadline_date=d["deadline_date"],
                    deadline_type=d["deadline_type"],
                    description=d.get("description"),
                    is_completed=d.get("is_completed", False),
                    days_until_deadline=d.get("days_until_deadline", 0),
                )
                if priority is None or d.get("priority") == priority:
                    result.append(item)
            return result


@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_case(self, user_id: int, input: CreateCaseInput) -> CaseType:
        from db.session import get_db_context
        with get_db_context() as db:
            case = create_case(
                db, user_id, input.case_number, input.case_type, input.jurisdiction, input.title
            )
            return CaseType.from_db(case)

    @strawberry.mutation
    def update_case_status(self, input: UpdateCaseStatusInput) -> Optional[CaseType]:
        from db.session import get_db_context
        with get_db_context() as db:
            case = update_case_status(db, input.case_id, CaseStatus(input.status.value))
            if case is None:
                return None
            return CaseType.from_db(case)

    @strawberry.mutation
    def delete_case(self, case_id: int) -> bool:
        from db.session import get_db_context
        with get_db_context() as db:
            return delete_case(db, case_id)


schema = strawberry.Schema(query=Query, mutation=Mutation)