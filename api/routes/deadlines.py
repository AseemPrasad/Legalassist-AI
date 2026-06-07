"""
Deadline Endpoints
GET /api/v1/deadlines/upcoming - Get user's upcoming deadlines
GET /api/v1/deadlines/{deadline_id} - Get deadline details
POST /api/v1/deadlines - Create new deadline
PUT /api/v1/deadlines/{deadline_id} - Update deadline
"""
from fastapi import APIRouter, HTTPException, status, Depends, Query
from api.models import DeadlineResponse, UpcomingDeadlinesResponse
from api.auth import get_current_user, CurrentUser
import structlog
from datetime import datetime, timezone, timedelta
from database import get_db, CaseDeadline

router = APIRouter(prefix="/api/v1/deadlines", tags=["deadlines"])
logger = structlog.get_logger(__name__)


def _derive_priority(days_until: int) -> str:
    if days_until <= 3:
        return "critical"
    if days_until <= 7:
        return "high"
    if days_until <= 14:
        return "medium"
    return "low"


def _compute_status(is_completed: bool, due_date: datetime) -> str:
    if is_completed:
        return "completed"
    if due_date < datetime.now(timezone.utc):
        return "overdue"
    return "pending"


def _deadline_to_response(d: CaseDeadline) -> DeadlineResponse:
    now = datetime.now(timezone.utc)
    due = d.deadline_date
    days_until = (due - now).days
    return DeadlineResponse(
        deadline_id=str(d.id),
        user_id=str(d.user_id),
        case_id=str(d.case_id) if d.case_id else None,
        title=d.case_title,
        description=d.description or f"{d.deadline_type} deadline",
        due_date=due,
        days_until_due=days_until,
        priority=_derive_priority(days_until),
        status=_compute_status(d.is_completed, due),
        reminder_enabled=True,
        reminder_days=7,
        created_at=d.created_at,
    )


@router.get(
    "/upcoming",
    response_model=UpcomingDeadlinesResponse,
    summary="Get user's upcoming deadlines"
)
async def get_upcoming_deadlines(
    days: int = Query(30, ge=1, le=365, description="Look-ahead window in days (max 365)"),
    current_user: CurrentUser = Depends(get_current_user)
) -> UpcomingDeadlinesResponse:
    logger.info(
        "Fetching upcoming deadlines",
        user_id=current_user.user_id,
        days=days,
        limit=limit,
        offset=offset,
    )

    db = None
    try:
        db = get_db()
        now = datetime.now(timezone.utc)
        look_ahead = now + timedelta(days=days)

        query = db.query(CaseDeadline).filter(
            CaseDeadline.user_id == int(current_user.user_id),
            CaseDeadline.deadline_date >= now,
            CaseDeadline.deadline_date <= look_ahead,
        ).order_by(CaseDeadline.deadline_date.asc())

        rows = query.all()

        deadlines = [_deadline_to_response(r) for r in rows]

        critical = sum(1 for d in deadlines if d.priority == "critical")
        high = sum(1 for d in deadlines if d.priority == "high")
        medium = sum(1 for d in deadlines if d.priority == "medium")
        low = sum(1 for d in deadlines if d.priority == "low")

        return UpcomingDeadlinesResponse(
            user_id=current_user.user_id,
            total_deadlines=len(deadlines),
            critical_count=critical,
            high_count=high,
            medium_count=medium,
            low_count=low,
            deadlines=deadlines,
            generated_at=datetime.now(timezone.utc),
        )
    finally:
        if db:
            db.close()


@router.get(
    "/{deadline_id}",
    response_model=DeadlineResponse,
    summary="Get deadline details"
)
async def get_deadline_details(
    deadline_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls),
) -> DeadlineResponse:
    logger.info(
        "Fetching deadline",
        deadline_id=deadline_id,
        user_id=current_user.user_id
    )

    db = None
    try:
        db = get_db()
        deadline = db.query(CaseDeadline).filter(
            CaseDeadline.id == int(deadline_id),
            CaseDeadline.user_id == int(current_user.user_id),
        ).first()

        if not deadline:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Deadline not found"
            )

        return _deadline_to_response(deadline)
    finally:
        if db:
            db.close()


@router.post(
    "",
    response_model=DeadlineResponse,
    summary="Create new deadline"
)
async def create_deadline(
    case_id: int,
    title: str,
    due_date: datetime,
    description: str = "",
    deadline_type: str = "filing",
    priority: str = "medium",
    reminder_enabled: bool = True,
    reminder_days: int = 7,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DeadlineResponse:
    logger.info(
        "Creating deadline",
        user_id=current_user.user_id,
        title=request.title
    )

    db = None
    try:
        db = get_db()
        now = datetime.now(timezone.utc)
        deadline = CaseDeadline(
            user_id=int(current_user.user_id),
            case_id=int(case_id) if case_id else None,
            case_title=title,
            deadline_date=due_date,
            deadline_type="custom",
            description=description,
            created_at=now,
            updated_at=now,
            is_completed=False,
        )
        db.add(deadline)
        db.commit()
        db.refresh(deadline)

        days_until = (deadline.deadline_date - datetime.now(timezone.utc)).days
        return DeadlineResponse(
            deadline_id=str(deadline.id),
            user_id=str(deadline.user_id),
            case_id=str(deadline.case_id) if deadline.case_id else None,
            title=deadline.case_title,
            description=deadline.description or f"{deadline.deadline_type} deadline",
            due_date=deadline.deadline_date,
            days_until_due=days_until,
            priority=priority,
            status="pending",
            reminder_enabled=True,
            reminder_days=reminder_days,
            created_at=deadline.created_at,
        )
    except Exception:
        if db:
            db.rollback()
        raise
    finally:
        if db:
            db.close()


@router.put(
    "/{deadline_id}",
    response_model=DeadlineResponse,
    summary="Update deadline"
)
async def update_deadline(
    deadline_id: int,
    title: str = None,
    due_date: datetime = None,
    deadline_type: str = None,
    priority: str = None,
    description: str = None,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DeadlineResponse:
    logger.info(
        "Updating deadline",
        deadline_id=deadline_id,
        user_id=current_user.user_id
    )

    db = None
    try:
        db = get_db()
        deadline = db.query(CaseDeadline).filter(
            CaseDeadline.id == int(deadline_id),
            CaseDeadline.user_id == int(current_user.user_id),
        ).first()

        if not deadline:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Deadline not found"
            )

        if title is not None:
            deadline.case_title = title
        if due_date is not None:
            deadline.deadline_date = due_date
        deadline.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(deadline)

        return _deadline_to_response(deadline)
    except Exception:
        if db:
            db.rollback()
        raise
    finally:
        if db:
            db.close()
