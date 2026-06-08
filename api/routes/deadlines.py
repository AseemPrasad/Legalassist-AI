"""
Deadline Endpoints
GET /api/v1/deadlines/upcoming - Get user's upcoming deadlines
GET /api/v1/deadlines/{deadline_id} - Get deadline details
POST /api/v1/deadlines - Create new deadline
"""
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from api.models import DeadlineResponse, UpcomingDeadlinesResponse
from api.auth import get_current_user, CurrentUser
import structlog
from datetime import datetime, timedelta, timezone

from database import Case, CaseDeadline, get_db

router = APIRouter(prefix="/api/v1/deadlines", tags=["deadlines"])
logger = structlog.get_logger(__name__)


def _deadline_priority(days_until_due: int) -> str:
    if days_until_due <= 3:
        return "critical"
    if days_until_due <= 10:
        return "high"
    if days_until_due <= 30:
        return "medium"
    return "low"


def _require_owned_case(case_id: str | None, current_user: CurrentUser, db: Session) -> Case | None:
    if case_id is None:
        return None

    try:
        case_id_int = int(case_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid case ID format")

    case = db.query(Case).filter(Case.id == case_id_int).first()
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    if current_user.role != "admin" and case.user_id != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    return case


def _require_owned_deadline(deadline_id: str, current_user: CurrentUser, db: Session) -> CaseDeadline:
    try:
        deadline_id_int = int(deadline_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid deadline ID format")

    deadline = db.query(CaseDeadline).filter(CaseDeadline.id == deadline_id_int).first()
    if not deadline:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Deadline not found")

    if current_user.role != "admin" and deadline.user_id != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Deadline not found")

    return deadline


@router.get(
    "/upcoming",
    response_model=UpcomingDeadlinesResponse,
    summary="Get user's upcoming deadlines"
)
async def get_upcoming_deadlines(
    days: int = 30,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UpcomingDeadlinesResponse:
    """
    Get upcoming deadlines for user
    
    - **days**: Look ahead N days (default 30)
    
    Returns sorted list of upcoming deadlines by urgency
    """
    
    logger.info(
        "Fetching upcoming deadlines",
        user_id=current_user.user_id,
        days=days
    )
    now = datetime.now(timezone.utc)
    target_date = (now + timedelta(days=days)).replace(hour=23, minute=59, second=59, microsecond=999999)

    deadline_rows = (
        db.query(CaseDeadline, Case)
        .join(Case, Case.id == CaseDeadline.case_id)
        .filter(
            CaseDeadline.user_id == current_user.user_id,
            CaseDeadline.is_completed.is_(False),
            CaseDeadline.deadline_date > now,
            CaseDeadline.deadline_date <= target_date,
        )
        .order_by(CaseDeadline.deadline_date.asc())
        .all()
    )

    deadlines = []
    for deadline, case in deadline_rows:
        due_date = deadline.deadline_date
        if due_date is not None and due_date.tzinfo is None:
            due_date = due_date.replace(tzinfo=timezone.utc)

        days_until_due = deadline.days_until_deadline()
        deadlines.append(
            DeadlineResponse(
                deadline_id=str(deadline.id),
                user_id=str(deadline.user_id),
                case_id=str(deadline.case_id),
                title=case.title or case.case_number,
                description=deadline.description or "",
                due_date=due_date or now,
                days_until_due=days_until_due,
                priority=_deadline_priority(days_until_due),
                status="pending",
                reminder_enabled=True,
                reminder_days=7,
                created_at=deadline.created_at,
            )
        )
    
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
        generated_at=datetime.now(timezone.utc)
    )


@router.get(
    "/{deadline_id}",
    response_model=DeadlineResponse,
    summary="Get deadline details"
)
async def get_deadline_details(
    deadline_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DeadlineResponse:
    """Get complete deadline details"""
    
    logger.info(
        "Fetching deadline",
        deadline_id=deadline_id,
        user_id=current_user.user_id
    )
    
    deadline = _require_owned_deadline(deadline_id, current_user, db)
    now = datetime.now(timezone.utc)
    due_date = deadline.deadline_date
    if due_date is not None and due_date.tzinfo is None:
        due_date = due_date.replace(tzinfo=timezone.utc)
    days_until = deadline.days_until_deadline()
    return DeadlineResponse(
        deadline_id=str(deadline.id),
        user_id=current_user.user_id,
        case_id=str(deadline.case_id),
        title=deadline.case_title,
        description=deadline.description or "",
        due_date=due_date or now,
        days_until_due=days_until,
        priority=_deadline_priority(days_until),
        status="completed" if deadline.is_completed else ("overdue" if due_date and due_date < now else "pending"),
        reminder_enabled=True,
        reminder_days=7,
        created_at=deadline.created_at
    )


@router.post(
    "",
    response_model=DeadlineResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new deadline"
)
async def create_deadline(
    title: str,
    due_date: datetime,
    description: str = "",
    priority: str = "medium",
    case_id: str = None,
    reminder_days: int = 7,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> DeadlineResponse:
    """Create a new deadline"""
    
    logger.info(
        "Creating deadline",
        user_id=current_user.user_id,
        title=title
    )

    case = _require_owned_case(case_id, current_user, db)

    # Ensure due_date is timezone-aware
    if due_date.tzinfo is None:
        due_date = due_date.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    days_until = max(0, (due_date.date() - now.date()).days)

    deadline = CaseDeadline(
        user_id=current_user.user_id,
        case_id=int(case_id),
        case_title=title,
        deadline_date=due_date,
        deadline_type=priority or _deadline_priority(days_until),
        description=description,
        is_completed=False,
    )
    db.add(deadline)
    db.commit()
    db.refresh(deadline)

    logger.info(
        "Deadline created",
        deadline_id=deadline.id,
        user_id=current_user.user_id,
    )

    return DeadlineResponse(
        deadline_id=str(deadline.id),
        user_id=current_user.user_id,
        case_id=str(deadline.case_id),
        title=deadline.case_title,
        description=deadline.description or "",
        due_date=deadline.deadline_date,
        days_until_due=deadline.days_until_deadline(),
        priority=_deadline_priority(deadline.days_until_deadline()),
        status="pending",
        reminder_enabled=True,
        reminder_days=reminder_days,
        created_at=deadline.created_at
    )


@router.put(
    "/{deadline_id}",
    response_model=DeadlineResponse,
    summary="Update deadline"
)
async def update_deadline(
    deadline_id: str,
    title: str = None,
    due_date: datetime = None,
    priority: str = None,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> DeadlineResponse:
    """Update a deadline"""
    
    logger.info(
        "Updating deadline",
        deadline_id=deadline_id,
        user_id=current_user.user_id
    )
    
    deadline = _require_owned_deadline(deadline_id, current_user, db)
    now = datetime.now(timezone.utc)
    effective_due_date = due_date or deadline.deadline_date
    if effective_due_date is not None and effective_due_date.tzinfo is None:
        effective_due_date = effective_due_date.replace(tzinfo=timezone.utc)
    days_until = ((effective_due_date or now) - now).days
    return DeadlineResponse(
        deadline_id=str(deadline.id),
        user_id=current_user.user_id,
        case_id=str(deadline.case_id),
        title=title or deadline.case_title,
        description=deadline.description or "",
        due_date=effective_due_date or now,
        days_until_due=days_until,
        priority=priority or _deadline_priority(days_until),
        status="completed" if deadline.is_completed else ("overdue" if effective_due_date and effective_due_date < now else "pending"),
        reminder_enabled=True,
        reminder_days=7,
        created_at=deadline.created_at
    )
