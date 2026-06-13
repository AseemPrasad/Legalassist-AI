"""
Deadline Endpoints
GET /api/v1/deadlines/upcoming - Get user's upcoming deadlines
GET /api/v1/deadlines/{deadline_id} - Get deadline details
POST /api/v1/deadlines - Create new deadline
PUT /api/v1/deadlines/{deadline_id} - Update deadline
"""
from fastapi import APIRouter, HTTPException, status, Depends, Query
from sqlalchemy.orm import Session
from api.models import DeadlineResponse, UpcomingDeadlinesResponse
from api.auth import get_current_user, CurrentUser
from api.dependencies import get_db_rls, evaluate_policy
from core.policy_engine import PolicyDecision
from db.models.cases import Case, CaseDeadline
import structlog
from datetime import datetime, timedelta, timezone
from core.clock import Clock
from database import get_db, UserPreference, CaseDeadline

router = APIRouter(prefix="/api/v1/deadlines", tags=["deadlines"])
logger = structlog.get_logger(__name__)


def _deadline_priority(days_until: int) -> str:
    if days_until <= 3:
        return "critical"
    if days_until <= 10:
        return "high"
    if days_until <= 30:
        return "medium"
    return "low"


def _load_reminder_settings(user_id: str) -> tuple:
    """Load reminder settings from user preferences. Returns (enabled, days)."""
    from database import get_db, UserPreference
    db = None
    try:
        db = get_db()
        pref = db.query(UserPreference).filter(
            UserPreference.user_id == int(user_id)
        ).first()
        if pref:
            enabled = pref.notify_1_day or pref.notify_3_days or pref.notify_10_days or pref.notify_30_days
            if pref.notify_1_day:
                days = 1
            elif pref.notify_3_days:
                days = 3
            elif pref.notify_10_days:
                days = 10
            elif pref.notify_30_days:
                days = 30
            else:
                days = 7
            return (enabled, days)
        return (True, 7)
    finally:
        if db:
            db.close()


def _deadline_priority(days_until: int) -> str:
    """Map days until deadline to priority level."""
    if days_until <= 3:
        return "critical"
    elif days_until <= 7:
        return "high"
    elif days_until <= 14:
        return "medium"
    return "low"


def _normalize_utc_datetime(value) -> datetime:
    """Ensure datetime is UTC-aware. Idempotent."""
    if value is None:
        return Clock.now()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return Clock.now()


def _days_until_due(due_date: datetime, now: datetime) -> int:
    """Calculate calendar days until due date."""
    if due_date is None:
        return 0
    due = _normalize_utc_datetime(due_date)
    now = _normalize_utc_datetime(now)
    return max(0, (due.date() - now.date()).days)


def _deadline_to_response(deadline) -> DeadlineResponse:
    """Convert a CaseDeadline DB object to a DeadlineResponse."""
    from db.session import _to_utc_datetime
    now = Clock.now()
    due_date = _to_utc_datetime(deadline.deadline_date)
    days_until = max(0, (due_date.date() - now.date()).days)
    return DeadlineResponse(
        deadline_id=str(deadline.id),
        user_id=str(deadline.user_id),
        case_id=str(deadline.case_id),
        title=deadline.case_title,
        description=deadline.description or "",
        due_date=due_date,
        days_until_due=days_until,
        priority=_deadline_priority(days_until),
        status=deadline.status.value if hasattr(deadline.status, "value") else str(deadline.status),
        reminder_enabled=True,
        reminder_days=7,
        created_at=deadline.created_at,
    )


@router.get(
    "/upcoming",
    response_model=UpcomingDeadlinesResponse,
    summary="Get user's upcoming deadlines"
)
async def get_upcoming_deadlines(
    days: int = Query(30, ge=1, le=365, description="Look-ahead window in days (max 365)"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls),
) -> UpcomingDeadlinesResponse:
    logger.info(
        "Fetching upcoming deadlines",
        user_id=current_user.user_id,
        days=days,
    )

    reminder_enabled, reminder_days = _load_reminder_settings(current_user.user_id)
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=days)

    deadline_rows = (
        db.query(CaseDeadline)
        .filter(
            CaseDeadline.user_id == int(current_user.user_id),
            CaseDeadline.deadline_date >= now,
            CaseDeadline.deadline_date <= cutoff,
            CaseDeadline.is_completed == False,
        )
        .order_by(CaseDeadline.deadline_date.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    deadlines = [_deadline_to_response(d) for d in deadline_rows]

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
        generated_at=Clock.now()
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
    logger.info(
        "Fetching deadline",
        deadline_id=deadline_id,
        user_id=current_user.user_id
    )
    
    reminder_enabled, reminder_days = _load_reminder_settings(current_user.user_id)
    now = Clock.now()
    return DeadlineResponse(
        deadline_id=deadline_id,
        user_id=current_user.user_id,
    )

    try:
        deadline_id_int = int(deadline_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid deadline ID format",
        )

    deadline = db.query(CaseDeadline).filter(CaseDeadline.id == deadline_id_int).first()
    if not deadline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deadline not found",
        )

    decision = evaluate_policy(current_user, "deadline", "view", deadline, db)
    if decision != PolicyDecision.ALLOW:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this deadline",
        )

    return _deadline_to_response(deadline)


@router.post(
    "",
    response_model=DeadlineResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new deadline"
)
async def create_deadline(
    case_id: int,
    title: str,
    due_date: datetime,
    description: str = "",
    reminder_days: int = 7,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls),
) -> DeadlineResponse:
    """Create a new deadline and persist it to the database."""

    if not case_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="case_id is required",
        )

    logger.info(
        "Creating deadline",
        user_id=current_user.user_id,
        title=title,
    )

    # Verify case access via policy engine
    case = db.query(Case).filter(Case.id == int(case_id)).first()
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found",
        )

    decision = evaluate_policy(current_user, "case", "add_deadline", case, db)
    if decision != PolicyDecision.ALLOW:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to add deadlines to this case",
        )

    now = datetime.now(timezone.utc)
        title=title
    )
    
    reminder_enabled, pref_reminder_days = _load_reminder_settings(current_user.user_id)
    now = Clock.now()
    days_until = max(0, (due_date.date() - now.date()).days)

    deadline = CaseDeadline(
        user_id=current_user.user_id,
        case_id=int(case_id),
        case_title=title,
        deadline_date=due_date,
        deadline_type=_deadline_priority(days_until),
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

    return _deadline_to_response(deadline)


@router.put(
    "/{deadline_id}",
    response_model=DeadlineResponse,
    summary="Update deadline"
)
async def update_deadline(
    deadline_id: int,
    title: str = None,
    due_date: datetime = None,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls),
) -> DeadlineResponse:
    logger.info(
        "Updating deadline",
        deadline_id=deadline_id,
        user_id=current_user.user_id,
    )

    deadline = db.query(CaseDeadline).filter(CaseDeadline.id == deadline_id).first()
    if not deadline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deadline not found",
        )

    decision = evaluate_policy(current_user, "deadline", "update", deadline, db)
    if decision != PolicyDecision.ALLOW:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to update this deadline",
        )

    if title is not None:
        deadline.case_title = title
    if due_date is not None:
        deadline.deadline_date = due_date
    deadline.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(deadline)

    return _deadline_to_response(deadline)
        user_id=current_user.user_id
    )
    
    now = Clock.now()
    effective_due_date = due_date or (now + timedelta(days=7))
    effective_due_date_utc = (
        effective_due_date.replace(tzinfo=timezone.utc)
        if effective_due_date.tzinfo is None
        else effective_due_date.astimezone(timezone.utc)
    )
    days_until = max(0, (effective_due_date_utc.date() - now.date()).days)
    
    db = get_db()
    try:
        deadline = db.query(CaseDeadline).filter(
            CaseDeadline.id == deadline_id,
            CaseDeadline.user_id == int(current_user.user_id)
        ).first()

    if title is not None:
        deadline.case_title = title
    if due_date is not None:
        # Validate via domain layer
        if not DeadlineEngine.validate_not_past(due_date):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Deadline date cannot be in the past",
            )
        deadline.deadline_date = due_date
    deadline.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(deadline)

        if title is not None:
            deadline.case_title = title
        if due_date is not None:
            deadline.deadline_date = due_date
        deadline.updated_at = Clock.now()
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
