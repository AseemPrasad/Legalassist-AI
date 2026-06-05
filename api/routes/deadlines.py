"""
Deadline Endpoints
GET /api/v1/deadlines/upcoming - Get user's upcoming deadlines
GET /api/v1/deadlines/{deadline_id} - Get deadline details
POST /api/v1/deadlines - Create new deadline
"""
from fastapi import APIRouter, HTTPException, status, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from api.models import DeadlineResponse, UpcomingDeadlinesResponse
from api.auth import get_current_user, CurrentUser
import structlog
from datetime import datetime, timedelta, timezone

from database import Case, CaseDeadline
from api.dependencies import get_db_rls
from core.deadline_engine import get_deadline_first_action

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

    query = "SELECT id, user_id, case_number, title FROM cases WHERE id = :case_id"
    params = {"case_id": case_id_int}
    if current_user.role != "admin":
        query += " AND user_id = :user_id"
        params["user_id"] = current_user.user_id

    case = db.execute(text(query), params).mappings().first()
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    return case


def _require_owned_deadline(deadline_id: str, current_user: CurrentUser, db: Session) -> CaseDeadline:
    try:
        deadline_id_int = int(deadline_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid deadline ID format")

    query = """
        SELECT id, user_id, case_id, case_title, deadline_date, deadline_type,
               description, created_at, updated_at, is_completed, status
        FROM case_deadlines
        WHERE id = :deadline_id
    """
    params = {"deadline_id": deadline_id_int}
    if current_user.role != "admin":
        query += " AND user_id = :user_id"
        params["user_id"] = current_user.user_id

    deadline = db.execute(text(query), params).mappings().first()
    if not deadline:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Deadline not found")

    return deadline


def _normalize_utc_datetime(value: datetime) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _days_until_due(due_date: datetime, now: datetime) -> int:
    normalized_due_date = _normalize_utc_datetime(due_date)
    normalized_now = _normalize_utc_datetime(now)
    return max(0, (normalized_due_date.date() - normalized_now.date()).days)


@router.get(
    "/upcoming",
    response_model=UpcomingDeadlinesResponse,
    summary="Get user's upcoming deadlines"
)
async def get_upcoming_deadlines_endpoint(
    days: int = 30,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls),
) -> UpcomingDeadlinesResponse:
    """Get upcoming deadlines for user"""

    logger.info(
        "Fetching upcoming deadlines",
        user_id=current_user.user_id,
        days=days,
        limit=limit,
        offset=offset,
    )
    now = datetime.now(timezone.utc)
    target_date = (now + timedelta(days=days)).replace(hour=23, minute=59, second=59, microsecond=999999)

    base_params = {"user_id": current_user.user_id, "now": now, "target_date": target_date}
    count_row = db.execute(
        text(
            """
            SELECT COUNT(*) AS total_deadlines
            FROM case_deadlines AS d
            JOIN cases AS c ON c.id = d.case_id
            WHERE d.user_id = :user_id
              AND d.status = 'active'
              AND d.deadline_date > :now
              AND d.deadline_date <= :target_date
            """
        ),
        base_params,
    ).mappings().first()
    total_deadlines = int(count_row["total_deadlines"]) if count_row else 0

    deadline_rows = db.execute(
        text(
            """
            SELECT
                d.id AS deadline_id,
                d.user_id,
                d.case_id,
                d.case_title,
                d.deadline_date,
                d.deadline_type,
                d.description,
                d.created_at,
                d.updated_at,
                d.is_completed,
                d.status,
                c.title AS case_title_from_case,
                c.case_number AS case_number
            FROM case_deadlines AS d
            JOIN cases AS c ON c.id = d.case_id
            WHERE d.user_id = :user_id
              AND d.status = 'active'
              AND d.deadline_date > :now
              AND d.deadline_date <= :target_date
                        ORDER BY d.deadline_date ASC, d.id ASC
                        LIMIT :limit OFFSET :offset
            """
        ),
                {**base_params, "limit": limit, "offset": offset},
    ).mappings().all()

    deadlines = []
    for deadline in deadline_rows:
        due_date = _normalize_utc_datetime(deadline["deadline_date"])
        days_until_due = _days_until_due(due_date, now)
        deadlines.append(
            DeadlineResponse(
                deadline_id=str(deadline["deadline_id"]),
                user_id=str(deadline["user_id"]),
                case_id=str(deadline["case_id"]),
                title=deadline["case_title_from_case"] or deadline["case_number"],
                description=deadline["description"] or "",
                due_date=due_date or now,
                days_until_due=days_until_due,
                priority=_deadline_priority(days_until_due),
                status=deadline["status"] or "active",
                reminder_enabled=True,
                reminder_days=7,
                created_at=deadline["created_at"],
            )
        )
    
    critical = sum(1 for d in deadlines if d.priority == "critical")
    high = sum(1 for d in deadlines if d.priority == "high")
    medium = sum(1 for d in deadlines if d.priority == "medium")
    low = sum(1 for d in deadlines if d.priority == "low")
    
    return UpcomingDeadlinesResponse(
        user_id=str(current_user.user_id),
        total_deadlines=total_deadlines,
        limit=limit,
        offset=offset,
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
    db: Session = Depends(get_db_rls),
) -> DeadlineResponse:
    """Get complete deadline details"""

    logger.info(
        "Fetching deadline",
        deadline_id=deadline_id,
        user_id=current_user.user_id
    )
    
    deadline = _require_owned_deadline(deadline_id, current_user, db)
    now = datetime.now(timezone.utc)
    due_date = _normalize_utc_datetime(deadline["deadline_date"])
    days_until = _days_until_due(due_date, now)
    return DeadlineResponse(
        deadline_id=str(deadline["id"]),
        user_id=str(current_user.user_id),
        case_id=str(deadline["case_id"]),
        title=deadline["case_title"],
        description=deadline["description"] or "",
        due_date=due_date or now,
        days_until_due=days_until,
        priority=_deadline_priority(days_until),
        status=deadline["status"] or ("completed" if deadline["is_completed"] else "active"),
        reminder_enabled=True,
        reminder_days=7,
        created_at=deadline["created_at"]
    )


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
    db: Session = Depends(get_db_rls)
) -> DeadlineResponse:
    """Create a new deadline"""

    logger.info(
        "Creating deadline",
        user_id=current_user.user_id,
        title=request.title
    )
    
    now = datetime.utcnow()
    if due_date.tzinfo is None:
        due_date = due_date.replace(tzinfo=timezone.utc)
    if due_date < now.replace(tzinfo=timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Deadline date must be in the future"
        )
    days_until = (due_date - now).days
    
    return DeadlineResponse(
        deadline_id=str(deadline_id),
        user_id=str(current_user.user_id),
        case_id=str(case["id"]),
        title=title,
        description=description,
        due_date=normalized_due_date,
        days_until_due=days_until,
        priority=priority or _deadline_priority(days_until),
        status="active",
        reminder_enabled=True,
        reminder_days=request.reminder_days,
        created_at=now
    )


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
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls)
) -> DeadlineResponse:
    """Update a deadline"""

    logger.info(
        "Updating deadline",
        deadline_id=deadline_id,
        user_id=current_user.user_id
    )
    
    # In production, fetch and update from database
    now = datetime.utcnow()
    if due_date is not None:
        if due_date.tzinfo is None:
            due_date = due_date.replace(tzinfo=timezone.utc)
        if due_date < now.replace(tzinfo=timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Deadline date must be in the future"
            )
    return DeadlineResponse(
        deadline_id=str(updated_deadline.id),
        user_id=str(updated_deadline.user_id),
        case_id=str(updated_deadline.case_id),
        title=updated_deadline.case_title,
        description=updated_deadline.description or "",
        due_date=due_date or now,
        days_until_due=days_until,
        priority=updated_deadline.deadline_type or _deadline_priority(days_until),
        status=updated_deadline.status,
        reminder_enabled=True,
        reminder_days=7,
        created_at=updated_deadline.created_at
    )


@router.post(
    "/{deadline_id}/reopen",
    response_model=DeadlineResponse,
    summary="Reopen a deadline"
)
async def reopen_deadline(
    deadline_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls)
) -> DeadlineResponse:
    """Reopen a completed deadline to active status with validation and audit trail"""
    
    logger.info(
        "Reopening deadline",
        deadline_id=deadline_id,
        user_id=current_user.user_id
    )
    
    # Require ownership / fetch
    deadline = _require_owned_deadline(deadline_id, current_user, db)
    
    from db.case_service import transition_deadline
    
    try:
        updated_deadline = transition_deadline(
            db=db,
            deadline_id=int(deadline_id),
            target_status="active",
            actor_user_id=current_user.user_id
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
        
    now = datetime.now(timezone.utc)
    due_date = _normalize_utc_datetime(updated_deadline.deadline_date)
    days_until = _days_until_due(due_date, now)
    
    return DeadlineResponse(
        deadline_id=str(updated_deadline.id),
        user_id=str(updated_deadline.user_id),
        case_id=str(updated_deadline.case_id),
        title=updated_deadline.case_title,
        description=updated_deadline.description or "",
        due_date=due_date or now,
        days_until_due=days_until,
        priority=updated_deadline.deadline_type or _deadline_priority(days_until),
        status=updated_deadline.status,
        reminder_enabled=True,
        reminder_days=7,
        created_at=updated_deadline.created_at
    )



