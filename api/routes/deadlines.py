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
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from database import get_db, CaseDeadline, Case

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
async def get_upcoming_deadlines(
    days: int = Query(30, ge=1, le=365, description="Look-ahead window in days (max 365)"),
    current_user: CurrentUser = Depends(get_current_user)
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
    db: Session = Depends(get_db),
) -> DeadlineResponse:
    """Create a new deadline and persist it to the database."""

    if not case_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="case_id is required",
        )

    if due_date.tzinfo is None:
        due_date = due_date.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    if due_date < now:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Deadline date must be in the future",
        )

    try:
        normalized_case_id = int(case_id)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="case_id must be an integer",
        )

    case = db.query(Case).filter(Case.id == normalized_case_id).first()
    if not case or str(case.user_id) != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found",
        )

    case_title = case.title or case.case_number

    try:
        deadline = create_case_deadline(
            db=db,
            user_id=int(current_user.user_id),
            case_id=normalized_case_id,
            case_title=case_title,
            deadline_date=due_date,
            deadline_type=priority,
            description=description,
        )
    except (ValueError, PermissionError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    days_until = (deadline.deadline_date - now).days

    logger.info(
        "Deadline created",
        deadline_id=deadline.id,
        user_id=current_user.user_id,
        case_id=normalized_case_id,
    )

    return DeadlineResponse(
        deadline_id=str(deadline.id),
        user_id=str(deadline.user_id),
        case_id=str(deadline.case_id),
        title=deadline.case_title,
        description=deadline.description or "",
        due_date=deadline.deadline_date,
        days_until_due=max(0, days_until),
        priority=deadline.deadline_type,
        status="completed" if deadline.is_completed else "pending",
        reminder_enabled=True,
        reminder_days=reminder_days,
        created_at=deadline.created_at,
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
    description: str = None,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DeadlineResponse:
    """Update a deadline and persist changes to the database."""

    try:
        deadline_pk = int(deadline_id)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="deadline_id must be an integer",
        )

    deadline = db.query(CaseDeadline).filter(CaseDeadline.id == deadline_pk).first()
    if not deadline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deadline not found",
        )

    case = db.query(Case).filter(Case.id == deadline.case_id).first()
    if not case or str(case.user_id) != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deadline not found",
        )

    if title is not None:
        deadline.case_title = title
    if due_date is not None:
        if due_date.tzinfo is None:
            due_date = due_date.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        if due_date < now:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Deadline date must be in the future",
            )
        deadline.deadline_date = due_date
    if priority is not None:
        deadline.deadline_type = priority
    if description is not None:
        deadline.description = description

    db.commit()
    db.refresh(deadline)

    now = datetime.now(timezone.utc)
    days_until = (deadline.deadline_date - now).days

    logger.info(
        "Deadline updated",
        deadline_id=deadline.id,
        user_id=current_user.user_id,
    )

    return DeadlineResponse(
        deadline_id=str(deadline.id),
        user_id=str(deadline.user_id),
        case_id=str(deadline.case_id),
        title=deadline.case_title,
        description=deadline.description or "",
        due_date=deadline.deadline_date,
        days_until_due=max(0, days_until),
        priority=deadline.deadline_type,
        status="completed" if deadline.is_completed else "pending",
        reminder_enabled=True,
        reminder_days=7,
        created_at=deadline.created_at,
    )



