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
               description, created_at, updated_at, is_completed
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
    days: int = 30,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls),
) -> UpcomingDeadlinesResponse:
    """
    Get upcoming deadlines for user
    
    - **days**: Look ahead N days (default 30)
    
    Returns sorted list of upcoming deadlines by urgency
    """
    
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
              AND d.is_completed = 0
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
                c.title AS case_title_from_case,
                c.case_number AS case_number
            FROM case_deadlines AS d
            JOIN cases AS c ON c.id = d.case_id
            WHERE d.user_id = :user_id
              AND d.is_completed = 0
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
                status="pending",
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
        status="completed" if deadline["is_completed"] else ("overdue" if due_date and due_date < now else "pending"),
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
    title: str,
    due_date: datetime,
    description: str = "",
    priority: str = "medium",
    case_id: str = None,
    reminder_days: int = 7,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls)
) -> DeadlineResponse:
    """Create a new deadline"""
    
    logger.info(
        "Creating deadline",
        user_id=current_user.user_id,
        title=title
    )
    
    if case_id is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="case_id is required")

    case = _require_owned_case(case_id, current_user, db)

    now = datetime.now(timezone.utc)
    normalized_due_date = _normalize_utc_datetime(due_date)
    days_until = _days_until_due(normalized_due_date, now)

    insert_result = db.execute(
        text(
            """
            INSERT INTO case_deadlines (
                user_id, case_id, case_title, deadline_date, deadline_type,
                description, created_at, updated_at, is_completed
            ) VALUES (
                :user_id, :case_id, :case_title, :deadline_date, :deadline_type,
                :description, :created_at, :updated_at, :is_completed
            )
            """
        ),
        {
            "user_id": current_user.user_id,
            "case_id": case["id"],
            "case_title": case["title"] or case["case_number"],
            "deadline_date": normalized_due_date,
            "deadline_type": priority or "manual",
            "description": description or title,
            "created_at": now,
            "updated_at": now,
            "is_completed": 0,
        },
    )
    deadline_id = insert_result.lastrowid
    db.commit()
    
    return DeadlineResponse(
        deadline_id=str(deadline_id),
        user_id=str(current_user.user_id),
        case_id=str(case["id"]),
        title=title,
        description=description,
        due_date=normalized_due_date,
        days_until_due=days_until,
        priority=priority or _deadline_priority(days_until),
        status="pending",
        reminder_enabled=True,
        reminder_days=reminder_days,
        created_at=now
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
    db: Session = Depends(get_db_rls)
) -> DeadlineResponse:
    """Update a deadline"""
    
    logger.info(
        "Updating deadline",
        deadline_id=deadline_id,
        user_id=current_user.user_id
    )
    
    deadline = _require_owned_deadline(deadline_id, current_user, db)
    now = datetime.now(timezone.utc)
    effective_due_date = _normalize_utc_datetime(due_date or deadline["deadline_date"])
    days_until = _days_until_due(effective_due_date, now)

    db.execute(
        text("""
            UPDATE case_deadlines
            SET case_title = :title,
                deadline_date = :due_date,
                updated_at = :now
            WHERE id = :deadline_id
        """),
        {
            "title": title or deadline["case_title"],
            "due_date": effective_due_date,
            "now": now,
            "deadline_id": deadline["id"],
        },
    )
    db.commit()

    return DeadlineResponse(
        deadline_id=str(deadline["id"]),
        user_id=str(current_user.user_id),
        case_id=str(deadline["case_id"]),
        title=title or deadline["case_title"],
        description=deadline["description"] or "",
        due_date=effective_due_date or now,
        days_until_due=days_until,
        priority=priority or _deadline_priority(days_until),
        status="completed" if deadline["is_completed"] else ("overdue" if effective_due_date and effective_due_date < now else "pending"),
        reminder_enabled=True,
        reminder_days=7,
        created_at=deadline["created_at"]
    )



