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

router = APIRouter(prefix="/api/v1/deadlines", tags=["deadlines"])
logger = structlog.get_logger(__name__)


def _derive_deadline_status(due_date: datetime) -> str:
    """Return 'overdue' if past due, 'pending' otherwise."""
    if due_date < datetime.now(timezone.utc):
        return "overdue"
    return "pending"


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
    
    now = datetime.now(timezone.utc)
    deadlines = [
        DeadlineResponse(
            deadline_id="dl_001",
            user_id=current_user.user_id,
            case_id="case_001",
            title="Motion Response Due",
            description="Response to plaintiff's motion for summary judgment",
            due_date=now + timedelta(days=3),
            days_until_due=3,
            priority="critical",
            status=_derive_deadline_status(now + timedelta(days=3)),
            reminder_enabled=True,
            reminder_days=7,
            created_at=now
        ),
        DeadlineResponse(
            deadline_id="dl_002",
            user_id=current_user.user_id,
            case_id="case_002",
            title="Filing Deadline",
            description="Appeal filing deadline",
            due_date=now + timedelta(days=10),
            days_until_due=10,
            priority="high",
            status=_derive_deadline_status(now + timedelta(days=10)),
            reminder_enabled=True,
            reminder_days=7,
            created_at=now
        ),
        DeadlineResponse(
            deadline_id="dl_003",
            user_id=current_user.user_id,
            case_id="case_003",
            title="Document Production",
            description="Produce documents per discovery order",
            due_date=now + timedelta(days=21),
            days_until_due=21,
            priority="medium",
            status=_derive_deadline_status(now + timedelta(days=21)),
            reminder_enabled=True,
            reminder_days=7,
            created_at=now
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
        for i, (title, desc, d) in enumerate(mock_items, start=1)
    ]
    
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
        generated_at=datetime.utcnow()
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
    logger.info(
        "Fetching deadline",
        deadline_id=deadline_id,
        user_id=current_user.user_id
    )
    
    now = datetime.now(timezone.utc)
    due = now + timedelta(days=5)
    return DeadlineResponse(
        deadline_id=deadline_id,
        user_id=current_user.user_id,
        case_id="case_001",
        title="Example Deadline",
        description="Example deadline description",
        due_date=due,
        days_until_due=5,
        priority="high",
        status=_derive_deadline_status(due),
        reminder_enabled=True,
        reminder_days=7,
        created_at=now
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
    case_id: str = None,
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

    logger.info(
        "Creating deadline",
        user_id=current_user.user_id,
        title=request.title
    )
    
    now = datetime.now(timezone.utc)
    days_until = (due_date - now).days
    
    return DeadlineResponse(
        deadline_id=str(deadline_id),
        user_id=str(current_user.user_id),
        case_id=str(case["id"]),
        title=title,
        description=description,
        due_date=normalized_due_date,
        days_until_due=days_until,
        priority=priority,
        status=_derive_deadline_status(due_date),
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
    current_user: CurrentUser = Depends(get_current_user)
) -> DeadlineResponse:
    logger.info(
        "Updating deadline",
        deadline_id=deadline_id,
        user_id=current_user.user_id
    )
    
    now = datetime.now(timezone.utc)
    effective_due_date = due_date or (now + timedelta(days=7))
    effective_due_date_utc = (
        effective_due_date.replace(tzinfo=timezone.utc)
        if effective_due_date.tzinfo is None
        else effective_due_date.astimezone(timezone.utc)
    )
    days_until = max(0, (effective_due_date_utc.date() - now.date()).days)
    
    return DeadlineResponse(
        deadline_id=deadline_id,
        user_id=current_user.user_id,
        case_id="case_001",
        title=title or "Updated Deadline",
        description="Updated description",
        due_date=effective_due_date_utc,
        days_until_due=days_until,
        priority=_deadline_priority(days_until),
        status="pending",
        reminder_enabled=True,
        reminder_days=7,
        created_at=updated_deadline.created_at
    )

    logger.info(
        "Updating deadline",
        deadline_id=deadline_id,
        user_id=current_user.user_id
    )
    
    now = datetime.now(timezone.utc)
    due = due_date or (now + timedelta(days=7))
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

    db.commit()
    db.refresh(deadline)

    now = datetime.now(timezone.utc)
    days_until = (deadline.deadline_date - now).days

    logger.info(
        "Reopening deadline",
        deadline_id=deadline_id,
        user_id=current_user.user_id,
        case_id="case_001",
        title=title or "Updated Deadline",
        description="Updated description",
        due_date=due,
        days_until_due=7,
        priority=priority or "medium",
        status=_derive_deadline_status(due),
        reminder_enabled=True,
        reminder_days=7,
        created_at=updated_deadline.created_at
    )


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
