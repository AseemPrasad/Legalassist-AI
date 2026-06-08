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


def _deadline_priority(days_until_due: int) -> str:
    if days_until_due <= 3:
        return "critical"
    if days_until_due <= 10:
        return "high"
    if days_until_due <= 30:
        return "medium"
    return "low"


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
    
    # Mock deadline data — priority is computed from days_until_due
    now = datetime.now(timezone.utc)
    mock_items = [
        ("Motion Response Due", "Response to plaintiff's motion for summary judgment", 3),
        ("Filing Deadline", "Appeal filing deadline", 10),
        ("Document Production", "Produce documents per discovery order", 21),
    ]
    deadlines = [
        DeadlineResponse(
            deadline_id=f"dl_{i:03d}",
            user_id=current_user.user_id,
            case_id=f"case_{i:03d}",
            title=title,
            description=desc,
            due_date=now + timedelta(days=d),
            days_until_due=d,
            priority=_deadline_priority(d),
            status="pending",
            reminder_enabled=True,
            reminder_days=7,
            created_at=now
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
    days_until = 5
    return DeadlineResponse(
        deadline_id=deadline_id,
        user_id=current_user.user_id,
        case_id="case_001",
        title="Example Deadline",
        description="Example deadline description",
        due_date=now + timedelta(days=days_until),
        days_until_due=days_until,
        priority=_deadline_priority(days_until),
        status="pending",
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
    logger.info(
        "Creating deadline",
        user_id=current_user.user_id,
        title=request.title
    )
    
    now = datetime.now(timezone.utc)
    due_date_utc = due_date.replace(tzinfo=timezone.utc) if due_date.tzinfo is None else due_date.astimezone(timezone.utc)
    days_until = max(0, (due_date_utc.date() - now.date()).days)
    
    return DeadlineResponse(
        deadline_id=str(deadline_id),
        user_id=str(current_user.user_id),
        case_id=str(case["id"]),
        title=title,
        description=description,
        due_date=due_date_utc,
        days_until_due=days_until,
        priority=_deadline_priority(days_until),
        status="pending",
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

    db = None
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
        priority=_deadline_priority(days_until),
        status=updated_deadline.status,
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
