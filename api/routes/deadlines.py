"""
Deadline Endpoints
GET /api/v1/deadlines/upcoming - Get user's upcoming deadlines
GET /api/v1/deadlines/{deadline_id} - Get deadline details
POST /api/v1/deadlines - Create new deadline
PUT /api/v1/deadlines/{deadline_id} - Update deadline
"""
from fastapi import APIRouter, HTTPException, status, Depends
from api.models import DeadlineResponse, UpcomingDeadlinesResponse
from api.auth import get_current_user, CurrentUser
import structlog
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from database import get_db, CaseDeadline, Case

router = APIRouter(prefix="/api/v1/deadlines", tags=["deadlines"])
logger = structlog.get_logger(__name__)


@router.get(
    "/upcoming",
    response_model=UpcomingDeadlinesResponse,
    summary="Get user's upcoming deadlines"
)
async def get_upcoming_deadlines(
    days: int = 30,
    current_user: CurrentUser = Depends(get_current_user)
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
    
    # Mock deadline data
    now = datetime.utcnow()
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
            status="pending",
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
            status="pending",
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
            status="pending",
            reminder_enabled=True,
            reminder_days=7,
            created_at=now
        ),
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
    current_user: CurrentUser = Depends(get_current_user)
) -> DeadlineResponse:
    """Get complete deadline details"""
    
    logger.info(
        "Fetching deadline",
        deadline_id=deadline_id,
        user_id=current_user.user_id
    )
    
    now = datetime.utcnow()
    return DeadlineResponse(
        deadline_id=deadline_id,
        user_id=current_user.user_id,
        case_id="case_001",
        title="Example Deadline",
        description="Example deadline description",
        due_date=now + timedelta(days=5),
        days_until_due=5,
        priority="high",
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
    title: str,
    due_date: datetime,
    description: str = "",
    priority: str = "medium",
    case_id: str = None,
    reminder_days: int = 7,
    current_user: CurrentUser = Depends(get_current_user)
) -> DeadlineResponse:
    """Create a new deadline"""
    
    logger.info(
        "Creating deadline",
        user_id=current_user.user_id,
        title=title
    )
    
    now = datetime.utcnow()
    days_until = (due_date - now).days
    
    return DeadlineResponse(
        deadline_id="dl_new",
        user_id=current_user.user_id,
        case_id=case_id,
        title=title,
        description=description,
        due_date=due_date,
        days_until_due=days_until,
        priority=priority,
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
