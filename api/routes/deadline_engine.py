from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from core.deadline_engine import calculate_deadline
from api.auth import get_current_user, CurrentUser

router = APIRouter(prefix="/api/deadline", tags=["deadline"])


class DeadlineRequest(BaseModel):
    start: str
    business_days: int
    timezone: Optional[str] = "UTC"
    exclude_weekends: Optional[bool] = True
    holidays: Optional[List[str]] = None
    jurisdiction: Optional[str] = None
    emergency_extension_days: Optional[int] = 0
    filing_time: Optional[str] = None


@router.post("/calculate")
async def calculate_endpoint(req: DeadlineRequest, current_user: CurrentUser = Depends(get_current_user)):
    try:
        res = calculate_deadline(
            start=req.start,
            business_days=req.business_days,
            timezone=req.timezone or "UTC",
            exclude_weekends=req.exclude_weekends,
            holidays=req.holidays,
            jurisdiction=req.jurisdiction,
            emergency_extension_days=req.emergency_extension_days,
            filing_time=req.filing_time,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
