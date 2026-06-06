from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, field_validator
from typing import Optional, Dict
from sqlalchemy.orm import Session

from core.efiling import EfilingClient
from api.auth import get_current_user, CurrentUser
from database import get_db
from db.models.efiling import UserFiling

router = APIRouter(prefix="/api/efiling", tags=["efiling"])

_MAX_BASE64_BYTES = 10 * 1024 * 1024  # 10 MB


class SubmitRequest(BaseModel):
    court: str
    file_base64: str
    metadata: Optional[Dict] = None

    @field_validator("file_base64")
    @classmethod
    def _limit_base64_size(cls, v: str) -> str:
        if len(v) > _MAX_BASE64_BYTES:
            raise ValueError(f"file_base64 exceeds maximum size of {_MAX_BASE64_BYTES} bytes")
        return v


@router.post("/submit")
async def submit_document(
    req: SubmitRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        res = EfilingClient.submit(req.court, req.file_base64, metadata=req.metadata or {})
        tid = res.get("tracking_id")
        if tid:
            db.add(UserFiling(user_id=int(current_user.user_id), tracking_id=str(tid)))
            db.commit()
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status/{tracking_id}")
async def status(
    tracking_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    filing = (
        db.query(UserFiling)
        .filter(
            UserFiling.user_id == int(current_user.user_id),
            UserFiling.tracking_id == tracking_id,
        )
        .first()
    )
    if not filing:
        raise HTTPException(status_code=404, detail="tracking id not found")
    try:
        res = EfilingClient.get_status(tracking_id)
        return res
    except KeyError:
        raise HTTPException(status_code=404, detail="tracking id not found")
