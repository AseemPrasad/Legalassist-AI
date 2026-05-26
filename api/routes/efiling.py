from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict
from core.efiling import EfilingClient
from api.auth import get_current_user, CurrentUser

router = APIRouter(prefix="/api/efiling", tags=["efiling"])


class SubmitRequest(BaseModel):
    court: str
    file_base64: str
    metadata: Optional[Dict] = None


@router.post("/submit")
async def submit_document(req: SubmitRequest, current_user: CurrentUser = Depends(get_current_user)):
    try:
        res = EfilingClient.submit(req.court, req.file_base64, metadata=req.metadata or {})
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status/{tracking_id}")
async def status(tracking_id: str, current_user: CurrentUser = Depends(get_current_user)):
    try:
        res = EfilingClient.get_status(tracking_id)
        return res
    except KeyError:
        raise HTTPException(status_code=404, detail="tracking id not found")
