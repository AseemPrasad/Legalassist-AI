"""API routes for audio transcription (voice-to-text)."""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from core.speech_transcription import TranscriptionEngine
from api.auth import get_current_user, CurrentUser
from api.validation import decode_base64_safe
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["transcription"])


class TranscribeRequest(BaseModel):
    audio_base64: str
    language: str = "en"


@router.post("/transcribe")
def transcribe(req: TranscribeRequest, current_user: CurrentUser = Depends(get_current_user)):
    try:
        audio_bytes = decode_base64_safe(req.audio_base64)
        engine = TranscriptionEngine()
        text = engine.transcribe_bytes(audio_bytes, language=req.language)
        return {"transcription": text}
    except Exception as e:
        logger.error("Transcription error: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to transcribe audio")
