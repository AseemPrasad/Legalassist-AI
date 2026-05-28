"""API routes for audio transcription (voice-to-text)."""
import base64
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from core.speech_transcription import TranscriptionEngine
from api.auth import get_current_user, CurrentUser
from api.validation import decode_base64_safe
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import CurrentUser, get_current_user
from core.speech_transcription import (
    TranscriptionInvalidAudio,
    TranscriptionProviderUnavailable,
    transcribe_audio,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["transcription"])


class TranscribeRequest(BaseModel):
    audio_base64: str
    language: str = "en"


@router.post("/transcribe")
def transcribe(
    req: TranscribeRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Transcribe base64-encoded audio using OpenAI Whisper.

    Returns
    -------
    200 ``{"transcription": "<text>"}``
        Successful transcription.
    400
        The supplied audio data is empty or invalid.
    503
        The transcription provider is unavailable or not configured.
    """
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data") from exc

    try:
        text = transcribe_audio(audio_bytes, language=req.language or None)
        return {"transcription": text}
    except TranscriptionInvalidAudio as exc:
        logger.warning("Invalid audio submitted: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TranscriptionProviderUnavailable as exc:
        logger.error("Transcription provider unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Transcription service unavailable") from exc
    except Exception as exc:
        logger.error("Unexpected transcription error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to transcribe audio") from exc
