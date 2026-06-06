"""Document registration and verification API routes (simulated blockchain)."""
import os
import logging

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, field_validator

from core.blockchain_sim import BlockchainSimulator
from core.document_verifier import register_document, verify_document
from api.auth import get_current_user, CurrentUser
from api.validation import decode_base64_safe

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["document_verification"])

_MAX_BASE64_BYTES = 10 * 1024 * 1024  # 10 MB

# Use Redis-backed blockchain so all workers share the same chain.
_redis_url = os.environ.get("REDIS_URL") or ""
if _redis_url:
    import redis as redis_lib
    _redis_client = redis_lib.from_url(_redis_url, socket_connect_timeout=5)
else:
    _redis_client = None

_GLOBAL_CHAIN = BlockchainSimulator(redis_client=_redis_client, redis_key="blockchain:documents")


class RegisterRequest(BaseModel):
    file_base64: str
    filename: str = "document"

    @field_validator("file_base64")
    @classmethod
    def _limit_base64_size(cls, v: str) -> str:
        if len(v) > _MAX_BASE64_BYTES:
            raise ValueError(f"file_base64 exceeds maximum size of {_MAX_BASE64_BYTES} bytes")
        return v


class VerifyRequest(BaseModel):
    file_base64: str

    @field_validator("file_base64")
    @classmethod
    def _limit_base64_size(cls, v: str) -> str:
        if len(v) > _MAX_BASE64_BYTES:
            raise ValueError(f"file_base64 exceeds maximum size of {_MAX_BASE64_BYTES} bytes")
        return v


@router.post("/documents/register")
def register(req: RegisterRequest, current_user: CurrentUser = Depends(get_current_user)):
    try:
        file_bytes = decode_base64_safe(req.file_base64)
        res = register_document(file_bytes, chain=_GLOBAL_CHAIN, metadata={"filename": req.filename})
        return res
    except Exception as e:
        logger.error("Register error: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to register document")


@router.post("/documents/verify")
def verify(req: VerifyRequest, current_user: CurrentUser = Depends(get_current_user)):
    try:
        file_bytes = decode_base64_safe(req.file_base64)
        res = verify_document(file_bytes, chain=_GLOBAL_CHAIN)
        return res
    except Exception as e:
        logger.error("Verify error: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to verify document")
