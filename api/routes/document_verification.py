"""Document registration and verification API routes (simulated blockchain)."""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from core.blockchain_sim import BlockchainSimulator
from core.document_verifier import register_document, verify_document
from api.auth import get_current_user, CurrentUser
from api.validation import decode_base64_safe
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["document_verification"])

# For simplicity, keep a process-global blockchain simulator instance.
_GLOBAL_CHAIN = BlockchainSimulator()


class RegisterRequest(BaseModel):
    file_base64: str
    filename: str = "document"


class VerifyRequest(BaseModel):
    file_base64: str


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
