from fastapi import Request
from starlette.responses import Response, JSONResponse
from typing import Callable
from database import db_session, reserve_idempotency_key, get_idempotency_response, set_idempotency_response
import structlog

logger = structlog.get_logger(__name__)


async def idempotency_middleware(request: Request, call_next: Callable):
    key = request.headers.get("Idempotency-Key") or request.headers.get("idempotency-key")
    # Only enforce idempotency for unsafe methods
    if not key or request.method not in ("POST", "PUT", "PATCH", "DELETE"):
        return await call_next(request)

    method = request.method
    path = str(request.url.path)

    with db_session() as db:
        ik, created = reserve_idempotency_key(db, key, method, path)
        if not created:
            resp = get_idempotency_response(db, key)
            if resp:
                logger.debug("Idempotency hit: returning stored response", key=key)
                headers = resp.get("headers", {})
                return Response(content=resp.get("body", ""), status_code=resp.get("status_code", 200), headers=headers)
            logger.debug("Idempotency in progress: rejecting duplicate request", key=key)
            return JSONResponse(status_code=409, content={"error": "Idempotency key in progress"})

    response = await call_next(request)

    # Capture body
    body_text = ""
    try:
        if hasattr(response, "body_iterator"):
            # Can't reliably capture streaming bodies
            body_text = ""
        else:
            # Starlette Response: .body is bytes attribute or await response.body()
            if hasattr(response, "body") and isinstance(response.body, (bytes, bytearray)):
                body_text = response.body.decode("utf-8") if response.body else ""
            else:
                body_bytes = await response.body()
                body_text = body_bytes.decode("utf-8") if isinstance(body_bytes, (bytes, bytearray)) else str(body_bytes)
    except Exception:
        body_text = ""

    try:
        with db_session() as db:
            set_idempotency_response(db, key, response.status_code, dict(response.headers), body_text)
    except Exception:
        logger.exception("Failed to persist idempotency response", key=key)

    return response
