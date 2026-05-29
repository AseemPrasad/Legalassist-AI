from __future__ import annotations

from typing import Callable

import structlog
from fastapi import Request, status
from fastapi.responses import JSONResponse

from api.middlewares._shared import ANALYTICS_PATH_PREFIXES, SKIP_PATHS, UPLOAD_PATH_PREFIXES
from api.validation import ValidationConfig

logger = structlog.get_logger(__name__)


def _request_size_limit_for_path(path: str) -> int:
    if any(path.startswith(prefix) for prefix in UPLOAD_PATH_PREFIXES):
        return ValidationConfig.MAX_UPLOAD_SIZE
    if any(path.startswith(prefix) for prefix in ANALYTICS_PATH_PREFIXES):
        return ValidationConfig.MAX_ANALYTICS_PAYLOAD
    return ValidationConfig.MAX_JSON_BODY


async def request_size_limit_middleware(request: Request, call_next: Callable):
    if request.url.path in SKIP_PATHS:
        return await call_next(request)

    transfer_encoding = request.headers.get("transfer-encoding", "").lower()
    content_length = request.headers.get("content-length")
    max_size = _request_size_limit_for_path(request.url.path)

    if any(request.url.path.startswith(p) for p in UPLOAD_PATH_PREFIXES):
        if content_length is None:
            return JSONResponse(
                status_code=status.HTTP_411_LENGTH_REQUIRED,
                content={
                    "error_code": "LENGTH_REQUIRED",
                    "message": "Content-Length header is required for upload endpoints.",
                },
            )

    if "chunked" in transfer_encoding:
        return JSONResponse(
            status_code=status.HTTP_411_LENGTH_REQUIRED,
            content={
                "error_code": "CHUNKED_ENCODING_NOT_SUPPORTED",
                "message": "Chunked transfer encoding is not supported. Provide Content-Length header.",
            },
        )

    if content_length is None:
        return JSONResponse(
            status_code=status.HTTP_411_LENGTH_REQUIRED,
            content={
                "error_code": "LENGTH_REQUIRED",
                "message": "Content-Length header is required for all requests.",
            },
        )

    try:
        content_length_bytes = int(content_length)
    except (TypeError, ValueError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error_code": "INVALID_CONTENT_LENGTH",
                "message": "Content-Length must be a valid integer.",
            },
        )

    if content_length_bytes > max_size:
        logger.warning(
            "request_size_limit_exceeded",
            path=request.url.path,
            content_length=content_length_bytes,
            max_size=max_size,
            size_mb=round(content_length_bytes / 1024 / 1024, 2),
        )
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "error_code": "PAYLOAD_TOO_LARGE",
                "message": (
                    f"Request body too large: {round(content_length_bytes / 1024 / 1024, 2)} MB "
                    f"(max {round(max_size / 1024 / 1024, 2)} MB)"
                ),
            },
        )

    body_bytes = bytearray()
    async for chunk in request.stream():
        body_bytes.extend(chunk)
        if len(body_bytes) > max_size:
            logger.warning(
                "request_body_exceeded_limit_during_stream",
                path=request.url.path,
                max_size=max_size,
                attempted_bytes=len(body_bytes),
            )
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error_code": "PAYLOAD_TOO_LARGE",
                    "message": f"Request body exceeded maximum allowed size of {round(max_size / 1024 / 1024, 2)} MB",
                },
            )

    request._body = bytes(body_bytes)

    return await call_next(request)
