"""Shared structured API error helpers."""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from observability.instrumentation import generate_correlation_id


class StructuredAPIError(HTTPException):
    """HTTP exception that carries a standard error code."""

    def __init__(self, status_code: int, error_code: str, message: str):
        super().__init__(status_code=status_code, detail=message)
        self.error_code = error_code
        self.message = message


def get_request_id(request: Optional[Request] = None, fallback: Optional[str] = None) -> str:
    if fallback:
        return fallback

    if request is not None:
        request_id = getattr(request.state, "request_id", None)
        if request_id:
            return request_id

        for header_name in ("X-Request-Id", "X-Correlation-Id"):
            header_value = request.headers.get(header_name)
            if header_value:
                return header_value

    return generate_correlation_id()


def build_error_payload(error_code: str, message: str, request_id: str) -> dict[str, str]:
    return {
        "error_code": error_code,
        "message": message,
        "request_id": request_id,
    }


def structured_error_response(
    status_code: int,
    error_code: str,
    message: str,
    request: Optional[Request] = None,
    request_id: Optional[str] = None,
) -> JSONResponse:
    resolved_request_id = get_request_id(request, request_id)
    return JSONResponse(
        status_code=status_code,
        content=build_error_payload(error_code, message, resolved_request_id),
        headers={"X-Request-Id": resolved_request_id},
    )


async def structured_api_error_handler(request: Request, exc: StructuredAPIError) -> JSONResponse:
    return structured_error_response(
        status_code=exc.status_code,
        error_code=exc.error_code,
        message=exc.message,
        request=request,
    )


def register_structured_error_handlers(app) -> None:
    app.add_exception_handler(StructuredAPIError, structured_api_error_handler)
