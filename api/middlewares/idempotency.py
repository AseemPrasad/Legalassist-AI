from __future__ import annotations

import json
import hashlib
from hashlib import sha256
from typing import Callable

import structlog
from fastapi import Request, status
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session

from api.auth import verify_api_key, verify_token
from api.config import get_settings
from api.idempotency import IdempotencyManager
from api.middlewares._shared import IDEMPOTENT_METHODS, SKIP_PATHS
from database import SessionLocal

settings = get_settings()
logger = structlog.get_logger(__name__)
http_idempotency_manager = IdempotencyManager()

SAFE_IDEMPOTENT_PREFIXES = (
    "/api/v1/cases",
    "/api/v1/reports",
    "/api/v1/deadlines",
    "/api/v1/analytics",
)


def is_safe_to_cache(path: str) -> bool:
    return any(path.startswith(p) for p in SAFE_IDEMPOTENT_PREFIXES)


def _response_contains_sensitive_fields(body_bytes: bytes, headers: dict) -> bool:
    content_type = (headers.get("content-type") or "").lower()
    if "application/json" not in content_type:
        return True

    try:
        import json as _json

        data = _json.loads(body_bytes.decode("utf-8"))
    except Exception:
        return True

    sensitive_keywords = {"document", "content", "text", "file", "pii", "ssn", "email", "phone", "dob"}

    def scan(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if any(k.lower().find(kw) != -1 for kw in sensitive_keywords):
                    return True
                if scan(v):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if scan(item):
                    return True
        return False

    return scan(data)


def _request_principal(request: Request) -> str:
    existing = getattr(request.state, "principal", None)
    if existing:
        return existing

    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header.split(None, 1)[1].strip()
        try:
            payload = verify_token(token)
            if payload:
                sub = payload.get("sub") or payload.get("user_id")
                if sub:
                    principal = f"user:{int(sub)}"
                    request.state.principal = principal
                    return principal
        except Exception:
            pass

    api_key_hdr = request.headers.get("x-api-key")
    if api_key_hdr and "." in api_key_hdr:
        key_id, secret = api_key_hdr.split(".", 1)
        try:
            from db.models import APIKey

            db: Session = SessionLocal()
            try:
                key_record = db.query(APIKey).filter(APIKey.key_id == key_id).first()
                if key_record and key_record.is_valid() and verify_api_key(secret, key_record.key_salt, key_record.key_hash):
                    if getattr(key_record, "user_id", None):
                        principal = f"user:{int(key_record.user_id)}"
                    else:
                        principal = f"api_key:{key_id}"
                    request.state.principal = principal
                    return principal
            finally:
                db.close()
        except Exception:
            pass

    ip = request.client.host if request.client is not None else "unknown"
    ua = request.headers.get("user-agent", "")
    fp = sha256(f"{ip}|{ua}".encode()).hexdigest()
    principal = f"anonymous:{fp}"
    request.state.principal = principal
    return principal


def _idempotency_exempt_path(path: str) -> bool:
    return path in SKIP_PATHS or path in {"/openapi.json", "/docs", "/redoc"}


def _response_headers_for_cache(response: Response) -> dict:
    headers = {}
    for key, value in response.headers.items():
        lower_key = key.lower()
        if lower_key in {"content-length", "transfer-encoding", "connection", "date", "server"}:
            continue
        headers[key] = value
    return headers


def _canonicalize_request_body(body: bytes, content_type: str) -> bytes:
    if "application/json" not in content_type.lower():
        return body or b""

    if not body:
        return b""

    try:
        parsed = json.loads(body.decode("utf-8"))
    except Exception:
        return body

    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return canonical.encode("utf-8")


async def idempotency_middleware(request: Request, call_next: Callable):
    if request.method.upper() not in IDEMPOTENT_METHODS or _idempotency_exempt_path(request.url.path):
        return await call_next(request)

    idempotency_key = request.headers.get("Idempotency-Key")
    if not idempotency_key:
        return JSONResponse(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            content={
                "error_code": "IDEMPOTENCY_KEY_REQUIRED",
                "message": "Idempotency-Key header is required for write operations.",
            },
        )

    body = await request.body()
    canonical_body = _canonicalize_request_body(body, request.headers.get("content-type", ""))
    body_fingerprint = hashlib.sha256(canonical_body or b"").hexdigest()
    principal = _request_principal(request)
    key = http_idempotency_manager.build_http_key(
        method=request.method,
        path=request.url.path,
        idempotency_key=idempotency_key,
        principal=principal,
        body_fingerprint=body_fingerprint,
    )

    cached = http_idempotency_manager.get_http_response(key)
    if cached:
        if cached.get("request_fingerprint") and cached["request_fingerprint"] != body_fingerprint:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "error_code": "IDEMPOTENCY_KEY_REUSED",
                    "message": "This Idempotency-Key was already used with a different request body.",
                },
            )
        headers = {**cached["headers"], "X-Idempotent-Replay": "true"}
        if cached.get("body_stripped"):
            headers["X-Idempotent-Body-Stripped"] = "true"
            return Response(
                content=b"",
                status_code=cached["status_code"],
                headers=headers,
                media_type=cached.get("media_type") or cached["headers"].get("content-type"),
            )

        return Response(
            content=cached["body"],
            status_code=cached["status_code"],
            headers=headers,
            media_type=cached.get("media_type") or cached["headers"].get("content-type"),
        )

    if not http_idempotency_manager.acquire_http(key, ttl=86400):
        cached = http_idempotency_manager.get_http_response(key)
        if cached:
            return Response(
                content=cached["body"],
                status_code=cached["status_code"],
                headers={**cached["headers"], "X-Idempotent-Replay": "true"},
                media_type=cached.get("media_type") or cached["headers"].get("content-type"),
            )
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error_code": "IDEMPOTENCY_IN_PROGRESS",
                "message": "A request with this idempotency key is already in progress.",
            },
        )

    response = await call_next(request)

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    headers = _response_headers_for_cache(response)
    body_to_store = b""
    body_stripped = True

    if response.status_code < 400 and not _response_contains_sensitive_fields(response_body, headers):
        body_to_store = response_body
        body_stripped = False

    cache_payload = {
        "status_code": response.status_code,
        "headers": headers,
        "body": body_to_store,
        "media_type": response.media_type or headers.get("content-type"),
        "request_fingerprint": body_fingerprint,
        "body_stripped": body_stripped,
    }

    if response.status_code < 400:
        http_idempotency_manager.store_http_response(key, cache_payload, ttl=86400)
    else:
        http_idempotency_manager.release_http_lock(key)

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=headers,
        media_type=response.media_type or headers.get("content-type"),
    )
