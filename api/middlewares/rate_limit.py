from __future__ import annotations

from typing import Callable

import structlog
from fastapi import Request

from api.config import get_settings
from api.limiter import (
    build_rate_limit_response,
    get_rate_limit_policy,
    is_whitelisted,
    limiter,
    resolve_rate_limit_identifier,
)
from api.middlewares._shared import SKIP_PATHS

settings = get_settings()
logger = structlog.get_logger(__name__)


async def rate_limit_middleware(request: Request, call_next: Callable):
    if not settings.RATE_LIMIT_ENABLED:
        return await call_next(request)

    path = request.url.path
    if path in SKIP_PATHS:
        return await call_next(request)

    identifier = resolve_rate_limit_identifier(request)
    request.state.rate_limit_identifier = identifier
    request.state.user_id = identifier

    if is_whitelisted(identifier):
        response = await call_next(request)
        response.headers["X-RateLimit-Scope"] = "whitelist"
        return response

    rule, matched_override = get_rate_limit_policy(path, request.method)
    allowed = await limiter.check_rate_limit(
        identifier=identifier,
        endpoint=f"{request.method.upper()} {path}",
        limit=rule.requests,
        window_seconds=rule.window,
        request_id=getattr(request.state, "request_id", None),
    )

    if not allowed:
        retry_after = await limiter.get_remaining_ttl(identifier, f"{request.method.upper()} {path}", rule.window)
        logger.warning(
            "rate_limit_exceeded",
            identifier=identifier,
            path=path,
            method=request.method,
            limit=rule.requests,
            window_seconds=rule.window,
            retry_after=retry_after,
        )
        return build_rate_limit_response(
            retry_after=retry_after,
            message=f"Rate limit exceeded. Limit is {rule.requests} requests per {rule.window} seconds.",
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(rule.requests)
    response.headers["X-RateLimit-Window"] = str(rule.window)
    response.headers["X-RateLimit-Scope"] = "endpoint" if matched_override else "global"
    return response
