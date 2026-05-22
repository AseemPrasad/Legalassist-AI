"""Redis-backed sliding-window rate limiting for the API.

This module provides:
- strict identifier resolution from verified JWTs or IPs,
- atomic Redis Lua sliding-window enforcement,
- per-path limit presets for sensitive endpoints,
- fail-closed behavior on Redis/Lua errors for protected API routes.
"""

from __future__ import annotations

import hashlib
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Callable

import redis.asyncio as redis
import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.auth import verify_token
from api.config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)


SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
local clear_before = now - window

redis.call('ZREMRANGEBYSCORE', key, 0, clear_before)
local current_count = redis.call('ZCARD', key)

if current_count < limit then
    redis.call('ZADD', key, now, member)
    redis.call('PEXPIRE', key, window)
    return {1, current_count + 1}
else
    return {0, current_count}
end
"""


@dataclass(frozen=True)
class RateLimitRule:
    requests: int
    window: int


RATE_LIMIT_RULES: list[tuple[str, str, str, RateLimitRule]] = [
    ("POST", "/api/v1/auth/token", "exact", RateLimitRule(5, 60)),
    ("POST", "/api/v1/analyze/upload", "exact", RateLimitRule(5, 300)),
    ("POST", "/api/v1/analyze/document", "exact", RateLimitRule(10, 300)),
    ("GET", "/api/cases/search/text", "exact", RateLimitRule(30, 60)),
    ("GET", "/api/cases/", "prefix", RateLimitRule(20, 60)),
    ("GET", "/api/v1/analytics/", "prefix", RateLimitRule(20, 60)),
]

WHITELIST = {
    "127.0.0.1",
    "::1",
    "localhost",
    "internal-admin-service",
    "internal-ingest-service",
}


class RateLimitExceeded(HTTPException):
    def __init__(self, retry_after: int, message: str = "Rate limit exceeded"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error_code": "RATE_LIMIT_EXCEEDED",
                "message": message,
                "retry_after": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )


class DistributedRateLimiter:
    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._script = None
        self.enabled = settings.RATE_LIMIT_ENABLED
        self.redis_url = settings.REDIS_URL

    async def get_redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            self._script = self._redis.register_script(SLIDING_WINDOW_SCRIPT)
        return self._redis

    def _generate_key(self, identifier: str, endpoint: str) -> str:
        endpoint_hash = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:12]
        return f"ratelimit:v3:{endpoint_hash}:{identifier}"

    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: str,
        limit: int,
        window_seconds: int,
        request_id: Optional[str] = None,
    ) -> bool:
        if not self.enabled:
            return True

        try:
            await self.get_redis()
            key = self._generate_key(identifier, endpoint)
            now_ms = int(time.time() * 1000)
            window_ms = window_seconds * 1000
            member = request_id or secrets.token_hex(16)
            result = await self._script(keys=[key], args=[now_ms, window_ms, limit, member])
            allowed = bool(int(result[0]))
            current_count = int(result[1])

            if not allowed:
                logger.warning(
                    "rate_limit_triggered",
                    identifier=identifier,
                    endpoint=endpoint,
                    limit=limit,
                    window_seconds=window_seconds,
                    current_count=current_count,
                )

            return allowed
        except Exception as exc:
            logger.error(
                "rate_limiter_error",
                error=str(exc),
                identifier=identifier,
                endpoint=endpoint,
                fail_open=False,
            )
            return False

    async def get_remaining_ttl(self, identifier: str, endpoint: str, window_seconds: int) -> int:
        try:
            client = await self.get_redis()
            key = self._generate_key(identifier, endpoint)
            oldest = await client.zrange(key, 0, 0, withscores=True)
            if not oldest:
                return window_seconds

            oldest_ts = oldest[0][1]
            now_ms = int(time.time() * 1000)
            window_ms = window_seconds * 1000
            expires_in = int((oldest_ts + window_ms - now_ms) / 1000)
            return max(1, expires_in)
        except Exception:
            return window_seconds


limiter = DistributedRateLimiter()


def is_whitelisted(identifier: str) -> bool:
    return identifier in WHITELIST


def resolve_rate_limit_identifier(request: Request) -> str:
    """Prefer verified JWT user_id; otherwise use source IP."""
    authorization = request.headers.get("Authorization")
    if authorization:
        token = authorization.removeprefix("Bearer ").strip()
        if token:
            try:
                payload = verify_token(token)
                user_id = payload.get("sub") or payload.get("user_id")
                if user_id is not None:
                    return f"user:{user_id}"
            except HTTPException:
                pass

    if request.client and request.client.host:
        return f"ip:{request.client.host}"

    # Use X-Forwarded-For as fallback if present (trusted proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
        if client_ip:
            return f"ip:{client_ip}"

    # Last resort: unique per-request identifier so unknown clients
    # do not share a single bucket that can be exhausted by one attacker.
    return f"anon:{uuid.uuid4().hex}"


def _rule_matches(rule_method: str, rule_key: str, rule_type: str, request_method: str, request_path: str) -> bool:
    if rule_method != "*" and rule_method != request_method:
        return False

    if rule_type == "exact":
        return request_path == rule_key
    return request_path.startswith(rule_key)


def get_rate_limit_policy(path: str, method: str) -> tuple[RateLimitRule, bool]:
    request_method = method.upper()
    for rule_method, rule_key, rule_type, rule in RATE_LIMIT_RULES:
        if _rule_matches(rule_method, rule_key, rule_type, request_method, path):
            return rule, True
    if path.startswith("/api/v1/auth/"):
        return RateLimitRule(settings.AUTH_RATE_LIMIT_REQUESTS, settings.AUTH_RATE_LIMIT_WINDOW), False
    return RateLimitRule(settings.RATE_LIMIT_REQUESTS, settings.RATE_LIMIT_WINDOW), False


def build_rate_limit_response(retry_after: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error_code": "RATE_LIMIT_EXCEEDED",
            "message": message,
            "retry_after": retry_after,
        },
        headers={"Retry-After": str(retry_after)},
    )


def RateLimit(
    requests: int = None,
    window: int = None,
    use_auth_defaults: bool = False,
    scope: str = "endpoint",
):
    """FastAPI dependency factory for route-specific throttling."""
    limit_req = requests or (settings.AUTH_RATE_LIMIT_REQUESTS if use_auth_defaults else settings.RATE_LIMIT_REQUESTS)
    limit_win = window or (settings.AUTH_RATE_LIMIT_WINDOW if use_auth_defaults else settings.RATE_LIMIT_WINDOW)

    async def rate_limit_dependency(request: Request):
        identifier = resolve_rate_limit_identifier(request)
        request.state.rate_limit_identifier = identifier

        if is_whitelisted(identifier):
            return True

        endpoint = request.url.path if scope == "endpoint" else "GLOBAL"
        allowed = await limiter.check_rate_limit(
            identifier=identifier,
            endpoint=endpoint,
            limit=limit_req,
            window_seconds=limit_win,
            request_id=getattr(request.state, "request_id", None),
        )

        if not allowed:
            retry_after = await limiter.get_remaining_ttl(identifier, endpoint, limit_win)
            raise RateLimitExceeded(
                retry_after=retry_after,
                message=f"Too many requests. Limit is {limit_req} per {limit_win} seconds.",
            )

        return True

    return rate_limit_dependency


async def cleanup_limiter():
    if limiter._redis:
        await limiter._redis.close()
        limiter._redis = None
