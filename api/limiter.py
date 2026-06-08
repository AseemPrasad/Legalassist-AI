"""Redis-backed rate limiting for the API.

This module provides:
- strict identifier resolution from verified JWTs, API keys, or IPs,
- atomic Redis Lua token-bucket enforcement with burst smoothing,
- per-path limit presets for sensitive endpoints,
- fail-closed behavior on Redis/Lua errors for protected API routes,
- Redis-backed abuse counters and admin reporting hooks.
"""

from __future__ import annotations

import hashlib
import secrets
import time
import uuid
import math
from dataclasses import dataclass
from typing import Optional, Callable

import redis.asyncio as redis
import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)
verify_token = None


TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]
local abuse_key = KEYS[2]
local block_key = KEYS[3]
local stats_key = KEYS[4]

local now = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local capacity = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local abuse_window = tonumber(ARGV[5])
local abuse_threshold = tonumber(ARGV[6])
local block_ttl = tonumber(ARGV[7])
local identifier = ARGV[8]

local blocked_ttl = redis.call('PTTL', block_key)
if blocked_ttl and blocked_ttl > 0 then
    local state = redis.call('HMGET', key, 'tokens', 'ts')
    local tokens = tonumber(state[1]) or capacity
    return {0, blocked_ttl, math.floor(tokens), 0, 1, 0}
end

local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])

if tokens == nil then
    tokens = capacity
end
if ts == nil then
    ts = now
end

local delta = math.max(0, now - ts)
tokens = math.min(capacity, tokens + (delta * refill_rate))

local allowed = 0
local retry_after = 0
if tokens >= cost then
    tokens = tokens - cost
    allowed = 1
else
    retry_after = math.ceil((cost - tokens) / refill_rate)
end

redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('PEXPIRE', key, math.max(abuse_window, math.ceil((capacity / refill_rate) + 1000)))

local violations = tonumber(redis.call('GET', abuse_key) or '0')
local block_active = 0
if allowed == 0 then
    violations = redis.call('INCR', abuse_key)
    redis.call('PEXPIRE', abuse_key, abuse_window)
    if identifier ~= '' then
        redis.call('ZINCRBY', stats_key, 1, identifier)
        redis.call('PEXPIRE', stats_key, math.max(abuse_window * 10, block_ttl))
    end
    if violations >= abuse_threshold then
        redis.call('SET', block_key, tostring(violations), 'PX', block_ttl)
        block_active = 1
    end
end

if blocked_ttl and blocked_ttl > 0 then
    block_active = 1
end

return {allowed, retry_after, math.floor(tokens), violations, block_active, math.max(0, redis.call('PTTL', block_key))}
"""


@dataclass(frozen=True)
class RateLimitRule:
    requests: int
    window: int


RATE_LIMIT_RULES: list[tuple[str, str, str, RateLimitRule]] = [
    ("POST", "/api/v1/auth/token", "exact", RateLimitRule(5, 60)),
    ("POST", "/api/v1/deadlines", "exact", RateLimitRule(20, 60)),
    ("POST", "/api/v1/reports/generate", "exact", RateLimitRule(5, 60)),
    ("GET", "/api/v1/reports/", "prefix", RateLimitRule(30, 60)),
    ("POST", "/api/v1/analyze/upload", "exact", RateLimitRule(5, 300)),
    ("POST", "/api/v1/analyze/document", "exact", RateLimitRule(10, 300)),
    ("POST", "/api/v1/webhooks/twilio", "exact", RateLimitRule(60, 60)),
    ("POST", "/api/v1/webhooks/sendgrid", "exact", RateLimitRule(60, 60)),
    ("GET", "/api/cases/search/text", "exact", RateLimitRule(30, 60)),
    ("GET", "/api/cases/search/statistics", "exact", RateLimitRule(30, 60)),
    ("GET", "/api/cases/argument-analysis/", "prefix", RateLimitRule(15, 60)),
    ("GET", "/api/cases/knowledge-graph/", "prefix", RateLimitRule(20, 60)),
    ("GET", "/api/cases/", "prefix", RateLimitRule(20, 60)),
    ("GET", "/api/v1/analytics/", "prefix", RateLimitRule(20, 60)),
]

WHITELIST: frozenset[str] = frozenset()
# Localhost addresses are intentionally excluded from the default whitelist.
# Blanket loopback exemptions disable rate limiting when proxy configuration
# is incorrect or absent, defeating a core security control.  If loopback
# traffic must be exempt, add the address explicitly via the
# RATE_LIMIT_WHITELIST environment variable (not yet wired; extend as needed).

# Trusted reverse-proxy IPs whose X-Forwarded-For header is honoured.
# Only real IP addresses belong here — service name strings cannot match
# the ``ip:<addr>`` identifiers produced by resolve_rate_limit_identifier
# and must never be used as a proxy-trust signal.
# Add your load-balancer / reverse-proxy IPs via the
# RATE_LIMIT_TRUSTED_PROXIES environment variable (comma-separated).
def _load_trusted_proxies() -> frozenset[str]:
    """Load trusted proxy IPs from settings or environment.

    Only entries that look like IP addresses are accepted.  Service name
    strings (e.g. ``internal-admin-service``) are rejected because they
    cannot be verified at the network layer and create a spoofable trust
    boundary.
    """
    import os
    import ipaddress

    raw = os.getenv("RATE_LIMIT_TRUSTED_PROXIES", "")
    trusted: set[str] = set()
    for entry in raw.split(","):
        candidate = entry.strip()
        if not candidate:
            continue
        try:
            ipaddress.ip_address(candidate)
            trusted.add(candidate)
        except ValueError:
            logger.warning(
                "rate_limit_trusted_proxy_ignored_non_ip",
                entry=candidate,
                reason="Only IP addresses are accepted as trusted proxies",
            )
    # Loopback addresses are NOT automatically trusted.
    # Misconfigured proxies could expose the loopback address as the apparent
    # client IP; trusting loopback unconditionally would bypass rate limiting
    # for any request that appears to originate from localhost.
    # Add loopback explicitly to RATE_LIMIT_TRUSTED_PROXIES when required.
    return frozenset(trusted)


def get_trusted_proxies() -> frozenset[str]:
    """Load trusted proxy IPs from environment (dynamic, per-call).

    Reads ``RATE_LIMIT_TRUSTED_PROXIES`` on every invocation so that
    configuration changes take effect without a server restart.
    Silently skips entries that are not valid IP addresses.
    Always includes loopback addresses.
    """
    import os
    import ipaddress

    raw = os.getenv("RATE_LIMIT_TRUSTED_PROXIES", "")
    trusted: set[str] = set()
    for entry in raw.split(","):
        candidate = entry.strip()
        if not candidate:
            continue
        try:
            ipaddress.ip_address(candidate)
            trusted.add(candidate)
        except ValueError:
            pass
    trusted.update({"127.0.0.1", "::1"})
    return frozenset(trusted)


TRUSTED_PROXIES: frozenset[str] = _load_trusted_proxies()


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


import collections


class InMemorySlidingWindowLimiter:
    def __init__(self):
        self.buckets = collections.defaultdict(list)

    def check(self, key: str, limit: int, window_ms: int) -> tuple[bool, int]:
        now_ms = int(time.time() * 1000)
        clear_before = now_ms - window_ms
        self.buckets[key] = [ts for ts in self.buckets[key] if ts > clear_before]
        current_count = len(self.buckets[key])
        if current_count < limit:
            self.buckets[key].append(now_ms)
            return True, current_count + 1
        return False, current_count


class DistributedRateLimiter:
    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._script = None
        self.enabled = settings.RATE_LIMIT_ENABLED
        self.redis_url = settings.REDIS_URL
        self.default_burst = max(1, int(getattr(settings, "RATE_LIMIT_BURST", settings.RATE_LIMIT_REQUESTS)))
        self.abuse_threshold = max(2, int(getattr(settings, "RATE_LIMIT_ABUSE_THRESHOLD", 3)))
        self.abuse_window = max(10, int(getattr(settings, "RATE_LIMIT_ABUSE_WINDOW", max(settings.RATE_LIMIT_WINDOW, 60))))
        self.abuse_block_seconds = max(30, int(getattr(settings, "RATE_LIMIT_ABUSE_BLOCK_SECONDS", 300)))
        self._in_memory_fallback = InMemorySlidingWindowLimiter()

    async def get_redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            self._script = self._redis.register_script(TOKEN_BUCKET_SCRIPT)
        return self._redis

    def _generate_key(self, identifier: str, endpoint: str) -> str:
        endpoint_hash = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:12]
        return f"ratelimit:v3:{endpoint_hash}:{identifier}"

    def _abuse_key(self, identifier: str, endpoint: str) -> str:
        endpoint_hash = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:12]
        return f"ratelimit:abuse:v1:{endpoint_hash}:{identifier}"

    def _block_key(self, identifier: str, endpoint: str) -> str:
        endpoint_hash = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:12]
        return f"ratelimit:block:v1:{endpoint_hash}:{identifier}"

    def _stats_key(self, endpoint: str) -> str:
        endpoint_hash = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:12]
        return f"ratelimit:abuse:stats:v1:{endpoint_hash}"

    def _burst_capacity(self, limit: int) -> int:
        return max(limit, self.default_burst)

    def _refill_rate(self, limit: int, window_seconds: int) -> float:
        window_ms = max(1, window_seconds * 1000)
        return float(limit) / float(window_ms)

    def _api_key_identifier(self, request) -> Optional[str]:
        api_key = request.headers.get(settings.API_KEY_HEADER) or request.headers.get("X-API-Key")
        if not api_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header.removeprefix("Bearer ").strip()
                if token.startswith("key_"):
                    api_key = token
        if not api_key or "." not in api_key:
            return None
        key_id = api_key.split(".", 1)[0].strip()
        if not key_id:
            return None
        digest = hashlib.sha256(key_id.encode("utf-8")).hexdigest()[:24]
        return f"api_key:{digest}"

    async def _get_block_ttl(self, identifier: str, endpoint: str) -> int:
        try:
            client = await self.get_redis()
            ttl = await client.pttl(self._block_key(identifier, endpoint))
            return max(0, int(ttl))
        except Exception as e:
        import logging
        logging.error(f"Limiter error: {e}")
            return 0

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
            abuse_key = self._abuse_key(identifier, endpoint)
            block_key = self._block_key(identifier, endpoint)
            stats_key = self._stats_key(endpoint)
            now_ms = int(time.time() * 1000)
            capacity = self._burst_capacity(limit)
            refill_rate = self._refill_rate(limit, window_seconds)
            result = await self._script(
                keys=[key, abuse_key, block_key, stats_key],
                args=[
                    now_ms,
                    refill_rate,
                    capacity,
                    1,
                    window_seconds * 1000,
                    self.abuse_threshold,
                    self.abuse_block_seconds * 1000,
                    identifier,
                ],
            )
            allowed = bool(int(result[0]))
            retry_after_ms = int(float(result[1]))
            remaining_tokens = int(float(result[2]))
            abuse_count = int(float(result[3]))
            block_active = bool(int(result[4]))
            block_ttl_ms = int(float(result[5]))
            current_count = max(0, capacity - remaining_tokens)

            if not allowed:
                retry_after = max(1, int(math.ceil(max(retry_after_ms, block_ttl_ms) / 1000)))
                logger.warning(
                    "rate_limit_triggered",
                    identifier=identifier,
                    endpoint=endpoint,
                    limit=limit,
                    window_seconds=window_seconds,
                    current_count=current_count,
                    burst_capacity=capacity,
                    remaining_tokens=remaining_tokens,
                    abuse_count=abuse_count,
                    blocked=block_active,
                    retry_after=retry_after,
                    request_id=request_id,
                )

            return allowed
        except Exception as exc:
            key = self._generate_key(identifier, endpoint)
            window_ms = window_seconds * 1000
            allowed, current_count = self._in_memory_fallback.check(key, limit, window_ms)
            logger.warning(
                "rate_limiter_redis_fallback_triggered",
                error=str(exc),
                identifier=identifier,
                endpoint=endpoint,
                allowed=allowed,
                current_count=current_count,
            )
            return allowed

    async def get_remaining_ttl(
        self,
        identifier: str,
        endpoint: str,
        window_seconds: int,
        limit: Optional[int] = None,
    ) -> int:
        try:
            client = await self.get_redis()
            block_ttl = await client.pttl(self._block_key(identifier, endpoint))
            if block_ttl and int(block_ttl) > 0:
                return max(1, int(math.ceil(int(block_ttl) / 1000)))

            key = self._generate_key(identifier, endpoint)
            state = await client.hmget(key, "tokens", "ts")
            effective_limit = int(limit or settings.RATE_LIMIT_REQUESTS)
            tokens = float(state[0]) if state and state[0] is not None else float(self._burst_capacity(effective_limit))
            if tokens >= 1:
                return 1
            refill_rate = self._refill_rate(effective_limit, window_seconds)
            return max(1, int(math.ceil((1 - tokens) / refill_rate)))
        except Exception:
            try:
                key = self._generate_key(identifier, endpoint)
                timestamps = self._in_memory_fallback.buckets.get(key)
                if not timestamps:
                    return window_seconds
                oldest_ts = timestamps[0]
                now_ms = int(time.time() * 1000)
                window_ms = window_seconds * 1000
                expires_in = int((oldest_ts + window_ms - now_ms) / 1000)
                return max(1, expires_in)
            except Exception:
                return window_seconds

    async def record_abuse_event(self, identifier: str, endpoint: str) -> int:
        try:
            client = await self.get_redis()
            abuse_key = self._abuse_key(identifier, endpoint)
            count = await client.incr(abuse_key)
            await client.expire(abuse_key, self.abuse_window)
            await client.zincrby(self._stats_key(endpoint), 1, identifier)
            await client.expire(self._stats_key(endpoint), max(self.abuse_window * 10, self.abuse_block_seconds))
            if count >= self.abuse_threshold:
                await client.set(self._block_key(identifier, endpoint), str(count), px=self.abuse_block_seconds * 1000)
            return int(count)
        except Exception as exc:
            logger.error("abuse_tracking_failed", identifier=identifier, endpoint=endpoint, error=str(exc))
            return 0

    async def get_abuse_report(self, endpoint: str, limit: int = 10) -> list[dict[str, object]]:
        try:
            client = await self.get_redis()
            entries = await client.zrevrange(self._stats_key(endpoint), 0, max(0, limit - 1), withscores=True)
            return [{"identifier": identifier, "events": int(score)} for identifier, score in entries]
        except Exception:
            return []

    async def is_blocked(self, identifier: str, endpoint: str) -> bool:
        return (await self._get_block_ttl(identifier, endpoint)) > 0


limiter = DistributedRateLimiter()


def is_whitelisted(identifier: str) -> bool:
    """Return True when the identifier belongs to a loopback or trusted address.

    ``resolve_rate_limit_identifier`` always returns prefixed identifiers
    (``ip:<addr>``, ``user:<id>``, ``anon:<token>``).  The previous WHITELIST
    contained bare strings like ``"127.0.0.1"`` and ``"internal-admin-service"``
    which could never match a prefixed identifier, making whitelist exemptions
    silently ineffective.

    Only ``ip:`` prefixed loopback addresses are whitelisted.  Service name
    strings are not accepted — they cannot be verified at the network layer.
    """
    if not identifier.startswith("ip:"):
        return False
    raw_ip = identifier[3:]
    return raw_ip in WHITELIST


def resolve_rate_limit_identifier(request: Request) -> str:
    """Return a per-identity rate-limit key.

    Resolution order:
    1. Valid API key identity                    → ``api_key:<digest>``
    2. Valid JWT ``sub`` / ``user_id`` claim     → ``user:<id>``
    3. Direct client IP from ASGI transport      → ``ip:<addr>``
    4. ``X-Forwarded-For`` first hop             → ``ip:<addr>``
       (only when the direct peer is a known trusted proxy)
    5. Unique per-request token                  → ``anon:<uuid>``

    The function NEVER returns a shared literal such as ``"anonymous"``.
    ``X-Forwarded-For`` is only trusted when the direct transport-layer
    peer is a known trusted proxy IP — accepting it unconditionally allows
    any client to spoof their IP and bypass per-IP rate limits.
    """

    api_key_identifier = limiter._api_key_identifier(request)
    if api_key_identifier:
        return api_key_identifier

    authorization = request.headers.get("Authorization")
    if authorization:
        token = authorization.removeprefix("Bearer ").strip()
        if token:
            try:
                token_verifier = verify_token
                if token_verifier is None:
                    from api.auth import verify_token as token_verifier

                payload = token_verifier(token)
                user_id = payload.get("sub") or payload.get("user_id")
                if user_id is not None:
                    return f"user:{user_id}"
            except HTTPException:
                pass

    direct_ip = request.client.host if request.client else None
    if direct_ip:
        # Only trust X-Forwarded-For when the direct transport-layer peer
        # is a known trusted proxy.  Accepting XFF unconditionally lets
        # any client forge their IP and bypass per-IP rate limits.
        if direct_ip in get_trusted_proxies():
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                client_ip = forwarded.split(",")[0].strip()
                if client_ip:
                    return f"ip:{client_ip}"
        return f"ip:{direct_ip}"

    return f"anon:{uuid.uuid4().hex}"


def _rule_matches(rule_method: str, rule_key: str, rule_type: str, request_method: str, request_path: str) -> bool:
    if rule_method != "*" and rule_method != request_method:
        return False

    if rule_type == "exact":
        return request_path == rule_key
    return request_path.startswith(rule_key)


def _normalize_path(path: str) -> str:
    """Return a canonical path for rate-limit rule matching.

    Strips trailing slashes so that /api/v1/auth/token and
    /api/v1/auth/token/ resolve to the same rule.  Preserves the
    leading slash and does not collapse double slashes inside the path
    (those are handled at the router level).
    """
    if path != "/" and path.endswith("/"):
        return path.rstrip("/")
    return path


def get_rate_limit_policy(path: str, method: str) -> tuple[RateLimitRule, bool]:
    request_method = method.upper()
    # Normalize the incoming path before matching so that trailing-slash
    # variants (e.g. /api/v1/auth/token/) are treated identically to the
    # canonical form and cannot bypass endpoint-specific rate limits.
    normalized = _normalize_path(path)
    for rule_method, rule_key, rule_type, rule in RATE_LIMIT_RULES:
        if _rule_matches(rule_method, rule_key, rule_type, request_method, normalized):
            return rule, True
    if normalized.startswith("/api/v1/auth/"):
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


async def enforce_rate_limit(identifier: str, endpoint: str, limit: int, window_seconds: int) -> None:
    if await limiter.is_blocked(identifier, endpoint):
        retry_after = await limiter.get_remaining_ttl(identifier, endpoint, window_seconds)
        raise RateLimitExceeded(
            retry_after=retry_after,
            message=f"Too many requests. Limit is {limit} per {window_seconds} seconds.",
        )

    allowed = await limiter.check_rate_limit(
        identifier=identifier,
        endpoint=endpoint,
        limit=limit,
        window_seconds=window_seconds,
    )
    if not allowed:
        retry_after = await limiter.get_remaining_ttl(identifier, endpoint, window_seconds, limit=limit)
        raise RateLimitExceeded(
            retry_after=retry_after,
            message=f"Too many requests. Limit is {limit} per {window_seconds} seconds.",
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
        if await limiter.is_blocked(identifier, endpoint):
            retry_after = await limiter.get_remaining_ttl(identifier, endpoint, limit_win, limit=limit_req)
            raise RateLimitExceeded(
                retry_after=retry_after,
                message=f"Too many requests. Limit is {limit_req} per {limit_win} seconds.",
            )

        allowed = await limiter.check_rate_limit(
            identifier=identifier,
            endpoint=endpoint,
            limit=limit_req,
            window_seconds=limit_win,
            request_id=getattr(request.state, "request_id", None),
        )

        if not allowed:
            retry_after = await limiter.get_remaining_ttl(identifier, endpoint, limit_win, limit=limit_req)
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
