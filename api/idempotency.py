"""
Redis-backed idempotency manager for Celery tasks.
"""
from __future__ import annotations

import base64
import hashlib
import os
import json
import time
from typing import Optional, Any
import structlog

try:
    import redis
except Exception:  # pragma: no cover - runtime dependency may not be present in tests
    redis = None

logger = structlog.get_logger(__name__)


class IdempotencyManager:
    """Simple Redis-backed idempotency manager.

    Usage:
        manager = IdempotencyManager()
        if not manager.acquire(key, ttl=60):
            return manager.get_result(key)
        try:
            result = do_work()
            manager.mark_completed(key, result)
            return result
        finally:
            manager.release_lock(key)
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if redis is None:
                raise RuntimeError("redis library is required for idempotency manager")
            self._client = redis.from_url(self.redis_url, decode_responses=False)
        return self._client

    def _key_lock(self, key: str) -> str:
        return f"idemp:lock:{key}"

    def _key_result(self, key: str) -> str:
        return f"idemp:result:{key}"

    def _key_http_result(self, key: str) -> str:
        return f"idemp:http:result:{key}"

    def _key_http_lock(self, key: str) -> str:
        return f"idemp:http:lock:{key}"

    @staticmethod
    def build_http_key(
        *,
        method: str,
        path: str,
        idempotency_key: str,
        principal: str,
        body_fingerprint: str,
    ) -> str:
        principal_hash = hashlib.sha256(principal.encode("utf-8")).hexdigest()
        return f"{method.upper()}:{path}:{principal_hash}:{idempotency_key}:{body_fingerprint}"

    def acquire(self, key: str, ttl: int = 60) -> bool:
        """Acquire a lock for the given idempotency key. Returns True if acquired."""
        lock_key = self._key_lock(key)
        try:
            # SET NX with expiry
            acquired = self.client.set(lock_key, b"1", nx=True, ex=ttl)
            if acquired:
                logger.info("idempotency_lock_acquired", key=key)
            else:
                logger.info("idempotency_lock_exists", key=key)
            return bool(acquired)
        except Exception as e:
            logger.error("idempotency_acquire_failed", key=key, error=str(e))
            # Fail open: if Redis is unavailable, allow processing
            return True

    def mark_completed(self, key: str, result: Any, ttl: int = 3600) -> None:
        """Mark the key as completed and store the serialized result."""
        res_key = self._key_result(key)
        try:
            payload = json.dumps({"result": result, "timestamp": int(time.time())}).encode("utf-8")
            self.client.set(res_key, payload, ex=ttl)
            # Release the lock key if present
            try:
                self.client.delete(self._key_lock(key))
            except Exception:
                pass
            logger.info("idempotency_marked_completed", key=key)
        except Exception as e:
            logger.error("idempotency_mark_completed_failed", key=key, error=str(e))

    def get_result(self, key: str) -> Optional[Any]:
        """Return stored result for a completed idempotency key, or None."""
        res_key = self._key_result(key)
        try:
            raw = self.client.get(res_key)
            if not raw:
                return None
            data = json.loads(raw.decode("utf-8"))
            return data.get("result")
        except Exception as e:
            logger.error("idempotency_get_result_failed", key=key, error=str(e))
            return None

    def release_lock(self, key: str) -> None:
        try:
            self.client.delete(self._key_lock(key))
        except Exception:
            pass

    def acquire_http(self, key: str, ttl: int = 3600) -> bool:
        """Acquire a lock for a HTTP idempotency key."""
        lock_key = self._key_http_lock(key)
        try:
            acquired = self.client.set(lock_key, b"1", nx=True, ex=ttl)
            if acquired:
                logger.info("http_idempotency_lock_acquired", key=key)
            else:
                logger.info("http_idempotency_lock_exists", key=key)
            return bool(acquired)
        except Exception as e:
            logger.error("http_idempotency_acquire_failed", key=key, error=str(e))
            return True

    def store_http_response(
        self,
        key: str,
        response: dict,
        ttl: int = 86400,
    ) -> None:
        """Persist the serialized HTTP response for replay on retry."""
        payload = {
            "response": {
                "status_code": int(response.get("status_code", 200)),
                "headers": response.get("headers", {}),
                "body_b64": base64.b64encode(response.get("body", b"")).decode("ascii"),
                "media_type": response.get("media_type"),
            },
            "request_fingerprint": response.get("request_fingerprint"),
            "timestamp": int(time.time()),
        }
        try:
            self.client.set(self._key_http_result(key), json.dumps(payload).encode("utf-8"), ex=ttl)
            try:
                self.client.delete(self._key_http_lock(key))
            except Exception:
                pass
            logger.info("http_idempotency_response_stored", key=key)
        except Exception as e:
            logger.error("http_idempotency_store_failed", key=key, error=str(e))

    def get_http_response(self, key: str) -> Optional[dict]:
        """Return a stored HTTP response payload for the given key."""
        try:
            raw = self.client.get(self._key_http_result(key))
            if not raw:
                return None
            data = json.loads(raw.decode("utf-8"))
            response = data.get("response") or {}
            body_b64 = response.get("body_b64") or ""
            return {
                "status_code": int(response.get("status_code", 200)),
                "headers": response.get("headers", {}),
                "body": base64.b64decode(body_b64.encode("ascii")) if body_b64 else b"",
                "media_type": response.get("media_type"),
                "request_fingerprint": data.get("request_fingerprint"),
            }
        except Exception as e:
            logger.error("http_idempotency_get_failed", key=key, error=str(e))
            return None

    def release_http_lock(self, key: str) -> None:
        try:
            self.client.delete(self._key_http_lock(key))
        except Exception:
            pass
