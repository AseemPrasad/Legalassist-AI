"""
Redis-backed idempotency manager for Celery tasks.
"""
from __future__ import annotations

import base64
import hashlib
import os
import json
import time
import uuid
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
        self._owners: dict[str, str] = {}

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

    def _key_state(self, key: str) -> str:
        return f"idemp:state:{key}"

    def _serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, separators=(",", ":")).encode("utf-8")

    def _deserialize(self, raw: Optional[bytes]) -> Optional[dict[str, Any]]:
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    def _read_state(self, key: str) -> Optional[dict[str, Any]]:
        try:
            return self._deserialize(self.client.get(self._key_state(key)))
        except Exception:
            return None

    def _write_state(self, key: str, payload: dict[str, Any], ttl: int) -> None:
        self.client.set(self._key_state(key), self._serialize(payload), ex=ttl)

    def _is_stale(self, state: Optional[dict[str, Any]], stale_after: int) -> bool:
        if not state or state.get("status") != "pending":
            return False
        heartbeat = int(state.get("heartbeat", 0) or 0)
        return heartbeat > 0 and (time.time() - heartbeat) > stale_after

    def _current_owner(self, key: str) -> Optional[str]:
        return self._owners.get(key)

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

    def acquire(self, key: str, ttl: int = 60, stale_after: Optional[int] = None) -> bool:
        """Acquire a lock for the given idempotency key.

        A successful acquire writes a durable pending state with an owner token.
        If the lock expired but a pending state is stale, the next caller can take over.
        """
        lock_key = self._key_lock(key)
        state_key = self._key_state(key)
        stale_after = stale_after or max(ttl * 2, ttl + 5)
        token = uuid.uuid4().hex
        try:
            acquired = self.client.set(lock_key, token.encode("ascii"), nx=True, ex=ttl)
            if acquired:
                self._owners[key] = token
                self._write_state(
                    key,
                    {
                        "status": "pending",
                        "token": token,
                        "ttl": ttl,
                        "started": int(time.time()),
                        "heartbeat": int(time.time()),
                    },
                    ttl=max(ttl * 3, stale_after),
                )
                logger.info("idempotency_lock_acquired", key=key)
                return True

            state = self._read_state(key)
            if state and state.get("status") == "completed":
                logger.info("idempotency_lock_exists", key=key)
                return False

            if self._is_stale(state, stale_after):
                try:
                    self.client.delete(lock_key)
                except Exception:
                    pass
                try:
                    self.client.set(state_key, self._serialize({**state, "status": "stale", "stale_at": int(time.time())}), ex=max(ttl * 3, stale_after))
                except Exception:
                    pass
                acquired = self.client.set(lock_key, token.encode("ascii"), nx=True, ex=ttl)
                if acquired:
                    self._owners[key] = token
                    self._write_state(
                        key,
                        {
                            "status": "pending",
                            "token": token,
                            "ttl": ttl,
                            "started": int(time.time()),
                            "heartbeat": int(time.time()),
                            "recovered_from_stale": True,
                        },
                        ttl=max(ttl * 3, stale_after),
                    )
                    logger.info("idempotency_lock_recovered", key=key)
                    return True

            logger.info("idempotency_lock_exists", key=key)
            return False
        except Exception as e:
            logger.error("idempotency_acquire_failed", key=key, error=str(e))
            # Fail open: if Redis is unavailable, allow processing
            return True

    def heartbeat(self, key: str, ttl: int = 60) -> bool:
        """Renew the lease for a running task if this process owns the key."""
        token = self._current_owner(key)
        if not token:
            state = self._read_state(key)
            token = state.get("token") if state else None
        if not token:
            return False

        try:
            state = self._read_state(key)
            if not state or state.get("status") != "pending" or state.get("token") != token:
                return False
            self.client.expire(self._key_lock(key), ttl)
            state.update({"heartbeat": int(time.time()), "ttl": ttl})
            self._write_state(key, state, ttl=max(ttl * 3, ttl + 5))
            return True
        except Exception as e:
            logger.error("idempotency_heartbeat_failed", key=key, error=str(e))
            return False

    def mark_completed(self, key: str, result: Any, ttl: int = 3600) -> None:
        """Mark the key as completed and store the serialized result."""
        res_key = self._key_result(key)
        try:
            payload = json.dumps({"result": result, "timestamp": int(time.time())}).encode("utf-8")
            self.client.set(res_key, payload, ex=ttl)
            state = self._read_state(key) or {}
            state.update({
                "status": "completed",
                "result": result,
                "completed_at": int(time.time()),
            })
            self._write_state(key, state, ttl=ttl)
            # Release the lock key if present
            try:
                self.client.delete(self._key_lock(key))
            except Exception:
                pass
            self._owners.pop(key, None)
            logger.info("idempotency_marked_completed", key=key)
        except Exception as e:
            logger.error("idempotency_mark_completed_failed", key=key, error=str(e))

    def get_result(self, key: str) -> Optional[Any]:
        """Return stored result for a completed idempotency key, or None."""
        res_key = self._key_result(key)
        try:
            raw = self.client.get(res_key)
            if not raw:
                state = self._read_state(key)
                if state and state.get("status") == "completed":
                    return state.get("result")
                return None
            data = json.loads(raw.decode("utf-8"))
            return data.get("result")
        except Exception as e:
            logger.error("idempotency_get_result_failed", key=key, error=str(e))
            return None

    def release_lock(self, key: str) -> None:
        try:
            state = self._read_state(key)
            token = self._current_owner(key)
            if state and state.get("status") == "pending" and (not token or state.get("token") == token):
                self.client.delete(self._key_lock(key))
                self.client.delete(self._key_state(key))
            else:
                self.client.delete(self._key_lock(key))
            self._owners.pop(key, None)
        except Exception:
            pass

    def reconcile_stale_pending(self, stale_after: int = 600, limit: int = 1000) -> int:
        """Mark stale pending entries so the next worker can safely take over."""
        reclaimed = 0
        try:
            for raw_key in self.client.scan_iter(match=self._key_state("*"), count=limit):
                key = raw_key.decode("utf-8") if isinstance(raw_key, (bytes, bytearray)) else str(raw_key)
                raw_state = self.client.get(key)
                state = self._deserialize(raw_state)
                if not self._is_stale(state, stale_after):
                    continue
                self.client.delete(key.replace("idemp:state:", "idemp:lock:", 1))
                state.update({"status": "stale", "stale_at": int(time.time())})
                self.client.set(key, self._serialize(state), ex=max(stale_after, int(state.get("ttl", stale_after)) * 3))
                reclaimed += 1
        except Exception as e:
            logger.error("idempotency_reconcile_failed", error=str(e))
        return reclaimed

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
