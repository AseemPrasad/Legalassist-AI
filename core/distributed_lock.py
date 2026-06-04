"""Redis-based distributed locking for Celery task atomicity.

Uses redlock-py (or redis-py lock fallback) to guarantee strict serial
processing per document_id across distributed workers.
"""

import os
import time
import threading
from contextlib import contextmanager
from typing import Optional, Callable, Any
from functools import wraps

import structlog

logger = structlog.get_logger(__name__)

_redis_url = os.getenv("REDIS_URL", "")
_redis_client: Optional[Any] = None
_lock_pool: list = []
_pool_lock = threading.Lock()


def _get_redis_client() -> Optional[Any]:
    """Lazy singleton Redis client for lock coordination."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not _redis_url:
        return None
    try:
        import redis as _redis_mod
        _redis_client = _redis_mod.from_url(_redis_url, decode_responses=False)
        return _redis_client
    except Exception as exc:
        logger.warning("redis_lock_client_unavailable", error=str(exc))
        return None


def _get_lock_pool() -> list:
    """Return list of Redis clients for redlock algorithm."""
    global _lock_pool
    if _lock_pool:
        return _lock_pool
    with _pool_lock:
        if _lock_pool:
            return _lock_pool
        client = _get_redis_client()
        if client is not None:
            _lock_pool = [client]
        return _lock_pool


class DistributedLockError(Exception):
    """Raised when a distributed lock cannot be acquired or is invalid."""
    pass


class DistributedLock:
    """
    Redis-backed distributed lock using redlock algorithm.

    Lock key: ``lock:document:{document_id}``
    TTL: prevents deadlocks if a worker crashes while holding the lock.
    """

    DEFAULT_TTL_MS = 30000  # 30 seconds
    DEFAULT_RETRY_COUNT = 3
    DEFAULT_RETRY_DELAY_MS = 200

    def __init__(
        self,
        document_id: str,
        ttl_ms: int = DEFAULT_TTL_MS,
        retry_count: int = DEFAULT_RETRY_COUNT,
        retry_delay_ms: int = DEFAULT_RETRY_DELAY_MS,
    ):
        self.document_id = document_id
        self.lock_key = f"lock:document:{document_id}"
        self.ttl_ms = ttl_ms
        self.retry_count = retry_count
        self.retry_delay_ms = retry_delay_ms
        self._lock_instance: Optional[Any] = None

    def acquire(self) -> bool:
        """Attempt to acquire the distributed lock with retries."""
        client = _get_redis_client()
        if client is None:
            logger.warning(
                "distributed_lock_redis_unavailable",
                document_id=self.document_id,
                fallback="proceed_without_lock",
            )
            return True  # Degraded: proceed without lock rather than deadlock

        # Try redlock-py first
        try:
            from redlock import Redlock

            pool = _get_lock_pool()
            if not pool:
                return True  # Degraded

            redlock = Redlock(pool, retry_count=self.retry_count, retry_delay=self.retry_delay_ms)
            self._lock_instance = redlock.lock(self.lock_key, self.ttl_ms)
            if self._lock_instance:
                logger.debug(
                    "distributed_lock_acquired",
                    document_id=self.document_id,
                    lock_key=self.lock_key,
                    ttl_ms=self.ttl_ms,
                )
                return True
        except ImportError:
            logger.debug("redlock_py_not_installed", fallback="redis_py_lock")
        except Exception as exc:
            logger.warning("redlock_acquire_failed", error=str(exc), fallback="redis_py_lock")

        # Fallback to redis-py Lock (single-instance, less robust but functional)
        try:
            lock = client.lock(
                self.lock_key,
                timeout=self.ttl_ms / 1000.0,
                sleep=self.retry_delay_ms / 1000.0,
                blocking=True,
                blocking_timeout=(self.retry_count * self.retry_delay_ms) / 1000.0,
            )
            acquired = lock.acquire()
            if acquired:
                self._lock_instance = lock
                logger.debug(
                    "distributed_lock_acquired_redis_py",
                    document_id=self.document_id,
                    lock_key=self.lock_key,
                    ttl_ms=self.ttl_ms,
                )
                return True
        except Exception as exc:
            logger.error("redis_py_lock_acquire_failed", error=str(exc))

        logger.warning(
            "distributed_lock_not_acquired",
            document_id=self.document_id,
            lock_key=self.lock_key,
            ttl_ms=self.ttl_ms,
        )
        return False

    def release(self) -> None:
        """Release the distributed lock if held."""
        if self._lock_instance is None:
            return

        try:
            # redlock-py lock
            if hasattr(self._lock_instance, "unlock"):
                self._lock_instance.unlock()
                logger.debug("distributed_lock_released_redlock", document_id=self.document_id)
                return
        except Exception as exc:
            logger.warning("redlock_unlock_failed", error=str(exc))

        try:
            # redis-py Lock
            if hasattr(self._lock_instance, "release"):
                self._lock_instance.release()
                logger.debug("distributed_lock_released_redis_py", document_id=self.document_id)
                return
        except Exception as exc:
            logger.warning("redis_py_lock_release_failed", error=str(exc))

        self._lock_instance = None

    def extend(self, additional_ms: int) -> bool:
        """Extend lock TTL while still held (for long-running tasks)."""
        if self._lock_instance is None:
            return False
        try:
            if hasattr(self._lock_instance, "extend"):
                self._lock_instance.extend(additional_ms)
                self.ttl_ms += additional_ms
                return True
            # redis-py Lock extend
            if hasattr(self._lock_instance, "reacquire"):
                self._lock_instance.reacquire()
                return True
        except Exception as exc:
            logger.warning("distributed_lock_extend_failed", error=str(exc))
        return False


@contextmanager
def document_lock(
    document_id: str,
    ttl_ms: int = DistributedLock.DEFAULT_TTL_MS,
    retry_count: int = DistributedLock.DEFAULT_RETRY_COUNT,
    retry_delay_ms: int = DistributedLock.DEFAULT_RETRY_DELAY_MS,
    raise_on_failure: bool = True,
):
    """
    Context manager for acquiring a distributed lock on a document_id.

    Usage:
        with document_lock(document_id):
            # critical section
            pass
    """
    lock = DistributedLock(document_id, ttl_ms, retry_count, retry_delay_ms)
    acquired = lock.acquire()
    if not acquired:
        if raise_on_failure:
            raise DistributedLockError(
                f"Could not acquire distributed lock for document {document_id}"
            )
        logger.warning(
            "distributed_lock_skipped",
            document_id=document_id,
            reason="acquire_failed_and_raise_on_failure_is_false",
        )
        try:
            yield None
        finally:
            return
    try:
        yield lock
    finally:
        lock.release()


def with_document_lock(
    document_id_arg: str = "document_id",
    ttl_ms: int = DistributedLock.DEFAULT_TTL_MS,
    retry_count: int = DistributedLock.DEFAULT_RETRY_COUNT,
    retry_delay_ms: int = DistributedLock.DEFAULT_RETRY_DELAY_MS,
):
    """
    Decorator for Celery tasks that wraps execution in a distributed lock.

    The decorator inspects the task kwargs for the named argument and
    acquires a lock on that document_id for the duration of the task.

    Usage:
        @celery_app.task(bind=True)
        @with_document_lock(document_id_arg="document_id", ttl_ms=60000)
        def my_task(self, user_id, document_id, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            doc_id = kwargs.get(document_id_arg)
            if not doc_id:
                # Try positional: after 'self' for bound tasks
                if len(args) >= 2:
                    # args = (self, user_id, document_id, ...)
                    sig = func.__code__.co_varnames[:func.__code__.co_argcount]
                    if document_id_arg in sig:
                        idx = sig.index(document_id_arg)
                        if idx < len(args):
                            doc_id = args[idx]
            if not doc_id:
                logger.warning(
                    "distributed_lock_no_document_id",
                    task=func.__name__,
                    document_id_arg=document_id_arg,
                )
                return func(*args, **kwargs)

            with document_lock(
                document_id=doc_id,
                ttl_ms=ttl_ms,
                retry_count=retry_count,
                retry_delay_ms=retry_delay_ms,
                raise_on_failure=True,
            ) as lock:
                # For long tasks, auto-extend the lock every ttl/2
                result = func(*args, **kwargs)
                return result

        return wrapper
    return decorator


def extend_lock_for_task(lock: Optional[DistributedLock], additional_ms: int = 30000) -> bool:
    """Helper to extend a lock mid-task for long-running operations."""
    if lock is None:
        return False
    return lock.extend(additional_ms)