"""Graceful shutdown for the API process.

uvicorn stops accepting connections and drains in-flight requests when it gets
SIGTERM/SIGINT (see ``--timeout-graceful-shutdown`` in Dockerfile.api). What is
left for the application is releasing the resources it owns: Redis clients, the
database pool and buffered log output. Each stage is logged and isolated, so a
failure in one never stops the others, and the whole sequence is bounded by
``API_SHUTDOWN_TIMEOUT`` seconds.
"""
import asyncio
import inspect
import logging
import os
from typing import Awaitable, Callable, List, Tuple, Union

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_SHUTDOWN_TIMEOUT = 25.0

Stage = Tuple[str, Callable[[], Union[None, Awaitable[None]]]]


def get_shutdown_timeout() -> float:
    """Seconds allowed for cleanup, from API_SHUTDOWN_TIMEOUT (default 25)."""
    raw = os.getenv("API_SHUTDOWN_TIMEOUT")
    if not raw:
        return DEFAULT_SHUTDOWN_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_SHUTDOWN_TIMEOUT
    return value if value > 0 else DEFAULT_SHUTDOWN_TIMEOUT


async def _close_rate_limiter() -> None:
    from api.limiter import cleanup_limiter

    await cleanup_limiter()


def _dispose_db_pool() -> None:
    from db.session import engine

    engine.dispose()


def _flush_logs() -> None:
    logging.shutdown()


def default_stages() -> List[Stage]:
    return [
        ("redis_rate_limiter", _close_rate_limiter),
        ("database_pool", _dispose_db_pool),
        ("log_handlers", _flush_logs),
    ]


async def _run_stage(name: str, func: Callable) -> bool:
    logger.info("shutdown_stage_start", stage=name)
    try:
        result = func()
        if inspect.isawaitable(result):
            await result
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("shutdown_stage_failed", stage=name, error=str(exc))
        return False
    logger.info("shutdown_stage_done", stage=name)
    return True


async def run_shutdown(stages: List[Stage] = None, timeout: float = None) -> bool:
    """Run the cleanup stages in order. Returns True if all finished in time."""
    stages = default_stages() if stages is None else stages
    timeout = get_shutdown_timeout() if timeout is None else timeout

    async def _all() -> bool:
        ok = True
        for name, func in stages:
            ok = await _run_stage(name, func) and ok
        return ok

    try:
        return await asyncio.wait_for(_all(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("shutdown_timed_out", timeout_seconds=timeout)
        return False
