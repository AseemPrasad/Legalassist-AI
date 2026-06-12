"""Asyncio-based pub/sub bus for real-time job progress updates.

Mirrors the timeline_realtime bus pattern for document analysis jobs.
"""

import asyncio
from typing import Dict, Optional
import structlog

logger = structlog.get_logger(__name__)


class JobRealtimeBus:
    """In-memory pub/sub bus for job progress events keyed by job_id."""

    def __init__(self, queue_maxsize: int = 100) -> None:
        try:
            from api.config import get_settings
            max_size = get_settings().TIMELINE_REALTIME_QUEUE_SIZE
            self._queue_maxsize = max(1, int(max_size))
        except Exception:
            self._queue_maxsize = max(1, queue_maxsize)
        self._queues: Dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        """Subscribe to events for a specific job_id."""
        async with self._lock:
            if job_id not in self._queues:
                self._queues[job_id] = asyncio.Queue(maxsize=self._queue_maxsize)
            return self._queues[job_id]

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        """Unsubscribe a specific queue from a job_id."""
        async with self._lock:
            q = self._queues.get(job_id)
            if q is queue:
                del self._queues[job_id]

    async def publish(self, job_id: str, payload: dict) -> None:
        """Publish an event to all subscribers of a job_id."""
        async with self._lock:
            queue = self._queues.get(job_id)
        if queue is None:
            return
        if queue.full():
            try:
                queue.get_nowait()
                logger.warning("job_event_queue_dropped_oldest", job_id=job_id)
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(payload)
        except Exception as e:
            logger.error("job_event_publish_failed", job_id=job_id, error=str(e))


# Global singleton — imported by Celery signal handlers and WebSocket handlers
job_realtime_bus = JobRealtimeBus()