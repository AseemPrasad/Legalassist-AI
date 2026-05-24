from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

import structlog

from core.timeline_payloads import TimelineEventPayload


logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class _SubscriberConnection:
    queue: "asyncio.Queue[Dict[str, Any]]"
    loop: asyncio.AbstractEventLoop
    thread_id: int


@dataclass
class _CaseChannel:
    connections: Set[_SubscriberConnection] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    dropped_messages: int = 0


class TimelineRealtimeBus:
    """
    Simple in-memory case-scoped pub/sub bus.

    - Each websocket connection subscribes by providing an asyncio.Queue
    - Writers broadcast a JSON-serializable payload to all subscribers of
      the given case_id.
    """

    def __init__(self, queue_maxsize: int = 100) -> None:
        self._queue_maxsize = max(1, int(queue_maxsize))
        self._channels: Dict[int, _CaseChannel] = {}
        self._global_lock = asyncio.Lock()
        self._drop_lock = threading.Lock()
        self._dropped_messages_total = 0

    @property
    def queue_maxsize(self) -> int:
        return self._queue_maxsize

    @property
    def dropped_messages_total(self) -> int:
        with self._drop_lock:
            return self._dropped_messages_total

    def _record_drop(self, case_id: int, channel: _CaseChannel) -> None:
        with self._drop_lock:
            self._dropped_messages_total += 1
            total_dropped = self._dropped_messages_total

        channel.dropped_messages += 1
        logger.warning(
            "timeline_realtime_queue_dropped",
            case_id=case_id,
            queue_maxsize=self._queue_maxsize,
            dropped_messages=1,
            total_dropped_messages=total_dropped,
            case_dropped_messages=channel.dropped_messages,
            policy="drop_oldest_keep_latest",
        )

    @staticmethod
    def _deliver_message(
        *,
        queue: asyncio.Queue[Dict[str, Any]],
        message: Dict[str, Any],
        case_id: int,
        channel: _CaseChannel,
        record_drop,
    ) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            else:
                record_drop(case_id, channel)

        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            # If another producer filled the queue between eviction and put,
            # drop the oldest item once more and keep the newest payload.
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            record_drop(case_id, channel)
            queue.put_nowait(message)

    async def _get_or_create_channel(self, case_id: int) -> _CaseChannel:
        async with self._global_lock:
            if case_id not in self._channels:
                self._channels[case_id] = _CaseChannel()
            return self._channels[case_id]

    async def subscribe(self, case_id: int) -> asyncio.Queue[Dict[str, Any]]:
        channel = await self._get_or_create_channel(case_id)
        q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=self._queue_maxsize)
        loop = asyncio.get_running_loop()
        subscriber = _SubscriberConnection(queue=q, loop=loop, thread_id=threading.get_ident())
        async with channel.lock:
            channel.connections.add(subscriber)
        return q

    async def unsubscribe(self, case_id: int, q: asyncio.Queue[Dict[str, Any]]) -> None:
        async with self._global_lock:
            channel = self._channels.get(case_id)
            if channel is None:
                return
        async with channel.lock:
            channel.connections = {subscriber for subscriber in channel.connections if subscriber.queue is not q}
            if not channel.connections:
                async with self._global_lock:
                    if self._channels.get(case_id) is channel:
                        del self._channels[case_id]

    async def close(self) -> None:
        async with self._global_lock:
            self._channels.clear()

    async def publish(self, case_id: int, payload: Dict[str, Any]) -> None:
        channel = await self._get_or_create_channel(case_id)
        validated_payload = TimelineEventPayload.model_validate(payload)
        message = validated_payload.model_dump(mode="json")
        current_loop = asyncio.get_running_loop()
        current_thread_id = threading.get_ident()
        async with channel.lock:
            targets = list(channel.connections)
            dead_targets = [subscriber for subscriber in targets if subscriber.loop.is_closed()]
            if dead_targets:
                channel.connections.difference_update(dead_targets)
                logger.warning(
                    "timeline_realtime_dead_subscribers_removed",
                    case_id=case_id,
                    dead_subscribers=len(dead_targets),
                    subscriber_count=len(targets),
                    remaining_subscribers=len(channel.connections),
                )
                targets = [subscriber for subscriber in targets if not subscriber.loop.is_closed()]

        # fan-out outside lock
        for subscriber in targets:
            q = subscriber.queue
            loop = subscriber.loop

            deliver = lambda q=q: self._deliver_message(
                queue=q,
                message=message,
                case_id=case_id,
                channel=channel,
                record_drop=self._record_drop,
            )

            if loop is current_loop:
                deliver()
            else:
                logger.debug(
                    "timeline_realtime_cross_loop_delivery",
                    case_id=case_id,
                    subscriber_count=len(targets),
                    target_loop_running=loop.is_running(),
                    target_loop_closed=loop.is_closed(),
                    target_thread_id=subscriber.thread_id,
                    publisher_thread_id=current_thread_id,
                )

                if loop.is_closed():
                    continue

                try:
                    loop.call_soon_threadsafe(deliver)
                except RuntimeError:
                    if not loop.is_closed():
                        raise

                    async with channel.lock:
                        if subscriber in channel.connections:
                            channel.connections.remove(subscriber)
                            logger.warning(
                                "timeline_realtime_dead_subscriber_removed",
                                case_id=case_id,
                                subscriber_count=len(channel.connections),
                                target_thread_id=subscriber.thread_id,
                                publisher_thread_id=current_thread_id,
                            )


timeline_queue_maxsize = int(os.getenv("TIMELINE_REALTIME_QUEUE_MAXSIZE", "100"))
timeline_realtime_bus = TimelineRealtimeBus(queue_maxsize=timeline_queue_maxsize)


def publish_timeline_event_best_effort(payload: Dict[str, Any]) -> Optional[asyncio.Task[Any]]:
    """Publish a timeline event without depending on the caller's loop state."""
    case_id = payload["case_id"]
    publish_coro = timeline_realtime_bus.publish(case_id=case_id, payload=payload)

    def _log_publish_task_failure(task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return

        exc = task.exception()
        if exc is not None:
            logger.error(
                "timeline_realtime_publish_failed",
                case_id=case_id,
                error_type=type(exc).__name__,
                error=str(exc),
                exc_info=exc,
            )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        task = loop.create_task(publish_coro)
        task.add_done_callback(_log_publish_task_failure)
        return task

    fallback_loop = asyncio.new_event_loop()
    try:
        fallback_loop.run_until_complete(publish_coro)
    finally:
        fallback_loop.close()
    return None
