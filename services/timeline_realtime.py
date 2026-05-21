from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Set


@dataclass
class _CaseChannel:
    connections: Set["asyncio.Queue[str]"] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class TimelineRealtimeBus:
    """
    Simple in-memory case-scoped pub/sub bus.

    - Each websocket connection subscribes by providing an asyncio.Queue
    - Writers broadcast a JSON-serializable payload to all subscribers of
      the given case_id.
    """

    def __init__(self) -> None:
        self._channels: Dict[int, _CaseChannel] = {}
        self._global_lock = asyncio.Lock()

    async def _get_or_create_channel(self, case_id: int) -> _CaseChannel:
        async with self._global_lock:
            if case_id not in self._channels:
                self._channels[case_id] = _CaseChannel()
            return self._channels[case_id]

    async def subscribe(self, case_id: int) -> asyncio.Queue[str]:
        channel = await self._get_or_create_channel(case_id)
        q: asyncio.Queue[str] = asyncio.Queue()
        async with channel.lock:
            channel.connections.add(q)
        return q

    async def unsubscribe(self, case_id: int, q: asyncio.Queue[str]) -> None:
        async with self._global_lock:
            channel = self._channels.get(case_id)
            if channel is None:
                return
        async with channel.lock:
            channel.connections.discard(q)
            if not channel.connections:
                async with self._global_lock:
                    if self._channels.get(case_id) is channel:
                        del self._channels[case_id]

    async def close(self) -> None:
        async with self._global_lock:
            self._channels.clear()

    async def publish(self, case_id: int, payload: Dict[str, Any]) -> None:
        channel = await self._get_or_create_channel(case_id)
        message = json.dumps(payload, default=str)
        async with channel.lock:
            targets = list(channel.connections)

        # fan-out outside lock
        for q in targets:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                # drop message for that slow consumer
                pass


timeline_realtime_bus = TimelineRealtimeBus()


def publish_timeline_event_best_effort(payload: Dict[str, Any]) -> None:
    """Publish a timeline event without depending on the caller's loop state."""
    case_id = payload["case_id"]
    publish_coro = timeline_realtime_bus.publish(case_id=case_id, payload=payload)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        loop.create_task(publish_coro)
        return

    fallback_loop = asyncio.new_event_loop()
    try:
        fallback_loop.run_until_complete(publish_coro)
    finally:
        fallback_loop.close()
