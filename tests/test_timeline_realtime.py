import asyncio
from datetime import datetime, timezone

import pytest

from core.timeline_payloads import TimelineEventPayload
from services.timeline_realtime import TimelineRealtimeBus
from pydantic import ValidationError


def test_unsubscribe_removes_empty_channel():
    bus = TimelineRealtimeBus()

    async def scenario() -> None:
        queue = await bus.subscribe(42)
        assert 42 in bus._channels

        await bus.unsubscribe(42, queue)
        assert 42 not in bus._channels

    asyncio.run(scenario())


def test_close_clears_all_channels():
    bus = TimelineRealtimeBus()

    async def scenario() -> None:
        await bus.subscribe(1)
        await bus.subscribe(2)
        assert set(bus._channels) == {1, 2}

        await bus.close()
        assert bus._channels == {}

    asyncio.run(scenario())


def test_publish_normalizes_datetimes_to_utc_iso():
    bus = TimelineRealtimeBus()

    async def scenario() -> None:
        queue = await bus.subscribe(7)

        await bus.publish(
            7,
            {
                "type": "timeline_event",
                "case_id": 7,
                "event_type": "deadline_created",
                "description": "Manual deadline added",
                "timestamp": datetime(2026, 5, 22, 10, 30, tzinfo=timezone.utc),
                "metadata": {
                    "nested_timestamp": datetime(2026, 5, 22, 10, 31),
                },
                "event_id": 555,
            },
        )

        payload = await queue.get()
        validated = TimelineEventPayload.model_validate(payload)
        assert validated.type == "timeline_event"
        assert validated.case_id == 7
        assert validated.event_type == "deadline_created"
        assert validated.description == "Manual deadline added"
        assert validated.timestamp == datetime(2026, 5, 22, 10, 30, tzinfo=timezone.utc)
        assert validated.event_id == 555
        assert validated.model_dump(mode="json")["metadata"]["nested_timestamp"] == "2026-05-22T10:31:00+00:00"

    asyncio.run(scenario())


def test_publish_rejects_invalid_payload_shape():
    bus = TimelineRealtimeBus()

    async def scenario() -> None:
        with pytest.raises(ValidationError):
            await bus.publish(
                7,
                {
                    "type": "timeline_event",
                    "case_id": 7,
                    "event_type": "deadline_created",
                    "description": "Manual deadline added",
                    "timestamp": datetime(2026, 5, 22, 10, 30, tzinfo=timezone.utc),
                    "metadata": {},
                },
            )

    asyncio.run(scenario())