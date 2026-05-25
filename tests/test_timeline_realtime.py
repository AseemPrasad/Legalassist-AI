import asyncio
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from core.timeline_payloads import TimelineEventPayload
from services.timeline_realtime import TimelineRealtimeBus, _SubscriberConnection, publish_timeline_event_best_effort
from pydantic import ValidationError


def test_unsubscribe_removes_empty_channel():
    bus = TimelineRealtimeBus()

    async def scenario() -> None:
        queue = await bus.subscribe(42)
        assert 42 in bus._channels

        await bus.unsubscribe(42, queue)
        assert 42 not in bus._channels

    asyncio.run(scenario())


def test_unsubscribe_prunes_closed_subscribers():
    bus = TimelineRealtimeBus()

    class ClosedLoop:
        def is_closed(self):
            return True

    async def scenario() -> None:
        queue = await bus.subscribe(42)
        channel = await bus._get_or_create_channel(42)

        async with channel.lock:
            channel.connections.add(
                _SubscriberConnection(
                    queue=asyncio.Queue(maxsize=1),
                    loop=ClosedLoop(),
                    thread_id=999,
                )
            )

        await bus.unsubscribe(42, queue)

        assert 42 not in bus._channels
        assert bus.dropped_connections_total == 2

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
                "schema_version": 2,
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
        assert set(TimelineEventPayload.model_fields) == {
            "schema_version",
            "type",
            "case_id",
            "event_type",
            "description",
            "timestamp",
            "metadata",
            "event_id",
        }
        assert set(validated.model_dump(mode="json")) == {
            "schema_version",
            "type",
            "case_id",
            "event_type",
            "description",
            "timestamp",
            "metadata",
            "event_id",
        }
        assert validated.schema_version == TimelineEventPayload.CURRENT_SCHEMA_VERSION
        assert validated.type == "timeline_event"
        assert validated.case_id == 7
        assert validated.event_type == "deadline_created"
        assert validated.description == "Manual deadline added"
        assert validated.timestamp == datetime(2026, 5, 22, 10, 30, tzinfo=timezone.utc)
        assert validated.event_id == 555
        assert validated.model_dump(mode="json")["metadata"]["nested_timestamp"] == "2026-05-22T10:31:00+00:00"

    asyncio.run(scenario())


def test_timeline_event_payload_accepts_legacy_version_and_ignores_extra_fields():
    payload = TimelineEventPayload.model_validate(
        {
            "type": "timeline_event",
            "case_id": 7,
            "event_type": "deadline_created",
            "description": "Manual deadline added",
            "timestamp": datetime(2026, 5, 22, 10, 30, tzinfo=timezone.utc),
            "metadata": {},
            "event_id": 555,
            "unexpected": "value",
        }
    )

    assert payload.schema_version == TimelineEventPayload.LEGACY_SCHEMA_VERSION
    assert payload.model_dump(mode="json") == {
        "schema_version": TimelineEventPayload.LEGACY_SCHEMA_VERSION,
        "type": "timeline_event",
        "case_id": 7,
        "event_type": "deadline_created",
        "description": "Manual deadline added",
        "timestamp": "2026-05-22T10:30:00+00:00",
        "metadata": {},
        "event_id": 555,
    }


def test_timeline_event_payload_supports_schema_version_two_and_aliases():
    payload = TimelineEventPayload.model_validate(
        {
            "schemaVersion": 2,
            "type": "timeline_event",
            "caseId": 7,
            "eventType": "deadline_created",
            "description": "Manual deadline added",
            "timestamp": datetime(2026, 5, 22, 10, 30),
            "metadata": None,
            "eventId": 555,
            "extra": "ignored",
        }
    )

    assert payload.schema_version == 2
    assert payload.metadata == {}
    assert payload.timestamp == datetime(2026, 5, 22, 10, 30, tzinfo=timezone.utc)


def test_publish_rejects_invalid_payload_shape():
    bus = TimelineRealtimeBus()

    async def scenario() -> None:
        with pytest.raises(ValidationError):
            await bus.publish(
                7,
                {
                    "schema_version": 2,
                    "type": "timeline_event",
                    "case_id": 7,
                    "event_type": "deadline_created",
                    "description": "Manual deadline added",
                    "timestamp": datetime(2026, 5, 22, 10, 30, tzinfo=timezone.utc),
                    "metadata": {},
                },
            )

    asyncio.run(scenario())


def test_publish_keeps_latest_message_when_queue_is_full():
    bus = TimelineRealtimeBus(queue_maxsize=1)

    async def scenario() -> None:
        queue = await bus.subscribe(7)
        assert queue.maxsize == 1
        assert bus.queue_maxsize == 1

        with patch("services.timeline_realtime.logger.warning") as mock_warning:
            await bus.publish(
                7,
                {
                    "schema_version": 2,
                    "type": "timeline_event",
                    "case_id": 7,
                    "event_type": "deadline_created",
                    "description": "Oldest message",
                    "timestamp": datetime(2026, 5, 22, 10, 30, tzinfo=timezone.utc),
                    "metadata": {},
                    "event_id": 1,
                },
            )
            await bus.publish(
                7,
                {
                    "schema_version": 2,
                    "type": "timeline_event",
                    "case_id": 7,
                    "event_type": "deadline_created",
                    "description": "Newest message",
                    "timestamp": datetime(2026, 5, 22, 10, 31, tzinfo=timezone.utc),
                    "metadata": {},
                    "event_id": 2,
                },
            )

            payload = await queue.get()
            validated = TimelineEventPayload.model_validate(payload)

            assert validated.description == "Newest message"
            assert validated.event_id == 2
            assert validated.schema_version == TimelineEventPayload.CURRENT_SCHEMA_VERSION
            assert bus.dropped_messages_total == 1
            assert bus._channels[7].dropped_messages == 1
            assert mock_warning.call_count == 1
            assert mock_warning.call_args.kwargs["policy"] == "drop_oldest_keep_latest"
            assert mock_warning.call_args.kwargs["case_id"] == 7

    asyncio.run(scenario())


def test_publish_prunes_closed_subscribers_before_fanout():
    bus = TimelineRealtimeBus()

    class FakeLiveLoop:
        def is_closed(self):
            return False

        def is_running(self):
            return True

        def call_soon_threadsafe(self, callback):
            callback()

    async def scenario() -> None:
        queue = await bus.subscribe(7)
        channel = await bus._get_or_create_channel(7)

        dead_loop = asyncio.new_event_loop()
        dead_loop.close()
        fake_live_loop = FakeLiveLoop()

        async with channel.lock:
            channel.connections.add(
                _SubscriberConnection(
                    queue=asyncio.Queue(maxsize=1),
                    loop=dead_loop,
                    thread_id=999,
                )
            )
            channel.connections.add(
                _SubscriberConnection(
                    queue=asyncio.Queue(maxsize=1),
                    loop=fake_live_loop,
                    thread_id=1000,
                )
            )

        with patch("services.timeline_realtime.logger.warning") as mock_warning:
            await bus.publish(
                7,
                {
                    "schema_version": 2,
                    "type": "timeline_event",
                    "case_id": 7,
                    "event_type": "deadline_created",
                    "description": "Manual deadline added",
                    "timestamp": datetime(2026, 5, 22, 10, 30, tzinfo=timezone.utc),
                    "metadata": {},
                    "event_id": 555,
                },
            )

            payload = await queue.get()
            validated = TimelineEventPayload.model_validate(payload)
            assert validated.case_id == 7
            assert len(channel.connections) == 1
            assert bus.dropped_connections_total == 1
            assert channel.dropped_connections == 1
            assert mock_warning.call_args.kwargs["dropped_connections"] == 1
            assert mock_warning.call_args.kwargs["reason"] == "publish_prune"

        dead_loop.close()

    asyncio.run(scenario())


def test_publish_logs_cross_loop_delivery_metadata():
    bus = TimelineRealtimeBus()

    class FakeLoop:
        def __init__(self):
            self.calls = []

        def is_closed(self):
            return False

        def is_running(self):
            return True

        def call_soon_threadsafe(self, callback):
            self.calls.append(callback)
            callback()

    async def scenario() -> None:
        channel = await bus._get_or_create_channel(7)
        target_loop = FakeLoop()
        queue = asyncio.Queue(maxsize=1)

        async with channel.lock:
            channel.connections.add(
                _SubscriberConnection(
                    queue=queue,
                    loop=target_loop,
                    thread_id=12345,
                )
            )

        with patch("services.timeline_realtime.logger.debug") as mock_debug:
            await bus.publish(
                7,
                {
                    "schema_version": 2,
                    "type": "timeline_event",
                    "case_id": 7,
                    "event_type": "deadline_created",
                    "description": "Manual deadline added",
                    "timestamp": datetime(2026, 5, 22, 10, 30, tzinfo=timezone.utc),
                    "metadata": {},
                    "event_id": 555,
                },
            )

            payload = await queue.get()
            validated = TimelineEventPayload.model_validate(payload)
            assert validated.case_id == 7
            assert target_loop.calls
            assert mock_debug.call_args.kwargs["subscriber_count"] == 1
            assert mock_debug.call_args.kwargs["target_loop_running"] is True
            assert mock_debug.call_args.kwargs["target_loop_closed"] is False

    asyncio.run(scenario())


def test_publish_timeline_event_best_effort_logs_async_task_failures():
    async def scenario() -> None:
        payload = {
            "schema_version": 2,
            "type": "timeline_event",
            "case_id": 7,
            "event_type": "deadline_created",
            "description": "Manual deadline added",
            "timestamp": datetime(2026, 5, 22, 10, 30, tzinfo=timezone.utc),
            "metadata": {},
            "event_id": 555,
        }

        async def failing_publish(*args, **kwargs):
            raise ValueError("boom")

        with patch.object(
            TimelineRealtimeBus,
            "publish",
            failing_publish,
        ):
            with patch("services.timeline_realtime.logger.error") as mock_error:
                task = publish_timeline_event_best_effort(payload)
                assert task is not None

                with pytest.raises(ValueError, match="boom"):
                    await task

                assert mock_error.call_count == 1
                assert mock_error.call_args.args[0] == "timeline_realtime_publish_failed"
                assert mock_error.call_args.kwargs["case_id"] == 7
                assert mock_error.call_args.kwargs["error_type"] == "ValueError"
                assert mock_error.call_args.kwargs["error"] == "boom"

    asyncio.run(scenario())


def test_publish_timeline_event_best_effort_handles_missing_case_id():
    with patch("services.timeline_realtime.logger.error") as mock_error:
        result = publish_timeline_event_best_effort(
            {
                "schema_version": 2,
                "type": "timeline_event",
                "event_type": "deadline_created",
                "description": "Malformed payload",
                "timestamp": datetime(2026, 5, 22, 10, 30, tzinfo=timezone.utc),
                "metadata": {},
                "event_id": 555,
            }
        )

        assert result is None
        assert mock_error.call_count == 1
        assert mock_error.call_args.args[0] == "timeline_realtime_publish_malformed_payload"
        assert mock_error.call_args.kwargs["error_type"] == "KeyError"
        assert mock_error.call_args.kwargs["error"] == "case_id missing"
        assert mock_error.call_args.kwargs["payload_keys"] == [
            "description",
            "event_id",
            "event_type",
            "metadata",
            "schema_version",
            "timestamp",
            "type",
        ]