import asyncio

from services.timeline_realtime import TimelineRealtimeBus


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