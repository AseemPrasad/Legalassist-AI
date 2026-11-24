import asyncio

import pytest

from api import shutdown


def run(coro):
    return asyncio.run(coro)


def test_stages_run_in_order_and_support_sync_and_async():
    order = []

    async def a():
        order.append("a")

    def b():
        order.append("b")

    assert run(shutdown.run_shutdown([("a", a), ("b", b)], timeout=5)) is True
    assert order == ["a", "b"]


def test_failing_stage_does_not_stop_later_stages():
    order = []

    def boom():
        raise RuntimeError("redis gone")

    def after():
        order.append("after")

    assert run(shutdown.run_shutdown([("boom", boom), ("after", after)], timeout=5)) is False
    assert order == ["after"]


def test_slow_stage_hits_timeout():
    async def slow():
        await asyncio.sleep(5)

    assert run(shutdown.run_shutdown([("slow", slow)], timeout=0.05)) is False


@pytest.mark.parametrize(
    "raw, expected",
    [(None, 25.0), ("", 25.0), ("x", 25.0), ("0", 25.0), ("-1", 25.0), ("8", 8.0)],
)
def test_timeout_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("API_SHUTDOWN_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("API_SHUTDOWN_TIMEOUT", raw)
    assert shutdown.get_shutdown_timeout() == expected


def test_default_stages_named():
    names = [n for n, _ in shutdown.default_stages()]
    assert names == ["redis_rate_limiter", "database_pool", "log_handlers"]
