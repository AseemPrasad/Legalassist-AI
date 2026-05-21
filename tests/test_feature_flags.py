"""Tests for FeatureFlagManager"""

import os
import pytest
from unittest.mock import MagicMock

from api.feature_flags import FeatureFlagManager, FeatureFlagDefinition, get_feature_flag_manager


def test_defaults_and_env_override(monkeypatch):
    defaults = {"NEW_UI": False}
    manager = FeatureFlagManager(defaults=defaults)

    # default false
    assert manager.is_enabled("new_ui") is False

    # env override
    monkeypatch.setenv("FEATURE_NEW_UI", "1")
    assert manager.is_enabled("new_ui") is True


def test_redis_override(monkeypatch):
    fake_redis = MagicMock()
    fake_redis.get.return_value = "1"

    manager = FeatureFlagManager(defaults={"X": False}, redis_url="redis://fake")
    manager._client = fake_redis

    assert manager.is_enabled("x") is True
    fake_redis.get.assert_called()


def test_set_flag_without_redis(monkeypatch):
    manager = FeatureFlagManager()
    # no redis configured -> set_flag should return False
    assert manager.set_flag("feature", True) is False


def test_get_feature_flag_manager_singleton():
    m1 = get_feature_flag_manager()
    m2 = get_feature_flag_manager()
    assert m1 is m2


def test_deterministic_user_bucketing():
    definition = FeatureFlagDefinition(
        name="knowledge_status_dashboard",
        rollout_percent=10,
        targeting_rules={"roles": ["user", "admin"]},
    )
    manager = FeatureFlagManager(definitions=[definition])

    bucket_a_1 = manager._deterministic_bucket("knowledge_status_dashboard", "user-123")
    bucket_a_2 = manager._deterministic_bucket("knowledge_status_dashboard", "user-123")
    bucket_b = manager._deterministic_bucket("knowledge_status_dashboard", "user-456")

    assert bucket_a_1 == bucket_a_2
    assert 0 <= bucket_a_1 < 100
    assert 0 <= bucket_b < 100

    enabled_first = manager.is_enabled_for_user(
        "knowledge_status_dashboard",
        "user-123",
        attributes={"role": "user"},
        surface="ui",
        record_event=False,
    )
    enabled_second = manager.is_enabled_for_user(
        "knowledge_status_dashboard",
        "user-123",
        attributes={"role": "user"},
        surface="ui",
        record_event=False,
    )

    assert enabled_first == enabled_second


def test_targeting_rules_and_flag_events(monkeypatch):
    definition = FeatureFlagDefinition(
        name="beta_panel",
        rollout_percent=100,
        targeting_rules={"roles": ["admin"], "attributes": {"region": {"equals": "in"}}},
    )
    manager = FeatureFlagManager(definitions=[definition])

    events = []

    def capture_event(event, flag_name, *, surface="api", variant="control"):
        events.append((event, flag_name, surface, variant))

    monkeypatch.setattr("api.feature_flags.record_feature_flag_event", capture_event)

    assert manager.is_enabled_for_user(
        "beta_panel",
        "user-1",
        attributes={"role": "admin", "region": "in"},
        surface="api",
    ) is True
    manager.mark_flag_used("beta_panel", user_id="user-1", surface="api")

    assert manager.is_enabled_for_user(
        "beta_panel",
        "user-2",
        attributes={"role": "user", "region": "in"},
        surface="ui",
    ) is False

    assert any(event[0] == "flag_shown" and event[1] == "BETA_PANEL" and event[2] == "api" for event in events)
    assert any(event[0] == "flag_used" and event[1] == "BETA_PANEL" and event[2] == "api" for event in events)
