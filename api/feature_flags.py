"""
Feature flag manager with optional Redis backend and env overrides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from typing import Optional, Dict, Any, Iterable, Mapping
import structlog

from observability.instrumentation import record_feature_flag_event

try:
    import redis
except Exception:  # pragma: no cover
    redis = None

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class FeatureFlagDefinition:
    """Configuration schema for a feature flag rollout."""

    name: str
    rollout_percent: int = 100
    targeting_rules: Dict[str, Any] = field(default_factory=dict)
    enabled: Optional[bool] = None

    def __post_init__(self):
        normalized_name = self.name.upper().strip()
        if not normalized_name:
            raise ValueError("Feature flag name cannot be empty")
        if self.rollout_percent < 0 or self.rollout_percent > 100:
            raise ValueError("rollout_percent must be between 0 and 100")
        object.__setattr__(self, "name", normalized_name)


DEFAULT_FEATURE_FLAG_DEFINITIONS = [
    FeatureFlagDefinition(
        name="knowledge_status_dashboard",
        rollout_percent=int(os.getenv("FEATURE_KNOWLEDGE_STATUS_DASHBOARD_ROLLOUT", "10")),
        targeting_rules={},
    ),
]


def _normalize_flag_name(name: str) -> str:
    return str(name or "").upper().strip()


def _coerce_target_list(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return {str(value)}


class FeatureFlagManager:
    """Simple feature flag manager.

    Priority order:
      1. Redis backend (if configured and available)
      2. Environment variables (FEATURE_<NAME>=1)
      3. Default values provided at init
    """

    def __init__(
        self,
        defaults: Optional[Dict[str, bool]] = None,
        redis_url: Optional[str] = None,
        definitions: Optional[Iterable[FeatureFlagDefinition | Mapping[str, Any]]] = None,
    ):
        self.defaults = defaults or {}
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self._client = None
        self.definitions: Dict[str, FeatureFlagDefinition] = {}

        for definition in definitions or []:
            self.register_flag_definition(definition)

    @property
    def client(self):
        if self._client is None and self.redis_url:
            if redis is None:
                logger.warning("redis_not_installed_for_feature_flags")
                self._client = None
            else:
                try:
                    self._client = redis.from_url(self.redis_url, decode_responses=True)
                except Exception as e:
                    logger.error("feature_flags_redis_init_failed", error=str(e))
                    self._client = None
        return self._client

    def _redis_key(self, name: str) -> str:
        return f"feature:{name}"

    def _redis_definition_key(self, name: str) -> str:
        return f"feature:def:{name}"

    def register_flag_definition(self, definition: FeatureFlagDefinition | Mapping[str, Any]) -> FeatureFlagDefinition:
        if not isinstance(definition, FeatureFlagDefinition):
            definition = FeatureFlagDefinition(**dict(definition))
        self.definitions[definition.name] = definition
        client = self.client
        if client:
            try:
                client.set(self._redis_definition_key(definition.name), json.dumps({
                    "name": definition.name,
                    "rollout_percent": definition.rollout_percent,
                    "targeting_rules": definition.targeting_rules,
                    "enabled": definition.enabled,
                }))
            except Exception as e:
                logger.warning("feature_flags_definition_persist_failed", name=definition.name, error=str(e))
        return definition

    def get_flag_definition(self, name: str) -> Optional[FeatureFlagDefinition]:
        name_up = _normalize_flag_name(name)
        if name_up in self.definitions:
            return self.definitions[name_up]

        client = self.client
        if client:
            try:
                raw = client.get(self._redis_definition_key(name_up))
                if raw:
                    payload = json.loads(raw)
                    definition = FeatureFlagDefinition(**payload)
                    self.definitions[name_up] = definition
                    return definition
            except Exception as e:
                logger.warning("feature_flags_definition_load_failed", name=name_up, error=str(e))
        return None

    def _deterministic_bucket(self, name: str, user_id: str, salt: str = "") -> int:
        payload = f"{_normalize_flag_name(name)}:{user_id}:{salt}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return int(digest[:8], 16) % 100

    def _matches_targeting_rules(self, rules: Dict[str, Any], *, user_id: Optional[str], attributes: Optional[Dict[str, Any]]) -> bool:
        if not rules:
            return True

        attributes = attributes or {}
        user_id_str = str(user_id) if user_id is not None else None

        include_user_ids = _coerce_target_list(rules.get("user_ids") or rules.get("include_user_ids"))
        exclude_user_ids = _coerce_target_list(rules.get("exclude_user_ids") or rules.get("deny_user_ids"))
        roles = _coerce_target_list(rules.get("roles"))
        include_emails = _coerce_target_list(rules.get("emails") or rules.get("include_emails"))

        if include_user_ids and user_id_str not in include_user_ids:
            return False
        if user_id_str and user_id_str in exclude_user_ids:
            return False

        if roles:
            role = str(attributes.get("role", ""))
            if role not in roles:
                return False

        if include_emails:
            email = str(attributes.get("email", ""))
            if email not in include_emails:
                return False

        attribute_rules = rules.get("attributes") or {}
        for key, expected in attribute_rules.items():
            actual = attributes.get(key)
            if isinstance(expected, dict):
                if "equals" in expected and actual != expected["equals"]:
                    return False
                if "in" in expected and actual not in set(expected["in"]):
                    return False
                if "not_in" in expected and actual in set(expected["not_in"]):
                    return False
            elif actual != expected:
                return False

        return True

    def _evaluate_definition(
        self,
        definition: FeatureFlagDefinition,
        *,
        user_id: Optional[str],
        attributes: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, int, str]:
        attributes = attributes or {}

        if definition.enabled is not None:
            enabled = bool(definition.enabled)
            return enabled, 0 if enabled else 100, "forced_on" if enabled else "forced_off"

        if not self._matches_targeting_rules(definition.targeting_rules, user_id=user_id, attributes=attributes):
            return False, 100, "targeting_excluded"

        if user_id is None:
            return bool(self.defaults.get(definition.name, False)), 0, "anonymous"

        salt = str(definition.targeting_rules.get("salt", ""))
        bucket = self._deterministic_bucket(definition.name, user_id, salt=salt)
        return bucket < definition.rollout_percent, bucket, f"bucket_{bucket:02d}"

    def is_enabled(self, name: str) -> bool:
        name_up = name.upper()
        # 1) Redis override
        try:
            client = self.client
            if client:
                val = client.get(self._redis_key(name_up))
                if val is not None:
                    return str(val).lower() in ("1", "true", "yes", "on")
        except Exception as e:
            logger.warning("feature_flags_redis_unavailable", error=str(e))

        # 2) Env var override
        env_key = f"FEATURE_{name_up}"
        env_val = os.getenv(env_key)
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")

        # 3) Defaults
        return bool(self.defaults.get(name_up, False))

    def is_enabled_for_user(
        self,
        name: str,
        user_id: Optional[str],
        *,
        attributes: Optional[Dict[str, Any]] = None,
        surface: str = "api",
        record_event: bool = True,
    ) -> bool:
        name_up = _normalize_flag_name(name)
        definition = self.get_flag_definition(name_up)

        if definition is None:
            enabled = self.is_enabled(name_up)
            if record_event and user_id is not None:
                record_feature_flag_event("flag_shown", name_up, surface=surface, variant="default_enabled" if enabled else "default_disabled")
            return enabled

        enabled, _bucket, variant = self._evaluate_definition(definition, user_id=user_id, attributes=attributes)
        if record_event:
            record_feature_flag_event("flag_shown", name_up, surface=surface, variant=variant)
        return enabled

    def mark_flag_used(
        self,
        name: str,
        *,
        user_id: Optional[str] = None,
        surface: str = "api",
        variant: str = "used",
    ) -> None:
        record_feature_flag_event("flag_used", _normalize_flag_name(name), surface=surface, variant=variant)

    def set_flag(self, name: str, enabled: bool) -> bool:
        name_up = name.upper()
        client = self.client
        if not client:
            logger.warning("feature_flags_no_redis", name=name_up)
            return False
        try:
            client.set(self._redis_key(name_up), "1" if enabled else "0")
            return True
        except Exception as e:
            logger.error("feature_flags_set_failed", name=name_up, error=str(e))
            return False


# singleton
_manager: Optional[FeatureFlagManager] = None


def get_feature_flag_manager(defaults: Optional[Dict[str, bool]] = None) -> FeatureFlagManager:
    global _manager
    if _manager is None:
        _manager = FeatureFlagManager(defaults=defaults, definitions=DEFAULT_FEATURE_FLAG_DEFINITIONS)
    return _manager


def is_feature_enabled_for_user(
    name: str,
    user_id: Optional[str],
    *,
    attributes: Optional[Dict[str, Any]] = None,
    surface: str = "api",
) -> bool:
    return get_feature_flag_manager().is_enabled_for_user(
        name,
        user_id,
        attributes=attributes,
        surface=surface,
    )
