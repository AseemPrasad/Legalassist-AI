"""Secret manager adapter supporting HashiCorp Vault (KV v2) with env fallback.

Usage:
    from utils.secret_manager import SecretManager
    sm = SecretManager()
    secret = sm.get_secret('sendgrid_api_key')

Behavior:
    - If VAULT_ADDR and VAULT_TOKEN are set and `hvac` is installed, attempts to read
      KV v2 at path: `${VAULT_KV_MOUNT:-secret}/data/${secret_name}` and return the `data.data` value
      or the field named `value` inside it.
    - Otherwise falls back to environment variables (uppercased key) and to .env via existing config loader.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import hvac
except Exception:
    hvac = None


class SecretManager:
    def __init__(self):
        self.vault_addr = os.getenv("VAULT_ADDR")
        self.vault_token = os.getenv("VAULT_TOKEN")
        self.kv_mount = os.getenv("VAULT_KV_MOUNT", "secret")
        self.client = None
        if self.vault_addr and self.vault_token and hvac is not None:
            try:
                self.client = hvac.Client(url=self.vault_addr, token=self.vault_token)
                if not self.client.is_authenticated():
                    logger.warning("Vault client not authenticated; falling back to env")
                    self.client = None
            except Exception as e:
                logger.warning("Failed to create Vault client, falling back to env: %s", e)
                self.client = None

    def _env_fallback(self, name: str) -> Optional[str]:
        # canonical env key e.g., SENDGRID_API_KEY
        env_key = name.upper()
        return os.getenv(env_key)

    def get_secret(self, name: str) -> Optional[str]:
        """Return secret value for `name`.

        If Vault is available, read from `${kv_mount}/data/{name}` and return either
        the `data.data['value']` or if `data.data` is a mapping with a key matching
        the secret name, return that. Otherwise, fall back to an environment variable.
        """
        # First, try Vault KV v2
        if self.client:
            try:
                path = f"{self.kv_mount}/data/{name}"
                secret = self.client.secrets.kv.v2.read_secret_version(path=name, mount_point=self.kv_mount)
                data = secret.get("data", {}).get("data", {})
                # Prefer common field names
                for key in ("value", name, "secret", "api_key", "key"):
                    if key in data:
                        return data.get(key)
                # If a single-string value stored under a known field
                if isinstance(data, str):
                    return data
                # If data contains a single item, return its value
                if isinstance(data, dict) and len(data) == 1:
                    return list(data.values())[0]
            except Exception as e:
                logger.debug("Vault read failed for %s: %s", name, e)

        # Fallback to environment variables
        return self._env_fallback(name)


_default_manager: Optional[SecretManager] = None


def get_secret_manager() -> SecretManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = SecretManager()
    return _default_manager


def get_secret(name: str) -> Optional[str]:
    return get_secret_manager().get_secret(name)
