"""Tests for the proxy whitelist and X-Forwarded-For fix (#1573).

Verifies that:
- WHITELIST no longer contains service name strings.
- is_whitelisted correctly matches prefixed ip: identifiers.
- Service name strings cannot match is_whitelisted.
- X-Forwarded-For is only trusted when direct peer is a trusted proxy.
- X-Forwarded-For is ignored when direct peer is an untrusted client.
- _load_trusted_proxies rejects non-IP entries.
- resolve_rate_limit_identifier never returns a shared 'anonymous' bucket.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-value-12345")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost")

for _mod in ("streamlit", "pytesseract", "pdf2image"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from fastapi import HTTPException
from api.limiter import (
    WHITELIST,
    TRUSTED_PROXIES,
    is_whitelisted,
    resolve_rate_limit_identifier,
    _load_trusted_proxies,
)


class _FakeClient:
    def __init__(self, host: str):
        self.host = host


class _FakeRequest:
    def __init__(self, headers=None, client_host: str | None = "203.0.113.10"):
        self.headers = headers or {}
        self.client = _FakeClient(client_host) if client_host else None


# ---------------------------------------------------------------------------
# WHITELIST contents
# ---------------------------------------------------------------------------

def test_whitelist_contains_only_ip_addresses():
    """WHITELIST must contain only IP addresses, no service name strings."""
    import ipaddress
    for entry in WHITELIST:
        try:
            ipaddress.ip_address(entry)
        except ValueError:
            raise AssertionError(
                f"WHITELIST contains non-IP entry {entry!r}. "
                "Service name strings cannot be verified at the network layer "
                "and must not be used as a trust signal."
            )


def test_whitelist_does_not_contain_service_names():
    """Service name strings must have been removed from WHITELIST."""
    forbidden = {"internal-admin-service", "internal-ingest-service", "localhost"}
    overlap = WHITELIST & forbidden
    assert not overlap, (
        f"WHITELIST still contains service name strings: {overlap}. "
        "These can never match ip: prefixed identifiers and create a false "
        "sense of security."
    )


def test_whitelist_contains_loopback_addresses():
    """Loopback addresses must remain in WHITELIST."""
    assert "127.0.0.1" in WHITELIST
    assert "::1" in WHITELIST


# ---------------------------------------------------------------------------
# is_whitelisted with prefixed identifiers
# ---------------------------------------------------------------------------

def test_is_whitelisted_matches_prefixed_loopback():
    """is_whitelisted must match 'ip:127.0.0.1' (prefixed form)."""
    assert is_whitelisted("ip:127.0.0.1") is True
    assert is_whitelisted("ip:::1") is True


def test_is_whitelisted_rejects_bare_ip_without_prefix():
    """Bare IP strings without 'ip:' prefix must not be whitelisted."""
    assert is_whitelisted("127.0.0.1") is False


def test_is_whitelisted_rejects_service_names():
    """Service name strings must never be whitelisted."""
    assert is_whitelisted("internal-admin-service") is False
    assert is_whitelisted("internal-ingest-service") is False
    assert is_whitelisted("localhost") is False


def test_is_whitelisted_rejects_user_identifiers():
    """Authenticated user identifiers must not be whitelisted."""
    assert is_whitelisted("user:42") is False


def test_is_whitelisted_rejects_external_ip():
    """External IP addresses must not be whitelisted."""
    assert is_whitelisted("ip:203.0.113.10") is False


# ---------------------------------------------------------------------------
# X-Forwarded-For only trusted from trusted proxies
# ---------------------------------------------------------------------------

def test_xff_ignored_when_direct_client_is_untrusted():
    """X-Forwarded-For must be ignored when direct peer is not a trusted proxy."""
    request = _FakeRequest(
        headers={"X-Forwarded-For": "1.2.3.4"},
        client_host="203.0.113.99",  # not a trusted proxy
    )
    key = resolve_rate_limit_identifier(request)
    # Must use the real transport-layer IP, not the XFF value
    assert key == "ip:203.0.113.99"
    assert "1.2.3.4" not in key


def test_xff_trusted_when_direct_client_is_trusted_proxy(monkeypatch):
    """X-Forwarded-For IS used when the direct peer is a known trusted proxy."""
    monkeypatch.setenv("RATE_LIMIT_TRUSTED_PROXIES", "10.0.0.1,127.0.0.1,::1")

    request = _FakeRequest(
        headers={"X-Forwarded-For": "5.6.7.8"},
        client_host="10.0.0.1",
    )
    key = resolve_rate_limit_identifier(request)
    # Direct peer 10.0.0.1 is a trusted proxy → use XFF first hop
    assert key == "ip:5.6.7.8"

    # Without a direct client IP, proxy trust cannot be verified → anon
    request_no_direct = _FakeRequest(
        headers={"X-Forwarded-For": "5.6.7.8"},
        client_host=None,
    )
    key_no_direct = resolve_rate_limit_identifier(request_no_direct)
    assert key_no_direct.startswith("anon:")


def test_xff_multiple_hops_uses_first(monkeypatch):
    """With multiple XFF hops, only the first (leftmost) client IP is used."""
    monkeypatch.setenv("RATE_LIMIT_TRUSTED_PROXIES", "127.0.0.1,::1")
    request = _FakeRequest(
        headers={"X-Forwarded-For": "9.8.7.6, 10.0.0.1, 10.0.0.2"},
        client_host="127.0.0.1",  # loopback — trusted proxy
    )
    key = resolve_rate_limit_identifier(request)
    # Direct peer 127.0.0.1 is a trusted proxy → use first XFF hop
    assert key == "ip:9.8.7.6"


# ---------------------------------------------------------------------------
# resolve_rate_limit_identifier never returns 'anonymous'
# ---------------------------------------------------------------------------

def test_no_shared_anonymous_bucket():
    """resolve_rate_limit_identifier must never return the literal 'anonymous'."""
    request = _FakeRequest(client_host="10.0.0.5")
    key = resolve_rate_limit_identifier(request)
    assert key != "anonymous"
    assert key == "ip:10.0.0.5"


def test_no_direct_ip_returns_unique_anon_token():
    """When no IP is available, each call must produce a unique anon token."""
    request = _FakeRequest(client_host=None)
    k1 = resolve_rate_limit_identifier(request)
    k2 = resolve_rate_limit_identifier(request)
    assert k1.startswith("anon:")
    assert k2.startswith("anon:")
    assert k1 != k2


# ---------------------------------------------------------------------------
# _load_trusted_proxies rejects non-IP entries
# ---------------------------------------------------------------------------

def test_load_trusted_proxies_rejects_service_names(monkeypatch):
    """_load_trusted_proxies must silently drop non-IP entries."""
    monkeypatch.setenv(
        "RATE_LIMIT_TRUSTED_PROXIES",
        "10.0.0.1,internal-load-balancer,not-an-ip,192.168.1.1",
    )
    result = _load_trusted_proxies()
    assert "10.0.0.1" in result
    assert "192.168.1.1" in result
    assert "internal-load-balancer" not in result
    assert "not-an-ip" not in result
    # Loopback always included
    assert "127.0.0.1" in result
    assert "::1" in result


def test_load_trusted_proxies_empty_env(monkeypatch):
    """_load_trusted_proxies with empty env must return only loopback addresses."""
    monkeypatch.setenv("RATE_LIMIT_TRUSTED_PROXIES", "")
    result = _load_trusted_proxies()
    assert "127.0.0.1" in result
    assert "::1" in result
