"""Utility facade to retrieve and rotate secrets.

This module centralizes secret retrieval. It prefers environment variables
and falls back to the DB-backed SecretStore implemented in `core.secrets`.
"""
import os
import secrets as _secrets
import logging
from typing import Optional
from core.secrets import SecretStore

logger = logging.getLogger(__name__)


def get_secret(name: str) -> Optional[str]:
    """Return secret value: prefer environment variable, then DB-backed store."""
    v = os.getenv(name.upper())
    if v:
        return v

    db = None
    try:
        # Lazy import to avoid pulling in heavy DB ORM at module import time
        from database import SessionLocal
        db = SessionLocal()
        store = SecretStore(db)
        return store.get_secret(name)
    except Exception as e:
        logger.exception("Failed to get secret %s: %s", name, e)
        return None
    finally:
        if db:
            try:
                db.close()
            except Exception:
                pass


def rotate_secret(name: str, new_value: Optional[str] = None, rotated_by: Optional[str] = None, reason: Optional[str] = None):
    """Rotate or set a secret. If new_value is None, generate a random token."""
    db = None
    try:
        from database import SessionLocal
        db = SessionLocal()
        store = SecretStore(db)
        if new_value is None:
            new_value = _secrets.token_urlsafe(32)
        return store.rotate_secret(name, new_value, rotated_by=rotated_by, reason=reason)
    finally:
        if db:
            try:
                db.close()
            except Exception:
                pass
