"""Secrets store abstraction with local encrypted fallback and rotation support.

Provides:
- SecretStore: get/set/delete secrets (persists encrypted values in DB)
- LocalKeyManager: manages master encryption key in a file (dev only)
"""
import os
import base64
import logging
from typing import Optional
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.orm import Session
from db.models.secrets import SecretEntry, SecretRotationLog

logger = logging.getLogger(__name__)


class LocalKeyManager:
    """Manage a local master key stored in a file for development environments.

    The path may be overridden with the `SECRETS_MASTER_KEY_PATH` environment
    variable to support test isolation.
    """
    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = os.getenv("SECRETS_MASTER_KEY_PATH", ".secrets_master_key")
        self.path = path

    def get_or_create_key(self) -> bytes:
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                return f.read().strip()
        key = Fernet.generate_key()
        with open(self.path, "wb") as f:
            f.write(key)
        os.chmod(self.path, 0o600)
        return key


class SecretStore:
    def __init__(self, db: Session, master_key: Optional[bytes] = None):
        self.db = db
        if master_key is None:
            mgr = LocalKeyManager()
            master_key = mgr.get_or_create_key()
        self.fernet = Fernet(master_key)

    def set_secret(self, name: str, value: str, rotated_by: Optional[str] = None, reason: Optional[str] = None):
        enc = self.fernet.encrypt(value.encode("utf-8")).decode("utf-8")
        existing = self.db.query(SecretEntry).filter(SecretEntry.name == name).first()
        if existing:
            prev_version = existing.version
            existing.value_enc = enc
            existing.version = existing.version + 1
            existing.rotated_at = None
            self.db.add(existing)
            self.db.commit()
            log = SecretRotationLog(secret_id=existing.id, previous_version=prev_version, new_version=existing.version, rotated_by=rotated_by, reason=reason)
            self.db.add(log)
            self.db.commit()
            return existing
        entry = SecretEntry(name=name, value_enc=enc)
        self.db.add(entry)
        self.db.commit()
        return entry

    def get_secret(self, name: str) -> Optional[str]:
        entry = self.db.query(SecretEntry).filter(SecretEntry.name == name).first()
        if not entry:
            return None
        try:
            val = self.fernet.decrypt(entry.value_enc.encode("utf-8")).decode("utf-8")
            return val
        except InvalidToken:
            logger.exception("Failed to decrypt secret %s - invalid key", name)
            return None

    def rotate_secret(self, name: str, new_value: str, rotated_by: Optional[str] = None, reason: Optional[str] = None):
        return self.set_secret(name, new_value, rotated_by=rotated_by, reason=reason)

    def delete_secret(self, name: str):
        entry = self.db.query(SecretEntry).filter(SecretEntry.name == name).first()
        if entry:
            self.db.delete(entry)
            self.db.commit()
            return True
        return False
