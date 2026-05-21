"""
End-to-End Encryption (E2EE) Utilities for LegalAssist AI

Provides client-side and server-side utilities for encrypting documents
using AES-256-GCM with PBKDF2-derived keys. The server never sees plaintext.

Key management:
    - Keys are derived from user passphrase + per-file salt using PBKDF2
    - Master key wrapper stored encrypted in session
    - Individual file keys generated per upload with random salt
    - Decryption happens entirely in the client browser
"""

import base64
import hashlib
import os
import secrets
from dataclasses import dataclass
from typing import Optional

AESGCM_IV_LEN = 12
AESGCM_TAG_LEN = 16
PBKDF2_ITERATIONS = 600_000
KEY_SIZE = 32  # 256-bit


@dataclass
class EncryptedPayload:
    ciphertext_b64: str
    iv_b64: str
    salt_b64: str
    version: int = 1

    def to_dict(self) -> dict:
        return {
            "v": self.version,
            "ct": self.ciphertext_b64,
            "iv": self.iv_b64,
            "s": self.salt_b64,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EncryptedPayload":
        return cls(
            version=d.get("v", 1),
            ciphertext_b64=d["ct"],
            iv_b64=d["iv"],
            salt_b64=d["s"],
        )

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict())


def derive_key(passphrase: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
        dklen=KEY_SIZE,
    )


def generate_salt(n: int = 32) -> bytes:
    return secrets.token_bytes(n)


def encrypt_bytes(plaintext: bytes, key: bytes) -> tuple[bytes, bytes]:
    iv = secrets.token_bytes(AESGCM_IV_LEN)
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(iv, plaintext, None)
    return ciphertext, iv


def decrypt_bytes(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(iv, ciphertext, None)


def encrypt_file(plaintext: bytes, passphrase: str) -> EncryptedPayload:
    salt = generate_salt()
    key = derive_key(passphrase, salt)
    ciphertext, iv = encrypt_bytes(plaintext, key)
    return EncryptedPayload(
        ciphertext_b64=base64.b64encode(ciphertext).decode("ascii"),
        iv_b64=base64.b64encode(iv).decode("ascii"),
        salt_b64=base64.b64encode(salt).decode("ascii"),
    )


def decrypt_file(payload: EncryptedPayload, passphrase: str) -> bytes:
    salt = base64.b64decode(payload.salt_b64)
    key = derive_key(passphrase, salt)
    ciphertext = base64.b64decode(payload.ciphertext_b64)
    iv = base64.b64decode(payload.iv_b64)
    return decrypt_bytes(ciphertext, key, iv)


def encrypt_bytes_to_b64(plaintext: bytes, passphrase: str) -> str:
    return encrypt_file(plaintext, passphrase).to_json()


def decrypt_bytes_from_b64(encrypted_json: str, passphrase: str) -> bytes:
    import json
    d = json.loads(encrypted_json)
    payload = EncryptedPayload.from_dict(d)
    return decrypt_file(payload, passphrase)


def generate_file_key() -> str:
    return base64.b64encode(secrets.token_bytes(KEY_SIZE)).decode("ascii")


def wrap_file_key(file_key: str, master_key: str) -> str:
    salt = generate_salt(16)
    key = derive_key(master_key, salt)
    wrapped, iv = encrypt_bytes(file_key.encode("utf-8"), key)
    return base64.b64encode(salt + iv + wrapped).decode("ascii")


def unwrap_file_key(wrapped: str, master_key: str) -> str:
    data = base64.b64decode(wrapped)
    salt, iv, ciphertext = data[:16], data[16:16 + AESGCM_IV_LEN], data[16 + AESGCM_IV_LEN:]
    key = derive_key(master_key, salt)
    return decrypt_bytes(ciphertext, key, iv).decode("utf-8")