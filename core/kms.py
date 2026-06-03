"""KMS abstraction and a simple local file-backed KMS for development.

Implements a minimal AES-GCM based wrapper that acts as a root key provider.
This is NOT a production KMS; in production plug in a cloud KMS (AWS KMS, GCP KMS,
HashiCorp Vault) via the `KMSProvider` interface.
"""
import os
import base64
from abc import ABC, abstractmethod
from typing import Optional


class KMSProvider(ABC):
    @abstractmethod
    def wrap(self, plaintext: bytes) -> str:
        pass

    @abstractmethod
    def unwrap(self, ciphertext_b64: str) -> bytes:
        pass


class LocalFileKMS(KMSProvider):
    """A simple local KMS that stores a single root key in a file (dev only).

    The root key is a 32-byte value stored base64-encoded in `path`. If the file
    doesn't exist, it will be created with a random key.
    """

    def __init__(self, path: Optional[str] = None):
        import secrets
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        self.path = path or os.environ.get("E2EE_ROOT_KEY_FILE", ".e2ee_root_key")
        if not os.path.exists(self.path):
            key = secrets.token_bytes(32)
            with open(self.path, "wb") as f:
                f.write(base64.b64encode(key))
        raw = base64.b64decode(open(self.path, "rb").read())
        if len(raw) != 32:
            raise ValueError("Root key file must contain 32 raw bytes base64-encoded")
        self._root = raw
        self._aesgcm = AESGCM(self._root)

    def _iv(self):
        import secrets
        return secrets.token_bytes(12)

    def wrap(self, plaintext: bytes) -> str:
        iv = self._iv()
        ct = self._aesgcm.encrypt(iv, plaintext, None)
        return base64.b64encode(iv + ct).decode("ascii")

    def unwrap(self, ciphertext_b64: str) -> bytes:
        data = base64.b64decode(ciphertext_b64)
        iv, ct = data[:12], data[12:]
        return self._aesgcm.decrypt(iv, ct, None)
