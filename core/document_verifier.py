"""Document hashing and verification utilities.

Provides a simple interface to compute document hashes and register/verify
them using the `BlockchainSimulator`.
"""
from __future__ import annotations

import hashlib
from typing import Optional, Dict, Any

from core.blockchain_sim import BlockchainSimulator


def compute_document_hash(file_bytes: bytes) -> str:
    """Compute SHA-256 hex digest of document bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def register_document(file_bytes: bytes, chain: Optional[BlockchainSimulator] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Register a document on the given chain (or a new simulator if none).

    Returns the appended block info along with the data_hash.
    """
    if chain is None:
        chain = BlockchainSimulator()
    data_hash = compute_document_hash(file_bytes)
    block = chain.append_data_hash(data_hash, metadata={**(metadata or {}), "filename": (metadata or {}).get("filename")})
    return {"data_hash": data_hash, "block": dict(block)}


def verify_document(file_bytes: bytes, chain: BlockchainSimulator) -> Dict[str, Any]:
    """Verify if a document exists on the provided chain.

    Returns a dict with `found: bool`, optional `block`, and `proof` (list of blocks) when found.
    """
    data_hash = compute_document_hash(file_bytes)
    block = chain.find_by_data_hash(data_hash)
    if not block:
        return {"found": False, "data_hash": data_hash}
    proof = chain.export_proof(data_hash)
    return {"found": True, "data_hash": data_hash, "block": dict(block), "proof": [dict(b) for b in proof]}
