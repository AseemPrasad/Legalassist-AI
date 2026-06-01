"""A minimal, append-only blockchain simulator for storing document hashes.

This is NOT a real blockchain. It's a deterministic, append-only ledger
useful for proof-of-concept testing and local verification.
When a *redis_client* is provided the ledger is persisted in Redis so that
all workers in a multi-process deployment share the same chain.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import List, Optional, Dict, Any


class Block(dict):
    pass


class BlockchainSimulator:
    def __init__(self, redis_client=None, redis_key: str = "blockchain:default"):
        self.redis_client = redis_client
        self.redis_key = redis_key
        self.ledger: List[Block] = []

        if redis_client:
            self._load_from_redis()

        if not self.ledger:
            self._append_block(data="genesis")

    def _load_from_redis(self):
        raw = self.redis_client.get(self.redis_key)
        if raw:
            try:
                self.ledger = [Block(b) for b in json.loads(raw)]
            except (json.JSONDecodeError, TypeError):
                self.ledger = []

    def _flush_to_redis(self):
        if self.redis_client:
            self.redis_client.set(self.redis_key, json.dumps(self.ledger, default=str))

    def _compute_block_hash(self, index: int, prev_hash: str, timestamp: float, data_hash: str) -> str:
        m = hashlib.sha256()
        m.update(f"{index}|{prev_hash}|{timestamp}|{data_hash}".encode("utf-8"))
        return m.hexdigest()

    def _append_block(self, data: Any, data_hash: Optional[str] = None) -> Block:
        index = len(self.ledger)
        prev_hash = self.ledger[-1]["block_hash"] if self.ledger else "0" * 64
        timestamp = time.time()
        dh = data_hash or (hashlib.sha256(str(data).encode("utf-8")).hexdigest())
        block_hash = self._compute_block_hash(index, prev_hash, timestamp, dh)
        block: Block = Block({
            "index": index,
            "prev_hash": prev_hash,
            "data_hash": dh,
            "block_hash": block_hash,
            "timestamp": timestamp,
            "data": data,
        })
        self.ledger.append(block)
        self._flush_to_redis()
        return block

    def append_data_hash(self, data_hash: str, metadata: Optional[Dict[str, Any]] = None) -> Block:
        """Append a new block containing the provided data_hash.

        Returns the appended block.
        """
        return self._append_block(data=metadata or {}, data_hash=data_hash)

    def find_by_data_hash(self, data_hash: str) -> Optional[Block]:
        for block in self.ledger:
            if block.get("data_hash") == data_hash:
                return block
        return None

    def export_proof(self, data_hash: str) -> Optional[List[Block]]:
        """Return the chain from the matching block to genesis as a simple proof."""
        block = self.find_by_data_hash(data_hash)
        if not block:
            return None
        idx = block["index"]
        return self.ledger[: idx + 1]
