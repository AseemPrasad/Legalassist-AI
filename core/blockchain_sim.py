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


class MerkleNode:
    def __init__(self, val: str, left: Optional[MerkleNode] = None, right: Optional[MerkleNode] = None):
        self.val = val
        self.left = left
        self.right = right


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

    def _build_merkle_tree(self) -> Optional[MerkleNode]:
        hashes = [b["data_hash"] for b in self.ledger if "data_hash" in b]
        if not hashes:
            return None
        nodes = [MerkleNode(h) for h in hashes]
        while len(nodes) > 1:
            next_level = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                if i + 1 < len(nodes):
                    right = nodes[i + 1]
                    parent_hash = hashlib.sha256((left.val + right.val).encode("utf-8")).hexdigest()
                    next_level.append(MerkleNode(parent_hash, left, right))
                else:
                    next_level.append(left)
            nodes = next_level
        return nodes[0]

    def get_merkle_root(self) -> str:
        """Return the Merkle root hash for all data_hashes in the ledger."""
        root = self._build_merkle_tree()
        return root.val if root else "0" * 64

    def get_merkle_proof(self, data_hash: str) -> List[Dict[str, Any]]:
        """Generate a Merkle path proof for a given data_hash."""
        root = self._build_merkle_tree()
        if not root:
            return []
        proof = []
        def _find_path(node: Optional[MerkleNode], target_val: str) -> bool:
            if not node:
                return False
            if not node.left and not node.right:
                return node.val == target_val
            if _find_path(node.left, target_val):
                if node.right:
                    proof.append({"sibling": node.right.val, "is_left": False})
                return True
            if _find_path(node.right, target_val):
                if node.left:
                    proof.append({"sibling": node.left.val, "is_left": True})
                return True
            return False

        _find_path(root, data_hash)
        return proof

    @staticmethod
    def verify_merkle_proof(target: str, proof: List[Dict[str, Any]], root_hash: str) -> bool:
        """Verify that a target data_hash resides in the Merkle Tree with the given root_hash."""
        current = target
        for step in proof:
            sibling = step["sibling"]
            if step["is_left"]:
                current = hashlib.sha256((sibling + current).encode("utf-8")).hexdigest()
            else:
                current = hashlib.sha256((current + sibling).encode("utf-8")).hexdigest()
        return current == root_hash
