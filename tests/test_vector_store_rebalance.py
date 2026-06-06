import os
import threading
import importlib
import sys
import types

import numpy as np

import core.vector_store as vector_store_module
from core.vector_store import ShardedVectorStore


def test_rebalance_compacts_and_preserves_search(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store_module, "STORAGE_DIR", str(tmp_path))

    store = ShardedVectorStore(num_shards=2, dimension=4)
    store.add_batch(
        [
            (1, [1.0, 0.0, 0.0, 0.0]),
            (2, [0.0, 1.0, 0.0, 0.0]),
            (3, [0.0, 0.0, 1.0, 0.0]),
            (4, [0.0, 0.0, 0.0, 1.0]),
        ]
    )
    store.set_metadata(3, {"excerpt": "case three"})

    # Simulate a duplicated/corrupted entry that compaction should remove.
    shard = store.shard_for_id(3)
    with store._locks[shard]:
        store._shards[shard]["ids"].append(3)
        store._shards[shard]["vectors"] = np.vstack(
            [store._shards[shard]["vectors"], np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)]
        )

    result = store.rebalance_shards(num_shards=3, compact=True)

    assert result["previous_shards"] == 2
    assert result["current_shards"] == 3
    assert result["total_vectors"] == 4

    for shard_idx in range(3):
        assert os.path.exists(tmp_path / f"shard_{shard_idx}.npz")
        assert os.path.exists(tmp_path / f"shard_{shard_idx}_meta.json")

    # Search should continue to work after the atomic swap.
    results = store.search([0.0, 0.0, 1.0, 0.0], top_k=1)
    assert results[0][0] == 3

    # Metadata should survive the rebalance.
    assert store._shards[store.shard_for_id(3)]["metadatas"][3]["excerpt"] == "case three"

    # No duplicate IDs remain after compaction.
    all_ids = []
    for shard_idx in range(store.num_shards):
        all_ids.extend(store._shards[shard_idx]["ids"])
    assert all_ids.count(3) == 1


def test_search_during_rebalance_does_not_error(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store_module, "STORAGE_DIR", str(tmp_path))

    store = ShardedVectorStore(num_shards=2, dimension=4)
    store.add_batch([(10, [1.0, 0.0, 0.0, 0.0]), (11, [0.0, 1.0, 0.0, 0.0])])

    errors = []

    def search_loop():
        try:
            for _ in range(200):
                store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        except Exception as exc:  # pragma: no cover - we want the assertion below to surface this
            errors.append(exc)

    thread = threading.Thread(target=search_loop)
    thread.start()
    store.rebalance_shards(num_shards=4, compact=True)
    thread.join()

    assert not errors


def test_search_fans_out_across_shards_in_parallel(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store_module, "STORAGE_DIR", str(tmp_path))

    store = ShardedVectorStore(num_shards=4, dimension=4)
    store.add_batch(
        [
            (1, [1.0, 0.0, 0.0, 0.0]),
            (2, [0.0, 1.0, 0.0, 0.0]),
            (3, [0.0, 0.0, 1.0, 0.0]),
            (4, [0.0, 0.0, 0.0, 1.0]),
        ]
    )

    barrier = threading.Barrier(4)
    original_search = store._search_single_shard

    def wrapped_search_single_shard(*args, **kwargs):
        barrier.wait(timeout=1)
        return original_search(*args, **kwargs)

    monkeypatch.setattr(store, "_search_single_shard", wrapped_search_single_shard)

    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)

    assert results[0][0] == 1


def test_get_vector_store_reinitializes_when_shape_changes(monkeypatch):
    fake_database = types.ModuleType("database")
    fake_database.CaseEmbedding = object
    fake_database.CaseDocument = object
    fake_database.Case = object
    fake_database.SessionLocal = object
    monkeypatch.setitem(sys.modules, "database", fake_database)

    embedding_engine_module = importlib.import_module("core.embedding_engine")
    monkeypatch.setattr(embedding_engine_module, "_vector_store", None)

    first = embedding_engine_module.get_vector_store(num_shards=2, dimension=4)
    second = embedding_engine_module.get_vector_store(num_shards=3, dimension=4)
    third = embedding_engine_module.get_vector_store(num_shards=3, dimension=8)

    assert first is not second
    assert second is not third
    assert second.num_shards == 3
    assert third.dimension == 8
