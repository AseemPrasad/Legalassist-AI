"""
Simple sharded vector store implementation.
- Shards are chosen by `shard_for_id(case_id) = case_id % num_shards`.
- Each shard keeps in-memory arrays and a metadata dict mapping ids to indices.
- Provides `add_batch` for batched ingestion and `search` for nearest neighbor search (brute-force on shard).

This is a lightweight implementation suitable for local testing and small scale; can be extended to use FAISS or cloud vector DBs.
"""
from typing import List, Tuple, Dict, Any, Optional
import asyncio
import numpy as np
import os
from threading import Lock, RLock
import json
import math
import logging
import tempfile
from collections import OrderedDict
import copy
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

STORAGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'vector_shards')
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR, exist_ok=True)


class ShardedVectorStore:
    def __init__(self, num_shards: int = 4, dimension: int = 1536):
        self.num_shards = max(1, num_shards)
        self.dimension = dimension
        self._state_lock = RLock()
        # in-memory structures
        self._shards: Dict[int, Dict[str, Any]] = {}
        self._locks: Dict[int, Lock] = {}
        for s in range(self.num_shards):
            self._shards[s] = {
                'ids': [],  # list of case_ids
                'vectors': np.zeros((0, self.dimension), dtype=np.float32),
                'id_to_index': {},
                'metadatas': {},  # id -> metadata
            }
            self._locks[s] = Lock()
        self._load_persisted_shards()

    def _load_persisted_shards(self):
        for shard in range(self.num_shards):
            self.load_shard(shard)

    def shard_for_id(self, case_id: int) -> int:
        return int(case_id) % self.num_shards

    def _normalize_shard_payload(self, shard_data: Dict[str, Any]) -> Dict[str, Any]:
        ids = shard_data.get('ids', [])
        vectors = shard_data.get('vectors', np.zeros((0, self.dimension), dtype=np.float32))
        metadatas = shard_data.get('metadatas', {}) or {}

        deduped: "OrderedDict[int, Tuple[np.ndarray, Any]]" = OrderedDict()
        for idx, cid in enumerate(ids):
            cid_int = int(cid)
            vector = vectors[idx].astype(np.float32, copy=True)
            metadata = metadatas.get(cid_int, metadatas.get(str(cid_int), {}))
            deduped[cid_int] = (vector, metadata)

        if not deduped:
            return {
                'ids': [],
                'vectors': np.zeros((0, self.dimension), dtype=np.float32),
                'id_to_index': {},
                'metadatas': {},
            }

        new_ids = list(deduped.keys())
        new_vectors = np.stack([entry[0] for entry in deduped.values()], axis=0).astype(np.float32, copy=False)
        new_metadatas = {cid: deduped[cid][1] for cid in new_ids}
        return {
            'ids': new_ids,
            'vectors': new_vectors,
            'id_to_index': {cid: i for i, cid in enumerate(new_ids)},
            'metadatas': new_metadatas,
        }

    def _snapshot_store(self) -> Tuple[int, Dict[int, Dict[str, Any]]]:
        with self._state_lock:
            snapshot = {}
            for shard in range(self.num_shards):
                with self._locks[shard]:
                    shard_data = self._shards[shard]
                    snapshot[shard] = {
                        'ids': list(shard_data['ids']),
                        'vectors': np.array(shard_data['vectors'], copy=True),
                        'metadatas': copy.deepcopy(shard_data.get('metadatas', {})),
                    }
            return self.num_shards, snapshot

    def _search_single_shard(
        self,
        shard: int,
        query_vec: np.ndarray,
        top_k: int,
        shard_data: Dict[str, Any],
    ) -> List[Tuple[int, float]]:
        lock = self._locks.get(shard)
        if lock is None:
            return []

        with lock:
            vectors = np.array(shard_data['vectors'], copy=True)
            ids = list(shard_data['ids'])

        if vectors.shape[0] == 0:
            return []

        # 1. Coarse filtering: compute dot products (fast, O(N) operations)
        dots = vectors.dot(query_vec)

        # Select a candidate pool (e.g. 5x top_k, at least 100)
        n_candidates = vectors.shape[0]
        coarse_k = min(n_candidates, max(top_k * 5, 100))

        # Use argpartition to find top coarse_k indices in O(N) time
        candidate_indices = np.argpartition(-dots, coarse_k - 1)[:coarse_k]

        # 2. Fine evaluation: compute exact cosine similarities only on the candidate pool
        candidate_vectors = vectors[candidate_indices]
        candidate_dots = dots[candidate_indices]

        query_norm = np.linalg.norm(query_vec)
        norms = np.linalg.norm(candidate_vectors, axis=1) * query_norm

        sims = np.zeros_like(candidate_dots)
        nonzero = norms != 0
        sims[nonzero] = candidate_dots[nonzero] / norms[nonzero]
        sims = np.clip((sims + 1) / 2, 0.0, 1.0)

        # Sort the candidate pool
        shard_top_k = min(top_k, len(candidate_indices))
        sorted_candidates_idx = np.argsort(-sims)[:shard_top_k]

        return [(int(ids[candidate_indices[idx]]), float(sims[idx])) for idx in sorted_candidates_idx]

    def _build_rebalanced_state(self, snapshot: Dict[int, Dict[str, Any]], target_num_shards: int) -> Dict[int, Dict[str, Any]]:
        target_num_shards = max(1, int(target_num_shards))
        buckets: Dict[int, Dict[str, Any]] = {}
        for shard in range(target_num_shards):
            buckets[shard] = {
                'ids': [],
                'vectors': [],
                'metadatas': {},
            }

        all_entries: Dict[int, Tuple[np.ndarray, Any]] = {}
        for shard in sorted(snapshot.keys()):
            shard_data = self._normalize_shard_payload(snapshot[shard])
            for idx, cid in enumerate(shard_data['ids']):
                all_entries[int(cid)] = (
                    shard_data['vectors'][idx].astype(np.float32, copy=True),
                    shard_data['metadatas'].get(int(cid), {}),
                )

        for cid, (vector, metadata) in all_entries.items():
            target_shard = int(cid) % target_num_shards
            bucket = buckets[target_shard]
            bucket['ids'].append(int(cid))
            bucket['vectors'].append(vector)
            bucket['metadatas'][int(cid)] = metadata

        result: Dict[int, Dict[str, Any]] = {}
        for shard, bucket in buckets.items():
            vectors = np.stack(bucket['vectors'], axis=0).astype(np.float32, copy=False) if bucket['vectors'] else np.zeros((0, self.dimension), dtype=np.float32)
            ids = bucket['ids']
            result[shard] = {
                'ids': ids,
                'vectors': vectors,
                'id_to_index': {cid: i for i, cid in enumerate(ids)},
                'metadatas': bucket['metadatas'],
            }
        return result

    def _write_rebalanced_state(self, new_state: Dict[int, Dict[str, Any]], target_num_shards: int):
        os.makedirs(STORAGE_DIR, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix='vector_rebalance_', dir=STORAGE_DIR)
        try:
            for shard in range(target_num_shards):
                shard_data = new_state[shard]
                shard_path = os.path.join(tmp_dir, f"shard_{shard}.npz")
                meta_path = os.path.join(tmp_dir, f"shard_{shard}_meta.json")
                np.savez_compressed(
                    shard_path,
                    vectors=shard_data['vectors'],
                    ids=np.array(shard_data['ids'], dtype=np.int64),
                )
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump({str(k): v for k, v in shard_data.get('metadatas', {}).items()}, f)

            for shard in range(target_num_shards):
                temp_npz = os.path.join(tmp_dir, f"shard_{shard}.npz")
                temp_meta = os.path.join(tmp_dir, f"shard_{shard}_meta.json")
                final_npz = os.path.join(STORAGE_DIR, f"shard_{shard}.npz")
                final_meta = os.path.join(STORAGE_DIR, f"shard_{shard}_meta.json")
                os.replace(temp_npz, final_npz)
                os.replace(temp_meta, final_meta)

            for name in os.listdir(STORAGE_DIR):
                if not name.startswith('shard_'):
                    continue
                if name.endswith('.npz'):
                    base = name[len('shard_'):-len('.npz')]
                elif name.endswith('_meta.json'):
                    base = name[len('shard_'):-len('_meta.json')]
                else:
                    continue
                try:
                    shard_idx = int(base)
                except ValueError:
                    continue
                if shard_idx >= target_num_shards:
                    try:
                        os.remove(os.path.join(STORAGE_DIR, name))
                    except FileNotFoundError:
                        pass
        finally:
            try:
                for entry in os.listdir(tmp_dir):
                    try:
                        os.remove(os.path.join(tmp_dir, entry))
                    except Exception:
                        pass
                os.rmdir(tmp_dir)
            except Exception:
                pass

    def add_batch(self, items: List[Tuple[int, List[float]]]):
        """Add a batch of (case_id, vector) pairs into appropriate shards.
        Overwrites existing vectors for known ids.
        """
        grouped: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        with self._state_lock:
            for case_id, vec in items:
                shard = self.shard_for_id(case_id)
                arr = np.array(vec, dtype=np.float32)
                if arr.shape[0] != self.dimension:
                    raise ValueError(f"Vector dimension mismatch for case {case_id}: expected {self.dimension}, got {arr.shape[0]}")
                grouped.setdefault(shard, []).append((case_id, arr))

            # Insert per shard while holding lock
            for shard, entries in grouped.items():
                lock = self._locks[shard]
                with lock:
                    shard_data = self._shards[shard]
                    ids = shard_data['ids']
                    vectors = shard_data['vectors']
                    id_to_index = shard_data['id_to_index']

                    # For simplicity, append new vectors and update mappings
                    new_vectors = np.stack([e[1] for e in entries], axis=0)
                    new_ids = [e[0] for e in entries]

                    # Update existing ids: if id exists, replace vector in place
                    for i, cid in enumerate(new_ids):
                        if cid in id_to_index:
                            idx = id_to_index[cid]
                            vectors[idx] = new_vectors[i]
                        else:
                            # append
                            id_to_index[cid] = vectors.shape[0]
                            vectors = np.vstack([vectors, new_vectors[i:i+1]])
                            ids.append(cid)
                            # store metadata placeholder if not present
                            if cid not in shard_data['metadatas']:
                                shard_data['metadatas'][cid] = {}

                    shard_data['vectors'] = vectors
                    shard_data['ids'] = ids
                    shard_data['id_to_index'] = id_to_index
                    shard_data['metadatas'] = shard_data.get('metadatas', {})

                    # Persist shard metadata to disk for durability
                    self._persist_shard(shard)

    def _persist_shard(self, shard: int):
        shard_path = os.path.join(STORAGE_DIR, f"shard_{shard}.npz")
        shard_data = self._shards[shard]
        try:
            np.savez_compressed(shard_path, vectors=shard_data['vectors'], ids=np.array(shard_data['ids'], dtype=np.int64))
            # persist metadata separately
            meta_path = os.path.join(STORAGE_DIR, f"shard_{shard}_meta.json")
            try:
                import json
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump({str(k): v for k, v in shard_data.get('metadatas', {}).items()}, f)
            except Exception:
                pass
        except Exception as e:
            logger.exception("Failed to persist shard %s: %s", shard, e)

    def load_shard(self, shard: int):
        shard_path = os.path.join(STORAGE_DIR, f"shard_{shard}.npz")
        if not os.path.exists(shard_path):
            return
        with self._locks[shard]:
            data = np.load(shard_path)
            vectors = data['vectors']
            ids = data['ids'].tolist()
            id_to_index = {int(cid): i for i, cid in enumerate(ids)}
            # load metadata if present
            meta_path = os.path.join(STORAGE_DIR, f"shard_{shard}_meta.json")
            metadatas = {}
            if os.path.exists(meta_path):
                try:
                    import json
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        raw = json.load(f)
                        metadatas = {int(k): v for k, v in raw.items()}
                except Exception:
                    metadatas = {}

            self._shards[shard]['vectors'] = vectors
            self._shards[shard]['ids'] = ids
            self._shards[shard]['id_to_index'] = id_to_index
            self._shards[shard]['metadatas'] = metadatas

    def rebalance_shards(self, num_shards: Optional[int] = None, compact: bool = True) -> Dict[str, Any]:
        """Rebuild shard files into a new layout without stopping reads."""
        target_num_shards = max(1, int(num_shards or self.num_shards))
        with self._state_lock:
            current_num_shards, snapshot = self._snapshot_store()
            target_state = self._build_rebalanced_state(snapshot, target_num_shards)

            if compact:
                target_state = {
                    shard: self._normalize_shard_payload(shard_data)
                    for shard, shard_data in target_state.items()
                }

            self._write_rebalanced_state(target_state, target_num_shards)

            self.num_shards = target_num_shards
            self._shards = target_state
            self._locks = {shard: Lock() for shard in range(target_num_shards)}

            return {
                'previous_shards': current_num_shards,
                'current_shards': target_num_shards,
                'compacted': compact,
                'total_vectors': sum(len(state['ids']) for state in target_state.values()),
            }

    def compact_shards(self) -> Dict[str, Any]:
        """Compact the current shard layout without changing shard count."""
        return self.rebalance_shards(num_shards=self.num_shards, compact=True)

    def set_metadata(self, case_id: int, metadata: Dict[str, Any]):
        shard = self.shard_for_id(case_id)
        with self._state_lock:
            with self._locks[shard]:
                self._shards[shard]['metadatas'][case_id] = metadata
                self._persist_shard(shard)

    def delete(self, case_id: int) -> bool:
        """Remove a case and its vector/metadata from its corresponding shard."""
        shard = self.shard_for_id(case_id)
        with self._state_lock:
            lock = self._locks[shard]
            with lock:
                shard_data = self._shards[shard]
                ids = shard_data['ids']
                id_to_index = shard_data['id_to_index']
                if case_id in id_to_index:
                    idx = id_to_index[case_id]
                    # Delete vector and id
                    vectors = shard_data['vectors']
                    vectors = np.delete(vectors, idx, axis=0)
                    ids.pop(idx)
                    # Rebuild id_to_index
                    id_to_index = {int(cid): i for i, cid in enumerate(ids)}
                    # Remove metadata
                    shard_data['metadatas'].pop(case_id, None)
                    
                    shard_data['vectors'] = vectors
                    shard_data['ids'] = ids
                    shard_data['id_to_index'] = id_to_index
                    self._persist_shard(shard)
                    return True
        return False

    def similarity_search_with_score(self, query: str, k: int = 10):
        """Search by query string using an attached embedder if available.

        Returns list of (DocumentLike, score) where DocumentLike has `page_content` and `metadata`.
        """
        if not hasattr(self, 'embedder') or self.embedder is None:
            raise RuntimeError('No embedder attached to vector store for similarity_search_with_score')
        qvec = None
        try:
            qvec = self.embedder.embed_query(query)
        except Exception:
            try:
                qvec = self.embedder.embed_documents([query])[0]
            except Exception as e:
                raise

        results = self.search(qvec, top_k=k)
        # convert to Document-like objects
        out = []
        for cid, score in results:
            meta = self._shards[self.shard_for_id(cid)]['metadatas'].get(cid, {})
            class Doc:
                def __init__(self, content, metadata):
                    self.page_content = content
                    self.metadata = metadata
            content = meta.get('excerpt', '')
            out.append((Doc(content, meta), score))
        return out

    def search(self, query_vec: List[float], top_k: int = 10, shard_ids: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """Search nearest neighbors across selected shards (or all shards).
        Returns list of (case_id, score) sorted by descending similarity.
        """
        q = np.array(query_vec, dtype=np.float32)
        if q.shape[0] != self.dimension:
            raise ValueError("Query vector dimension mismatch")

        with self._state_lock:
            current_num_shards = self.num_shards
            shard_snapshot = {shard: self._shards[shard] for shard in range(current_num_shards)}

        shards_to_search = [shard for shard in (shard_ids if shard_ids is not None else list(range(current_num_shards))) if shard in shard_snapshot]
        if not shards_to_search:
            return []

        max_workers = min(len(shards_to_search), max(1, os.cpu_count() or 4))
        shard_results: List[List[Tuple[int, float]]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._search_single_shard, shard, q, top_k, shard_snapshot[shard]) for shard in shards_to_search]
            for future in futures:
                shard_results.append(future.result())

        results: List[Tuple[int, float]] = [item for batch in shard_results for item in batch]
        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:top_k]

    async def search_async(self, query_vec: List[float], top_k: int = 10, shard_ids: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """Async retrieval that fans shard work out concurrently."""
        q = np.array(query_vec, dtype=np.float32)
        if q.shape[0] != self.dimension:
            raise ValueError("Query vector dimension mismatch")

        with self._state_lock:
            current_num_shards = self.num_shards
            shard_snapshot = {shard: self._shards[shard] for shard in range(current_num_shards)}

        shards_to_search = [shard for shard in (shard_ids if shard_ids is not None else list(range(current_num_shards))) if shard in shard_snapshot]
        if not shards_to_search:
            return []

        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(None, self._search_single_shard, shard, q, top_k, shard_snapshot[shard]) for shard in shards_to_search]
        shard_results = await asyncio.gather(*tasks)
        results: List[Tuple[int, float]] = [item for batch in shard_results for item in batch]
        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:top_k]