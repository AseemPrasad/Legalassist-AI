"""
Simple sharded vector store implementation.
- Shards are chosen by `shard_for_id(case_id) = case_id % num_shards`.
- Each shard keeps in-memory arrays and a metadata dict mapping ids to indices.
- Provides `add_batch` for batched ingestion and `search` for nearest neighbor search (brute-force on shard).

This is a lightweight implementation suitable for local testing and small scale; can be extended to use FAISS or cloud vector DBs.
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import os
from threading import Lock
import json
import math
import logging

logger = logging.getLogger(__name__)

STORAGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'vector_shards')
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR, exist_ok=True)


class ShardedVectorStore:
    def __init__(self, num_shards: int = 4, dimension: int = 1536):
        self.num_shards = max(1, num_shards)
        self.dimension = dimension
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

    def shard_for_id(self, case_id: int) -> int:
        return int(case_id) % self.num_shards

    def add_batch(self, items: List[Tuple[int, List[float]]]):
        """Add a batch of (case_id, vector) pairs into appropriate shards.
        Overwrites existing vectors for known ids.
        """
        grouped: Dict[int, List[Tuple[int, np.ndarray]]] = {}
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

    def set_metadata(self, case_id: int, metadata: Dict[str, Any]):
        shard = self.shard_for_id(case_id)
        with self._locks[shard]:
            self._shards[shard]['metadatas'][case_id] = metadata
            self._persist_shard(shard)

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

        shards_to_search = shard_ids if shard_ids is not None else list(range(self.num_shards))
        results: List[Tuple[int, float]] = []
        for shard in shards_to_search:
            shard_data = self._shards[shard]
            vectors = shard_data['vectors']
            ids = shard_data['ids']
            if vectors.shape[0] == 0:
                continue
            # compute cosine similarity
            norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(q)
            dots = vectors.dot(q)
            sims = np.zeros_like(dots)
            nonzero = norms != 0
            sims[nonzero] = dots[nonzero] / norms[nonzero]
            # normalize -1..1 to 0..1
            sims = np.clip((sims + 1) / 2, 0.0, 1.0)
            # get top k in this shard
            idxs = np.argsort(-sims)[:top_k]
            for idx in idxs:
                results.append((ids[idx], float(sims[idx])))

        # merge and return top_k overall
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]