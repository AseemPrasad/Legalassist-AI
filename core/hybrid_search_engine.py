"""
Cross-Jurisdictional Hybrid Search Engine
==========================================
Combines semantic vector similarity (from :class:`~core.vector_store.ShardedVectorStore`)
with lexical BM25 scoring across distinct regional vector database shards.

Results are fused using **Reciprocal Rank Fusion (RRF)** before being passed
to the cross-encoder re-ranker for final high-relevance scoring.

Key Components
--------------
* ``JurisdictionShardRouter``   — maps jurisdiction metadata to specific shard IDs
* ``BM25Scorer``                — pure-Python BM25 implementation; no external deps
* ``ReciprocalRankFusion``      — standard RRF( k=60 ) combiner
* ``HybridSearchEngine``        — orchestrator: shard-route → semantic + BM25 → RRF → re-rank

Design Decisions
----------------
* BM25 operates on excerpt text stored in shard metadata (``meta["excerpt"]``).
  No separate inverted index is required for this scale; at larger volumes this
  should be replaced with Elasticsearch / Solr BM25.
* The shard router uses ``metadata["jurisdiction"]`` stored per chunk to select
  the correct regional shard.  If metadata is absent all shards are queried
  (graceful degradation).
* RRF is O(N log N) and purely in-memory — suitable for top-K of a few thousand
  candidates before the cross-encoder narrows to top-20.
"""
from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.vector_store import ShardedVectorStore, STORAGE_DIR

logger = logging.getLogger(__name__)


# ============================================================================
# Data types
# ============================================================================

@dataclass
class SearchChunk:
    """A single retrieved text chunk with its source metadata and scores."""
    chunk_id: str              # "<shard_id>:<case_id>:<chunk_index>"
    case_id: int
    shard_id: int
    text: str
    jurisdiction: str = ""
    court_level: str = ""      # e.g. "High Court" | "District Court" | "Supreme Court"
    source_hash: str = ""
    chunk_index: int = 0
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def final_score(self) -> float:
        """Best available score: rerank → rrf → semantic."""
        if self.rerank_score > 0:
            return self.rerank_score
        if self.rrf_score > 0:
            return self.rrf_score
        return self.semantic_score


@dataclass
class HybridSearchResult:
    """Top-level result returned to the API layer."""
    query: str
    jurisdiction_filter: Optional[str]
    court_level_filter: Optional[str]
    chunks: List[SearchChunk]
    total_semantic_candidates: int = 0
    total_bm25_candidates: int = 0
    fusion_strategy: str = "rrf"
    reranked: bool = False


# ============================================================================
# Jurisdiction → Shard routing
# ============================================================================

# Default routing table.  Maps lower-cased jurisdiction/court keywords to shard IDs.
# Operators can override via env ``JURISDICTION_SHARD_MAP`` (JSON).
_DEFAULT_JURISDICTION_SHARD_MAP: Dict[str, List[int]] = {
    "high court": [0, 1],
    "district court": [2, 3],
    "supreme court": [0],
    "tribunal": [2],
    "civil": [0, 2],
    "criminal": [1, 3],
}


def _load_jurisdiction_map() -> Dict[str, List[int]]:
    import json, os
    raw = os.getenv("JURISDICTION_SHARD_MAP", "")
    if raw:
        try:
            loaded = json.loads(raw)
            return {k.lower(): v for k, v in loaded.items()}
        except Exception:
            logger.warning("JURISDICTION_SHARD_MAP env var is malformed JSON — using defaults")
    return _DEFAULT_JURISDICTION_SHARD_MAP


class JurisdictionShardRouter:
    """
    Routes a query to the appropriate regional shard IDs based on case metadata.

    Parameters
    ----------
    num_shards:
        Total number of shards in the :class:`ShardedVectorStore`.
    """

    def __init__(self, num_shards: int = 4) -> None:
        self._num_shards = num_shards
        self._map = _load_jurisdiction_map()

    def shards_for_query(
        self,
        *,
        jurisdiction: Optional[str] = None,
        court_level: Optional[str] = None,
    ) -> List[int]:
        """
        Return the shard IDs to query given optional filter values.

        Falls back to all shards when no mapping matches.
        """
        if not jurisdiction and not court_level:
            return list(range(self._num_shards))

        candidates: set[int] = set()
        probe_terms = []
        if jurisdiction:
            probe_terms.append(jurisdiction.lower())
        if court_level:
            probe_terms.append(court_level.lower())

        for term in probe_terms:
            for key, shards in self._map.items():
                if key in term or term in key:
                    # Filter to valid shard IDs for this store
                    candidates.update(s for s in shards if s < self._num_shards)

        if not candidates:
            # No match — search all shards
            return list(range(self._num_shards))

        return sorted(candidates)


# ============================================================================
# BM25 scorer
# ============================================================================

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Scorer:
    """
    Okapi BM25 implementation over a fixed corpus of text chunks.

    Parameters
    ----------
    corpus:
        List of ``(chunk_id, text)`` tuples.  The corpus is built once per
        hybrid search call from the metadata stored in the vector shards.
    k1:
        BM25 term-frequency saturation parameter (default 1.5).
    b:
        BM25 length normalisation parameter (default 0.75).
    """

    def __init__(
        self,
        corpus: List[Tuple[str, str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._ids = [c[0] for c in corpus]
        tokenized = [_tokenize(c[1]) for c in corpus]
        self._n = len(tokenized)
        self._avgdl = sum(len(t) for t in tokenized) / max(1, self._n)
        # df: document frequency per term
        self._df: Dict[str, int] = defaultdict(int)
        for tokens in tokenized:
            for term in set(tokens):
                self._df[term] += 1
        # tf per document
        self._tf: List[Dict[str, int]] = []
        for tokens in tokenized:
            freq: Dict[str, int] = defaultdict(int)
            for t in tokens:
                freq[t] += 1
            self._tf.append(dict(freq))
        self._len = [len(t) for t in tokenized]

    def score(self, query: str) -> List[Tuple[str, float]]:
        """
        Score all corpus documents against *query*.

        Returns
        -------
        list of ``(chunk_id, bm25_score)`` sorted descending.
        """
        query_terms = _tokenize(query)
        scores = []
        for i, (chunk_id, tf_dict, dl) in enumerate(
            zip(self._ids, self._tf, self._len)
        ):
            s = 0.0
            for term in query_terms:
                if term not in tf_dict:
                    continue
                tf = tf_dict[term]
                df = self._df.get(term, 0)
                idf = math.log(
                    (self._n - df + 0.5) / (df + 0.5) + 1
                )
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * dl / self._avgdl
                )
                s += idf * (numerator / denominator)
            scores.append((chunk_id, s))
        scores.sort(key=lambda x: -x[1])
        return scores


# ============================================================================
# Reciprocal Rank Fusion
# ============================================================================

class ReciprocalRankFusion:
    """
    Standard RRF combiner.

    RRF score for document d = sum over ranked lists r of  1 / (k + rank_r(d))

    Parameters
    ----------
    k:
        Ranking constant (default 60, as in the original Cormack et al. 2009 paper).
    """

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(
        self,
        *ranked_lists: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Fuse *ranked_lists* using RRF.

        Each input is a list of ``(chunk_id, score)`` already sorted descending
        by its own relevance signal.  Output is sorted descending by RRF score.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        for ranked in ranked_lists:
            for rank, (chunk_id, _) in enumerate(ranked, start=1):
                rrf_scores[chunk_id] += 1.0 / (self.k + rank)
        return sorted(rrf_scores.items(), key=lambda x: -x[1])


# ============================================================================
# Shard metadata reader
# ============================================================================

def _read_all_shard_chunks(
    store: ShardedVectorStore,
    shard_ids: List[int],
) -> List[SearchChunk]:
    """
    Extract all :class:`SearchChunk` objects from the specified shards.

    The vector store's metadata dict is expected to have keys like::

        {
            "excerpt": "...",
            "jurisdiction": "High Court",
            "court_level": "High Court",
            "source_hash": "abc123",
            "chunk_index": 0,
        }
    """
    chunks: List[SearchChunk] = []
    for shard_id in shard_ids:
        if shard_id >= store.num_shards:
            continue
        with store._locks[shard_id]:
            shard_data = store._shards[shard_id]
            ids = list(shard_data["ids"])
            metadatas = dict(shard_data.get("metadatas", {}))

        for case_id in ids:
            meta = metadatas.get(case_id, {})
            if isinstance(meta, dict):
                # Each case_id can have multiple chunks stored in metadata.
                # Format A: single chunk in top-level dict
                if "excerpt" in meta:
                    chunk_index = int(meta.get("chunk_index", 0))
                    chunk_id = f"{shard_id}:{case_id}:{chunk_index}"
                    chunks.append(SearchChunk(
                        chunk_id=chunk_id,
                        case_id=int(case_id),
                        shard_id=shard_id,
                        text=meta.get("excerpt", ""),
                        jurisdiction=str(meta.get("jurisdiction", "")),
                        court_level=str(meta.get("court_level", "")),
                        source_hash=str(meta.get("source_hash", "")),
                        chunk_index=chunk_index,
                        metadata=meta,
                    ))
                # Format B: multiple chunks in a "chunks" sub-list
                elif "chunks" in meta:
                    for sub in meta.get("chunks", []):
                        chunk_index = int(sub.get("chunk_index", 0))
                        chunk_id = f"{shard_id}:{case_id}:{chunk_index}"
                        chunks.append(SearchChunk(
                            chunk_id=chunk_id,
                            case_id=int(case_id),
                            shard_id=shard_id,
                            text=sub.get("excerpt", ""),
                            jurisdiction=str(sub.get("jurisdiction", "")),
                            court_level=str(sub.get("court_level", "")),
                            source_hash=str(sub.get("source_hash", "")),
                            chunk_index=chunk_index,
                            metadata=sub,
                        ))
    return chunks


# ============================================================================
# Hybrid Search Engine
# ============================================================================

class HybridSearchEngine:
    """
    Cross-jurisdictional hybrid retrieval engine.

    Pipeline
    --------
    1. **Shard routing** — :class:`JurisdictionShardRouter` selects shards
       based on *jurisdiction* / *court_level* metadata filters.
    2. **Semantic search** — brute-force cosine similarity against the selected
       shards via :meth:`ShardedVectorStore.search`.
    3. **BM25 search** — lexical scoring over the same shards' excerpt texts.
    4. **RRF fusion** — :class:`ReciprocalRankFusion` merges the two ranked
       lists into a single unified ranking.
    5. **Cross-encoder re-ranking** — :class:`~core.cross_encoder_reranker.CrossEncoderReranker`
       scores the top-20 RRF candidates and keeps only high-relevance results.
    """

    def __init__(
        self,
        store: ShardedVectorStore,
        *,
        rrf_k: int = 60,
        semantic_top_k: int = 50,
        bm25_top_k: int = 50,
        rerank_top_n: int = 20,
        rerank_threshold: float = 0.0,
        embedder=None,
    ) -> None:
        self._store = store
        self._router = JurisdictionShardRouter(num_shards=store.num_shards)
        self._rrf = ReciprocalRankFusion(k=rrf_k)
        self._semantic_top_k = semantic_top_k
        self._bm25_top_k = bm25_top_k
        self._rerank_top_n = rerank_top_n
        self._rerank_threshold = rerank_threshold
        self._embedder = embedder

        # Lazily instantiate the cross-encoder re-ranker
        self._reranker = None

    def _get_reranker(self):
        if self._reranker is None:
            try:
                from core.cross_encoder_reranker import CrossEncoderReranker
                self._reranker = CrossEncoderReranker()
            except Exception as e:
                logger.warning("CrossEncoderReranker unavailable: %s", e)
                self._reranker = None
        return self._reranker

    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        """Embed a query string using the attached embedder."""
        if self._embedder is None:
            return None
        try:
            vec = self._embedder.embed_query(query)
            return np.array(vec, dtype=np.float32)
        except Exception:
            try:
                vec = self._embedder.embed_documents([query])[0]
                return np.array(vec, dtype=np.float32)
            except Exception as e:
                logger.warning("Failed to embed query: %s", e)
                return None

    def search(
        self,
        query: str,
        *,
        query_vector: Optional[List[float]] = None,
        jurisdiction: Optional[str] = None,
        court_level: Optional[str] = None,
        top_k: int = 10,
        enable_reranking: bool = True,
    ) -> HybridSearchResult:
        """
        Execute a hybrid search.

        Parameters
        ----------
        query:
            Free-text search query.
        query_vector:
            Pre-computed query embedding.  If ``None``, the engine will try to
            embed *query* using the attached embedder.
        jurisdiction:
            Optional jurisdiction filter — routes the query to matching shards.
        court_level:
            Optional court-level filter (e.g. ``"High Court"``).
        top_k:
            Final number of results to return (after re-ranking).
        enable_reranking:
            Set to ``False`` to skip the cross-encoder step.

        Returns
        -------
        :class:`HybridSearchResult`
        """
        # 1. Shard routing
        target_shards = self._router.shards_for_query(
            jurisdiction=jurisdiction,
            court_level=court_level,
        )
        logger.debug(
            "hybrid_search_shard_routing",
            query=query[:80],
            shards=target_shards,
            jurisdiction=jurisdiction,
            court_level=court_level,
        )

        # 2. Build chunk corpus from selected shards
        all_chunks = _read_all_shard_chunks(self._store, target_shards)
        chunk_by_id: Dict[str, SearchChunk] = {c.chunk_id: c for c in all_chunks}

        if not all_chunks:
            return HybridSearchResult(
                query=query,
                jurisdiction_filter=jurisdiction,
                court_level_filter=court_level,
                chunks=[],
            )

        # 3. Semantic search
        semantic_ranked: List[Tuple[str, float]] = []
        qvec = query_vector or (
            self._embed_query(query) if self._embedder else None
        )
        if qvec is not None:
            raw = self._store.search(
                qvec, top_k=self._semantic_top_k, shard_ids=target_shards
            )
            # Map (case_id, score) → chunk_ids (one case may have many chunks)
            case_score: Dict[int, float] = {int(cid): sc for cid, sc in raw}
            for c in all_chunks:
                if c.case_id in case_score:
                    c.semantic_score = case_score[c.case_id]
            # Build semantic ranking per chunk (use case score for all chunks of same case)
            semantic_ranked = [
                (c.chunk_id, c.semantic_score)
                for c in all_chunks
                if c.semantic_score > 0
            ]
            semantic_ranked.sort(key=lambda x: -x[1])
            semantic_ranked = semantic_ranked[: self._semantic_top_k]

        # 4. BM25 search
        corpus = [(c.chunk_id, c.text) for c in all_chunks]
        bm25 = BM25Scorer(corpus)
        bm25_raw = bm25.score(query)
        # Normalise BM25 scores to [0, 1]
        max_bm25 = bm25_raw[0][1] if bm25_raw else 1.0
        if max_bm25 == 0:
            max_bm25 = 1.0
        for chunk_id, score in bm25_raw:
            if chunk_id in chunk_by_id:
                chunk_by_id[chunk_id].bm25_score = score / max_bm25
        bm25_ranked = [(cid, s) for cid, s in bm25_raw if s > 0]
        bm25_ranked = bm25_ranked[: self._bm25_top_k]

        # 5. RRF fusion
        lists_to_fuse = []
        if semantic_ranked:
            lists_to_fuse.append(semantic_ranked)
        if bm25_ranked:
            lists_to_fuse.append(bm25_ranked)

        if not lists_to_fuse:
            # No signal at all — return top-k by semantic_score
            all_chunks.sort(key=lambda c: -c.semantic_score)
            return HybridSearchResult(
                query=query,
                jurisdiction_filter=jurisdiction,
                court_level_filter=court_level,
                chunks=all_chunks[:top_k],
                total_semantic_candidates=0,
                total_bm25_candidates=0,
            )

        fused = self._rrf.fuse(*lists_to_fuse)
        for chunk_id, rrf_score in fused:
            if chunk_id in chunk_by_id:
                chunk_by_id[chunk_id].rrf_score = rrf_score

        # Top-N candidates for re-ranking
        top_candidates = [
            chunk_by_id[cid]
            for cid, _ in fused[: self._rerank_top_n]
            if cid in chunk_by_id
        ]

        # 6. Cross-encoder re-ranking
        reranked = False
        reranker = self._get_reranker() if enable_reranking else None
        if reranker and top_candidates:
            try:
                scored = reranker.rerank(query, top_candidates)
                for chunk, score in scored:
                    chunk.rerank_score = float(score)
                # Filter by threshold and keep high-relevance context
                top_candidates = [
                    c for c, _ in scored
                    if c.rerank_score >= self._rerank_threshold
                ]
                reranked = True
            except Exception as e:
                logger.warning("Re-ranking failed; falling back to RRF order: %s", e)

        # Sort by final_score and truncate to top_k
        top_candidates.sort(key=lambda c: -c.final_score)
        top_candidates = top_candidates[:top_k]

        return HybridSearchResult(
            query=query,
            jurisdiction_filter=jurisdiction,
            court_level_filter=court_level,
            chunks=top_candidates,
            total_semantic_candidates=len(semantic_ranked),
            total_bm25_candidates=len(bm25_ranked),
            fusion_strategy="rrf",
            reranked=reranked,
        )


# ============================================================================
# Singleton helper
# ============================================================================

_engine: Optional[HybridSearchEngine] = None


def get_hybrid_search_engine(embedder=None) -> HybridSearchEngine:
    """Return the process-level :class:`HybridSearchEngine` singleton."""
    global _engine
    if _engine is None:
        store = ShardedVectorStore(num_shards=4, dimension=1536)
        _engine = HybridSearchEngine(
            store=store,
            embedder=embedder,
        )
    elif embedder is not None and _engine._embedder is None:
        _engine._embedder = embedder
    return _engine
