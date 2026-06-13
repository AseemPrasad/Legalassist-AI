"""
Hybrid Search API Routes
========================
FastAPI endpoints exposing the :class:`~core.hybrid_search_engine.HybridSearchEngine`:

  ``POST /api/v1/search/hybrid``
      Cross-jurisdictional hybrid search (RRF + cross-encoder re-ranking).

  ``GET /api/v1/search/hybrid/shards``
      Describe the jurisdiction→shard routing table.

  ``GET /api/v1/search/hybrid/health``
      Health check — returns engine configuration and re-ranker status.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.auth import get_current_user
from core.hybrid_search_engine import (
    HybridSearchEngine,
    HybridSearchResult,
    get_hybrid_search_engine,
    _load_jurisdiction_map,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/search", tags=["hybrid-search"])


# ============================================================================
# Request / Response schemas
# ============================================================================

class HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Search query text")
    jurisdiction: Optional[str] = Field(None, description="Filter by jurisdiction (e.g. 'High Court')")
    court_level: Optional[str] = Field(None, description="Filter by court level (e.g. 'District Court')")
    top_k: int = Field(10, ge=1, le=50, description="Number of final results to return")
    enable_reranking: bool = Field(True, description="Apply cross-encoder re-ranking to top-20 RRF candidates")
    query_vector: Optional[List[float]] = Field(
        None,
        description="Pre-computed query embedding (optional — engine will embed the query if omitted)",
    )


class ChunkResponse(BaseModel):
    chunk_id: str
    case_id: int
    shard_id: int
    text: str
    jurisdiction: str
    court_level: str
    chunk_index: int
    semantic_score: float
    bm25_score: float
    rrf_score: float
    rerank_score: float
    final_score: float
    metadata: Dict[str, Any]


class HybridSearchResponse(BaseModel):
    query: str
    jurisdiction_filter: Optional[str]
    court_level_filter: Optional[str]
    total_semantic_candidates: int
    total_bm25_candidates: int
    fusion_strategy: str
    reranked: bool
    result_count: int
    results: List[ChunkResponse]


# ============================================================================
# Helpers
# ============================================================================

def _to_chunk_response(c) -> ChunkResponse:
    return ChunkResponse(
        chunk_id=c.chunk_id,
        case_id=c.case_id,
        shard_id=c.shard_id,
        text=c.text,
        jurisdiction=c.jurisdiction,
        court_level=c.court_level,
        chunk_index=c.chunk_index,
        semantic_score=round(c.semantic_score, 6),
        bm25_score=round(c.bm25_score, 6),
        rrf_score=round(c.rrf_score, 6),
        rerank_score=round(c.rerank_score, 6),
        final_score=round(c.final_score, 6),
        metadata=c.metadata,
    )


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/hybrid", response_model=HybridSearchResponse, summary="Cross-jurisdictional hybrid search")
def hybrid_search(
    request: HybridSearchRequest,
    current_user=Depends(get_current_user),
) -> HybridSearchResponse:
    """
    Execute a cross-jurisdictional hybrid search.

    The engine:
    1. Routes the query to jurisdiction-specific regional shards.
    2. Runs semantic cosine-similarity search against those shards.
    3. Runs BM25 lexical matching over the same shard excerpts.
    4. Fuses both ranked lists with Reciprocal Rank Fusion (RRF, k=60).
    5. Applies a local cross-encoder model to the top-20 RRF candidates
       and retains only high-relevance context for downstream RAG.
    """
    engine = get_hybrid_search_engine()
    try:
        result: HybridSearchResult = engine.search(
            query=request.query,
            query_vector=request.query_vector,
            jurisdiction=request.jurisdiction,
            court_level=request.court_level,
            top_k=request.top_k,
            enable_reranking=request.enable_reranking,
        )
    except Exception as exc:
        logger.error("hybrid_search_failed", query=request.query, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid search failed: {exc}",
        )

    return HybridSearchResponse(
        query=result.query,
        jurisdiction_filter=result.jurisdiction_filter,
        court_level_filter=result.court_level_filter,
        total_semantic_candidates=result.total_semantic_candidates,
        total_bm25_candidates=result.total_bm25_candidates,
        fusion_strategy=result.fusion_strategy,
        reranked=result.reranked,
        result_count=len(result.chunks),
        results=[_to_chunk_response(c) for c in result.chunks],
    )


@router.get("/hybrid/shards", summary="Jurisdiction shard routing table")
def get_shard_routing(current_user=Depends(get_current_user)) -> Dict[str, Any]:
    """
    Return the jurisdiction-to-shard routing table currently in use.

    Useful for ops tooling and debugging unexpected shard selections.
    """
    engine = get_hybrid_search_engine()
    return {
        "num_shards": engine._store.num_shards,
        "routing_table": _load_jurisdiction_map(),
    }


@router.get("/hybrid/health", summary="Hybrid search engine health")
def hybrid_search_health(current_user=Depends(get_current_user)) -> Dict[str, Any]:
    """
    Return engine health status including re-ranker configuration.
    """
    engine = get_hybrid_search_engine()
    reranker_info: Dict[str, Any] = {"available": False}
    try:
        from core.cross_encoder_reranker import CrossEncoderReranker
        r = engine._get_reranker()
        if r is not None:
            reranker_info = {
                "available": True,
                "model": r.model_name,
                "is_fallback": r.is_fallback,
            }
    except Exception:
        pass

    return {
        "status": "ok",
        "num_shards": engine._store.num_shards,
        "semantic_top_k": engine._semantic_top_k,
        "bm25_top_k": engine._bm25_top_k,
        "rrf_k": engine._rrf.k,
        "rerank_top_n": engine._rerank_top_n,
        "reranker": reranker_info,
    }
