"""
API routes for Case Search and Precedent Matching
Endpoints for finding similar cases, precedents, comparisons, and knowledge graph queries.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from typing import List, Optional

from api.auth import get_current_user, CurrentUser
from database import CaseRecord, CaseOutcome, Case, get_db
from analytics_engine import CaseSimilarityCalculator

router = APIRouter(prefix="/api/v1/cases", tags=["case-search"])


@router.get("/{case_id}/search-similar")
def search_similar_cases(
    case_id: int,
    limit: int = Query(10, ge=1, le=50),
    min_similarity: float = Query(0.5, ge=0, le=1),
    case_type: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    outcome: Optional[str] = None,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Find similar cases based on attribute similarity"""
    db = get_db()
    try:
        case = db.query(Case).filter(Case.id == case_id, Case.user_id == int(current_user.user_id)).first()
        if not case:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

        query = db.query(CaseRecord)
        if case_type:
            query = query.filter(CaseRecord.case_type == case_type)
        if jurisdiction:
            query = query.filter(CaseRecord.jurisdiction == jurisdiction)
        if outcome:
            query = query.filter(CaseRecord.outcome == outcome)

        candidates = query.limit(500).all()
        reference = CaseRecord(
            hashed_case_id=str(case.id),
            case_type=case.case_type or "general",
            jurisdiction=case.jurisdiction or "unknown",
            plaintiff_type="",
            defendant_type="",
            case_value="",
            outcome="",
            judgment_summary="",
        )

        scored = []
        for c in candidates:
            score = CaseSimilarityCalculator.case_similarity_score(reference, c) / 100.0
            if score > min_similarity:
                scored.append({
                    "case_id": c.id,
                    "case_type": c.case_type,
                    "jurisdiction": c.jurisdiction,
                    "outcome": c.outcome,
                    "similarity": round(score, 4),
                })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return {"query_case_id": case_id, "similar_cases": scored[:limit], "count": min(len(scored), limit)}
    finally:
        db.close()


@router.get("/search/text")
def search_by_text(
    search_query: str = Query(..., min_length=10, alias="query"),
    limit: int = Query(10, ge=1, le=50),
    case_type: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Search for cases by free text"""
    db = get_db()
    try:
        q = db.query(CaseRecord)
        if case_type:
            q = q.filter(CaseRecord.case_type == case_type)
        if jurisdiction:
            q = q.filter(CaseRecord.jurisdiction == jurisdiction)

        candidates = q.limit(200).all()
        results = []
        for c in candidates:
            summary = c.judgment_summary or ""
            if search_query.lower() in summary.lower():
                results.append({
                    "case_id": c.id,
                    "case_type": c.case_type,
                    "jurisdiction": c.jurisdiction,
                    "outcome": c.outcome,
                    "summary": summary[:200],
                })

        return {"query": search_query, "results": results[:limit], "count": min(len(results), limit)}
    finally:
        db.close()


@router.get("/search/statistics")
def get_search_statistics(
    current_user: CurrentUser = Depends(get_current_user),
):
    """Get statistics about indexed cases"""
    db = get_db()
    try:
        total = db.query(CaseRecord).count()
        return {"total_indexed_cases": total}
    finally:
        db.close()
