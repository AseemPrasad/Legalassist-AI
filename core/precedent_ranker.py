"""Precedent ranking utilities.

Provides multi-factor ranking combining authority, recency, citation network and relevance.
"""
from datetime import datetime, timezone
import math
from typing import List, Dict, Any

from sqlalchemy.orm import Session

from database import Case, CaseDocument, KnowledgeGraphEdge


_COURT_AUTHORITY = [
    ("supreme", 1.0),
    ("high court", 0.9),
    ("tribunal", 0.8),
    ("district", 0.7),
    ("magistrate", 0.6),
]


def _authority_score(court_name: str) -> float:
    if not court_name:
        return 0.5
    name = court_name.lower()
    for key, score in _COURT_AUTHORITY:
        if key in name:
            return score
    return 0.6


def _recency_score(created_iso: str) -> float:
    try:
        dt = datetime.fromisoformat(created_iso)
        now = datetime.now(timezone.utc)
        years = max(0.0, (now - dt).days / 365.25)
        # Exponential decay: recent -> closer to 1
        return math.exp(-0.25 * years)
    except Exception:
        return 0.5


def _citation_count(db: Session, case_id: int) -> int:
    # Proxy: number of knowledge graph edges pointing to this case
    try:
        return db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.case_id == case_id).count()
    except Exception:
        return 0


def _overruled_penalty(db: Session, case_id: int) -> float:
    # Lightweight heuristic: if any document summary mentions 'overrule' or 'overruled', penalize
    try:
        docs = db.query(CaseDocument).filter(CaseDocument.case_id == case_id).all()
        for d in docs:
            summary = (d.summary or "").lower()
            if "overrule" in summary or "overruled" in summary:
                return 0.5
        return 0.0
    except Exception:
        return 0.0


def rank_precedents(
    db: Session,
    candidates: List[Dict[str, Any]],
    weights: Dict[str, float] = None,
) -> List[Dict[str, Any]]:
    """Compute multi-factor scores and sort candidates.

    candidates: list of result dicts from `PrecedentMatcher` with at least `case_id` and `weight` keys.
    weights: dict with keys 'relevance','authority','recency','citation' summing ideally to 1.0
    """
    if weights is None:
        weights = {"relevance": 0.45, "authority": 0.2, "recency": 0.2, "citation": 0.15}

    # compute citation max for normalization
    citation_counts = {}
    max_cite = 1
    for c in candidates:
        cid = c.get("case_id")
        cnt = _citation_count(db, cid)
        citation_counts[cid] = cnt
        if cnt > max_cite:
            max_cite = cnt

    ranked = []
    for c in candidates:
        cid = c.get("case_id")
        # relevance: use provided edge weight if available
        relevance = float(c.get("weight", 1.0))

        # authority from court_name if available
        court_name = c.get("court_name") or c.get("title") or ""
        authority = _authority_score(court_name)

        recency = _recency_score(c.get("created_at", datetime.now(timezone.utc).isoformat()))

        citation_norm = 0.0
        if max_cite > 0:
            citation_norm = citation_counts.get(cid, 0) / float(max_cite)

        overruled = _overruled_penalty(db, cid)

        score = (
            weights.get("relevance", 0.0) * relevance
            + weights.get("authority", 0.0) * authority
            + weights.get("recency", 0.0) * recency
            + weights.get("citation", 0.0) * citation_norm
        )

        # Apply overruled penalty multiplicatively
        if overruled:
            score = score * (1.0 - overruled)

        out = dict(c)
        out.update({
            "authority_score": round(authority, 3),
            "recency_score": round(recency, 3),
            "citation_count": citation_counts.get(cid, 0),
            "final_score": round(score, 4),
        })
        ranked.append(out)

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked
