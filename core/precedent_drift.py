"""Lightweight drift detection and retraining helpers for precedent matcher.

This module computes simple embedding-distribution statistics (centroid,
average nearest-neighbor similarity) and saves a baseline snapshot. A
drift detector compares recent embeddings against the baseline and triggers
retraining when thresholds are exceeded.
"""
import json
import logging
import os
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy.orm import Session

from db.models.analytics import CaseEmbedding
from db.models.cases import Case

logger = logging.getLogger(__name__)

DEFAULT_STATS_PATH = os.path.join("models", "precedent_model_stats.json")


def _load_stats(path: str = DEFAULT_STATS_PATH) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_stats(stats: Dict[str, Any], path: str = DEFAULT_STATS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f)


def compute_embedding_stats(db: Session, sample_size: int = 500) -> Optional[Dict[str, Any]]:
    """Compute centroid and pairwise similarity percentiles for embeddings.

    Returns a dict with centroid vector (list), mean_similarity, median_similarity.
    """
    try:
        q = db.query(CaseEmbedding).order_by(CaseEmbedding.id.desc()).limit(sample_size)
        rows = q.all()
        vecs = []
        for r in rows:
            try:
                arr = json.loads(r.embedding_vector)
                vecs.append(np.array(arr, dtype=np.float32))
            except Exception:
                continue

        if not vecs:
            logger.warning("No embeddings available to compute stats")
            return None

        mat = np.vstack(vecs)
        centroid = np.mean(mat, axis=0)

        # cosine similarities to centroid
        def cos_sim(a, b):
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        sims = [cos_sim(v, centroid) for v in vecs]
        sims = np.array(sims)

        stats = {
            "count": int(len(vecs)),
            "centroid": centroid.tolist(),
            "mean_sim": float(np.mean(sims)),
            "median_sim": float(np.median(sims)),
            "p10": float(np.percentile(sims, 10)),
            "p90": float(np.percentile(sims, 90)),
        }
        return stats
    except Exception as e:
        logger.exception("Failed to compute embedding stats: %s", e)
        return None


def detect_drift(db: Session, threshold_drop: float = 0.08, stats_path: str = DEFAULT_STATS_PATH) -> Dict[str, Any]:
    """Detect drift by comparing current stats to baseline saved in stats_path.

    threshold_drop: relative drop in mean similarity (e.g., 0.08 = 8%)
    Returns dict {drift: bool, baseline:..., current:..., reason: str}
    """
    baseline = _load_stats(stats_path)
    current = compute_embedding_stats(db)
    if current is None:
        return {"drift": False, "reason": "no_current_stats"}
    if not baseline:
        # No baseline - save current as baseline and return no drift
        try:
            _save_stats(current, stats_path)
            logger.info("Saved new baseline stats to %s", stats_path)
        except Exception:
            logger.exception("Failed to save baseline stats")
        return {"drift": False, "baseline": None, "current": current, "reason": "baseline_initialized"}

    # compare mean_sim relative drop
    base_mean = float(baseline.get("mean_sim", 0.0))
    cur_mean = float(current.get("mean_sim", 0.0))
    if base_mean <= 0:
        return {"drift": False, "baseline": baseline, "current": current, "reason": "invalid_baseline_mean"}

    relative_drop = (base_mean - cur_mean) / base_mean
    drift = relative_drop >= threshold_drop

    reason = f"relative_drop={relative_drop:.3f}"
    return {"drift": bool(drift), "baseline": baseline, "current": current, "relative_drop": relative_drop, "reason": reason}


def retrain_embeddings(db: Session, model: str = "text-embedding-3-small", batch_size: int = 64) -> Dict[str, Any]:
    """Retrain (re-embed) all cases and rebuild baseline stats.

    This re-computes embeddings for all cases, updates CaseEmbedding rows,
    and saves new baseline stats. Returns summary dict.
    """
    # Import EmbeddingEngine lazily to avoid heavy imports at module import time
    from core.embedding_engine import EmbeddingEngine

    engine = EmbeddingEngine(model=model)
    try:
        # Get all case ids
        case_ids = [c.id for c in db.query(Case).all()]
        if not case_ids:
            return {"retrained": False, "reason": "no_cases"}

        # regenerate embeddings in batches
        for i in range(0, len(case_ids), batch_size):
            batch = case_ids[i : i + batch_size]
            engine.embed_multiple_cases(db, batch, force_regenerate=True)

        # compute and persist new baseline
        stats = compute_embedding_stats(db)
        if stats:
            _save_stats(stats)

        return {"retrained": True, "count": len(case_ids), "stats": stats}
    except Exception as e:
        logger.exception("Retraining failed: %s", e)
        return {"retrained": False, "error": str(e)}
