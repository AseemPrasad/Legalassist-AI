"""
Local Cross-Encoder Re-ranker
==============================
Scores query–chunk pairs with a cross-encoder model to select only
high-relevance context before RAG ingestion.

Architecture
------------
* Prefers ``sentence-transformers`` ``CrossEncoder`` models (e.g.
  ``cross-encoder/ms-marco-MiniLM-L-6-v2``).
* Falls back to a lightweight TF-IDF cosine similarity scorer when
  ``sentence-transformers`` is not installed — ensuring the re-ranker
  is never a hard dependency.
* Scores are normalised to [0, 1] via sigmoid so that the threshold
  comparison in :class:`~core.hybrid_search_engine.HybridSearchEngine`
  is meaningful.

Configuration
-------------
``CROSS_ENCODER_MODEL``
    Name of the HuggingFace cross-encoder model.  Default:
    ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
``CROSS_ENCODER_DEVICE``
    ``cpu`` or ``cuda``.  Default: ``cpu``.
``CROSS_ENCODER_MAX_LENGTH``
    Maximum token length for the cross-encoder.  Default: 512.
"""
from __future__ import annotations

import logging
import math
import os
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_DEVICE = os.getenv("CROSS_ENCODER_DEVICE", "cpu")
_MAX_LENGTH = int(os.getenv("CROSS_ENCODER_MAX_LENGTH", "512"))


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1 + exp_x)


# ---------------------------------------------------------------------------
# TF-IDF fallback (zero-dependency)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _tfidf_cosine(query_tokens: List[str], doc_tokens: List[str]) -> float:
    """Unweighted token-overlap cosine — used as a no-dependency fallback."""
    if not query_tokens or not doc_tokens:
        return 0.0
    q_set = set(query_tokens)
    d_set = set(doc_tokens)
    overlap = q_set & d_set
    denominator = math.sqrt(len(q_set)) * math.sqrt(len(d_set))
    if denominator == 0:
        return 0.0
    return len(overlap) / denominator


class _TFIDFFallbackScorer:
    """Lightweight TF-IDF cosine scorer — no ML dependencies."""

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        scores = []
        for query, doc in pairs:
            q_tokens = _tokenize(query)
            d_tokens = _tokenize(doc)
            scores.append(_tfidf_cosine(q_tokens, d_tokens))
        return scores


# ---------------------------------------------------------------------------
# Cross-encoder wrapper
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Query–passage cross-encoder for high-relevance context selection.

    The re-ranker is used by :class:`~core.hybrid_search_engine.HybridSearchEngine`
    to score the top-N RRF-fused candidates and retain only those above a
    relevance threshold before RAG ingestion.

    Parameters
    ----------
    model_name:
        HuggingFace cross-encoder model identifier.
    device:
        ``"cpu"`` or ``"cuda"``.
    max_length:
        Maximum token length passed to the model tokeniser.
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        device: str = _DEVICE,
        max_length: int = _MAX_LENGTH,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._model = None
        self._is_fallback = False
        self._init_model()

    def _init_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
                max_length=self._max_length,
            )
            logger.info(
                "cross_encoder_loaded",
                model=self._model_name,
                device=self._device,
            )
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — "
                "using TF-IDF cosine fallback for cross-encoder re-ranking"
            )
            self._model = _TFIDFFallbackScorer()
            self._is_fallback = True
        except Exception as exc:
            logger.warning(
                "Failed to load cross-encoder model '%s': %s — using TF-IDF fallback",
                self._model_name,
                exc,
            )
            self._model = _TFIDFFallbackScorer()
            self._is_fallback = True

    def rerank(
        self,
        query: str,
        chunks: "List[Any]",  # List[SearchChunk] — avoid circular import
    ) -> "List[Tuple[Any, float]]":
        """
        Score each chunk in *chunks* against *query* and return sorted results.

        Parameters
        ----------
        query:
            The search query string.
        chunks:
            List of :class:`~core.hybrid_search_engine.SearchChunk` objects.

        Returns
        -------
        List of ``(chunk, normalised_score)`` sorted by descending score.
        """
        if not chunks:
            return []

        pairs = [(query, c.text) for c in chunks]

        try:
            raw_scores = self._model.predict(pairs)
        except Exception as exc:
            logger.warning("cross_encoder_predict_failed: %s", exc)
            return [(c, c.rrf_score) for c in chunks]

        # Normalise to [0, 1] via sigmoid (raw logits from CrossEncoder)
        if self._is_fallback:
            # Fallback already returns [0, 1]
            normalised = [float(s) for s in raw_scores]
        else:
            normalised = [_sigmoid(float(s)) for s in raw_scores]

        scored = list(zip(chunks, normalised))
        scored.sort(key=lambda x: -x[1])
        return scored

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_fallback(self) -> bool:
        """True when using the TF-IDF fallback (sentence-transformers not installed)."""
        return self._is_fallback
