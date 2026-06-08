from __future__ import annotations

from fastapi import Query, HTTPException

_UNPROCESSABLE = 422

# Common English stop words that add little semantic value
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "by", "with", "from", "up", "about", "into", "over", "after",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "need", "dare", "ought", "used",
    "it", "its", "it's", "this", "that", "these", "those", "i", "you",
    "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours",
    "hers", "ours", "theirs", "not", "no", "nor", "so", "very",
    "just", "as", "if", "then", "than", "too", "also", "only",
    "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "what", "which", "who", "whom", "whose",
})


def meaningful_search_query(
    query: str = Query(..., min_length=5),
) -> str:
    words = query.strip().split()
    if len(words) < 3:
        raise HTTPException(
            status_code=_UNPROCESSABLE,
            detail=f"Search query must contain at least 3 words, got {len(words)}",
        )

    unique_words = {w.lower() for w in words}
    if len(unique_words) < 2:
        raise HTTPException(
            status_code=_UNPROCESSABLE,
            detail="Search query must contain at least 2 unique words",
        )

    meaningful = [w for w in unique_words if w not in _STOP_WORDS and len(w) > 1]
    if len(meaningful) < 1:
        raise HTTPException(
            status_code=_UNPROCESSABLE,
            detail="Search query must contain at least one meaningful word beyond common stop words",
        )

    stop_word_ratio = sum(1 for w in words if w.lower() in _STOP_WORDS) / len(words)
    if stop_word_ratio > 0.75:
        raise HTTPException(
            status_code=_UNPROCESSABLE,
            detail="Search query contains too many common words and not enough meaningful content",
        )

    return query
