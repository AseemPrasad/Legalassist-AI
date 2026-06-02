from typing import Dict
import re


def _split_sentences(text: str):
    # Naive sentence splitter
    return [s.strip() for s in re.split(r'[\n\r]+|(?<=[.!?])\s+', text) if s.strip()]


def _count_keywords(text: str, keywords):
    text_l = text.lower()
    return sum(text_l.count(k) for k in keywords)


def score_argument(text: str, metadata: Dict = None) -> Dict:
    """Produce a heuristic argument strength score (0-100) and component breakdown.

    This is a deterministic, provider-free POC used for UI integration and testing.
    It combines simple signals: logical structure, evidence markers, precedent mentions,
    counter-argument anticipation, and clarity/conciseness.
    """
    if metadata is None:
        metadata = {}

    sentences = _split_sentences(text)
    num_sentences = max(1, len(sentences))
    words = text.split()
    num_words = max(1, len(words))

    # Logical structure heuristics: connective words indicating cause/conclusion
    logical_connectors = ["therefore", "thus", "hence", "because", "since", "so", "consequently"]
    logical_marks = _count_keywords(text, logical_connectors)
    logical_score = min(1.0, logical_marks / max(1, num_sentences * 0.5))

    # Evidence quality: explicit mentions of exhibits, witnesses, documents, numbers
    evidence_keywords = ["exhibit", "evidence", "witness", "affidavit", "document", "report"]
    evidence_marks = _count_keywords(text, evidence_keywords)
    numeric_refs = len(re.findall(r"\b\d{3,}\b", text))
    evidence_score = min(1.0, (evidence_marks + 0.5 * numeric_refs) / 2.0)

    # Precedent support: look for case-style tokens or 'v.' citations
    precedent_marks = len(re.findall(r"\b\w+ v\. \w+\b", text, flags=re.IGNORECASE))
    precedent_marks += len(re.findall(r"\bvs?\.\b", text, flags=re.IGNORECASE))
    precedent_score = min(1.0, precedent_marks / 2.0)

    # Counter-argument anticipation: presence of 'however', 'on the other hand', 'but'
    counter_words = ["however", "on the other hand", "but", "although", "nevertheless"]
    counter_marks = _count_keywords(text, counter_words)
    counter_score = min(1.0, counter_marks / 2.0)

    # Clarity / persuasiveness: short sentences, varied lengths, not overly long
    avg_sentence_len = num_words / num_sentences
    if avg_sentence_len < 12:
        clarity_score = 1.0
    elif avg_sentence_len < 20:
        clarity_score = 0.75
    elif avg_sentence_len < 35:
        clarity_score = 0.5
    else:
        clarity_score = 0.25

    # Weighted aggregation
    weights = {
        "logical": 0.25,
        "evidence": 0.25,
        "precedent": 0.2,
        "counter": 0.15,
        "clarity": 0.15,
    }

    final_score = (
        logical_score * weights["logical"] +
        evidence_score * weights["evidence"] +
        precedent_score * weights["precedent"] +
        counter_score * weights["counter"] +
        clarity_score * weights["clarity"]
    )

    # Scale to 0-100 and round
    final_score_100 = int(round(final_score * 100))

    return {
        "final_score": final_score_100,
        "components": {
            "logical_structure": int(round(logical_score * 100)),
            "evidence_quality": int(round(evidence_score * 100)),
            "precedent_support": int(round(precedent_score * 100)),
            "counter_argument_anticipation": int(round(counter_score * 100)),
            "clarity": int(round(clarity_score * 100)),
        },
        "metadata": metadata,
        "summary": {
            "num_sentences": num_sentences,
            "num_words": num_words,
        }
    }


__all__ = ["score_argument"]
