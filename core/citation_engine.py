"""Legal citation extraction and validation utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional


_CASE_REPORTER_PATTERN = re.compile(
    r"\b(?:AIR|SCC|SCR|SCALE|CriLJ|All\s+LJ|JT|DLT|KLT|KerLT|BomCR|SCC\s+OnLine|ILR)"
    r"\s*\d{4}\s*[A-Z]{0,6}\s*\d+\b",
    re.IGNORECASE,
)

_CASE_PARTY_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z0-9&.,'()\-/ ]{2,80}\s+v(?:\.|s\.|ersus)?\s+"
    r"[A-Z][A-Za-z0-9&.,'()\-/ ]{2,80}(?:,\s*(?:AIR\s*\d{4}\s*[A-Z]{1,8}\s*\d+|"
    r"(?:SCC|SCR|SCALE|CriLJ|All\s+LJ|JT|DLT|KLT|KerLT|BomCR|SCC\s+OnLine|ILR)\s*\d{4}\s*[A-Z]{0,6}\s*\d+))?",
    re.IGNORECASE,
)

_SECTION_PATTERN = re.compile(
    r"\b(?:Section|Sec\.?|S\.?|U\/s)\s*\d+[A-Za-z0-9()\-/]*"
    r"(?:\s*(?:of\s+the)?\s*(?:IPC|CrPC|CPC|Evidence\s+Act|Constitution(?:\s+of\s+India)?|"
    r"Companies\s+Act|Income\s+Tax\s+Act|Indian\s+Penal\s+Code|Code\s+of\s+Civil\s+Procedure|"
    r"Code\s+of\s+Criminal\s+Procedure|Motor\s+Vehicles\s+Act|Consumer\s+Protection\s+Act|"
    r"Negotiable\s+Instruments\s+Act|Information\s+Technology\s+Act|Transfer\s+of\s+Property\s+Act))?",
    re.IGNORECASE,
)

_ARTICLE_PATTERN = re.compile(r"\bArticle\s*\d+[A-Za-z0-9()\-/]*\b", re.IGNORECASE)
_RULE_PATTERN = re.compile(r"\bRule\s*\d+[A-Za-z0-9()\-/]*\b", re.IGNORECASE)
_REGULATION_PATTERN = re.compile(r"\b(?:Regulation|Reg\.?|Order)\s*\d+[A-Za-z0-9()\-/]*\b", re.IGNORECASE)

_NEGATIVE_SIGNALS = {
    "overruled": "deprecated",
    "reversed": "deprecated",
    "superseded": "deprecated",
    "distinguished": "limited",
    "followed": "authoritative",
    "applied": "authoritative",
    "relied on": "authoritative",
    "cited": "mentioned",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _window(text: str, start: int, end: int, padding: int = 120) -> str:
    left = max(0, start - padding)
    right = min(len(text), end + padding)
    return text[left:right].strip()


def _authority_rank(citation_type: str, status: str) -> float:
    base = {
        "case_law": 0.9,
        "statute": 0.95,
        "constitutional_article": 0.92,
        "rule": 0.84,
        "regulation": 0.8,
        "other": 0.6,
    }.get(citation_type, 0.6)

    if status == "deprecated":
        return max(0.1, base - 0.45)
    if status == "limited":
        return max(0.2, base - 0.2)
    if status == "authoritative":
        return min(1.0, base + 0.05)
    return base


def _classify_citation(citation: str) -> str:
    lowered = citation.lower()
    if _ARTICLE_PATTERN.search(citation):
        return "constitutional_article"
    if _SECTION_PATTERN.search(citation):
        return "statute"
    if _RULE_PATTERN.search(citation):
        return "rule"
    if _REGULATION_PATTERN.search(citation):
        return "regulation"
    if " v" in lowered or _CASE_REPORTER_PATTERN.search(citation):
        return "case_law"
    return "other"


def _citation_status(context: str, citation_type: str) -> str:
    if citation_type != "case_law":
        return "valid"
    lowered = _normalize(context)
    for signal, status in _NEGATIVE_SIGNALS.items():
        if signal in lowered:
            return status
    return "valid"


def _validation_notes(citation_type: str, citation: str, context: str) -> List[str]:
    notes: List[str] = []
    if citation_type == "case_law" and " v" not in citation.lower() and not _CASE_REPORTER_PATTERN.search(citation):
        notes.append("Case citation pattern is weak; manual review recommended.")
    if citation_type == "other":
        notes.append("Could not map citation to a known legal citation type.")
    if "see also" in _normalize(context):
        notes.append("Context includes secondary references; verify the primary authority.")
    return notes


@dataclass
class CitationHit:
    citation: str
    citation_type: str
    status: str
    confidence: float
    authority_rank: float
    is_valid: bool
    span: Dict[str, int]
    context: str
    notes: List[str]
    matched_reference: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation": self.citation,
            "normalized_citation": _normalize(self.citation),
            "citation_type": self.citation_type,
            "status": self.status,
            "confidence": round(self.confidence, 3),
            "authority_rank": round(self.authority_rank, 3),
            "is_valid": self.is_valid,
            "span": self.span,
            "context": self.context,
            "notes": self.notes,
            "matched_reference": self.matched_reference,
        }


class CitationEngine:
    """Extract, validate, and rank legal citations from text."""

    @staticmethod
    def analyze(
        text: str,
        known_references: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not text or not text.strip():
            return {
                "citations": [],
                "summary": {
                    "total_citations": 0,
                    "unique_citations": 0,
                    "validated_citations": 0,
                    "needs_review": 0,
                    "deprecated_citations": 0,
                    "by_type": {},
                },
                "network": {"nodes": [], "edges": []},
            }

        hits: List[CitationHit] = []
        seen = set()
        candidates = []

        for priority, pattern in enumerate(
            (
                _CASE_PARTY_PATTERN,
                _CASE_REPORTER_PATTERN,
                _SECTION_PATTERN,
                _ARTICLE_PATTERN,
                _RULE_PATTERN,
                _REGULATION_PATTERN,
            )
        ):
            for match in pattern.finditer(text):
                candidates.append((match.start(), match.end(), priority, match))

        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
        accepted_spans: List[tuple[int, int]] = []

        for start, end, _, match in candidates:
            if any(not (end <= span_start or start >= span_end) for span_start, span_end in accepted_spans):
                continue

            accepted_spans.append((start, end))
            citation = match.group(0).strip().rstrip(".,;")
            normalized = _normalize(citation)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)

            context = _window(text, match.start(), match.end())
            citation_type = _classify_citation(citation)
            status = _citation_status(context, citation_type)
            notes = _validation_notes(citation_type, citation, context)
            is_valid = citation_type != "other" and bool(citation)

            confidence = 0.65
            if citation_type in {"statute", "constitutional_article"}:
                confidence = 0.95
            elif citation_type == "rule":
                confidence = 0.88
            elif citation_type == "regulation":
                confidence = 0.82
            elif citation_type == "case_law":
                confidence = 0.9 if _CASE_REPORTER_PATTERN.search(citation) else 0.78

            if status in {"deprecated", "limited"}:
                confidence -= 0.1

            matched_reference = CitationEngine._match_known_reference(citation, known_references or [])
            if matched_reference:
                confidence = min(1.0, confidence + 0.05)

            hits.append(
                CitationHit(
                    citation=citation,
                    citation_type=citation_type,
                    status=status,
                    confidence=max(0.1, min(1.0, confidence)),
                    authority_rank=_authority_rank(citation_type, status),
                    is_valid=is_valid,
                    span={"start": match.start(), "end": match.end()},
                    context=context,
                    notes=notes,
                    matched_reference=matched_reference,
                )
            )

        citations = sorted(
            [hit.to_dict() for hit in hits],
            key=lambda item: (-item["authority_rank"], item["span"]["start"]),
        )

        summary = CitationEngine._build_summary(citations)
        network = CitationEngine._build_network(citations)
        return {"citations": citations, "summary": summary, "network": network}

    @staticmethod
    def _build_summary(citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        for item in citations:
            by_type[item["citation_type"]] = by_type.get(item["citation_type"], 0) + 1

        return {
            "total_citations": len(citations),
            "unique_citations": len({item["normalized_citation"] for item in citations}),
            "validated_citations": sum(1 for item in citations if item["is_valid"]),
            "needs_review": sum(1 for item in citations if item["status"] in {"limited", "deprecated"} or item["notes"]),
            "deprecated_citations": sum(1 for item in citations if item["status"] == "deprecated"),
            "by_type": by_type,
        }

    @staticmethod
    def _build_network(citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        nodes = [
            {"id": "document", "label": "Document", "type": "source"},
        ]
        edges = []
        for index, item in enumerate(citations, start=1):
            node_id = f"citation_{index}"
            nodes.append(
                {
                    "id": node_id,
                    "label": item["citation"],
                    "type": item["citation_type"],
                    "status": item["status"],
                    "authority_rank": item["authority_rank"],
                }
            )
            edges.append({"source": "document", "target": node_id, "relation": "mentions"})
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _match_known_reference(
        citation: str,
        known_references: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not known_references:
            return None

        citation_norm = _normalize(citation)
        best: Optional[Dict[str, Any]] = None
        best_score = 0.0

        for reference in known_references:
            if not reference:
                continue
            reference_text = " ".join(
                str(reference.get(field, "")) for field in ("case_number", "title", "citation", "name")
            ).strip()
            if not reference_text:
                continue
            score = SequenceMatcher(None, citation_norm, _normalize(reference_text)).ratio()
            if score > best_score:
                best_score = score
                best = {
                    "reference": reference,
                    "similarity": round(score, 3),
                }

        return best if best_score >= 0.6 else None


def analyze_legal_citations(
    text: str,
    known_references: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for citation analysis."""

    return CitationEngine.analyze(text, known_references=known_references)