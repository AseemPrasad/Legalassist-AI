from core.citation_engine import CitationEngine


def test_citation_engine_extracts_and_validates_common_legal_citations():
    text = """
    In State of Karnataka v. B. R. Gopal, AIR 1980 SC 1, the court held that the
    conviction could not stand. The authority was later overruled by a larger bench.

    The court applied Section 302 of the Indian Penal Code and Article 21 of the Constitution of India.
    It also referred to Rule 37 of the Code of Civil Procedure, 1908.
    Article 21 was mentioned again in the reasoning.
    """

    analysis = CitationEngine.analyze(
        text,
        known_references=[{"citation": "AIR 1980 SC 1", "title": "State of Karnataka v. B. R. Gopal"}],
    )

    summary = analysis["summary"]
    citations = analysis["citations"]

    assert summary["total_citations"] == 4
    assert summary["unique_citations"] == 4
    assert summary["validated_citations"] == 4
    assert summary["deprecated_citations"] == 1
    assert summary["by_type"]["case_law"] == 1
    assert summary["by_type"]["statute"] == 1
    assert summary["by_type"]["constitutional_article"] == 1
    assert summary["by_type"]["rule"] == 1

    case_citation = next(item for item in citations if item["citation_type"] == "case_law")
    assert case_citation["status"] == "deprecated"
    assert case_citation["matched_reference"] is not None
    assert case_citation["matched_reference"]["similarity"] >= 0.6

    statute_citation = next(item for item in citations if item["citation_type"] == "statute")
    assert statute_citation["is_valid"] is True
    assert statute_citation["authority_rank"] > 0.9


def test_citation_engine_handles_empty_text():
    analysis = CitationEngine.analyze("")

    assert analysis["citations"] == []
    assert analysis["summary"]["total_citations"] == 0
    assert analysis["network"]["edges"] == []
