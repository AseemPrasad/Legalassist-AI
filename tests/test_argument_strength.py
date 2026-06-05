from core.argument_strength import score_argument


def test_score_basic_returns_components():
    text = (
        "The defendant breached the contract because they failed to deliver the goods. "
        "Exhibit 1 shows delivery dates and invoices. Smith v. Jones supports that delay is a material breach. "
        "However, the plaintiff also delayed payment."
    )

    res = score_argument(text, metadata={"case_id": "T-123"})
    assert "final_score" in res
    assert 0 <= res["final_score"] <= 100
    comps = res.get("components", {})
    assert set(comps.keys()) == {
        "logical_structure",
        "evidence_quality",
        "precedent_support",
        "counter_argument_anticipation",
        "clarity",
    }


def test_score_short_vague_argument_low_score():
    text = "I think we should win."
    res = score_argument(text)
    assert res["final_score"] <= 50
