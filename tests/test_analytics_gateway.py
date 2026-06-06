"""Tests for the API-first analytics gateway."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base, CaseOutcome, CaseRecord
import services.analytics_gateway as analytics_gateway


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session_factory()


def _add_case(db, hashed_case_id: str, outcome: str, appeal_filed: bool, appeal_success: bool | None):
    case = CaseRecord(
        hashed_case_id=hashed_case_id,
        case_type="civil",
        jurisdiction="Delhi",
        outcome=outcome,
    )
    db.add(case)
    db.flush()

    db.add(
        CaseOutcome(
            case_id=case.id,
            appeal_filed=appeal_filed,
            appeal_success=appeal_success,
        )
    )
    db.commit()


def test_get_dashboard_summary_uses_api_when_configured(monkeypatch):
    payload = {
        "total_cases_processed": 12,
        "appeals_filed": 4,
        "appeal_rate_percent": 33.3,
        "plaintiff_wins": 5,
        "defendant_wins": 3,
        "settlements": 2,
        "dismissals": 2,
    }

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    def fake_get(url, timeout):
        assert url.endswith("/api/v1/analytics/dashboard")
        assert timeout == 5.0
        return FakeResponse()

    monkeypatch.setattr(analytics_gateway.Config, "API_BASE_URL", "http://backend.test")
    monkeypatch.setattr(analytics_gateway.requests, "get", fake_get)

    summary = analytics_gateway.get_dashboard_summary()

    assert summary == payload


def test_get_dashboard_summary_falls_back_to_local_db(monkeypatch):
    monkeypatch.setattr(analytics_gateway.Config, "API_BASE_URL", "")

    db = _make_session()
    try:
        _add_case(db, "case-1", "plaintiff_won", True, True)
        _add_case(db, "case-2", "defendant_won", True, False)

        summary = analytics_gateway.get_dashboard_summary(db)

        assert summary["total_cases_processed"] == 2
        assert summary["appeals_filed"] == 2
        assert summary["plaintiff_wins"] == 1
        assert summary["defendant_wins"] == 1
    finally:
        db.close()
