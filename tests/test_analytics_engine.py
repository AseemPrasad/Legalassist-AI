"""Regression tests for analytics engine null-safety."""

from analytics_engine import AnalyticsCalculator
from database import CaseRecord, CaseOutcome


class TestAnalyticsCalculator:
    """Test analytics calculations with defensive None handling."""

    def test_calculate_appeal_success_rate_ignores_missing_outcome_data(self):
        """Cases without outcome data should be ignored, not crash the calculation."""
        cases = [
            CaseRecord(
                hashed_case_id="case-none",
                case_type="civil",
                jurisdiction="Delhi",
                outcome="plaintiff_won",
            ),
            CaseRecord(
                hashed_case_id="case-success",
                case_type="civil",
                jurisdiction="Delhi",
                outcome="plaintiff_won",
            ),
            CaseRecord(
                hashed_case_id="case-failure",
                case_type="civil",
                jurisdiction="Delhi",
                outcome="defendant_won",
            ),
        ]

        cases[1].outcome_data = CaseOutcome(
            appeal_filed=True,
            appeal_success=True,
        )
        cases[2].outcome_data = CaseOutcome(
            appeal_filed=True,
            appeal_success=False,
        )

        assert AnalyticsCalculator.calculate_appeal_success_rate(cases) == 50.0
