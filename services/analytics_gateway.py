"""API-first access to dashboard analytics with a local fallback."""

from __future__ import annotations

import logging
from typing import Any, Dict
from urllib.parse import urljoin

from analytics_engine import AnalyticsAggregator
from config import Config
from core.api_client import call_with_fallback, APIResponse

logger = logging.getLogger(__name__)


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _dashboard_endpoint(base_url: str) -> str:
    return urljoin(f"{_normalize_base_url(base_url)}/", "api/v1/analytics/dashboard")


def _coerce_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    required_keys = {
        "total_cases_processed",
        "appeals_filed",
        "appeal_rate_percent",
        "plaintiff_wins",
        "defendant_wins",
        "settlements",
        "dismissals",
    }

    missing = required_keys - set(payload.keys())
    if missing:
        raise ValueError(f"Dashboard summary is missing keys: {sorted(missing)}")

    return {
        "total_cases_processed": int(payload["total_cases_processed"]),
        "appeals_filed": int(payload["appeals_filed"]),
        "appeal_rate_percent": float(payload["appeal_rate_percent"]),
        "plaintiff_wins": int(payload["plaintiff_wins"]),
        "defendant_wins": int(payload["defendant_wins"]),
        "settlements": int(payload["settlements"]),
        "dismissals": int(payload["dismissals"]),
    }


def get_dashboard_summary(db=None) -> Dict[str, Any]:
    """Fetch the dashboard summary from the API when possible, otherwise use local aggregates."""

    api_base_url = str(Config.API_BASE_URL or "").strip()
    if api_base_url:
        response = call_with_fallback(
            _dashboard_endpoint(api_base_url),
            timeout=Config.API_REQUEST_TIMEOUT_SECONDS,
            retries=2,
            fallback=lambda: _local_summary_fallback(db),
        )

        if response.success:
            return _coerce_summary(response.data)
        logger.warning(
            "Falling back to local analytics summary. api_base_url=%s, error=%s",
            api_base_url,
            response.error,
        )

    return _local_summary_fallback(db)


def _local_summary_fallback(db):
    """Return local analytics summary or raise if db unavailable."""
    if db is None:
        raise RuntimeError("A database session is required when the analytics API is unavailable.")
    return AnalyticsAggregator.get_dashboard_summary(db)
