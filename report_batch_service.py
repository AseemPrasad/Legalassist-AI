"""Batch report generation service.

Generates multiple reports in a single batch operation with
per-item progress tracking and consolidated result reporting.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from report_service import generate_report, GeneratedReport


@dataclass
class BatchReportItem:
    case_id: int
    report_type: str = "comprehensive"
    format: str = "pdf"
    include_remedies: bool = True
    include_timeline: bool = True
    style: str = "formal"
    watermark: Optional[str] = None


@dataclass
class BatchItemResult:
    case_id: int
    success: bool
    report: Optional[GeneratedReport] = None
    error: Optional[str] = None


@dataclass
class BatchReportResult:
    batch_id: str
    user_id: int
    total: int
    succeeded: int
    failed: int
    results: List[BatchItemResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


def generate_batch_report(
    *,
    user_id: int,
    items: List[BatchReportItem],
    report_id: Optional[str] = None,
) -> BatchReportResult:
    """Generate reports for multiple cases in a single batch."""

    batch_id = report_id or datetime.now(timezone.utc).strftime(
        "%Y%m%d%H%M%S%f"
    )

    result = BatchReportResult(
        batch_id=batch_id,
        user_id=user_id,
        total=len(items),
        succeeded=0,
        failed=0,
    )

    for item in items:
        try:
            single = generate_report(
                user_id=user_id,
                case_id=item.case_id,
                report_type=item.report_type,
                include_remedies=item.include_remedies,
                include_timeline=item.include_timeline,
                format=item.format,
                style=item.style,
                watermark=item.watermark,
                report_id=f"{batch_id}_{item.case_id}_{uuid.uuid4().hex[:8]}",
            )
            result.results.append(
                BatchItemResult(case_id=item.case_id, success=True, report=single)
            )
            result.succeeded += 1
        except Exception as e:
            result.results.append(
                BatchItemResult(case_id=item.case_id, success=False, error=str(e))
            )
            result.failed += 1

    result.completed_at = datetime.now(timezone.utc)
    return result
