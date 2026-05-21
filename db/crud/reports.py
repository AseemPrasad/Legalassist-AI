"""
CRUD operations for Report model.

Provides functions for creating, retrieving, and updating report records
in the database for reliable tracking of generated reports.
"""

import datetime as dt
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import and_
from db.models import Report, ReportStatus, ReportType, ReportFormat
from db.session import get_db


def create_report(
    db: Session,
    report_id: str,
    user_id: int,
    case_id: int,
    celery_task_id: str,
    report_type: str = "comprehensive",
    format: str = "pdf",
    style: str = "formal",
) -> Report:
    """
    Create a new report record.
    
    Args:
        db: Database session
        report_id: Unique report UUID
        user_id: Owner of the report
        case_id: Associated case ID
        celery_task_id: Celery task ID for tracking
        report_type: Type of report
        format: Output format
        style: Report style
        
    Returns:
        Created Report model instance
    """
    report = Report(
        report_id=report_id,
        user_id=user_id,
        case_id=case_id,
        celery_task_id=celery_task_id,
        report_type=ReportType(report_type),
        format=ReportFormat(format),
        style=style,
        status=ReportStatus.PENDING,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def get_report_by_id(
    db: Session,
    report_id: str,
    user_id: Optional[int] = None,
) -> Optional[Report]:
    """
    Retrieve a report by report_id.
    
    Args:
        db: Database session
        report_id: Report UUID
        user_id: Optional user ID to validate ownership
        
    Returns:
        Report model or None if not found
    """
    query = db.query(Report).filter(Report.report_id == report_id)
    
    if user_id is not None:
        query = query.filter(Report.user_id == user_id)
    
    return query.first()


def get_report_by_celery_task_id(
    db: Session,
    celery_task_id: str,
) -> Optional[Report]:
    """
    Retrieve a report by Celery task ID.
    
    Args:
        db: Database session
        celery_task_id: Celery task ID
        
    Returns:
        Report model or None if not found
    """
    return db.query(Report).filter(
        Report.celery_task_id == celery_task_id
    ).first()


def update_report_status(
    db: Session,
    report_id: str,
    status: str,
    file_path: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
    error_message: Optional[str] = None,
    started_at: Optional[dt.datetime] = None,
    completed_at: Optional[dt.datetime] = None,
) -> Optional[Report]:
    """
    Update a report's status and metadata.
    
    Args:
        db: Database session
        report_id: Report UUID
        status: New status (pending, processing, completed, failed)
        file_path: Path to the generated file
        file_size_bytes: Size of the generated file
        error_message: Error message if failed
        started_at: When task started
        completed_at: When task completed
        
    Returns:
        Updated Report model or None if not found
    """
    report = get_report_by_id(db, report_id)
    if not report:
        return None
    
    report.status = ReportStatus(status)
    if file_path is not None:
        report.file_path = file_path
    if file_size_bytes is not None:
        report.file_size_bytes = file_size_bytes
    if error_message is not None:
        report.error_message = error_message
    if started_at is not None:
        report.started_at = started_at
    if completed_at is not None:
        report.completed_at = completed_at
    
    report.updated_at = dt.datetime.now(dt.timezone.utc)
    db.commit()
    db.refresh(report)
    return report


def list_reports_by_user(
    db: Session,
    user_id: int,
    limit: int = 10,
    offset: int = 0,
    status: Optional[str] = None,
) -> tuple[List[Report], int]:
    """
    List reports for a user with optional filtering.
    
    Args:
        db: Database session
        user_id: User ID
        limit: Max results to return
        offset: Pagination offset
        status: Optional status filter
        
    Returns:
        Tuple of (reports list, total count)
    """
    query = db.query(Report).filter(Report.user_id == user_id)
    
    if status:
        query = query.filter(Report.status == ReportStatus(status))
    
    total = query.count()
    reports = query.order_by(Report.created_at.desc()).limit(limit).offset(offset).all()
    
    return reports, total


def list_reports_by_case(
    db: Session,
    case_id: int,
    user_id: Optional[int] = None,
) -> List[Report]:
    """
    List reports for a specific case.
    
    Args:
        db: Database session
        case_id: Case ID
        user_id: Optional user ID filter
        
    Returns:
        List of Report models
    """
    query = db.query(Report).filter(Report.case_id == case_id)
    
    if user_id is not None:
        query = query.filter(Report.user_id == user_id)
    
    return query.order_by(Report.created_at.desc()).all()
