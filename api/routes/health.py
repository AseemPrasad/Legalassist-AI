"""
Health Check Endpoints
GET /api/v1/health - Comprehensive health status (manual inspection)
GET /api/v1/health/ready - Readiness probe (Kubernetes readiness)
GET /api/v1/health/live - Liveness probe (Kubernetes liveness)
"""
from fastapi import APIRouter, Response
import structlog
from api.health_checks import get_health_manager

from db.models import SchedulerRun
from db.session import SessionLocal
from sqlalchemy import desc
router = APIRouter(prefix="/api/v1", tags=["health"])
logger = structlog.get_logger(__name__)


@router.get(
    "/health",
    summary="Comprehensive health status",
    response_description="Full health check with all component details"
)
async def health_check() -> dict:
    """
    Comprehensive health check for manual inspection and monitoring
    Returns detailed status of all components
    """
    manager = get_health_manager()
    result = await manager.deep_health_check(timeout=5)
    logger.info("health_check_completed", status=result["status"])
    return result


@router.get(
    "/health/ready",
    summary="Readiness probe",
    response_description="Kubernetes readiness probe endpoint"
)
async def readiness_check(response: Response) -> dict:
    """
    Readiness probe for Kubernetes
    - Returns 200 only if ALL dependencies (database, Redis, Celery) are healthy
    - Kubernetes removes pod from load balancer if it returns 503
    - Used for rolling updates and traffic management
    """
    manager = get_health_manager()
    result = await manager.readiness_check(timeout=5)
    
    status_code = result.pop("status_code", 200)
    response.status_code = status_code
    
    if status_code == 503:
        logger.warning("readiness_check_failed", checks=result["checks"])
    else:
        logger.info("readiness_check_passed")
    
    return result


@router.get(
    "/health/live",
    summary="Liveness probe",
    response_description="Kubernetes liveness probe endpoint"
)
async def liveness_check() -> dict:
    """
    Liveness probe for Kubernetes
    - Returns 200 if service process is running
    - Kubernetes restarts pod if it returns non-2xx status
    - Only checks if service itself is responsive (not dependencies)
    - Prevents restart loops if dependencies are temporarily down
    """
    manager = get_health_manager()
    result = await manager.liveness_check()
    logger.info("liveness_check_passed")
    return result

@router.get(
    "/health/scheduler",
    summary="Scheduler health metrics",
    response_description="Latest scheduler run metrics per job",
)
async def scheduler_health() -> dict:
    """Return recent scheduler run metrics for each job."""
    db = SessionLocal()
    try:
        job_names = [name[0] for name in db.query(SchedulerRun.job_name).distinct().all()]
        result: dict = {}
        for job in job_names:
            runs = (
                db.query(SchedulerRun)
                .filter(SchedulerRun.job_name == job)
                .order_by(desc(SchedulerRun.started_at))
                .limit(100)
                .all()
            )
            result[job] = [
                {
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                    "sent_count": r.sent_count,
                    "status": r.status.value,
                }
                for r in runs
            ]
        return result
    finally:
        db.close()
