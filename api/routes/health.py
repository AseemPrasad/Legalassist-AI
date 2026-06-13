"""
Health Check Endpoints
GET /api/v1/health - Comprehensive health status (manual inspection)
GET /api/v1/health/ready - Readiness probe (Kubernetes readiness)
GET /api/v1/health/live - Liveness probe (Kubernetes liveness)
"""
from datetime import datetime, timezone
from fastapi import APIRouter, Response, Depends, HTTPException, status
import structlog
from api.health_checks import get_health_manager
from api.auth import get_current_user, CurrentUser

from db.models import SchedulerRun
from db.session import SessionLocal
from sqlalchemy import desc
router = APIRouter(prefix="/api/v1", tags=["health"])
logger = structlog.get_logger(__name__)


async def get_current_admin_user(
    current_user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    """Verify current user has admin role."""
    if getattr(current_user, "role", None) != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


@router.get(
    "/health",
    summary="Comprehensive health status",
    response_description="Full health check with all component details"
)
async def health_check() -> dict:
    """API health status consolidating database, queue/workers, and scheduler metrics."""
    manager = get_health_manager()
    report = await manager.deep_health_check(timeout=5)
    
    # Consolidate scheduler metrics
    db = SessionLocal()
    try:
        job_names = [name[0] for name in db.query(SchedulerRun.job_name).distinct().all()]
        scheduler_metrics = {}
        for job in job_names:
            run = (
                db.query(SchedulerRun)
                .filter(SchedulerRun.job_name == job)
                .order_by(desc(SchedulerRun.started_at))
                .first()
            )
            if run:
                scheduler_metrics[job] = {
                    "last_run_started_at": run.started_at.isoformat() if run.started_at else None,
                    "last_run_finished_at": run.finished_at.isoformat() if run.finished_at else None,
                    "sent_count": run.sent_count,
                    "status": run.status.value,
                }
        report["scheduler"] = {
            "status": "healthy" if all(r["status"] == "success" for r in scheduler_metrics.values()) else "degraded",
            "jobs": scheduler_metrics
        }
    except Exception as e:
        report["scheduler"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    finally:
        db.close()
        
    return report


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


@router.post(
    "/admin/circuit-breaker/reset",
    summary="Manually reset LLM circuit breaker",
    response_description="Circuit breaker reset status and metrics snapshot",
    tags=["admin"],
)
async def reset_circuit_breaker(
    admin_user: CurrentUser = Depends(get_current_admin_user),
) -> dict:
    """
    Manually reset the LLM circuit breaker to CLOSED state.
    Requires admin authentication.
    """
    from core.circuit_breaker import get_llm_circuit_breaker
    
    breaker = get_llm_circuit_breaker()
    result = breaker.reset()
    
    logger.info(
        "circuit_breaker_manual_reset_by_admin",
        admin_user_id=getattr(admin_user, "user_id", None),
        previous_state=result["previous_state"],
    )
    
    return result


@router.get(
    "/admin/circuit-breaker/status",
    summary="Get LLM circuit breaker status",
    response_description="Current circuit breaker state and metrics",
    tags=["admin"],
)
async def circuit_breaker_status(
    admin_user: CurrentUser = Depends(get_current_admin_user),
) -> dict:
    """
    Get current status of the LLM circuit breaker.
    Requires admin authentication.
    """
    from core.circuit_breaker import get_llm_circuit_breaker
    
    breaker = get_llm_circuit_breaker()
    return breaker.get_status()
