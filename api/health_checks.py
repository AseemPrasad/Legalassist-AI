"""
Comprehensive health checks for Kubernetes probes (liveness & readiness)
Verifies: Database, Redis, Celery, and API responsiveness
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
import redis.asyncio as redis

logger = structlog.get_logger(__name__)


class HealthCheckResult:
    """Result of a health check"""
    
    def __init__(self, name: str, healthy: bool, message: str = ""):
        self.name = name
        self.healthy = healthy
        self.message = message
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": "healthy" if self.healthy else "unhealthy",
            "message": self.message
        }


class HealthCheckManager:
    """Manages all health checks"""
    
    def __init__(
        self,
        db_url: Optional[str] = None,
        redis_url: Optional[str] = None,
        celery_broker_url: Optional[str] = None,
    ):
        self.db_url = db_url
        self.redis_url = redis_url
        self.celery_broker_url = celery_broker_url
    
    async def check_database(self, timeout: int = 5) -> HealthCheckResult:
        """Check database connectivity"""
        try:
            if not self.db_url:
                return HealthCheckResult("database", True, "Not configured")
            
            engine = create_async_engine(
                self.db_url,
                echo=False,
                pool_pre_ping=True,
                pool_size=1,
                max_overflow=0,
            )
            
            try:
                async with engine.begin() as conn:
                    try:
                        await asyncio.wait_for(
                            conn.execute(text("SELECT 1")),
                            timeout=timeout
                        )
                        logger.info("database_check_success")
                        return HealthCheckResult("database", True, "Connected")
                    except asyncio.TimeoutError:
                        logger.error("database_check_timeout")
                        return HealthCheckResult("database", False, "Query timeout")
            finally:
                await engine.dispose()
        
        except Exception as e:
            logger.error("database_check_failed", error=str(e))
            return HealthCheckResult("database", False, f"Error: {str(e)[:50]}")
    
    async def check_redis(self, timeout: int = 5) -> HealthCheckResult:
        """Check Redis connectivity"""
        try:
            if not self.redis_url:
                return HealthCheckResult("redis", True, "Not configured")
            
            redis_client = redis.from_url(
                self.redis_url,
                socket_connect_timeout=timeout,
                socket_keepalive=True,
            )
            
            try:
                pong = await asyncio.wait_for(
                    redis_client.ping(),
                    timeout=timeout
                )
                if pong:
                    logger.info("redis_check_success")
                    return HealthCheckResult("redis", True, "Connected")
                else:
                    logger.warning("redis_check_failed_pong")
                    return HealthCheckResult("redis", False, "No PONG")
            except asyncio.TimeoutError:
                logger.error("redis_check_timeout")
                return HealthCheckResult("redis", False, "Connection timeout")
            finally:
                await redis_client.close()
        
        except Exception as e:
            logger.error("redis_check_failed", error=str(e))
            return HealthCheckResult("redis", False, f"Error: {str(e)[:50]}")
    
    async def check_celery(self, timeout: int = 5) -> HealthCheckResult:
        """Check Celery connectivity"""
        try:
            if not self.celery_broker_url:
                return HealthCheckResult("celery", True, "Not configured")
            
            # Try to inspect celery app stats
            from celery_app import app as celery_app
            
            try:
                # Attempt to get active tasks (non-blocking with timeout)
                inspector = celery_app.control.inspect(timeout=timeout)
                if inspector is None:
                    logger.warning("celery_check_inspector_none")
                    return HealthCheckResult("celery", False, "Inspector unavailable")
                
                # This should timeout if broker is unreachable
                stats = await asyncio.wait_for(
                    asyncio.to_thread(lambda: inspector.stats()),
                    timeout=timeout
                )
                
                if stats:
                    logger.info("celery_check_success", workers=len(stats))
                    return HealthCheckResult("celery", True, f"{len(stats)} workers")
                else:
                    logger.warning("celery_check_no_workers")
                    # Workers might be down but broker is up
                    return HealthCheckResult("celery", True, "No workers (broker ok)")
            
            except (asyncio.TimeoutError, TimeoutError):
                logger.error("celery_check_timeout")
                return HealthCheckResult("celery", False, "Broker timeout")
        
        except Exception as e:
            logger.error("celery_check_failed", error=str(e))
            return HealthCheckResult("celery", False, f"Error: {str(e)[:50]}")
    
    async def liveness_check(self) -> Dict[str, Any]:
        """
        Liveness probe: checks if service should be restarted
        - Returns 200 if service is running (regardless of dependency state)
        - Restarts container if it crashes
        """
        return {
            "alive": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "legal-assist-api"
        }
    
    async def readiness_check(self, timeout: int = 5) -> Dict[str, Any]:
        """
        Readiness probe: checks if service can handle traffic
        - Returns 200 only if ALL dependencies are healthy
        - Kubernetes removes from load balancer if unhealthy
        """
        checks = await asyncio.gather(
            self.check_database(timeout=timeout),
            self.check_redis(timeout=timeout),
            self.check_celery(timeout=timeout),
            return_exceptions=True
        )
        
        results = []
        all_healthy = True
        
        for check_result in checks:
            if isinstance(check_result, Exception):
                logger.error("readiness_check_exception", error=str(check_result))
                all_healthy = False
            elif isinstance(check_result, HealthCheckResult):
                results.append(check_result.to_dict())
                if not check_result.healthy:
                    all_healthy = False
        
        status_code = 200 if all_healthy else 503
        
        return {
            "ready": all_healthy,
            "status_code": status_code,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": results
        }
    
    async def deep_health_check(self, timeout: int = 5) -> Dict[str, Any]:
        """
        Comprehensive health check with all details
        For manual inspection and debugging
        """
        readiness_result = await self.readiness_check(timeout=timeout)
        
        return {
            "status": "healthy" if readiness_result["ready"] else "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "readiness": readiness_result,
            "checks": {
                check["name"]: check["status"]
                for check in readiness_result["checks"]
            }
        }


# Singleton instance
_health_manager: Optional[HealthCheckManager] = None


def get_health_manager() -> HealthCheckManager:
    """Get or create singleton health check manager"""
    global _health_manager
    if _health_manager is None:
        from api.config import get_settings
        settings = get_settings()
        _health_manager = HealthCheckManager(
            db_url=settings.DATABASE_URL,
            redis_url=settings.REDIS_URL if hasattr(settings, 'REDIS_URL') else None,
            celery_broker_url=settings.CELERY_BROKER_URL if hasattr(settings, 'CELERY_BROKER_URL') else None,
        )
    return _health_manager
