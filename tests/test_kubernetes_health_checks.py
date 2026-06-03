"""
Tests for Kubernetes health checks (liveness & readiness probes)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from api.health_checks import (
    HealthCheckResult,
    HealthCheckManager,
)


class TestHealthCheckResult:
    """Test health check result structure"""
    
    def test_healthy_result(self):
        """Test creating healthy result"""
        result = HealthCheckResult("database", True, "Connected")
        
        assert result.name == "database"
        assert result.healthy is True
        assert result.message == "Connected"
        assert result.to_dict() == {
            "name": "database",
            "status": "healthy",
            "message": "Connected"
        }
    
    def test_unhealthy_result(self):
        """Test creating unhealthy result"""
        result = HealthCheckResult("redis", False, "Connection refused")
        
        assert result.healthy is False
        assert result.to_dict()["status"] == "unhealthy"


class TestHealthCheckManager:
    """Test health check manager"""
    
    @pytest.mark.asyncio
    async def test_liveness_check(self):
        """Test liveness check always returns healthy"""
        manager = HealthCheckManager()
        result = await manager.liveness_check()
        
        assert result["alive"] is True
        assert "timestamp" in result
        assert result["service"] == "legal-assist-api"
    
    @pytest.mark.asyncio
    async def test_check_database_not_configured(self):
        """Test database check when not configured"""
        manager = HealthCheckManager(db_url=None)
        result = await manager.check_database()
        
        assert result.healthy is True
        assert "Not configured" in result.message
    
    @pytest.mark.asyncio
    async def test_check_redis_not_configured(self):
        """Test Redis check when not configured"""
        manager = HealthCheckManager(redis_url=None)
        result = await manager.check_redis()
        
        assert result.healthy is True
        assert "Not configured" in result.message
    
    @pytest.mark.asyncio
    async def test_check_celery_not_configured(self):
        """Test Celery check when not configured"""
        manager = HealthCheckManager(celery_broker_url=None)
        result = await manager.check_celery()
        
        assert result.healthy is True
        assert "Not configured" in result.message
    
    @pytest.mark.asyncio
    async def test_readiness_check_all_healthy(self):
        """Test readiness check when all dependencies healthy"""
        manager = HealthCheckManager()
        
        # Mock all checks to pass
        with patch.object(manager, 'check_database') as mock_db, \
             patch.object(manager, 'check_redis') as mock_redis, \
             patch.object(manager, 'check_celery') as mock_celery:
            
            mock_db.return_value = HealthCheckResult("database", True, "Connected")
            mock_redis.return_value = HealthCheckResult("redis", True, "Connected")
            mock_celery.return_value = HealthCheckResult("celery", True, "Ready")
            
            result = await manager.readiness_check(timeout=5)
            
            assert result["ready"] is True
            assert result["status_code"] == 200
            assert len(result["checks"]) == 3
            assert all(c["status"] == "healthy" for c in result["checks"])
    
    @pytest.mark.asyncio
    async def test_readiness_check_database_unhealthy(self):
        """Test readiness check when database is down"""
        manager = HealthCheckManager()
        
        with patch.object(manager, 'check_database') as mock_db, \
             patch.object(manager, 'check_redis') as mock_redis, \
             patch.object(manager, 'check_celery') as mock_celery:
            
            mock_db.return_value = HealthCheckResult("database", False, "Connection refused")
            mock_redis.return_value = HealthCheckResult("redis", True, "Connected")
            mock_celery.return_value = HealthCheckResult("celery", True, "Ready")
            
            result = await manager.readiness_check(timeout=5)
            
            assert result["ready"] is False
            assert result["status_code"] == 503
            assert result["checks"][0]["status"] == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_readiness_check_redis_unhealthy(self):
        """Test readiness check when Redis is down"""
        manager = HealthCheckManager()
        
        with patch.object(manager, 'check_database') as mock_db, \
             patch.object(manager, 'check_redis') as mock_redis, \
             patch.object(manager, 'check_celery') as mock_celery:
            
            mock_db.return_value = HealthCheckResult("database", True, "Connected")
            mock_redis.return_value = HealthCheckResult("redis", False, "Connection timeout")
            mock_celery.return_value = HealthCheckResult("celery", True, "Ready")
            
            result = await manager.readiness_check(timeout=5)
            
            assert result["ready"] is False
            assert result["status_code"] == 503
    
    @pytest.mark.asyncio
    async def test_readiness_check_celery_unhealthy(self):
        """Test readiness check when Celery broker is down"""
        manager = HealthCheckManager()
        
        with patch.object(manager, 'check_database') as mock_db, \
             patch.object(manager, 'check_redis') as mock_redis, \
             patch.object(manager, 'check_celery') as mock_celery:
            
            mock_db.return_value = HealthCheckResult("database", True, "Connected")
            mock_redis.return_value = HealthCheckResult("redis", True, "Connected")
            mock_celery.return_value = HealthCheckResult("celery", False, "Broker unreachable")
            
            result = await manager.readiness_check(timeout=5)
            
            assert result["ready"] is False
            assert result["status_code"] == 503
    
    @pytest.mark.asyncio
    async def test_deep_health_check(self):
        """Test comprehensive health check"""
        manager = HealthCheckManager()
        
        with patch.object(manager, 'readiness_check') as mock_readiness:
            mock_readiness.return_value = {
                "ready": True,
                "status_code": 200,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checks": [
                    {"name": "database", "status": "healthy", "message": "Connected"},
                    {"name": "redis", "status": "healthy", "message": "Connected"},
                    {"name": "celery", "status": "healthy", "message": "Ready"}
                ]
            }
            
            result = await manager.deep_health_check(timeout=5)
            
            assert result["status"] == "healthy"
            assert "timestamp" in result
            assert result["version"] == "1.0.0"
            assert result["checks"]["database"] == "healthy"
            assert result["checks"]["redis"] == "healthy"
            assert result["checks"]["celery"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_deep_health_check_degraded(self):
        """Test comprehensive health check when degraded"""
        manager = HealthCheckManager()
        
        with patch.object(manager, 'readiness_check') as mock_readiness:
            mock_readiness.return_value = {
                "ready": False,
                "status_code": 503,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checks": [
                    {"name": "database", "status": "unhealthy", "message": "Connection refused"},
                    {"name": "redis", "status": "healthy", "message": "Connected"},
                    {"name": "celery", "status": "healthy", "message": "Ready"}
                ]
            }
            
            result = await manager.deep_health_check(timeout=5)
            
            assert result["status"] == "degraded"
            assert result["checks"]["database"] == "unhealthy"


@pytest.mark.asyncio
async def test_health_manager_singleton():
    """Test health check manager singleton pattern"""
    from api.health_checks import get_health_manager
    
    manager1 = get_health_manager()
    manager2 = get_health_manager()
    
    assert manager1 is manager2


class TestHealthCheckIntegration:
    """Integration tests for health checks"""
    
    @pytest.mark.asyncio
    async def test_readiness_returns_503_on_error(self):
        """Test readiness check returns 503 status code on dependency error"""
        manager = HealthCheckManager()
        
        with patch.object(manager, 'check_database') as mock_db:
            mock_db.side_effect = Exception("Connection error")
            
            result = await manager.readiness_check(timeout=1)
            
            assert result["status_code"] == 503
            assert result["ready"] is False
    
    @pytest.mark.asyncio
    async def test_liveness_returns_200_always(self):
        """Test liveness check always returns 200 regardless of dependencies"""
        manager = HealthCheckManager()
        result = await manager.liveness_check()
        
        assert result["alive"] is True
        # Liveness doesn't depend on other services
