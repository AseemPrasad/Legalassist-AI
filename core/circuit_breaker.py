"""
Circuit Breaker for LLM API client.
Redis-backed shared state for multi-worker coordination.
"""
import time
import logging
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

from config import Config

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerMetrics:
    state_changes: int = 0
    rejected_requests: int = 0
    failed_requests: int = 0
    successful_requests: int = 0


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is OPEN and request is rejected."""
    pass


class CircuitBreaker:
    """
    Redis-backed circuit breaker for LLM API calls.
    
    States:
    - CLOSED: normal operation, requests pass through
    - OPEN: failure threshold exceeded, all requests fast-fail
    - HALF_OPEN: after cooldown, allows single probe to test recovery
    """
    
    def __init__(
        self,
        name: str = "llm_circuit_breaker",
        failure_threshold: float = 0.5,
        window_seconds: int = 30,
        cooldown_seconds: int = 60,
        redis_url: Optional[str] = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.metrics = CircuitBreakerMetrics()
        
        self._redis_url = redis_url or getattr(Config, "REDIS_URL", "") or getattr(Config, "REDIS_URL", "")
        self._redis: Optional[Any] = None
        self._local_only = False
        
        if _REDIS_AVAILABLE and self._redis_url:
            try:
                self._redis = redis.from_url(self._redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("circuit_breaker_redis_connected", name=name)
            except Exception as e:
                logger.warning("circuit_breaker_redis_unavailable", name=name, error=str(e))
                self._local_only = True
        else:
            self._local_only = True
            logger.warning("circuit_breaker_redis_not_configured", name=name)
    
    # Redis key helpers
    def _key(self, suffix: str) -> str:
        return f"circuit_breaker:{self.name}:{suffix}"
    
    def _get_state(self) -> CircuitState:
        """Get current circuit state from Redis or local fallback."""
        if self._redis:
            state = self._redis.get(self._key("state"))
            if state:
                return CircuitState(state)
        return CircuitState.CLOSED
    
    def _set_state(self, state: CircuitState) -> None:
        """Set circuit state in Redis with TTL."""
        if self._redis:
            self._redis.setex(
                self._key("state"),
                self.cooldown_seconds + self.window_seconds,
                state.value,
            )
        self.metrics.state_changes += 1
        logger.info(
            "circuit_breaker_state_change",
            name=self.name,
            new_state=state.value,
            total_state_changes=self.metrics.state_changes,
        )
    
    def _record_failure(self) -> None:
        """Record a failure in the sliding window."""
        now = time.time()
        if self._redis:
            key = self._key("failures")
            self._redis.zadd(key, {str(now): now})
            self._redis.expire(key, self.window_seconds)
    
    def _record_success(self) -> None:
        """Record a success in the sliding window."""
        now = time.time()
        if self._redis:
            key = self._key("successes")
            self._redis.zadd(key, {str(now): now})
            self._redis.expire(key, self.window_seconds)
        self.metrics.successful_requests += 1
    
    def _get_failure_rate(self) -> float:
        """Calculate failure rate in the current window."""
        if not self._redis:
            return 0.0
        
        now = time.time()
        window_start = now - self.window_seconds
        
        failures = self._redis.zcount(self._key("failures"), window_start, now)
        successes = self._redis.zcount(self._key("successes"), window_start, now)
        total = failures + successes
        
        if total == 0:
            return 0.0
        
        return failures / total
    
    def _get_last_opened(self) -> Optional[float]:
        """Get timestamp when circuit was last opened."""
        if self._redis:
            ts = self._redis.get(self._key("last_opened"))
            if ts:
                return float(ts)
        return None
    
    def _set_last_opened(self) -> None:
        """Record timestamp when circuit opened."""
        if self._redis:
            self._redis.setex(
                self._key("last_opened"),
                self.cooldown_seconds,
                str(time.time()),
            )
    
    def _is_cooldown_elapsed(self) -> bool:
        """Check if cooldown period has elapsed since circuit opened."""
        last_opened = self._get_last_opened()
        if not last_opened:
            return True
        return (time.time() - last_opened) >= self.cooldown_seconds
    
    def _get_probe_lock(self) -> bool:
        """Acquire lock for HALF_OPEN probe request."""
        if not self._redis:
            return True
        # Use NX to ensure only one worker gets the probe
        acquired = self._redis.set(
            self._key("probe_lock"),
            str(time.time()),
            nx=True,
            ex=self.cooldown_seconds,
        )
        return bool(acquired)
    
    def _release_probe_lock(self) -> None:
        """Release probe lock."""
        if self._redis:
            self._redis.delete(self._key("probe_lock"))
    
    def can_execute(self) -> bool:
        """
        Check if request can execute. Returns True if allowed, False if rejected.
        Handles state transitions: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.
        """
        state = self._get_state()
        
        if state == CircuitState.CLOSED:
            return True
        
        if state == CircuitState.OPEN:
            if self._is_cooldown_elapsed():
                # Transition to HALF_OPEN
                self._set_state(CircuitState.HALF_OPEN)
                return self._get_probe_lock()
            else:
                self.metrics.rejected_requests += 1
                logger.info(
                    "circuit_breaker_rejected",
                    name=self.name,
                    state=state.value,
                    rejected_count=self.metrics.rejected_requests,
                )
                return False
        
        if state == CircuitState.HALF_OPEN:
            # Only allow if we have the probe lock
            return self._get_probe_lock()
        
        return True
    
    def record_success(self) -> None:
        """Record successful request and handle state transitions."""
        self._record_success()
        state = self._get_state()
        
        if state == CircuitState.HALF_OPEN:
            # Probe succeeded, close the circuit
            self._set_state(CircuitState.CLOSED)
            self._release_probe_lock()
            logger.info("circuit_breaker_closed_after_probe", name=self.name)
    
    def record_failure(self) -> None:
        """Record failed request and handle state transitions."""
        self._record_failure()
        self.metrics.failed_requests += 1
        
        state = self._get_state()
        
        if state == CircuitState.HALF_OPEN:
            # Probe failed, re-open immediately
            self._set_state(CircuitState.OPEN)
            self._set_last_opened()
            self._release_probe_lock()
            logger.warning("circuit_breaker_reopened_after_probe_failure", name=self.name)
            return
        
        if state == CircuitState.CLOSED:
            failure_rate = self._get_failure_rate()
            if failure_rate >= self.failure_threshold:
                self._set_state(CircuitState.OPEN)
                self._set_last_opened()
                logger.warning(
                    "circuit_breaker_opened",
                    name=self.name,
                    failure_rate=failure_rate,
                    threshold=self.failure_threshold,
                )
    
    def reset(self) -> Dict[str, Any]:
        """Manually reset circuit breaker to CLOSED state. Returns current metrics."""
        current_state = self._get_state()
        
        if self._redis:
            pipe = self._redis.pipeline()
            pipe.delete(self._key("state"))
            pipe.delete(self._key("failures"))
            pipe.delete(self._key("successes"))
            pipe.delete(self._key("last_opened"))
            pipe.delete(self._key("probe_lock"))
            pipe.execute()
        
        self._set_state(CircuitState.CLOSED)
        
        metrics_snapshot = {
            "state_changes": self.metrics.state_changes,
            "rejected_requests": self.metrics.rejected_requests,
            "failed_requests": self.metrics.failed_requests,
            "successful_requests": self.metrics.successful_requests,
        }
        
        # Reset local metrics
        self.metrics = CircuitBreakerMetrics()
        
        logger.info("circuit_breaker_manual_reset", name=self.name, previous_state=current_state.value)
        
        return {
            "status": "reset",
            "previous_state": current_state.value,
            "new_state": CircuitState.CLOSED.value,
            "metrics": metrics_snapshot,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        state = self._get_state()
        failure_rate = self._get_failure_rate()
        
        return {
            "name": self.name,
            "state": state.value,
            "failure_rate": failure_rate,
            "failure_threshold": self.failure_threshold,
            "window_seconds": self.window_seconds,
            "cooldown_seconds": self.cooldown_seconds,
            "metrics": {
                "state_changes": self.metrics.state_changes,
                "rejected_requests": self.metrics.rejected_requests,
                "failed_requests": self.metrics.failed_requests,
                "successful_requests": self.metrics.successful_requests,
            },
        }


# Singleton instance for LLM circuit breaker
_llm_circuit_breaker: Optional[CircuitBreaker] = None


def get_llm_circuit_breaker() -> CircuitBreaker:
    """Get or create the singleton LLM circuit breaker instance."""
    global _llm_circuit_breaker
    if _llm_circuit_breaker is None:
        _llm_circuit_breaker = CircuitBreaker(
            name="llm_openrouter",
            failure_threshold=0.5,
            window_seconds=30,
            cooldown_seconds=60,
        )
    return _llm_circuit_breaker