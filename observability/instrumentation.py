"""
Observability instrumentation for Legalassist-AI
Includes Prometheus metrics, structured logging, and distributed tracing
"""

import logging
import time
import os
from functools import wraps
from typing import Callable, Any
from datetime import datetime
from contextlib import contextmanager
from contextvars import ContextVar
import uuid

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, multiprocess, generate_latest
from prometheus_client import start_http_server
import structlog
try:
    from jaeger_client import Config
except ModuleNotFoundError:  # pragma: no cover - optional in some environments
    Config = None

try:
    from opentelemetry import trace, metrics
    from opentelemetry.context import attach as otel_attach, detach as otel_detach
    from opentelemetry.propagate import extract as otel_extract, inject as otel_inject
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.instrumentation.flask import FlaskInstrumentor
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except Exception:
        FastAPIInstrumentor = None
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    try:
        from opentelemetry.instrumentation.celery import CeleryInstrumentor
    except Exception:
        CeleryInstrumentor = None
except ModuleNotFoundError:  # pragma: no cover - optional in some environments
    trace = None
    metrics = None
    otel_attach = None
    otel_detach = None
    otel_extract = None
    otel_inject = None
    PrometheusMetricReader = None
    JaegerExporter = None
    TracerProvider = None
    BatchSpanProcessor = None
    SERVICE_NAME = None
    Resource = None
    MeterProvider = None
    FlaskInstrumentor = None
    RequestsInstrumentor = None
    SQLAlchemyInstrumentor = None
    RedisInstrumentor = None

# ==================== Prometheus Metrics ====================
registry = CollectorRegistry()
_observability_initialized = False
_sentry_initialized = False
_REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)
_USER_ID: ContextVar[str | None] = ContextVar("user_id", default=None)
_SESSION_ID: ContextVar[str | None] = ContextVar("session_id", default=None)

# HTTP Metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    registry=registry
)

http_requests_queued = Gauge(
    'http_requests_queued',
    'Number of queued HTTP requests',
    registry=registry
)

# LLM Metrics
llm_tokens_used_total = Counter(
    'llm_tokens_used_total',
    'Total tokens used in LLM calls',
    ['model', 'type'],  # type: prompt, completion
    registry=registry
)

llm_api_calls_total = Counter(
    'llm_api_calls_total',
    'Total LLM API calls',
    ['model', 'status'],
    registry=registry
)

llm_api_call_duration_seconds = Histogram(
    'llm_api_call_duration_seconds',
    'LLM API call duration in seconds',
    ['model'],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    registry=registry
)

llm_api_costs_total = Gauge(
    'llm_api_costs_total',
    'Total cumulative API costs in dollars',
    ['model'],
    registry=registry
)

# Document Processing Metrics
document_processing_total = Counter(
    'document_processing_total',
    'Total documents processed',
    ['document_type', 'status'],
    registry=registry
)

document_processing_duration_seconds = Histogram(
    'document_processing_duration_seconds',
    'Document processing duration in seconds',
    ['document_type'],
    buckets=(0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
    registry=registry
)

document_processing_failures_total = Counter(
    'document_processing_failures_total',
    'Failed document processing attempts',
    ['document_type', 'reason'],
    registry=registry
)

# PDF Export Metrics
pdf_export_total = Counter(
    'pdf_export_total',
    'Total PDF exports',
    ['status'],
    registry=registry
)

pdf_export_duration_seconds = Histogram(
    'pdf_export_duration_seconds',
    'PDF export duration in seconds',
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0),
    registry=registry
)

pdf_export_failures_total = Counter(
    'pdf_export_failures_total',
    'Failed PDF exports',
    ['reason'],
    registry=registry
)

# Authentication Metrics
auth_attempts_total = Counter(
    'auth_attempts_total',
    'Total authentication attempts',
    ['status'],
    registry=registry
)

auth_failures_total = Counter(
    'auth_failures_total',
    'Failed authentication attempts',
    ['reason'],
    registry=registry
)

# Database Metrics
db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation', 'table'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0),
    registry=registry
)

db_connection_pool_size = Gauge(
    'db_connection_pool_size',
    'Database connection pool size',
    registry=registry
)

db_active_connections = Gauge(
    'db_active_connections',
    'Number of active database connections',
    registry=registry
)

# Cache Metrics
cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type'],
    registry=registry
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type'],
    registry=registry
)

cache_operation_duration_seconds = Histogram(
    'cache_operation_duration_seconds',
    'Cache operation duration in seconds',
    ['operation', 'cache_type'],
    buckets=(0.001, 0.01, 0.05, 0.1),
    registry=registry
)

# Business Metrics
active_cases = Gauge(
    'active_cases_total',
    'Total active cases',
    registry=registry
)

pending_deadlines = Gauge(
    'pending_deadlines_total',
    'Total pending deadlines',
    registry=registry
)

user_sessions_active = Gauge(
    'user_sessions_active',
    'Currently active user sessions',
    registry=registry
)

async_jobs_pending = Gauge(
    'async_jobs_pending',
    'Number of pending async jobs',
    registry=registry
)

api_errors_total = Counter(
    'api_errors_total',
    'Total API errors',
    ['endpoint', 'error_type'],
    registry=registry
)

feature_flag_events_total = Counter(
    'feature_flag_events_total',
    'Feature flag lifecycle events',
    ['event', 'flag_name', 'surface', 'variant'],
    registry=registry
)


# ==================== Structured Logging ====================
def setup_structured_logging():
    """Configure structlog for JSON-formatted logging with correlation IDs"""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=None,
        level=logging.INFO,
    )


# Get structlog logger
log = structlog.get_logger()


# ==================== Distributed Tracing ====================
def setup_jaeger_tracing(service_name: str = "legalassist-ai"):
    """Initialize Jaeger distributed tracing"""
    if trace is None or JaegerExporter is None or TracerProvider is None or BatchSpanProcessor is None or Resource is None:
        return None

    jaeger_exporter = JaegerExporter(
        agent_host_name=os.getenv("JAEGER_AGENT_HOST", "localhost"),
        agent_port=int(os.getenv("JAEGER_AGENT_PORT", "6831")),
    )
    
    resource = Resource.create({SERVICE_NAME: service_name})
    jaeger_provider = TracerProvider(resource=resource)
    jaeger_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
    trace.set_tracer_provider(jaeger_provider)
    
    return trace.get_tracer(__name__)


# Get tracer instance
tracer = setup_jaeger_tracing()


# ==================== Context Management ====================
class CorrelationContext:
    """Thread-local correlation context for request tracing"""
    def __init__(self):
        self.correlation_id = None
        self.user_id = None
        self.session_id = None
    
    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get(self):
        return {
            'correlation_id': self.correlation_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
        }


correlation_context = CorrelationContext()


def generate_correlation_id() -> str:
    """Generate W3C-compatible trace ID for request tracing.
    
    Returns a 32-hex-character trace ID suitable for traceparent header.
    """
    return uuid.uuid4().hex + uuid.uuid4().hex[:16]


def bind_request_context(*, request_id: str | None = None, user_id: str | None = None, session_id: str | None = None):
    """Bind request-scoped context for logs and traces."""
    if request_id is not None:
        _REQUEST_ID.set(request_id)
        correlation_context.correlation_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id, correlation_id=request_id)
    if user_id is not None:
        _USER_ID.set(user_id)
        structlog.contextvars.bind_contextvars(user_id=user_id)
    if session_id is not None:
        _SESSION_ID.set(session_id)
        structlog.contextvars.bind_contextvars(session_id=session_id)


def get_request_context() -> dict:
    return {
        "request_id": _REQUEST_ID.get(),
        "user_id": _USER_ID.get(),
        "session_id": _SESSION_ID.get(),
    }


def clear_request_context():
    _REQUEST_ID.set(None)
    _USER_ID.set(None)
    _SESSION_ID.set(None)
    correlation_context.correlation_id = None
    correlation_context.user_id = None
    correlation_context.session_id = None
    structlog.contextvars.clear_contextvars()


def get_current_trace_headers() -> dict[str, str]:
    """Inject the active trace context into HTTP-style headers."""
    if otel_inject is None:
        return {}

    carrier: dict[str, str] = {}
    try:
        otel_inject(carrier)
    except Exception:
        return {}

    return {key.lower(): value for key, value in carrier.items() if key.lower() in {"traceparent", "tracestate", "baggage"}}


@contextmanager
def use_extracted_trace_context(carrier: dict[str, str] | None):
    """Attach an incoming trace context for the lifetime of the request."""
    if otel_extract is None or otel_attach is None or otel_detach is None:
        yield None
        return

    token = None
    try:
        ctx = otel_extract(carrier or {})
        token = otel_attach(ctx)
        yield ctx
    finally:
        if token is not None:
            otel_detach(token)


def record_api_error(endpoint: str, error: Exception | str):
    error_type = error if isinstance(error, str) else type(error).__name__
    api_errors_total.labels(endpoint=endpoint, error_type=error_type).inc()


def set_queue_depth(pending_jobs: int):
    async_jobs_pending.set(max(0, pending_jobs))


def capture_exception(error: Exception, **context):
    """Capture an exception in Sentry if available, and always log it."""
    extra_context = {**get_request_context(), **context}
    log.error("exception_captured", error=str(error), **extra_context)

    try:
        import sentry_sdk  # type: ignore
    except ModuleNotFoundError:
        return

    if sentry_sdk.Hub.current.client is not None:
        sentry_sdk.capture_exception(error)


def setup_sentry_tracing():
    """Initialize Sentry if configured."""
    global _sentry_initialized
    if _sentry_initialized:
        return

    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        return

    try:
        import sentry_sdk  # type: ignore
    except ModuleNotFoundError:
        log.warning("sentry_sdk_not_installed", configured=True)
        return

    sentry_sdk.init(
        dsn=dsn,
        environment=os.getenv("ENVIRONMENT", "development"),
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
        send_default_pii=False,
    )
    _sentry_initialized = True


@contextmanager
def traced_operation(operation_name: str, attributes: dict = None):
    """Context manager for distributed tracing of operations"""
    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span(operation_name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            raise


# ==================== Decorators ====================
def track_http_request(endpoint: str = None):
    """Decorator to track HTTP request metrics"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            ep = endpoint or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                http_requests_total.labels(
                    method="POST",
                    endpoint=ep,
                    status="success"
                ).inc()
                return result
            except Exception as e:
                http_requests_total.labels(
                    method="POST",
                    endpoint=ep,
                    status="error"
                ).inc()
                log.error(
                    "http_request_failed",
                    endpoint=ep,
                    error=str(e),
                    correlation_id=correlation_context.correlation_id
                )
                raise
            finally:
                duration = time.time() - start_time
                http_request_duration_seconds.labels(
                    method="POST",
                    endpoint=ep
                ).observe(duration)
        
        return wrapper
    return decorator


def track_llm_call(model: str):
    """Decorator to track LLM API calls and token usage"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            with traced_operation(f"llm_call_{model}", {"model": model}):
                try:
                    result = func(*args, **kwargs)
                    
                    # Extract token usage if available in result
                    if isinstance(result, dict) and 'usage' in result:
                        usage = result['usage']
                        llm_tokens_used_total.labels(
                            model=model,
                            type="prompt"
                        ).inc(usage.get('prompt_tokens', 0))
                        llm_tokens_used_total.labels(
                            model=model,
                            type="completion"
                        ).inc(usage.get('completion_tokens', 0))
                    
                    llm_api_calls_total.labels(
                        model=model,
                        status="success"
                    ).inc()
                    
                    return result
                except Exception as e:
                    llm_api_calls_total.labels(
                        model=model,
                        status="error"
                    ).inc()
                    log.error(
                        "llm_call_failed",
                        model=model,
                        error=str(e),
                        correlation_id=correlation_context.correlation_id
                    )
                    raise
                finally:
                    duration = time.time() - start_time
                    llm_api_call_duration_seconds.labels(model=model).observe(duration)
        
        return wrapper
    return decorator


def track_document_processing(doc_type: str):
    """Decorator to track document processing metrics"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            with traced_operation(f"document_processing_{doc_type}", {"document_type": doc_type}):
                try:
                    result = func(*args, **kwargs)
                    document_processing_total.labels(
                        document_type=doc_type,
                        status="success"
                    ).inc()
                    return result
                except Exception as e:
                    reason = type(e).__name__
                    document_processing_failures_total.labels(
                        document_type=doc_type,
                        reason=reason
                    ).inc()
                    log.error(
                        "document_processing_failed",
                        document_type=doc_type,
                        error=str(e),
                        correlation_id=correlation_context.correlation_id
                    )
                    raise
                finally:
                    duration = time.time() - start_time
                    document_processing_duration_seconds.labels(
                        document_type=doc_type
                    ).observe(duration)
        
        return wrapper
    return decorator


def track_database_operation(operation: str, table: str):
    """Decorator to track database operation metrics"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                db_query_duration_seconds.labels(
                    operation=operation,
                    table=table
                ).observe(duration)
        
        return wrapper
    return decorator


def observe_request(endpoint: str, method: str, status_code: int, duration_seconds: float):
    """Record standardized request metrics."""
    http_requests_total.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
    http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration_seconds)


def observe_business_metrics(*, active_cases_count: int | None = None, pending_deadlines_count: int | None = None, active_users_count: int | None = None):
    if active_cases_count is not None:
        active_cases.set(max(0, active_cases_count))
    if pending_deadlines_count is not None:
        pending_deadlines.set(max(0, pending_deadlines_count))
    if active_users_count is not None:
        user_sessions_active.set(max(0, active_users_count))


def record_feature_flag_event(event: str, flag_name: str, *, surface: str = "api", variant: str = "control"):
    """Record feature flag exposure or usage for rollout analysis."""
    feature_flag_events_total.labels(
        event=event,
        flag_name=flag_name,
        surface=surface,
        variant=variant,
    ).inc()
    log.info(
        "feature_flag_event",
        event=event,
        flag_name=flag_name,
        surface=surface,
        variant=variant,
    )


# ==================== Metrics Endpoint ====================
def get_metrics():
    """Get Prometheus metrics in text format"""
    # Handle multiprocess mode if in use
    if os.environ.get('prometheus_multiproc_dir'):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry)
    return generate_latest(registry)


# ==================== Initialization ====================
def initialize_observability():
    """Initialize all observability components"""
    global _observability_initialized
    if _observability_initialized:
        return

    # Setup structured logging
    setup_structured_logging()

    setup_sentry_tracing()
    
    # Setup distributed tracing
    global tracer
    tracer = setup_jaeger_tracing()
    # Auto-instrument common libraries when opentelemetry instrumentations are available
    try:
        if RequestsInstrumentor is not None:
            RequestsInstrumentor().instrument()
            log.info("requests_instrumented")
    except Exception as e:
        log.warning("requests_instrumentation_failed", error=str(e))

    try:
        if SQLAlchemyInstrumentor is not None:
            SQLAlchemyInstrumentor().instrument()
            log.info("sqlalchemy_instrumented")
    except Exception as e:
        log.warning("sqlalchemy_instrumentation_failed", error=str(e))

    try:
        if RedisInstrumentor is not None:
            RedisInstrumentor().instrument()
            log.info("redis_instrumented")
    except Exception as e:
        log.warning("redis_instrumentation_failed", error=str(e))

    try:
        if CeleryInstrumentor is not None:
            CeleryInstrumentor().instrument()
            log.info("celery_instrumented")
    except Exception as e:
        log.warning("celery_instrumentation_failed", error=str(e))
    
    # Start Prometheus metrics server
    metrics_port = int(os.getenv("PROMETHEUS_METRICS_PORT", "9090"))
    try:
        start_http_server(metrics_port)
        log.info("prometheus_metrics_started", port=metrics_port)
    except OSError as e:
        log.warning("prometheus_metrics_already_running", error=str(e))
    
    log.info("observability_initialized")
    _observability_initialized = True
