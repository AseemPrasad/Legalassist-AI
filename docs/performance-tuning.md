# Performance Tuning Guide

## Database Performance

### Connection Pooling

The application uses SQLAlchemy with connection pooling. Tune pool settings in `db/session.py`:

```python
engine = create_engine(
    DATABASE_URL,
    pool_size=10,           # Default: 5. Increase for higher concurrency.
    max_overflow=20,        # Default: 10. Extra connections beyond pool_size.
    pool_timeout=30,        # Seconds to wait for a connection before error.
    pool_pre_ping=True,     # Verify connections before use (recommended).
    pool_recycle=3600,      # Recycle connections after 1 hour.
)
```

**Recommendations by deployment size:**

| Scale | Users | pool_size | max_overflow |
|-------|-------|-----------|--------------|
| Small | < 50 | 5 | 10 |
| Medium | 50-500 | 20 | 30 |
| Large | > 500 | 50 | 50 |

### Indexing Strategy

Ensure these indexes are present for common query patterns:

```sql
-- High-traffic query indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_cases_user_id ON cases(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_cases_updated_at ON cases(updated_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_deadlines_due_date ON case_deadlines(due_date);
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_notifications_user_id ON notification_logs(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_audit_log_timestamp ON audit_logs(timestamp DESC);
```

### Query Optimization

- Use `selectinload()` for eager loading of relationships (avoids N+1 queries)
- Use `contains_eager()` when filtering on joined tables
- Use `with_expression()` for computed columns
- Avoid calling `.all()` before filtering (filter first, then paginate)

```python
# Good: Filter in SQL, eager load relationships
cases = (
    db.query(Case)
    .options(selectinload(Case.deadlines))
    .filter(Case.user_id == user_id)
    .offset(offset)
    .limit(limit)
    .all()
)
```

## Celery Worker Tuning

### Worker Configuration

```python
# In celery_app.py or Celery config
celery_app.conf.update(
    worker_concurrency=4,           # Number of worker processes. Default: CPU count.
    worker_prefetch_multiplier=1,   # Process one task at a time per worker.
    task_acks_late=True,            # Re-deliver task if worker crashes.
    task_reject_on_worker_lost=True, # Reject unacknowledged tasks.
    task_soft_time_limit=300,       # Soft time limit (5 minutes).
    task_time_limit=600,            # Hard time limit (10 minutes).
)
```

**Recommendations:**

| Workload | concurrency | prefetch_multiplier |
|----------|------------|-------------------|
| CPU-bound | CPU cores | 1 |
| I/O-bound | CPU cores * 2 | 4 |
| Mixed | CPU cores | 2 |

### Task Routing

Route tasks to dedicated queues:

```python
celery_app.conf.task_routes = {
    'tasks.analyze_document': {'queue': 'cpu-heavy'},
    'tasks.send_notification': {'queue': 'io-fast'},
    'tasks.generate_report': {'queue': 'io-slow'},
}
```

## Redis Caching

### Memory Management

Configure Redis maxmemory policy:

```conf
# In redis.conf or docker-compose
maxmemory 2gb
maxmemory-policy allkeys-lru
```

### Cache Strategies

| Data Type | Strategy | TTL |
|-----------|----------|-----|
| Case list | Cache-aside | 5 minutes |
| Analytics | Write-through | 15 minutes |
| User session | Write-behind | 24 hours |
| Search results | Cache-aside | 10 minutes |

## Vector Store Optimization

- Use batch insertion (100-500 vectors per batch)
- Set `ef_construction` between 100-200 for HNSW index
- Set `M` between 16-32 for memory/recall tradeoff
- Monitor shard sizes in `vector_shards/` directory

## Monitoring

### Prometheus Metrics

Key metrics to monitor via `observability/`:

| Metric | What It Tells You | Alert Threshold |
|--------|-------------------|-----------------|
| `http_request_duration_seconds` | API latency | p99 > 2s |
| `db_pool_available_connections` | Connection pressure | < 3 |
| `celery_queue_depth` | Task backlog | > 100 |
| `redis_memory_used_bytes` | Cache memory | > 80% of maxmemory |

### Quick Health Check

```bash
# API health
curl http://localhost:8000/health

# Database connectivity
curl http://localhost:8000/health/db

# Celery worker status
celery -A celery_app status
```

## Common Bottlenecks

1. **N+1 queries**: Use SQLAlchemy's `joinedload()` or `selectinload()`
2. **Missing indexes**: Add indexes for filtered/joined columns
3. **Large payloads**: Implement pagination for list endpoints
4. **Sync I/O in async endpoints**: Use thread pool executor or switch to async drivers
5. **Unbounded result sets**: Always use `.limit()` on queries
