# Event Sourcing Architecture

## Overview

LegalAssist AI uses event sourcing as its core data pattern for case lifecycle management. Instead of storing the current state only, every state-changing operation is recorded as an immutable event in the event store. Projections read these events to materialize the current state for queries.

## Core Concepts

### Domain Events (core/domain_events.py)

Domain events represent significant occurrences in the system. Each event is immutable and timestamped.

```
Event Structure:
- event_id: UUID (unique identifier)
- event_type: str (e.g., "case.created", "deadline.assigned")
- aggregate_id: str (the entity this event belongs to)
- aggregate_type: str (e.g., "case", "deadline")
- data: dict (event-specific payload)
- metadata: dict (correlation ID, causation ID, actor info)
- timestamp: datetime (when the event occurred)
- version: int (monotonic sequence number per aggregate)
```

### Event Store (core/event_store.py)

The event store provides:

- `append_events(aggregate_id, events, expected_version)` — Atomically append events with optimistic concurrency control
- `read_events(aggregate_id)` — Read all events for an aggregate in order
- `read_events_by_type(event_type, after_timestamp)` — Query events by type for projections

### Projections (core/projections/)

Projections read events and build query-optimized views:

| Projection | File | Purpose |
|------------|------|---------|
| Case Detail | `projections/case_detail.py` | Current case state for API responses |
| Analytics | `projections/analytics.py` | Aggregated metrics for dashboards |
| Timeline | `projections/timeline.py` | Ordered event timeline for UI |

## Event Flow

```
User Action → Command Handler → Domain Event → Event Store → Projection Update → API Response
```

1. User performs an action (e.g., "create case")
2. Command handler validates and creates a domain event
3. Event is appended to the event store
4. Event bus publishes the event to subscribed projections
5. Projections update their materialized views
6. API query reads from projections

## Adding a New Event Type

1. Define the event class in `core/domain_events.py`:
```python
@dataclass
class CaseAmended(AbstractDomainEvent):
    field_name: str
    old_value: Any
    new_value: Any
```

2. Register the event type in the event registry
3. Update relevant projections to handle the new event
4. Add the command handler that creates the event
5. Write tests for the event flow

## Rebuilding Projections

To rebuild a projection from scratch:

```python
from core.event_store import EventStore
from core.projections.case_detail import CaseDetailProjection

store = EventStore()
projection = CaseDetailProjection()

for event in store.read_events_by_type("case.*"):
    projection.handle(event)
projection.save()
```

## Benefits

| Benefit | Description |
|---------|-------------|
| Complete audit trail | Every state change is recorded and immutable |
| Temporal queries | Query the state at any point in time |
| Debugging | Replay events to reproduce issues |
| Projection flexibility | Add new projections without migrating data |
| Concurrency | Optimistic locking prevents conflicting writes |

## Testing

Test event-sourced modules using the in-memory event store:

```python
from core.event_store import InMemoryEventStore

store = InMemoryEventStore()
store.append_events("case-1", [case_created_event], expected_version=0)
events = store.read_events("case-1")
assert len(events) == 1
```

Refer to `tests/test_event_sourcing.py` for comprehensive test examples.
