"""Timeline event creation and aggregation helpers."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from core.time_serialization import to_utc_iso
from core.timeline_payloads import TimelineEventPayload
from sqlalchemy.orm import Session

from db.models import CaseDocument, CaseDeadline, CaseTimeline, NotificationLog, NotificationStatus
from services.timeline_realtime import publish_timeline_event_best_effort


class TimelineService:
    def create_event(
        self,
        db: Session,
        case_id: int,
        event_type: str,
        description: str,
        metadata: Optional[dict] = None,
        event_date=None,
    ) -> CaseTimeline:
        event = CaseTimeline(
            case_id=case_id,
            event_type=event_type,
            description=description,
            event_date=event_date or dt.datetime.now(dt.timezone.utc),
            event_metadata=metadata,
        )
        db.add(event)
        db.commit()
        db.refresh(event)

        # Publish realtime update to connected websocket clients (case-scoped)
        # Fire-and-forget: timeline_realtime_bus is in-memory, so async scheduling
        # keeps DB write latency low.
        payload = TimelineEventPayload(
            schema_version=TimelineEventPayload.CURRENT_SCHEMA_VERSION,
            type="timeline_event",
            case_id=case_id,
            event_type=event.event_type,
            description=event.description,
            timestamp=event.event_date,
            metadata=event.event_metadata or {},
            event_id=event.id,
        )
        publish_timeline_event_best_effort(payload.model_dump(mode="json"))

        return event

    def get_case_timeline(self, db: Session, case_id: int) -> List[CaseTimeline]:
        return db.query(CaseTimeline).filter(CaseTimeline.case_id == case_id).order_by(CaseTimeline.event_date.desc()).all()

    def get_case_timeline_events(self, db: Session, case_id: int) -> List[Dict[str, Any]]:
        events = self.get_case_timeline(db, case_id)
        return [
            {
                "id": e.id,
                "event_type": e.event_type,
                "event_date": to_utc_iso(e.event_date),
                "description": e.description,
                "metadata": e.event_metadata,
            }
            for e in events
        ]

    def get_case_full_timeline(self, db: Session, case_id: int) -> List[Dict[str, Any]]:
        timelines = db.query(CaseTimeline).filter(CaseTimeline.case_id == case_id).all()
        documents = db.query(CaseDocument).filter(CaseDocument.case_id == case_id).all()
        deadlines = db.query(CaseDeadline).filter(CaseDeadline.case_id == case_id).all()

        notifications = []
        if deadlines:
            deadline_ids = [d.id for d in deadlines]
            notifications = db.query(NotificationLog).filter(NotificationLog.deadline_id.in_(deadline_ids)).all()

        notification_log_ids_in_timeline = set()
        for t in timelines:
            metadata = t.event_metadata if isinstance(t.event_metadata, dict) else {}
            notification_log_id = metadata.get("notification_log_id")
            if notification_log_id is not None:
                notification_log_ids_in_timeline.add(notification_log_id)

        items: List[Dict[str, Any]] = []

        for t in timelines:
            item = {
                "type": t.event_type,
                "timestamp": to_utc_iso(t.event_date),
                "description": t.description,
                "metadata": t.event_metadata or {},
                "source": "timeline",
            }
            if t.event_type == "reminder" and t.event_metadata and isinstance(t.event_metadata, dict):
                mp = t.event_metadata.get("message_preview") or t.event_metadata.get("message")
                if mp:
                    item["message_preview"] = mp
            items.append(item)

        for d in documents:
            items.append({
                "type": "document_uploaded",
                "timestamp": to_utc_iso(d.uploaded_at),
                "description": f"{d.document_type.value} uploaded",
                "metadata": {"document_id": d.id},
                "source": "document",
            })

        for d in deadlines:
            ts = to_utc_iso(d.created_at) if d.created_at else (to_utc_iso(d.deadline_date) if d.deadline_date else "")
            items.append({
                "type": "deadline_created",
                "timestamp": ts,
                "description": f"{d.deadline_type} - {d.description or ''}",
                "metadata": {"deadline_id": d.id},
                "source": "deadline",
            })

        for n in notifications:
            if n.id in notification_log_ids_in_timeline:
                continue
            notification_event_type = f"notification_{n.status.value}"
            items.append({
                "type": notification_event_type,
                "timestamp": to_utc_iso(n.created_at) if n.created_at else "",
                "description": f"Notification {n.status.value} via {n.channel.value}",
                "metadata": {"notification_log_id": n.id, "deadline_id": n.deadline_id, "days_before": n.days_before, "channel": n.channel.value, "status": n.status.value, "message_id": n.message_id},
                "message_preview": n.message_preview,
                "source": "notification",
            })

        return sorted(items, key=lambda x: x.get("timestamp") or "", reverse=True)

    def record_notification_event(
        self,
        db: Session,
        notification_log: NotificationLog,
        status: NotificationStatus,
        provider: str,
        event_date: Optional[dt.datetime] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[CaseTimeline]:
        case_id = getattr(getattr(notification_log, "deadline", None), "case_id", None)
        if case_id is None:
            return None

        event_metadata = {
            "notification_log_id": notification_log.id,
            "deadline_id": notification_log.deadline_id,
            "user_id": notification_log.user_id,
            "channel": notification_log.channel.value if hasattr(notification_log.channel, "value") else str(notification_log.channel),
            "status": status.value if hasattr(status, "value") else str(status),
            "message_id": notification_log.message_id,
            "days_before": notification_log.days_before,
            "provider": provider,
            "recipient": notification_log.recipient,
        }
        if metadata:
            event_metadata.update(metadata)

        description = f"Notification {event_metadata['status']} via {provider} ({event_metadata['channel']})"
        return self.create_event(
            db=db,
            case_id=case_id,
            event_type=f"notification_{event_metadata['status']}",
            description=description,
            metadata=event_metadata,
            event_date=event_date or dt.datetime.now(dt.timezone.utc),
        )


timeline_service = TimelineService()
