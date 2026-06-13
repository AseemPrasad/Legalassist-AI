"""
SendGrid Webhook Endpoint
POST /api/v1/webhooks/sendgrid - Receive SendGrid delivery event callbacks

Design: each event in the batch is processed in its own independent
database transaction so that a single malformed or unexpected event
cannot roll back delivery updates that were already applied successfully.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List

import structlog
from fastapi import APIRouter, Request, Response, status

from database import (
    NotificationLog,
    NotificationStatus,
    SessionLocal,
)

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Supported SendGrid event → NotificationStatus mapping
# ---------------------------------------------------------------------------

_SENDGRID_EVENT_STATUS: Dict[str, NotificationStatus] = {
    "delivered": NotificationStatus.SENT,
    "bounce":    NotificationStatus.BOUNCED,
    "bounced":   NotificationStatus.BOUNCED,
    "open":      NotificationStatus.OPENED,
    "opened":    NotificationStatus.OPENED,
}


# ---------------------------------------------------------------------------
# Per-event processing (isolated transaction)
# ---------------------------------------------------------------------------

def _process_single_event(event: Dict[str, Any]) -> bool:
    """Apply one SendGrid delivery event to the database.

    Each call opens its own SQLAlchemy session and commits or rolls back
    independently.  This is the core of the isolation guarantee: a failure
    inside this function (bad data, unexpected field type, DB constraint
    violation, etc.) only rolls back *this* event's changes — previously
    committed events in the same batch are unaffected.

    Args:
        event: A single dict from the SendGrid events array.

    Returns:
        True if the event was processed and committed successfully,
        False otherwise.
    """
    sg_event = (event.get("event") or "").lower().strip()
    message_id = (event.get("sg_message_id") or "").split(".")[0].strip()

    if not sg_event:
        logger.warning(
            "sendgrid_webhook_missing_event_type",
            raw_event=event,
        )
        return False

    new_status = _SENDGRID_EVENT_STATUS.get(sg_event)
    if new_status is None:
        # Informational events (click, unsubscribe, etc.) — not an error.
        logger.debug(
            "sendgrid_webhook_untracked_event",
            sg_event=sg_event,
            message_id=message_id,
        )
        return True  # Treated as success — nothing to persist.

    if not message_id:
        logger.warning(
            "sendgrid_webhook_missing_message_id",
            sg_event=sg_event,
        )
        return False

    # Use a dedicated session so commit/rollback is scoped to this event only.
    db = SessionLocal()
    try:
        log_entry: NotificationLog | None = (
            db.query(NotificationLog)
            .filter(NotificationLog.message_id == message_id)
            .first()
        )

        if log_entry is None:
            logger.info(
                "sendgrid_webhook_no_matching_log",
                message_id=message_id,
                sg_event=sg_event,
            )
            # Not an error — the log entry may not exist yet (race) or
            # may belong to a different service.
            return True

        log_entry.status = new_status

        if new_status == NotificationStatus.SENT:
            log_entry.delivered_at = dt.datetime.now(dt.timezone.utc)

        db.commit()

        logger.info(
            "sendgrid_webhook_event_applied",
            message_id=message_id,
            sg_event=sg_event,
            new_status=new_status.value,
            notification_log_id=log_entry.id,
        )
        return True

    except Exception as exc:  # noqa: BLE001
        # Roll back *only* this event's transaction.  The next event in the
        # batch will get a fresh session and a fresh transaction.
        db.rollback()
        logger.error(
            "sendgrid_webhook_event_failed",
            message_id=message_id,
            sg_event=sg_event,
            error=str(exc),
            exc_info=True,
        )
        return False

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/sendgrid",
    status_code=status.HTTP_200_OK,
    summary="Receive SendGrid delivery event callbacks",
)
async def sendgrid_webhook(request: Request) -> dict:
    """Process a batch of SendGrid delivery events.

    SendGrid sends a JSON array of event objects to this endpoint.  Each
    event is processed in an **isolated database transaction** so that one
    malformed or unexpected event never causes a batch-wide rollback.

    Contract:
    - Always returns HTTP 200 so SendGrid does not retry the entire batch.
    - Logs per-event successes, skips, and failures with structured context.
    - A summary of processed/skipped/failed counts is returned in the body.

    Isolation guarantee:
    ``_process_single_event`` opens a fresh ``SessionLocal`` session for
    each event and calls ``db.commit()`` or ``db.rollback()`` before
    returning.  No shared transaction object is passed between events.
    """
    try:
        events: List[Dict[str, Any]] = await request.json()
    except Exception as parse_exc:  # noqa: BLE001
        logger.error(
            "sendgrid_webhook_payload_parse_failed",
            error=str(parse_exc),
        )
        # Return 200 to prevent SendGrid from retrying with the same bad payload.
        return {"processed": 0, "skipped": 0, "failed": 0, "error": "invalid_json"}

    if not isinstance(events, list):
        logger.warning(
            "sendgrid_webhook_unexpected_payload_type",
            payload_type=type(events).__name__,
        )
        return {"processed": 0, "skipped": 0, "failed": 0, "error": "expected_array"}

    processed = 0
    skipped = 0
    failed = 0

    logger.info(
        "sendgrid_webhook_batch_received",
        event_count=len(events),
    )

    for event in events:
        if not isinstance(event, dict):
            skipped += 1
            logger.warning(
                "sendgrid_webhook_skipping_non_dict_event",
                event_type=type(event).__name__,
            )
            continue

        sg_event = (event.get("event") or "").lower().strip()

        # Silently skip informational events that we never store.
        if sg_event and sg_event not in _SENDGRID_EVENT_STATUS:
            skipped += 1
            logger.debug(
                "sendgrid_webhook_skipped_untracked_event",
                sg_event=sg_event,
            )
            continue

        # Each call is its own transaction — failure here does NOT affect
        # events already successfully committed earlier in the loop.
        success = _process_single_event(event)
        if success:
            processed += 1
        else:
            failed += 1

    logger.info(
        "sendgrid_webhook_batch_complete",
        total=len(events),
        processed=processed,
        skipped=skipped,
        failed=failed,
    )

    return {
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
    }
