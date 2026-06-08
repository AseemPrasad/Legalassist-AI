from __future__ import annotations

import json
import time
from urllib.parse import parse_qsl
from typing import Any



import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from api.errors import StructuredAPIError
from config import Config

from db.crud.notifications import update_notification_log_by_message_id
from api.dependencies import get_db_rls, get_db_rls_optional
from db.models.notifications import NotificationStatus
from services.timeline_service import timeline_service

try:
    from twilio.request_validator import RequestValidator
except ImportError:  # pragma: no cover - optional dependency
    RequestValidator = None

try:
    from sendgrid.helpers.eventwebhook import EventWebhook, EventWebhookHeader
except ImportError:  # pragma: no cover - optional dependency
    EventWebhook = None
    EventWebhookHeader = None

router = APIRouter(prefix="/api/v1/webhooks", tags=["notifications"])
logger = structlog.get_logger(__name__)


def _parse_twilio_webhook_payload(raw_body: str) -> dict[str, str]:
    return dict(parse_qsl(raw_body, keep_blank_values=True))


def _parse_sendgrid_webhook_payload(raw_body: str) -> list[dict[str, Any]]:
    try:
        events = json.loads(raw_body or "[]")
    except json.JSONDecodeError as exc:
        raise StructuredAPIError(status_code=status.HTTP_400_BAD_REQUEST, error_code="SENDGRID_WEBHOOK_INVALID_JSON", message="Invalid SendGrid webhook payload") from exc

    if isinstance(events, dict):
        events = [events]

    if not isinstance(events, list):
        raise StructuredAPIError(status_code=status.HTTP_400_BAD_REQUEST, error_code="SENDGRID_WEBHOOK_INVALID_JSON", message="Invalid SendGrid webhook payload")

    normalized_events: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            raise StructuredAPIError(status_code=status.HTTP_400_BAD_REQUEST, error_code="SENDGRID_WEBHOOK_INVALID_JSON", message="Invalid SendGrid webhook payload")
        normalized_events.append(event)

    return normalized_events


def _twilio_status_to_notification_status(message_status: str) -> NotificationStatus | None:
    normalized = (message_status or "").lower()
    if normalized == "delivered":
        return NotificationStatus.DELIVERED
    if normalized in {"failed", "undelivered", "canceled", "cancelled"}:
        return NotificationStatus.FAILED
    return None


def _sendgrid_event_to_notification_status(event_type: str) -> NotificationStatus | None:
    normalized = (event_type or "").lower()
    if normalized == "delivered":
        return NotificationStatus.DELIVERED
    if normalized in {"bounce", "dropped", "deferred", "blocked", "spamreport", "invalid"}:
        return NotificationStatus.FAILED
    if normalized in {"open", "click"}:
        return NotificationStatus.OPENED
    return None


def _message_id_candidates(message_id: str | None) -> list[str]:
    if not message_id:
        return []

    candidates: list[str] = []
    normalized = message_id.strip().strip("<>")
    if normalized:
        candidates.append(normalized)
        if "." in normalized:
            candidates.append(normalized.split(".", 1)[0])
        if "@" in normalized:
            candidates.append(normalized.split("@", 1)[0])

    return list(dict.fromkeys(candidates))


def _verify_twilio_signature(request: Request, params: dict[str, str]) -> bool:
    signature = request.headers.get("X-Twilio-Signature") or request.headers.get("x-twilio-signature")
    if not signature:
        raise StructuredAPIError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="TWILIO_SIGNATURE_MISSING", message="Missing Twilio signature header")

    if not Config.get_twilio_auth_token():
        raise StructuredAPIError(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, error_code="TWILIO_WEBHOOK_NOT_CONFIGURED", message="Twilio auth token is not configured")

    if RequestValidator is None:
        raise StructuredAPIError(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, error_code="TWILIO_WEBHOOK_VALIDATION_UNAVAILABLE", message="Twilio request validator is not installed")

    validator = RequestValidator(Config.get_twilio_auth_token())
    return bool(validator.validate(str(request.url), params, signature))


def _verify_sendgrid_signature(request: Request, payload: str) -> bool:
    signature_header = EventWebhookHeader.SIGNATURE if EventWebhookHeader else "X-Twilio-Email-Event-Webhook-Signature"
    timestamp_header = EventWebhookHeader.TIMESTAMP if EventWebhookHeader else "X-Twilio-Email-Event-Webhook-Timestamp"

    signature = request.headers.get(signature_header)
    timestamp = request.headers.get(timestamp_header)
    if not signature or not timestamp:
        raise StructuredAPIError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="SENDGRID_SIGNATURE_MISSING", message="Missing SendGrid signature headers")

    try:
        ts = int(timestamp)
    except ValueError:
        raise StructuredAPIError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="SENDGRID_WEBHOOK_INVALID_TIMESTAMP", message="Invalid SendGrid webhook timestamp")
    if abs(time.time() - ts) > 300:
        raise StructuredAPIError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="SENDGRID_WEBHOOK_EXPIRED", message="SendGrid webhook timestamp is too old, possible replay attack")

    public_key = Config.get_sendgrid_event_webhook_public_key()
    if not public_key:
        raise StructuredAPIError(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, error_code="SENDGRID_WEBHOOK_NOT_CONFIGURED", message="SendGrid event webhook public key is not configured")

    if EventWebhook is None:
        raise StructuredAPIError(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, error_code="SENDGRID_WEBHOOK_VALIDATION_UNAVAILABLE", message="SendGrid event webhook validator is not installed")

    verifier = EventWebhook(public_key)
    return bool(verifier.verify_signature(payload, signature, timestamp))


def _update_delivery_status(db: Session, message_id: str | None, status_value: NotificationStatus, error_message: str | None = None, message_preview: str | None = None):
    for candidate in _message_id_candidates(message_id):
        updated = update_notification_log_by_message_id(
            db=db,
            message_id=candidate,
            status=status_value,
            error_message=error_message,
            message_preview=message_preview,
        )
        if updated:
            try:
                timeline_service.record_notification_event(
                    db=db,
                    notification_log=updated,
                    status=status_value,
                    provider="twilio" if updated.channel.value == "sms" else "sendgrid",
                    metadata={
                        "error_message": error_message,
                        "message_preview": message_preview,
                    },
                )
            except Exception:
                logger.exception("notification_timeline_event_failed", notification_log_id=updated.id, message_id=candidate)
            return updated
    return None


@router.post("/twilio")
async def twilio_delivery_webhook(request: Request, db: Session = Depends(get_db_rls_optional)) -> dict:
    raw_bytes = await request.body()
    try:
        raw_body = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("twilio_webhook_invalid_utf8", body_length=len(raw_bytes))
        raw_body = raw_bytes.decode("utf-8", errors="replace")
    params = _parse_twilio_webhook_payload(raw_body)

    if not _verify_twilio_signature(request, params):
        raise StructuredAPIError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="TWILIO_SIGNATURE_INVALID", message="Invalid Twilio signature")

    message_sid = params.get("MessageSid") or params.get("SmsSid") or params.get("SmsMessageSid")
    message_status = (params.get("MessageStatus") or params.get("SmsStatus") or "").lower()
    error_code = params.get("ErrorCode")

    normalized_status = _twilio_status_to_notification_status(message_status)
    if normalized_status == NotificationStatus.DELIVERED:
        updated = _update_delivery_status(db, message_sid, NotificationStatus.DELIVERED)
    elif normalized_status == NotificationStatus.FAILED:
        updated = _update_delivery_status(db, message_sid, NotificationStatus.FAILED, error_message=f"Twilio status: {message_status}{f' ({error_code})' if error_code else ''}")
    else:
        updated = None

    logger.info("twilio_delivery_webhook_processed", message_id=message_sid, status=message_status, updated=bool(updated))
    return {"ok": True, "updated": bool(updated), "status": message_status}


@router.post("/sendgrid")
async def sendgrid_delivery_webhook(request: Request, db: Session = Depends(get_db_rls_optional)) -> dict:
    raw_bytes = await request.body()
    try:
        raw_body = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("sendgrid_webhook_invalid_utf8", body_length=len(raw_bytes))
        raw_body = raw_bytes.decode("utf-8", errors="replace")

    if not _verify_sendgrid_signature(request, raw_body):
        raise StructuredAPIError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="SENDGRID_SIGNATURE_INVALID", message="Invalid SendGrid signature")

    events = _parse_sendgrid_webhook_payload(raw_body)

    processed = 0
    updated = 0
    for event in events:
        processed += 1
        event_type = str(event.get("event", "")).lower()
        message_id = event.get("sg_message_id") or event.get("message_id") or event.get("smtp-id")
        normalized_status = _sendgrid_event_to_notification_status(event_type)
        if normalized_status == NotificationStatus.DELIVERED:
            result = _update_delivery_status(db, message_id, NotificationStatus.DELIVERED)
        elif normalized_status == NotificationStatus.FAILED:
            reason = event.get("reason") or event.get("response") or event_type
            result = _update_delivery_status(db, message_id, NotificationStatus.FAILED, error_message=str(reason))
        elif normalized_status == NotificationStatus.OPENED:
            result = _update_delivery_status(db, message_id, NotificationStatus.OPENED)
        else:
            result = None

        updated += int(bool(result))

    logger.info("sendgrid_delivery_webhook_processed", events=processed, updated=updated)
    return {"ok": True, "events": processed, "updated": updated}


# ============================================================================
# User Notification Preferences Router
# ============================================================================

from api.auth import get_current_user, CurrentUser
from api.models import UserPreferenceUpdate, UserPreferenceResponse
from db.notifications_service import create_or_update_user_preference
from db.models.notifications import UserPreference, NotificationChannel

pref_router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])

@pref_router.get("/preferences", response_model=UserPreferenceResponse)
async def get_user_preferences(
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls),
) -> UserPreferenceResponse:
    """Get the current authenticated user's notification preferences.

    Returns 404 when no preferences have been saved yet.
    Use PUT /preferences to create or update preferences explicitly.
    """
    pref = db.query(UserPreference).filter(UserPreference.user_id == current_user.user_id).first()
    if not pref:
        # Return 404 rather than implicitly creating default preferences.
        # GET must not modify persistent data; preference creation requires
        # explicit user action via PUT /preferences.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification preferences not found. Use PUT /preferences to create them.",
        )
    
    return UserPreferenceResponse(
        user_id=pref.user_id,
        email=pref.email,
        phone_number=pref.phone_number,
        notification_channel=pref.notification_channel.value if hasattr(pref.notification_channel, "value") else str(pref.notification_channel),
        timezone=pref.timezone,
        reminder_thresholds=pref.get_reminder_thresholds(),
        holiday_aware_reminders=pref.holiday_aware_reminders,
        holiday_country=pref.holiday_country,
        holiday_region=pref.holiday_region,
        holiday_calendar_json=pref.holiday_calendar_json,
        updated_at=pref.updated_at,
    )

@pref_router.put("/preferences", response_model=UserPreferenceResponse)
async def update_user_preferences(
    payload: UserPreferenceUpdate,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db_rls),
) -> UserPreferenceResponse:
    """Create or update the current authenticated user's notification preferences."""
    try:
        channel_enum = NotificationChannel(payload.notification_channel.lower())
    except ValueError:
        channel_enum = NotificationChannel.BOTH

    pref = create_or_update_user_preference(
        db=db,
        user_id=current_user.user_id,
        email=payload.email,
        phone_number=payload.phone_number,
        notification_channel=channel_enum,
        timezone=payload.timezone,
        holiday_aware_reminders=payload.holiday_aware_reminders,
        holiday_country=payload.holiday_country,
        holiday_region=payload.holiday_region,
        holiday_calendar_json=payload.holiday_calendar_json,
        reminder_thresholds=payload.reminder_thresholds,
    )

    return UserPreferenceResponse(
        user_id=pref.user_id,
        email=pref.email,
        phone_number=pref.phone_number,
        notification_channel=pref.notification_channel.value if hasattr(pref.notification_channel, "value") else str(pref.notification_channel),
        timezone=pref.timezone,
        reminder_thresholds=pref.get_reminder_thresholds(),
        holiday_aware_reminders=pref.holiday_aware_reminders,
        holiday_country=pref.holiday_country,
        holiday_region=pref.holiday_region,
        holiday_calendar_json=pref.holiday_calendar_json,
        updated_at=pref.updated_at,
    )


