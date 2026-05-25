from __future__ import annotations

import json
from urllib.parse import parse_qsl

import structlog
from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from api.errors import StructuredAPIError
from config import Config
from database import get_db
from db.crud.notifications import update_notification_log_by_message_id
from db.models.notifications import NotificationStatus

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
            return updated
    return None


@router.post("/twilio")
async def twilio_delivery_webhook(request: Request, db: Session = Depends(get_db)) -> dict:
    raw_body = (await request.body()).decode("utf-8")
    params = dict(parse_qsl(raw_body, keep_blank_values=True))

    if not _verify_twilio_signature(request, params):
        raise StructuredAPIError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="TWILIO_SIGNATURE_INVALID", message="Invalid Twilio signature")

    message_sid = params.get("MessageSid") or params.get("SmsSid") or params.get("SmsMessageSid")
    message_status = (params.get("MessageStatus") or params.get("SmsStatus") or "").lower()
    error_code = params.get("ErrorCode")

    if message_status == "delivered":
        updated = _update_delivery_status(db, message_sid, NotificationStatus.DELIVERED)
    elif message_status in {"failed", "undelivered", "canceled", "cancelled"}:
        updated = _update_delivery_status(db, message_sid, NotificationStatus.FAILED, error_message=f"Twilio status: {message_status}{f' ({error_code})' if error_code else ''}")
    else:
        updated = None

    logger.info("twilio_delivery_webhook_processed", message_id=message_sid, status=message_status, updated=bool(updated))
    return {"ok": True, "updated": bool(updated), "status": message_status}


@router.post("/sendgrid")
async def sendgrid_delivery_webhook(request: Request, db: Session = Depends(get_db)) -> dict:
    raw_body = (await request.body()).decode("utf-8")

    if not _verify_sendgrid_signature(request, raw_body):
        raise StructuredAPIError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="SENDGRID_SIGNATURE_INVALID", message="Invalid SendGrid signature")

    try:
        events = json.loads(raw_body or "[]")
    except json.JSONDecodeError as exc:
        raise StructuredAPIError(status_code=status.HTTP_400_BAD_REQUEST, error_code="SENDGRID_WEBHOOK_INVALID_JSON", message="Invalid SendGrid webhook payload") from exc

    if isinstance(events, dict):
        events = [events]

    processed = 0
    updated = 0
    for event in events:
        processed += 1
        event_type = str(event.get("event", "")).lower()
        message_id = event.get("sg_message_id") or event.get("message_id") or event.get("smtp-id")
        if event_type == "delivered":
            result = _update_delivery_status(db, message_id, NotificationStatus.DELIVERED)
        elif event_type in {"bounce", "dropped", "deferred", "blocked", "spamreport", "invalid"}:
            reason = event.get("reason") or event.get("response") or event_type
            result = _update_delivery_status(db, message_id, NotificationStatus.FAILED, error_message=str(reason))
        else:
            result = None

        updated += int(bool(result))

    logger.info("sendgrid_delivery_webhook_processed", events=processed, updated=updated)
    return {"ok": True, "events": processed, "updated": updated}