import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from core.clock import Clock

# Ensure the logs directory exists
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
AUDIT_LOG_FILE = LOG_DIR / "document_audit.log"

# Setup a dedicated logger for audit trails
audit_logger = logging.getLogger("legalassist.audit")
audit_logger.setLevel(logging.INFO)

# File handler for JSON structured logs
file_handler = logging.FileHandler(AUDIT_LOG_FILE)
file_handler.setLevel(logging.INFO)

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Expected to receive a dict in the `msg` field
        log_entry = {
            "timestamp": Clock.isoformat(),
            "level": record.levelname,
            "event": record.msg
        }
        return json.dumps(log_entry)

file_handler.setFormatter(JSONFormatter())
audit_logger.addHandler(file_handler)

# Prevent log propagation to the root logger to avoid console spam
audit_logger.propagate = False

class AuditService:
    """
    Service for creating structured audit trails for all sensitive
    document operations to comply with legal tech security standards.
    """

    @staticmethod
    def log_document_access(
        user_id: str,
        document_id: str,
        action: str,
        ip_address: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an audit event when a document is accessed or modified.
        
        :param user_id: ID of the user performing the action.
        :param document_id: ID of the target document.
        :param action: The action performed (e.g., READ, WRITE, DELETE, DOWNLOAD).
        :param ip_address: IP address of the requesting client.
        :param metadata: Any additional contextual metadata.
        """
        event_data = {
            "action": action.upper(),
            "user_id": user_id,
            "document_id": document_id,
            "ip_address": ip_address or "UNKNOWN",
            "metadata": metadata or {}
        }
        
        audit_logger.info(event_data)

# Example usage (for testing)
if __name__ == "__main__":
    AuditService.log_document_access(
        user_id="user_591823",
        document_id="doc_88491",
        action="READ",
        ip_address="192.168.1.104",
        metadata={"client_agent": "Mozilla/5.0", "duration_ms": 120}
    )
    print(f"Audit log entry written to {AUDIT_LOG_FILE}")
