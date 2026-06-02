"""In-memory job-to-user registry for WebSocket ownership checks.

Entries auto-expire after _TTL_SECONDS to prevent unbounded memory growth.
"""

import time as _time
import threading as _threading

_TTL_SECONDS = 3600  # 1 hour
_JOB_OWNERS: dict[str, tuple[int, float]] = {}
_CLEANUP_LOCK = _threading.Lock()


def register_job_owner(job_id: str, user_id: int) -> None:
    _JOB_OWNERS[job_id] = (user_id, _time.monotonic() + _TTL_SECONDS)


def get_job_owner(job_id: str) -> int | None:
    entry = _JOB_OWNERS.get(job_id)
    if entry is None:
        return None
    user_id, expiry = entry
    if _time.monotonic() > expiry:
        _try_evict(job_id)
        return None
    return user_id


def remove_job_owner(job_id: str) -> None:
    _JOB_OWNERS.pop(job_id, None)


def _try_evict(job_id: str) -> None:
    """Remove an entry only if it hasn't been updated since we checked."""
    with _CLEANUP_LOCK:
        entry = _JOB_OWNERS.get(job_id)
        if entry is not None and _time.monotonic() > entry[1]:
            del _JOB_OWNERS[job_id]


def cleanup_expired() -> int:
    """Remove all expired entries and return the count removed."""
    now = _time.monotonic()
    expired = [jid for jid, (_, exp) in _JOB_OWNERS.items() if now > exp]
    for jid in expired:
        _JOB_OWNERS.pop(jid, None)
    return len(expired)
