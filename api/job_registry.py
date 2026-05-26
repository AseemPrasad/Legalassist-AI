"""In-memory job-to-user registry for WebSocket ownership checks."""

_JOB_OWNERS: dict[str, int] = {}


def register_job_owner(job_id: str, user_id: int) -> None:
    _JOB_OWNERS[job_id] = user_id


def get_job_owner(job_id: str) -> int | None:
    return _JOB_OWNERS.get(job_id)
