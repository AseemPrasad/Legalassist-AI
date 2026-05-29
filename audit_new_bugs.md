# Security & Correctness Audit — 32 New Advanced Bugs

**Scope:** Full codebase audit excluding issues from earlier fix branches.  
**Methodology:** Source code static analysis with dependency-graph tracing.  
**Total bugs found:** 32  

---

## Subsystem: Core Deadline Engine

### Bug 1 — Weekend Detection Ignores Jurisdiction Parameter

**File:** `core/deadline_engine.py:24-25`

The `calculate_deadline()` function accepts a `jurisdiction` parameter but `_is_weekend()` always treats Saturday (5) and Sunday (6) as weekends. In jurisdictions like Israel (Sunday–Thursday), Bangladesh (Friday–Saturday), or UAE (Friday half-day), the weekend days differ. The `jurisdiction` parameter is only used for filing-cutoff logic (lines 78-89), never for weekend computation. A deadline calculated for these jurisdictions lands on the wrong day.

**Impact:** Incorrect deadline dates for non-Western jurisdictions, potentially causing missed court deadlines.

---

### Bug 2 — Zero Business Days Skips Weekend/Holiday Adjustment

**File:** `core/deadline_engine.py:28-103`

When `business_days=0`, the while loop body never executes and `current` stays at the start date. Weekend/holiday skipping is only done inside the loop. Meanwhile, `jurisdiction_adjustment` (lines 84-89) and `emergency_extension_days` (line 91) ARE applied even for zero-day deadlines. A zero-business-day deadline on a Saturday reports the Saturday as the deadline date instead of rolling to Monday.

**Impact:** Deadlines on weekends are not corrected when no business days are added.

---

### Bug 3 — Jurisdiction Adjustment and Emergency Extension Are Additive on the Same Axis

**File:** `core/deadline_engine.py:91`

```python
final = adjusted_for_weekends_holidays + timedelta(days=jurisdiction_adjustment + int(emergency_extension_days))
```

Jurisdiction adjustments (e.g., late-filing +1 day) and emergency extensions (e.g., 7-day court order) are summed into a single offset from the same base date. Jurisdiction rules should extend from the ORIGINAL calculated deadline, while emergency extensions should stack sequentially. If both apply, the misordered offsets produce a wrong calendar date.

**Impact:** Emergency extensions and jurisdiction penalties produce incorrect dates when both are present.

---

### Bug 4 — `fromisoformat` Crashes on Python 3.10 with SQLite Date Format

**File:** `core/deadline_engine.py:13`

```python
dt = datetime.fromisoformat(str(value))
```

If `value` is a naive `datetime` from SQLite (stored as `"2026-05-28 14:30:00"`), `datetime.fromisoformat()` in **Python 3.10** raises `ValueError` because the space-separated format without `T` is not ISO-8601 compliant. Python 3.11+ accepts this format. The project targets Python 3.10 (per `pyproject.toml`).

**Impact:** Code crashes on Python 3.10 when processing deadlines from SQLite (the default dev database).

---

### Bug 5 — Holiday List Strings Are Never Validated

**File:** `core/deadline_engine.py:51`

```python
holidays_set = set(holidays or [])
```

The `holidays` parameter is `Optional[List[str]]` with no validation. If an entry is in an unexpected format (e.g., `"2026/01/01"` instead of `"2026-01-01"`), the comparison `d.isoformat() in holidays_set` silently returns `False`, so the holiday is never skipped.

**Impact:** Holidays in incorrect format are silently ignored, causing deadlines to land on actual holidays.

---

## Subsystem: Notification Service

### Bug 6 — `send_sms_reminder` No Transaction Boundary Between Reservation and Update

**File:** `notification_service.py:611-641`

`reserve_notification()` creates a PENDING notification log entry with its own internal commit, then `send_sms()` is called (line 626), then `update_notification_result()` (line 631). There is no DB transaction wrapping these operations. If the process crashes between the SMS being sent and `update_notification_result`, the DB record remains PENDING even though the SMS was delivered. The reverse: if the SMS fails but `update_notification_result` raises, the PENDING record is orphaned with no retry mechanism.

**Impact:** Lost SMS deliveries or orphaned PENDING notification records.

---

### Bug 7 — `db.commit()` Race in Email Dispatch with Celery Worker

**File:** `notification_service.py:733-741`

After dispatching the Celery task (line 721), `db.commit()` runs at line 739 to set `message_id`. The Celery worker (`send_email_task`) starts IMMEDIATELY. If the worker executes `update_notification_result` before line 739's commit, the worker queries for a notification log row that does not yet exist (uncommitted). The worker fails silently. Conversely, if the commit runs first, there is a window where the caller returns success but the DB still has `message_id=NULL` (commit after task dispatch).

**Impact:** Lost notification status updates and duplicate sends from un-tracked delivery.

---

### Bug 8 — Tenacity Retry Uses String-Matched Error Messages Instead of HTTP Status Codes

**File:** `notification_service.py:123, 188`

```python
retry=tenacity.retry_if_exception(lambda e: '503' in str(e) or '429' in str(e))
```

The retry predicate searches for the substrings `"503"` or `"429"` inside `str(e)`. If the third-party library raises an `HTTPError` where the error message is `"Service Unavailable"` (no `"503"` in the string), or if the exception type is `TwilioRestException` with `status=503` but a human-readable message, the retry never fires. Real 503/429 responses are silently treated as permanent failures.

**Impact:** Transient Twilio/SendGrid outages cause permanent notification failures instead of graceful retries.

---

### Bug 9 — `deadline.case_title` Accessed Without Attribute Guarantee

**File:** `notification_service.py:414, 479, 595`

The code accesses `deadline.case_title` freely across multiple functions. If the `CaseDeadline` ORM model does not have a `case_title` column (some schemas store only the FK `case_id` and derive the title via a join), accessing it produces `AttributeError: 'CaseDeadline' object has no attribute 'case_title'`. SQLAlchemy queries that do not eagerly load joined fields will crash.

**Impact:** Runtime `AttributeError` crashes on notification dispatch when the ORM model lacks a direct `case_title` field.

---

### Bug 10 — Hardcoded Reminder Thresholds `[30, 10, 3, 1]` Are Inextensible

**File:** `notification_service.py:772`

```python
if days_left not in [30, 10, 3, 1]:
    return results
```

The reminder thresholds are hardcoded. There is no configuration, no DB-driven rule, and no per-jurisdiction customization. A jurisdiction that requires 60-day, 45-day, 7-day, and 2-day reminders cannot be supported. The only way to change thresholds is to modify source code.

**Impact:** Rigid notification scheduling — jurisdictions with different legal timeline requirements cannot customize reminders.

---

## Subsystem: Celery Background Tasks

### Bug 11 — `cleanup_old_tasks` Is a Permanent No-Op

**File:** `celery_app.py:1500-1517`

```python
backend = getattr(celery_app, "backend", None)
cleanup_fn = getattr(backend, "cleanup", None)
if callable(cleanup_fn):
    cleanup_fn()
```

The standard Celery Redis backend (`RedisBackend`) does NOT have a `cleanup()` method. Only the `filesystem` and `cache` backends support this. The task runs every 24 hours (line 326) and always logs "noop". Old task results accumulate in Redis indefinitely, consuming memory with no cleanup mechanism.

**Impact:** Unbounded Redis memory growth from stale task results that are never pruned.

---

### Bug 12 — `export_data_task` Returns Null Instead of Raising on Invalid Format

**File:** `celery_app.py:1039-1048`

```python
if format not in ("csv", "json"):
    return {
        "export_id": None,
        "file_path": None,
        "file_size_bytes": 0,
        "format": format,
        "expires_in_hours": None,
        ...
    }
```

When an unsupported format is requested, the task returns a dict with `None` values rather than raising a `ValueError`. Celery marks this as SUCCESS. The caller sees `status="completed"` with all null fields and cannot distinguish between "export succeeded" and "format was rejected." Downstream code receiving `file_path=None` would crash with `AttributeError`.

**Impact:** Silent data corruption — the caller believes export succeeded, then crashes on the null file path.

---

### Bug 13 — `generate_report_task` Idempotency Key Excludes `report_id`

**File:** `celery_app.py:856`

```python
idempotency_key = f"report:{user_id}:{case_id}:{report_type}:{format}:{privacy_profile}"
```

Two different `report_id` values for the same user/case/type/format/profile are treated as duplicates. The second request silently returns the cached result of the first. The second report is never generated, the second `report_id` has no DB record, and the caller receives a report with the wrong `report_id` and stale timestamps.

**Impact:** Lost report generation — users receive reports with incorrect `report_id` and stale data.

---

### Bug 14 — Export `mask_recipient` Misaligned with Phone Formatting Characters

**File:** `celery_app.py:1081-1086`

```python
digits = [c for c in recipient if c.isdigit()]
if len(digits) >= 7:
    return recipient[:3] + "*" * (len(recipient) - 7) + recipient[-4:]
```

For a phone `+1 (555) 123-4567` (11 digits, 17 chars with formatting):
- `len(digits) == 11 >= 7` → true
- `recipient[:3]` = `"+1 "`, stars count = `17 - 7 = 10`, `recipient[-4:]` = `"4567"`
- Output: `"+1 **********4567"` — 10 stars consume formatting characters, not just digits

Phone numbers with different formatting produce inconsistently masked outputs, and the actual digit content is obscured unpredictably.

**Impact:** Inconsistent PII masking — phone numbers with different formatting produce non-deterministic output; legitimate legal references get corrupted.

---

### Bug 15 — Content Hash Missing for `file_path` and `file_url` Sources in Analyzer

**File:** `celery_app.py:511-516`

```python
content_parts = []
if file_bytes:
    content_parts.append(hashlib.sha256(file_bytes).hexdigest())
if text:
    content_parts.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
content_hash = ... if content_parts else ""
```

When a document is loaded via `file_path` or `file_url` (without `file_bytes` or `text` being set simultaneously), `content_parts` is empty and `content_hash` is `""`. Multiple different files loaded by path or URL produce the same idempotency key `f"analyze:{user_id}:{document_id}:"`, causing one analysis to silently replace or block another.

**Impact:** File-path-based or URL-based document analysis is not uniquely content-keyed — re-analyzing different files returns stale cached results.

---

### Bug 16 — Idempotency Lock TTL Shorter Than Real Analysis Time

**File:** `celery_app.py:520`

```python
if not idemp.acquire(idempotency_key, ttl=300):
```

The idempotency lock expires after 300 seconds (5 minutes). Document analysis involving LLM calls can easily exceed 5 minutes for large documents. When the lock expires, a second request for the same document acquires the lock and starts duplicate processing. Worse, `idemp.mark_completed()` overwrites the first worker's result with the second worker's (possibly still-running) result.

**Impact:** Duplicate document analysis for long-running tasks, potential data corruption from overlapping writes.

---

### Bug 17 — `send_notification_task` Opens Two Separate DB Sessions

**File:** `celery_app.py:1377-1381, 1447-1452`

The email path opens a session to look up User (line 1377) and closes it (line 1381). The SMS path opens a second session to look up UserPreference (line 1447). Both could be loaded in a single session via a relationship join. Two sessions means two connection pool checkouts, two backend round-trips, and potential read-consistency issues if the user's data changes between the two queries.

**Impact:** Inefficient connection pool usage and potential read-consistency issues for notification sends.

---

### Bug 18 — `send_notification_task` Duplicate Retry Mechanism Conflicts with Celery's Built-in Retry

**File:** `celery_app.py:1349-1355, 1431-1432, 1463-1464`

The decorator declares `max_retries=3` AND `default_retry_delay=60`. Inside the task, the code manually calls `self.retry()` at lines 1432/1464. When `self.retry()` is called, Celery counts it as one of the `max_retries` attempts. The `default_retry_delay=60` conflicts with the exponential backoff logic used elsewhere (`send_email_task` line 328: `(2 ** self.request.retries) * 60`). The inconsistent retry strategies create unpredictable delivery delays.

**Impact:** Notification retry timing is unpredictable — some retries use fixed 60s delay, others use exponential backoff, making SLAs impossible to guarantee.

---

### Bug 19 — No Transaction Boundary on `process_case_document_upload_task`

**File:** `celery_app.py:743-819`

`update_case_document()` (line 781) commits internally, then `attachment.document_id = doc.id` (line 791), then `session.commit()` (line 792). If the process crashes between lines 791-792, the `update_case_document` write persists but `attachment.document_id` does not. The database is left with a document that has no attachment reference, causing broken FK relationships.

**Impact:** Database inconsistency — documents updated but attachment references missing.

---

## Subsystem: API & Authentication

### Bug 20 — CSRF Cookie Is Never Refreshed After Auth State Change

**File:** `api/csrf.py:218`

```python
if request.method in SAFE_METHODS and not csrf_cookie:
    session_id = secrets.token_urlsafe(16)
    token = generate_csrf_token(int(user_id) if str(user_id).isdigit() else 0, session_id)
```

The CSRF cookie is only set on safe methods when it does not already exist. If a user visits as anonymous → gets CSRF cookie bound to `user_id=0` → then logs in, the CSRF cookie is still bound to `user_id=0`. The bound CSRF token for `user_id=0` cannot validate requests for the authenticated user. Similarly, if user A logs in and gets a CSRF token, then user B logs in on the same browser, user B inherits user A's CSRF token bound to user A's ID.

**Impact:** CSRF token binding is stale after login/logout — unsafe requests from authenticated sessions may be erroneously rejected.

---

### Bug 21 — CSRF Exempt Path Matching Is Exact, Fails for Trailing Slashes

**File:** `api/csrf.py:198`

```python
if path in self.exempt_paths:
    return await call_next(request)
```

Routes like `/docs/` (with trailing slash) do not match `/docs`. FastAPI normalizes routes, but if a reverse proxy or client sends `/docs/`, the CSRF middleware rejects it with a 403 StructuredAPIError instead of passing through.

**Impact:** CSRF middleware incorrectly blocks routes that have trailing slashes, including OpenAPI docs accessed via certain proxies.

---

### Bug 22 — CSRF Origin Validation `is_same_origin` Does Not Verify Scheme

**File:** `api/csrf.py:96-115`

The function compares `parsed.hostname == host` but never checks the SCHEME (`http` vs `https`). The same-origin policy is defined as scheme + host + port. An attacker serving HTTP content at `http://evil.example.com` would pass the origin check for a target served at `https://example.com`. The scheme is lost after `urlparse()` if only `hostname` is compared.

**Impact:** CSRF same-origin check is incomplete — HTTP-origin pages can forge requests against HTTPS endpoints on the same hostname.

---

### Bug 23 — `get_db_rls` Non-Digit User ID Check Silently Skips RLS

**File:** `api/dependencies.py:75-76`

```python
user_id_str = str(current_user.user_id)
if user_id_str.isdigit():
    apply_rls_context(db, int(user_id_str))
```

If the system ever migrates to UUID or ULID user IDs (e.g., for GDPR-compliant pseudonymous identifiers or horizontal sharding), `isdigit()` returns `False` and RLS is silently skipped. PostgreSQL RLS with no `app.current_user_id` set may either block nothing or block everything, depending on policy. Either way, data isolation is broken.

**Impact:** Complete RLS bypass for non-integer user IDs — security vulnerability that activates the moment user IDs stop being integers.

---

### Bug 24 — Analytics Endpoints Return Hardcoded Mock Data

**File:** `api/routes/analytics.py:46-67, 84-106, 134-149`

Three endpoints (`/costs`, `/overview`, `/usage`) return hardcoded values:

```python
total_cost=125.50,
llm_api_cost=75.00,
...
api_calls=5432,
```

These values never reflect actual usage and are identical for all users. A user seeing "5 active cases" when they have 0, or "5432 API calls" when they have never used the API, receives fabricated information.

**Impact:** Fraudulent analytics data — users (including legal professionals) make case strategy decisions based on fabricated metrics.

---

### Bug 25 — `upload_document_file` Reads File Twice; Second Read Returns Empty Bytes

**File:** `api/routes/documents.py:265-279`

```python
bytes_read = await validate_file_upload_streaming(file, ...)  # consumes the file stream
# ...
file_content = await file.read()  # cursor at EOF → returns b""
```

`validate_file_upload_streaming()` reads the entire file to validate size. After that, the file's cursor is at EOF. `file.read()` returns `b""`. For text files: `text = file_content.decode("utf-8")` → `""`, so the analysis receives empty content. For binary files: `file_bytes = file_content` → `b""`.

**Impact:** All uploaded documents are analyzed with zero content — every analysis result is empty.

---

### Bug 26 — `file_path` Documents Loaded Without Ownership Verification

**File:** `api/routes/documents.py:119` → `celery_app.py:566-574`

When a user submits a `file_path` for analysis, `validate_file_path()` only checks that the path is within allowed directories. It does NOT verify that `current_user` OWNS the file or its associated case. User A can analyze user B's documents by guessing the file path within `/attachments/`.

**Impact:** Path-traversal-like information disclosure — a user can analyze another user's documents by guessing the file path within the allowed directory.

---

### Bug 27 — `cancel_analysis` Endpoint Has No Ownership Check

**File:** `api/routes/documents.py:216-230`

```python
async def cancel_analysis(
    job_id: str,
    current_user: CurrentUser = Depends(get_current_user)
) -> dict:
    success = TaskStatus.revoke_task(job_id)
```

The endpoint accepts any `job_id` from any authenticated user. It calls `celery_app.control.revoke(task_id, terminate=True)` with no check that `current_user` owns the job. User A can cancel user B's running analysis or report generation by guessing their `job_id`.

**Impact:** Arbitrary task cancellation — any authenticated user can disrupt another user's background jobs.

---

### Bug 28 — Request Size Middleware Requires Content-Length for ALL Requests Including GET/DELETE

**File:** `api/middlewares/request_size.py:50-57`

```python
if content_length is None:
    return JSONResponse(status_code=status.HTTP_411_LENGTH_REQUIRED, ...)
```

This check runs for EVERY request not in `SKIP_PATHS` (which only contains 5 paths: health, ready, live, metrics, root). HTTP GET, DELETE, OPTIONS, and HEAD requests typically do NOT send a `Content-Length` header. The middleware rejects them with 411 "Length Required." Every GET endpoint — `/api/v1/analytics/overview`, `/api/v1/cases/search`, `/api/v1/deadlines/upcoming` — returns 411. Additionally, chunked transfer encoding (common for streaming uploads) is also rejected (line 41-48).

**Impact:** Complete API unavailability for GET/DELETE/OPTIONS requests — effectively all read endpoints are broken.

---

## Subsystem: Infrastructure & Configuration

### Bug 29 — Module-Level Settings Evaluated at Import Time

**File:** `celery_app.py:130, 134`, `api/main.py:267`, `database.py:171, 197`

Multiple modules evaluate `Config` values at IMPORT TIME rather than at function call time:

- `celery_app.py:130`: `settings = get_settings()` at module level
- `celery_app.py:134`: `initialize_observability_for_environment()` at import time
- `database.py:171`: `engine = create_engine(DATABASE_URL, ...)` at import time
- `api/main.py:267`: `app = create_app()` at import time

If environment variables change at runtime (e.g., pytest fixtures that set different DB URLs), the stale import-time values persist. Pytest `@pytest.fixture(autouse)` that sets env vars may not execute before these modules are imported during test collection.

**Impact:** Tests that change env vars see stale configuration; cannot test against multiple databases without reloading modules.

---

### Bug 30 — `init_db` Creates Tables But Is Never Called by API or Streamlit

**File:** `database.py:196-199`

```python
def init_db():
    Base.metadata.create_all(bind=engine)
```

This function is defined, but its only caller is in `cli.py`. The FastAPI app (via `api/main.py`) and the Streamlit app (via `app.py`) never call `init_db()` on startup. If the database has not been pre-initialized by the CLI or Alembic migration, the first endpoint that accesses a table gets `sqlalchemy.exc.OperationalError: no such table: cases`.

**Impact:** Fresh deployments crash on first request because database tables do not exist.

---

### Bug 31 — `validate_file_path` Does Not Check File Existence

**File:** `api/routes/documents.py:36-77`

```python
resolved = raw.resolve(strict=False)
```

Uses `strict=False`, so the path is resolved WITHOUT checking if it exists. A non-existent file path passes validation. The Celery task then fails at `celery_app.py:567` with `os.path.getsize(file_path)` → `FileNotFoundError`, but only after the request is accepted and a `job_id` returned. The user sees "analysis pending" followed by "analysis failed" with no clear error.

**Impact:** Users receive task failures instead of immediate 400 errors when providing non-existent file paths.

---

### Bug 32 — `app = create_app()` Called at Module Level, Cannot Be Re-created with Different Settings

**File:** `api/main.py:267`

```python
app = create_app()
```

The FastAPI application instance is created at module import time. There is no factory function that accepts parameters. Every test that imports `api.main` gets the same app instance. Integration tests that need a different configuration (e.g., different DB, different rate limits) cannot create a fresh app. This also means `ValidationConfig.from_settings()` is called once and never re-initialized.

**Impact:** Integration tests cannot reconfigure the API; all tests share the same global app state.
