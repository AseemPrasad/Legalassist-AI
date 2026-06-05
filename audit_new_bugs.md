# Database Layer Security & Correctness Audit — NEW Findings

*Audited files: `database.py`, `database_resolved.py`, `db_shim_clean.py`, `db/session.py`, `db/base.py`, `db/case_service.py`, `db/attachments_service.py`, `db/notifications_service.py`, `db/otp_service.py`, `db/immutable_audit_log.py`, `db/retention_models.py`, `db/retention_service.py`, `db/crud/*.py`, `db/repositories/*.py`, `db/models/*.py`*

---

## BUG 1 — CRITICAL: `append_audit_entry` references undefined variable `db`

**File:** `db/immutable_audit_log.py:101`
**Type:** Runtime NameError / Broken immutable audit chain

```python
def append_audit_entry(
    event_type: str,
    action: str,
    ...
) -> dict:
    from db.session import db_session, _is_sqlite

    db.commit()  # <-- NameError: `db` is not defined anywhere in this scope
```

**Description:** The function has no `db` parameter, no `db` local variable, and no `db` import. Line 101 calls `db.commit()` on an undefined name. This is immediately reached at the start of the function, before the inner `with db_session() as db:` block (line 103) creates a local `db`. The `from db.session import ...` on line 84 does not import `db`.

**Impact:** Any call to `append_audit_entry()` (which is called from `record_audit_event()` → `record_immutable_audit_event()` → `append_audit_entry()`) raises `NameError: name 'db' is not defined`. In `record_audit_event` (`db/crud/audit.py:118`), this is caught and logged but silently swallowed. Result: **every audit event silently fails to write to the immutable audit chain**, breaking the entire tamper-evident audit integrity guarantee. The chain becomes permanently broken on first call.

**Exploit/Trigger:** Any audited action (case creation, user login, document upload, etc.) triggers `record_audit_event`, which triggers the broken chain append. Always fails.

**Severity:** CRITICAL — Zero immutable audit records are ever written. All integrity verification is completely disabled without any error reaching the caller.

---

## BUG 2 — CRITICAL: `IdempotencyKey` model and `IdempotencyKeyStatus` enum do not exist

**File:** `database.py:390,394,402,406-407,409,413,422`
**Type:** Missing model definition → NameError at runtime

```python
def reserve_idempotency_key(db: Session, key: str, method: str, path: str) -> Tuple[IdempotencyKey, bool]:
    from sqlalchemy.exc import IntegrityError
    ik = IdempotencyKey(key=key, method=method, path=path, status=IdempotencyKeyStatus.IN_PROGRESS)
    ...
    existing = db.query(IdempotencyKey).filter(IdempotencyKey.key == key).first()
    ...

def set_idempotency_response(db: Session, key: str, ...) -> IdempotencyKey:
    ik = db.query(IdempotencyKey).filter(IdempotencyKey.key == key)...first()
    ...
    ik.status = IdempotencyKeyStatus.COMPLETED

def get_idempotency_response(db: Session, key: str):
    ik = db.query(IdempotencyKey).filter(IdempotencyKey.key == key, IdempotencyKey.status == IdempotencyKeyStatus.COMPLETED).first()
```

**Description:** The identifiers `IdempotencyKey` and `IdempotencyKeyStatus` are used as if they were SQLAlchemy model and enum classes but **neither is defined anywhere in the codebase**. There is no `class IdempotencyKey(Base)` or `class IdempotencyKeyStatus(enum.Enum)` definition in any model file, nor are they imported from any third-party library. The closest thing is `api/idempotency.py` which has `IdempotencyManager` (a Redis client, not an SQLAlchemy model).

**Impact:** Any import path that triggers `database.py` module loading will not fail at import time (Python resolves names lazily in function bodies). But the first HTTP request to any endpoint using `reserve_idempotency_key`, `set_idempotency_response`, or `get_idempotency_response` will raise `NameError`. The `idempotency_middleware.py` imports all three and uses them on every POST/PUT/PATCH/DELETE request — **every such request crashes**.

**Exploit/Trigger:** Send any POST/PUT/PATCH/DELETE request to the API. The idempotency middleware (`api/idempotency_middleware.py:20`) calls `reserve_idempotency_key()` → `NameError` → HTTP 500.

**Severity:** CRITICAL — All mutating HTTP requests fail with 500. The entire write path of the API is broken.

---

## BUG 3 — CRITICAL: `OTPToken` model does not exist

**File:** `db/retention_service.py:176-182`
**Type:** ImportError at runtime

```python
def purge_expired_otl_tokens(db: Session, cutoff_days: int, dry_run: bool = False) -> tuple[list, int]:
    from db.models.auth import OTPToken  # <-- No such model!
    ...
    q = db.query(OTPToken).filter(...)
```

**Description:** Line 178 imports `OTPToken` from `db.models.auth`. The `auth.py` model file defines `UserRole`, `User`, `OTPVerification`, and `APIKey`. There is no `OTPToken` class anywhere in the codebase. The correct model for OTP is `OTPVerification`.

**Impact:** The first time any retention job scheduler calls `purge_expired_otl_tokens()` (or when the function is imported during module loading depending on Python version), an `ImportError` is raised. This prevents the entire retention service from running, causing expired OTP tokens to accumulate indefinitely.

**Severity:** CRITICAL — Unfixable ImportError blocks all retention enforcement for OTP records. OTP records leak PII forever.

---

## BUG 4 — HIGH: `get_db()` in `database.py` silently discards all writes

**File:** `database.py:218-227`
**Type:** Missing commit/rollback → silent data loss

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**Compare with identical function in `db/session.py:80-89` (the correct version):**
```python
def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
```

**Description:** The `database.py` version of `get_db` never commits or rolls back the session. `db.close()` at line 227 rolls back any uncommitted transaction (SQLAlchemy default behavior). Any code importing `get_db` from `database` (e.g., `from database import get_db`) uses the broken version. `api/sso.py:42` imports it this way.

**Impact:** All database writes made within the `get_db` context are silently rolled back on `close()`. Data that the caller believes was committed is silently discarded. No error is raised.

**Exploit/Trigger:** Call any endpoint that uses `from database import get_db` and performs writes. Data appears to succeed but is gone on next read.

**Severity:** HIGH — Silent data loss in production for all code paths importing `get_db` from `database.py`.

---

## BUG 5 — HIGH: `log_notification` in `db/crud/notifications.py` never commits

**File:** `db/crud/notifications.py:269-298`
**Type:** Missing commit → data never persisted

```python
def log_notification(...) -> NotificationLog:
    log = NotificationLog(...)
    db.add(log)
    db.flush()       # <-- only flush, no commit!
    db.refresh(log)
    return log
```

**Compare with `log_notification` in `database.py:359-387` (the correct version):**
```python
    db.add(log)
    db.commit()      # <-- correct
    db.refresh(log)
```

**Description:** This function uses `flush()` instead of `commit()`. `flush()` synchronizes the ORM session state to the database connection buffer but does NOT commit the transaction. The data is only visible to the current transaction and is rolled back when the session closes unless an outer scope commits.

**Impact:** The `database_resolved.py` re-exports `log_notification` from this module (line 64: `from db.crud.notifications import log_notification`). All callers that switched to `database_resolved` or explicitly import from `db.crud.notifications` get the broken version. Notification logs are silently swallowed.

**Exploit/Trigger:** Any code path that calls `log_notification` from `db.crud.notifications` — notification history appears empty despite ostensibly successful delivery.

**Severity:** HIGH — GDPR-relevant notification delivery logs are silently lost.

---

## BUG 6 — HIGH: `get_similarity_feedback` missing return statement

**File:** `database.py:1178-1193`
**Type:** Missing return → function returns `None`

```python
def get_similarity_feedback(
    db: Session,
    user_id: Optional[str] = None,
    query_signature: Optional[str] = None,
    candidate_case_id: Optional[int] = None,
    limit: int = 100,
) -> List[SimilarityFeedback]:
    query = db.query(SimilarityFeedback)
    if user_id is not None:
        query = query.filter(SimilarityFeedback.user_id == str(user_id))
    if query_signature is not None:
        query = query.filter(SimilarityFeedback.query_signature == query_signature)
    if candidate_case_id is not None:
        query = query.filter(SimilarityFeedback.candidate_case_id == candidate_case_id)

def create_case_comment(
```

**Description:** The function builds the query object but never calls `.all()` or has a `return` statement. It flows directly into the `create_case_comment` function definition. Python implicitly returns `None`.

**Impact:** Any caller expecting a `List[SimilarityFeedback]` gets `None`, causing `AttributeError: 'NoneType' object has no attribute '...'` or similar downstream crashes. Similarity feedback retrieval is completely broken.

**Exploit/Trigger:** Call `get_similarity_feedback()` from `database` — always returns `None`.

**Severity:** HIGH — Completely broken function signature, type-unsafe.

---

## BUG 7 — HIGH: Duplicate `get_user_stats` overwrites working version

**File:** `database.py:963-985` (first definition) and `database.py:1295-1297` (second, broken definition)
**Type:** Overwritten function → returns `None`

```python
# First definition (lines 963-985) - COMPLETE
def get_user_stats(db: Session, user_id: int) -> dict:
    """Calculate high-level stats for user dashboard"""
    cases = get_user_cases(db, user_id)
    ...
    return { ... }   # <-- returns dict

# Second definition (lines 1295-1297) - INCOMPLETE
def get_user_stats(db: Session, user_id: int) -> dict:
    """Get statistics for a user's cases"""
    cases = get_user_cases(db, user_id)
    # NO RETURN - falls through to register_slow_query_listener
```

**Description:** Due to code duplication in this compatibility shim, `get_user_stats` is defined twice. The second definition at line 1295 overwrites the first. The second definition only has `cases = get_user_cases(db, user_id)` and then falls off without a return, so it implicitly returns `None`.

**Impact:** Any caller of `get_user_stats` from `database` gets `None` instead of the expected dictionary. Dashboard pages that call this function crash with `TypeError: 'NoneType' object is not subscriptable` when trying to access keys like `["total_cases"]`.

**Exploit/Trigger:** Call `get_user_stats()` from `database` — always returns `None`.

**Severity:** HIGH — Dashboard analytics completely broken for all users.

---

## BUG 8 — HIGH: SQLite does not enforce foreign keys (missing PRAGMA)

**File:** `db/session.py:23-24`
**Type:** Missing database configuration → silent FK violation

```python
if _is_sqlite:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
```

**No `PRAGMA foreign_keys = ON` anywhere in the codebase.**

**Description:** SQLite by default does NOT enforce foreign key constraints. `PRAGMA foreign_keys = ON` must be set per-connection. This is missing entirely from the engine configuration, connection setup, and `init_db()`. All FK-level `ondelete="CASCADE"` definitions become dead letter — deleting a `Case` does NOT cascade-delete related records. The relationship-level `cascade="all, delete-orphan"` relies on SQLAlchemy loading the related objects first, which is unreliable for bulk operations.

**Impact:** Orphaned records accumulate in `case_documents`, `case_deadlines`, `attachments`, `case_timeline`, `case_notes`, `anonymized_share_tokens`, `case_comments`, `case_presence`, `audit_events`, `knowledge_invalidations`, `case_embeddings`, `case_issues`, `case_arguments`, `precedent_matches`, `user_feedback`, and `notification_logs` tables when a parent `Case` or `User` is deleted. Data integrity violation: referential integrity is completely absent for SQLite environments.

**Exploit/Trigger:** Deploy with SQLite (default for local dev/testing). Delete a user or case. All related records become orphans.

**Severity:** HIGH — Systematic referential integrity violation across the entire database schema. Data retention and GDPR compliance are compromised.

---

## BUG 9 — HIGH: `reserve_notification` and `reserve_idempotency_key` roll back caller's entire transaction

**File:** `db/crud/notifications.py:139-151` and `database.py:390-403,432-469`
**Type:** Transaction scope corruption → silent data loss

```python
# db/crud/notifications.py:139-151
def reserve_notification(...):
    try:
        db.add(log)
        db.commit()
        return log, True
    except IntegrityError:
        db.rollback()   # <-- ROLLBACKS CALLER'S TRANSACTION!
        existing = db.query(NotificationLog).filter(...).first()
        return existing, False
```

**Description:** When a unique constraint violation occurs (concurrent duplicate notification reservation), the `except IntegrityError: db.rollback()` rolls back the **entire database session transaction**, not just the failed INSERT. Any pending changes the caller made before calling `reserve_notification` are silently discarded.

**Impact:** If a caller does:
```python
doc = create_case_document(db, ...)
reserve_notification(db, ...)  # fails with IntegrityError due to race
```
→ `db.rollback()` in `reserve_notification` also rolls back the `create_case_document` insert! The case document is silently lost.

**Exploit/Trigger:** Race condition where two concurrent workers try to send the same notification. One wins, the other's IntegrityError rolls back all caller-side changes.

**Severity:** HIGH — Transaction corruption pattern replicated in 3 separate functions across the codebase.

---

## BUG 10 — HIGH: Bulk UPDATE in `archive_expired_cases` bypasses optimistic locking

**File:** `db/retention_service.py:72-76`
**Type:** Stale write via skipped version_id_col

```python
if ids:
    db.query(Case).filter(Case.id.in_(ids)).update(
        {Case.status: CaseStatus.CLOSED}, synchronize_session="fetch"
    )
    db.commit()
```

**Description:** The `Case` model uses SQLAlchemy's `version_id_col` for optimistic concurrency control (`cases.py:81,88-90`). This mechanism automatically adds `WHERE version = :old_version` to individual UPDATE statements and increments the version. Bulk `update()` bypasses this entirely — no version check is performed and no version increment occurs.

**Impact:** A concurrent write to a Case that is being archived can silently overwrite the user's changes. For example:
1. User edits case title → version checked, OK
2. Retention job bulk-sets status to CLOSED on same case → no version check
3. User's edit is saved → version check succeeds because version was never incremented in step 2
4. But the status was reverted to ACTIVE by the user → the bulk CLOSED status is silently overwritten

**Exploit/Trigger:** Run retention archival while a user is actively editing their case. User's changes survive but the archival status is lost.

**Severity:** HIGH — Optimistic locking guarantee is violated, enabling stale writes and data races.

---

## BUG 11 — HIGH: `archive_expired_cases` deletes wrong cases (logic error in status filter)

**File:** `db/retention_service.py:55-79`
**Type:** Wrong WHERE clause → deletes wrong records

```python
def archive_expired_cases(db, cutoff_days, dry_run=False):
    ...
    active_statuses = {CaseStatus.ACTIVE, CaseStatus.PENDING, CaseStatus.APPEALED}
    q = (
        db.query(Case)
        .filter(Case.status.notin_(active_statuses))   # <-- INVERTED LOGIC
        .filter(Case.updated_at < cutoff)
    )
```

**Description:** The variable is named `active_statuses` and contains statuses considered "active" (`ACTIVE`, `PENDING`, `APPEALED`). The filter uses `.notin_(active_statuses)` — it selects cases whose status is NOT in this set. This means cases that are **already** `CLOSED` (the only non-active status) are selected for archival. But the UPDATE then sets status to... `CLOSED` (line 75: `{Case.status: CaseStatus.CLOSED}`). The function should be selecting `ACTIVE/PENDING/APPEALED` cases older than the cutoff and archiving them, but instead it selects already-closed cases and redundantly re-sets their status to CLOSED.

**Impact:** The filter logic is inverted. Active cases that should be archived are skipped. Already-closed cases are redundantly touched each run. The archive function is completely ineffective at its intended purpose.

**Exploit/Trigger:** Run the retention archival job. It archives zero active cases and wastes cycles on already-closed cases.

**Severity:** HIGH — Retention archival for cases is entirely non-functional. Expired data is never transitioned to archived state.

---

## BUG 12 — MEDIUM: `Report.case_id` has no FK — string column, no referential integrity

**File:** `db/models/reports.py:31-32` and `db/crud/reports.py:45`
**Type:** No foreign key → data integrity violation

```python
class Report(Base):
    __tablename__ = "reports"
    ...
    case_id = Column(String(255), nullable=False)   # <-- No ForeignKey!
```

**While every other model uses:** `Column(Integer, ForeignKey("cases.id", ...))`

**Description:** `Report.case_id` is a plain `String(255)` with no `ForeignKey` constraint. It can reference non-existent cases, contain arbitrary string values, and cannot enforce referential integrity. Reports can be created with `case_id=None` (the function signature says `int` but Rust-style type narrowing doesn't apply — the column happily accepts `None` or strings). The `create_report` function passes it as `case_id: int` but writes it to a `String` column — type coercion is database-dependent.

**Impact:** Database-level referential integrity is missing for reports. Queries joining reports to cases produce incorrect results. Reports can be orphaned. Type mismatch between Python `int` and DB `String` causes hidden type coercion on every write.

**Exploit/Trigger:** Create a report with a missing case — no error is raised. The report points to nothing.

**Severity:** MEDIUM — Referential integrity gap, potential data corruption in reporting.

---

## BUG 13 — MEDIUM: `fetch_next_deadlines_per_case` may return multiple deadlines per case

**File:** `db/repositories/case_queries.py:74-103`
**Type:** Non-deterministic results on date ties

```python
subquery = (
    db.query(
        CaseDeadline.case_id,
        func.min(CaseDeadline.deadline_date).label("next_deadline_date"),
    )
    ...
    .group_by(CaseDeadline.case_id)
    .subquery()
)

next_deadlines = db.query(CaseDeadline).join(
    subquery,
    and_(
        CaseDeadline.case_id == subquery.c.case_id,
        CaseDeadline.deadline_date == subquery.c.next_deadline_date,
    ),
).all()
```

**Description:** If two deadlines for the same case share the exact same `deadline_date` (same day, different times), `func.min()` returns the earlier one, but the join on `deadline_date == next_deadline_date` matches BOTH rows. The function's contract says it returns "the next upcoming deadline" (singular) per case, but the join can return multiple.

**Impact:** Downstream code that expects exactly one `CaseDeadline` per case may see duplicates, causing UI rendering issues or incorrect deadline counts.

**Exploit/Trigger:** Create two deadlines for the same case with the same date → `fetch_next_deadlines_per_case` returns both.

**Severity:** MEDIUM — Non-deterministic behavior under edge condition.

---

## BUG 14 — MEDIUM: `set_idempotency_response` race when row doesn't exist

**File:** `database.py:406-416`
**Type:** TOCTOU race on idempotency key update

```python
def set_idempotency_response(db, key, ...):
    ik = db.query(IdempotencyKey).filter(IdempotencyKey.key == key).with_for_update(read=True).first()
    if not ik:
        ik = IdempotencyKey(key=key, method="POST", path="unknown")
    ...
```

**Description:** The `with_for_update(read=True)` only locks an existing row. If `reserve_idempotency_key` was NOT called first (e.g., if a caller skips it), two concurrent `set_idempotency_response` calls both get `None`, both create a new `IdempotencyKey` object, and both try to commit. The second commit hits a unique constraint violation or silently overwrites. Note: this is contingent on `IdempotencyKey` model existing (Bug 2), which it doesn't, but if fixed, this race remains.

**Impact:** Idempotency guarantee broken under concurrent requests when the reservation step is bypassed.

**Exploit/Trigger:** Send concurrent requests with the same idempotency key for a path that doesn't call `reserve_idempotency_key` first.

**Severity:** MEDIUM — Idempotency violation under concurrency, but requires the reservation step to be skipped.

---

## BUG 15 — MEDIUM: `update_notification_result` creates log with `recipient="unknown"` as fallback

**File:** `db/crud/notifications.py:183-191`

```python
return get_or_create_notification_log(
    db=db,
    ...
    recipient="unknown",   # <-- Hardcoded placeholder
    ...
)[0]
```

**And `database.py:506-517`:**
```python
return log_notification(
    db=db,
    ...
    recipient="unknown",
    ...
)
```

**Description:** When `update_notification_result` doesn't find an existing notification log (the race condition path), it creates a new log with `recipient="unknown"`. This hardcoded string is stored in the database as the actual recipient. If the notification was an email, `recipient="unknown"` means the delivery address is lost forever.

**Impact:** Notification delivery records with missing recipient information. Compliance issue for audit trails that need to track exactly where notifications were sent.

**Severity:** MEDIUM — Audit trail completeness issue for notification delivery.

---

## BUG 16 — MEDIUM: `purge_expired_notifications` uses `sent_at` instead of `created_at`

**File:** `db/retention_service.py:162`

```python
q = db.query(NotificationLog).filter(NotificationLog.sent_at < cutoff)
```

**Description:** `sent_at` is `NULL` for notifications that were never sent (still `PENDING` or `FAILED`). The SQL comparison `NULL < cutoff` evaluates to `false` (actually NULL in SQL). Notifications with `sent_at IS NULL` are never purged, even if they're years old.

**Impact:** `PENDING`/`FAILED` notification logs with `sent_at IS NULL` accumulate indefinitely, never subject to retention. The function should use `created_at` or `COALESCE(sent_at, created_at)`.

**Severity:** MEDIUM — Retention policy enforcement gap for unsent notifications.

---

## BUG 17 — LOW: `retention_service.py:180` minutes vs days confusion

**File:** `db/retention_service.py:176,180`

```python
def purge_expired_otl_tokens(db, cutoff_days, dry_run=False):
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=cutoff_days)
```

**Description:** Parameter is named `cutoff_days` but the implementation uses `timedelta(minutes=...)`. This is likely a copy-paste error. For the default call with `cutoff_days=7`, it purges tokens older than 7 minutes instead of 7 days. Combined with the `OTPToken` model existing bug, this function is doubly broken.

**Severity:** LOW (already broken by Bug 3)

---
