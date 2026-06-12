# LegalAssist AI — Technical Issue Backlog

This backlog contains verified, actionable engineering issues identified across the LegalAssist AI codebase. These issues range from critical runtime bugs and security vulnerabilities to refactoring, performance, and architectural improvements.

---

## Issue 1

**Title:** Home Page Summary & Remedies Display Hidden for English Users

**Category:** Bug

**Severity:** High

**Difficulty:** Intermediate

**Affected Files:** [0_Home.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/pages/0_Home.py)

**Short Description:** The entire UI display, confidence score, remedies rendering, drafting center, audio player, and timeline rendering logic is nested inside the `if language.lower() != "english" and output_language_mismatch_detected(...)` block. If the user's selected language is English, or if no translation mismatch is detected, the summary is successfully generated but never displayed, leaving English users with a blank output screen.

---

## Issue 2

**Title:** Comment Posting Form Unreachable (Dead Code) in Case Details Page

**Category:** Bug

**Severity:** High

**Difficulty:** Beginner

**Affected Files:** [2_Case_Details.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/pages/2_Case_Details.py)

**Short Description:** The Streamlit comment form and reply target selection logic is placed after the `return None` statement inside the `_create_anonymized_share_link()` helper function. Because of this incorrect indentation and placement, the comment-submission widget is treated as dead code and is never rendered, making it impossible for users to post collaboration comments.

---

## Issue 3

**Title:** Vector Store Contamination and Cross-User Data Leakage in RAG Fallback

**Category:** Security

**Severity:** High

**Difficulty:** Advanced

**Affected Files:** [rag_engine.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/core/rag_engine.py), [vector_store.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/core/vector_store.py)

**Short Description:** When Chroma is unavailable, the RAG engine falls back to `ShardedVectorStore` for ephemeral chat indexing. However, `ShardedVectorStore` automatically loads existing case shards from the global disk directory and persists new chat vectors to these shared shard files. This permanently contaminates the production database and leaks confidential document segments across different users' chat sessions.

---

## Issue 4

**Title:** Celery Worker Startup Crashes due to Undefined `get_settings` Import

**Category:** Bug

**Severity:** High

**Difficulty:** Beginner

**Affected Files:** [celery_app.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/celery_app.py)

**Short Description:** The Celery configuration file calls `settings = get_settings()` on line 33 but does not import `get_settings` from the API config module. This immediately raises a `NameError: name 'get_settings' is not defined` and crashes any background Celery worker process on startup.

---

## Issue 5

**Title:** Celery Module Import Fails on Undefined `initialize_observability_for_environment`

**Category:** Bug

**Severity:** High

**Difficulty:** Beginner

**Affected Files:** [celery_app.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/celery_app.py)

**Short Description:** The background task runner calls `initialize_observability_for_environment()` on line 37, but this telemetry setup function is never imported or defined in the module, resulting in a runtime `NameError` that prevents importing tasks.

---

## Issue 6

**Title:** Celery Context Propagator Fails on Undefined `generate_correlation_id`

**Category:** Bug

**Severity:** High

**Difficulty:** Beginner

**Affected Files:** [celery_app.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/celery_app.py)

**Short Description:** The `build_task_context_headers` helper tries to generate unique transaction identifiers using `generate_correlation_id()`. Because this utility is not imported or defined in the file, it raises a `NameError` whenever a task is enqueued with correlation context.

---

## Issue 7

**Title:** Task Status Query Crashes on Undefined `AsyncResult` Reference

**Category:** Bug

**Severity:** High

**Difficulty:** Beginner

**Affected Files:** [celery_app.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/celery_app.py)

**Short Description:** The `TaskStatus.get_task_status` utility method attempts to query task states by instantiating `AsyncResult(task_id, app=celery_app)`. However, `AsyncResult` is never imported from `celery.result`, raising a `NameError` and breaking task lifecycle checks.

---

## Issue 8

**Title:** Undefined `idempotency_key` Raises NameError during Document Analysis

**Category:** Bug

**Severity:** High

**Difficulty:** Beginner

**Affected Files:** [celery_app.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/celery_app.py)

**Short Description:** `analyze_document_task` repeatedly calls `idemp.heartbeat(idempotency_key, ttl=300)` and `idemp.mark_completed(idempotency_key, ...)` without ever defining or initializing the `idempotency_key` variable. This raises a `NameError` as soon as the task enters its first processing phase.

---

## Issue 9

**Title:** Undefined `start_time` Causes NameError on Analysis Finalization

**Category:** Bug

**Severity:** High

**Difficulty:** Beginner

**Affected Files:** [celery_app.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/celery_app.py)

**Short Description:** In the finalization phase of `analyze_document_task`, the task calculates total execution time by referencing a non-existent `start_time` variable. This produces a `NameError` and crashes the worker right before saving the successfully computed analysis results.

---

## Issue 10

**Title:** Task Execution Fails due to Mismatched `update_case_document` Arguments

**Category:** Bug

**Severity:** High

**Difficulty:** Intermediate

**Affected Files:** [celery_app.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/celery_app.py), [database.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/database.py)

**Short Description:** `process_case_document_upload_task` calls `update_case_document` passing extra keyword arguments like `extracted_metadata`, `extraction_method`, and `ocr_used`. Since the database helper signature only accepts content, summary, and remedies, this mismatch raises a runtime `TypeError` and blocks document upload processing.

---

## Issue 11

**Title:** Undefined `get_db` Call Crashes Asynchronous Report Generation

**Category:** Bug

**Severity:** High

**Difficulty:** Beginner

**Affected Files:** [celery_app.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/celery_app.py)

**Short Description:** The `generate_report_task` background worker tries to query the database by invoking `db = next(get_db())`. Since `get_db` is not imported or defined in `celery_app.py`, the task crashes with a `NameError` when transitioning into active status.

---

## Issue 12

**Title:** Stale Column Reference in Report Database Helpers Causes Operational Crashes

**Category:** Bug

**Severity:** High

**Difficulty:** Intermediate

**Affected Files:** [reports.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/db/crud/reports.py), [reports.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/db/models/reports.py)

**Short Description:** `create_report` and `get_report_by_celery_task_id` reference a column named `celery_task_id`. However, the SQLAlchemy `Report` model defines this column as `job_id`. This mismatch raises immediate `TypeError` on insert and `AttributeError` on queries, breaking report generation tracking.

---

## Issue 13

**Title:** `get_similarity_feedback` Lacks Return Statement and Always Returns None

**Category:** Bug

**Severity:** Medium

**Difficulty:** Beginner

**Affected Files:** [database.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/database.py)

**Short Description:** The `get_similarity_feedback` helper filters and queries the database for user search evaluation records but lacks a `return` statement. It returns `None` implicitly, rendering the API search-optimization loop completely non-functional.

---

## Issue 14

**Title:** Mismatched Primary Key Types Between Cases and Reports

**Category:** Bug

**Severity:** Medium

**Difficulty:** Intermediate

**Affected Files:** [reports.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/db/models/reports.py), [cases.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/db/models/cases.py)

**Short Description:** The `Report` model represents its associated case as a `String(255)` column named `case_id`, while the `Case` model uses an auto-incrementing `Integer` primary key. This prevents adding foreign key constraints and requires unsafe string casting during reports compilation.

---

## Issue 15

**Title:** Duplicate APIKey Import in Authentication Routes

**Category:** Refactor

**Severity:** Low

**Difficulty:** Beginner

**Affected Files:** [auth.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/api/routes/auth.py)

**Short Description:** The `APIKey` model is imported twice consecutively from `db.models` on lines 12 and 13. This redundancy should be cleaned up to ensure codebase hygiene and compliance with styling standards.

---

## Issue 16

**Title:** Unused CryptContext Instance Initialized in API Auth Modules

**Category:** Refactor

**Severity:** Low

**Difficulty:** Beginner

**Affected Files:** [auth.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/api/routes/auth.py)

**Short Description:** A module-level `_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")` is initialized on line 24 but is never used since actual password hashing and verification are delegated to `api.auth`. This incurs unnecessary memory overhead and dependency clutter.

---

## Issue 17

**Title:** double-read Stream Consumption Causes Empty Document Uploads

**Category:** Bug

**Severity:** High

**Difficulty:** Intermediate

**Affected Files:** [documents.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/api/routes/documents.py)

**Short Description:** The file upload endpoint calls `validate_file_upload_streaming()` which consumes the request stream, and then calls `await file.read()`. The second read yields an empty byte string `b""` because the cursor is at EOF, causing all uploaded documents to contain blank content.

---

## Issue 18

**Title:** Missing Ownership Checks Permit Unauthorized File Path Analysis (IDOR)

**Category:** Security

**Severity:** High

**Difficulty:** Intermediate

**Affected Files:** [documents.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/api/routes/documents.py)

**Short Description:** When a user requests document analysis via `file_path`, the endpoint only checks if the path lies in allowed directories. It fails to check if the requesting user owns the target file or case, allowing users to inspect other litigators' private attachments.

---

## Issue 19

**Title:** IDOR Vulnerability via Missing Owner Verification in `cancel_analysis`

**Category:** Security

**Severity:** High

**Difficulty:** Intermediate

**Affected Files:** [documents.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/api/routes/documents.py)

**Short Description:** The `cancel_analysis` endpoint revokes Celery tasks using a user-supplied `job_id` but lacks ownership validation. Any authenticated user can guess or sniff a job ID and cancel another user's running analysis or report compilation.

---

## Issue 20

**Title:** Mandatory Content-Length Checks Block all API GET and DELETE Requests

**Category:** Bug

**Severity:** High

**Difficulty:** Intermediate

**Affected Files:** [request_size.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/api/middlewares/request_size.py)

**Short Description:** The request size middleware rejects all incoming requests that lack a `Content-Length` header with a 411 Length Required status. Because HTTP GET, DELETE, and OPTIONS requests typically do not carry body payloads or size headers, this breaks almost all read and delete endpoints.

---

## Issue 21

**Title:** Module-Level Configurations Prevent Dynamic Settings and Test Isolation

**Category:** Refactor

**Severity:** Medium

**Difficulty:** Intermediate

**Affected Files:** [celery_app.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/celery_app.py), [database.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/database.py)

**Short Description:** Critical configuration evaluation, database engine instantiation, and observability setup run immediately at module import time instead of lazily. This prevents environment variable overrides, breaking test collection and isolation across integration tests.

---

## Issue 22

**Title:** Streamlit and FastAPI Fail to Initialize Missing Database Schema Tables

**Category:** Bug

**Severity:** Medium

**Difficulty:** Beginner

**Affected Files:** [database.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/database.py)

**Short Description:** `init_db()` is defined to generate SQLAlchemy tables but is only invoked in the CLI utilities. Streamlit and FastAPI never call it on startup, causing fresh deployments to crash with operational database errors unless manually seeded first.

---

## Issue 23

**Title:** Permissive Path Traversal Resolution via Strict=False in Path Validation

**Category:** Bug

**Severity:** Low

**Difficulty:** Beginner

**Affected Files:** [documents.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/api/routes/documents.py)

**Short Description:** `validate_file_path` resolves paths using `strict=False`, allowing non-existent paths to pass. This defers file-not-found exceptions to background celery tasks rather than returning an immediate 400 Bad Request to the client.

---

## Issue 24

**Title:** Incomplete CSRF Same-Origin Validation Omits Protocol Scheme Checks

**Category:** Security

**Severity:** Medium

**Difficulty:** Intermediate

**Affected Files:** [csrf.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/api/csrf.py)

**Short Description:** The `is_same_origin` utility compares hostnames but originally omitted verifying the URL protocol scheme (`http` vs `https`). This allows pages served over unsecured HTTP on the same host to forge request contexts against HTTPS endpoints.

---

## Issue 25

**Title:** Unbounded Redis Memory Growth from Stale Background Task Results

**Category:** Performance

**Severity:** Medium

**Difficulty:** Intermediate

**Affected Files:** [celery_app.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/celery_app.py)

**Short Description:** The daily `cleanup_old_tasks` job attempts to invoke `backend.cleanup()`, which is unsupported on the standard Celery Redis backend. Stale task metadata accumulates in memory indefinitely, causing gradual performance decay and potential memory exhaust.

---

## Issue 26

**Title:** Short Idempotency Lock TTL Interrupted by Long-Running Document Analysis

**Category:** Performance

**Severity:** Medium

**Difficulty:** Beginner

**Affected Files:** [celery_app.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/celery_app.py)

**Short Description:** The idempotency lock has a hardcoded TTL of 300 seconds. Large legal filings can exceed 5 minutes in LLM processing, allowing locks to expire and triggering duplicate concurrent processing for the same document.

---

## Issue 27

**Title:** Unnecessary Double Database Session Checkouts in Notification Delivery

**Category:** Performance

**Severity:** Medium

**Difficulty:** Beginner

**Affected Files:** [celery_app.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/celery_app.py)

**Short Description:** `send_notification_task` opens and commits two separate connection pool sessions back-to-back to fetch users and preferences. This doubles connection overhead and risks read-consistency issues if preference schemas are updated mid-execution.

---

## Issue 28

**Title:** Conflicting Exponential and Static Retries in Celery Task Configurations

**Category:** Refactor

**Severity:** Low

**Difficulty:** Beginner

**Affected Files:** [celery_app.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/celery_app.py)

**Short Description:** Task decorators declare static delays while task bodies manually compute exponential backoff and raise custom `self.retry()` triggers. This produces overlapping retry triggers and makes notification delivery latency completely unpredictable.

---

## Issue 29

**Title:** Stale CSRF Session Binding Vulnerability on User Login

**Category:** Security

**Severity:** Medium

**Difficulty:** Intermediate

**Affected Files:** [csrf.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/api/csrf.py)

**Short Description:** The double-submit CSRF cookie is not refreshed or regenerated during authentication state transitions. An attacker can seed a valid anonymous CSRF cookie on a shared terminal and hijack the session once the user logs in.

---

## Issue 30

**Title:** Missing Indexes on Large Audit Logs and User Feedback Tables

**Category:** Performance

**Severity:** Low

**Difficulty:** Beginner

**Affected Files:** [analytics.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/db/models/analytics.py)

**Short Description:** Tables like `audit_logs` and `user_feedback` lack database indexes on query-heavy columns such as `user_id` and `created_at`. This will degrade dashboard load times as the tables grow to thousands of rows.

---

## Issue 31

**Title:** Phone Number Masking Utility Obscures Essential Formatting Characters

**Category:** Refactor

**Severity:** Low

**Difficulty:** Beginner

**Affected Files:** [celery_app.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/celery_app.py)

**Short Description:** The PII masking utility relies on naive string indexing to apply mask characters. When phone numbers contain country codes or symbols (e.g. `+1 (555) 123-4567`), formatting characters are corrupted instead of masking only the digit payload.

---

## Issue 32

**Title:** Zero-Day Business Deadlines Fail to Normalize Weekend Placements

**Category:** Bug

**Severity:** Medium

**Difficulty:** Intermediate

**Affected Files:** [deadline_engine.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/core/deadline_engine.py)

**Short Description:** When the `business_days` offset is configured as 0, the deadline engine bypasses holiday and weekend adjustment loops. If a client registers a filing with an offset of 0, the deadline will remain on a weekend or holiday, violating legal procedure.

---

## Issue 33

**Title:** Missing Cascade Constraints Risk Orphaned Attachments on Case Deletion

**Category:** Refactor

**Severity:** Low

**Difficulty:** Beginner

**Affected Files:** [cases.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/db/models/cases.py)

**Short Description:** Foreign key declarations for file attachments lack explicit `ondelete="CASCADE"` hooks. Deleting a case leaves orphaned attachment rows in the database, wasting space and violating storage cleanup policies.

---

## Issue 34

**Title:** Global API Application Declaration Blocks Dynamically Mocked Unit Tests

**Category:** Testing

**Severity:** Medium

**Difficulty:** Intermediate

**Affected Files:** [main.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/api/main.py)

**Short Description:** The FastAPI application instance is declared globally at module load time rather than using a factory pattern. This prevents unit tests from loading the application with mocked config settings or isolated databases.

---

## Issue 35

**Title:** Stale Audit Logs Generated on Silently Swapped OTP Deliveries

**Category:** Refactor

**Severity:** Low

**Difficulty:** Beginner

**Affected Files:** [auth.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/auth.py)

**Short Description:** When SendGrid is not configured in development, the system logs successful OTP deliveries without specifying that they were mocked. This creates misleading logs and incorrect compliance records.

---

## Issue 36

**Title:** Bulk Deletions Avoid SQLAlchemy Cascades in OTP Expiry Operations

**Category:** Refactor

**Severity:** Low

**Difficulty:** Beginner

**Affected Files:** [auth.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/auth.py)

**Short Description:** OTP cleanup operations use bulk SQL deletes (`query.delete()`) which bypass ORM relationship cascading hooks. If the OTP models are ever extended with active session links, it will lead to constraint failures.

---

## Issue 37

**Title:** Aggressive Session Wipes Cause Transient Glitches in Streamlit Components

**Category:** Refactor

**Severity:** Low

**Difficulty:** Beginner

**Affected Files:** [auth.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/auth.py)

**Short Description:** `verify_login` calls `st.session_state.clear()` which removes not only credentials but also user theme selections and page paths. This causes sudden visual flashes and page redirects that degrade user experience.

---

## Issue 38

**Title:** Subprocess Spawn in Maintenance Tasks Prone to System-Level Blocking

**Category:** Bug

**Severity:** Low

**Difficulty:** Intermediate

**Affected Files:** [scheduler.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/scheduler.py)

**Short Description:** `managed_subprocess` spawns system commands using list parameters without a shell context. In Windows environments, executing scripts without wrapper processes can fail or hang, locking system file descriptors indefinitely.

---

## Issue 39

**Title:** Lack of Exception Logging in OTP Reset Operations

**Category:** Refactor

**Severity:** Low

**Difficulty:** Beginner

**Affected Files:** [auth.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/auth.py)

**Short Description:** OTP lockout reset operations swallow all database exceptions and return `False` without generating telemetry. This makes it impossible to diagnose underlying write conflicts or transaction timeouts during high-load logins.

---

## Issue 40

**Title:** Hardcoded Default Cost Breakdown Metrics in Analytics Endpoint

**Category:** Refactor

**Severity:** Low

**Difficulty:** Beginner

**Affected Files:** [analytics.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/api/routes/analytics.py)

**Short Description:** Cost tracking returns a hardcoded 0.0 value for costs and API request aggregates. Users are presented with static, non-functional telemetry that does not reflect their actual resource usage.

---

## Issue 41

**Title:** `LegalRAG` Fallback Method Raises AttributeError on Missing LangChain Retriever

**Category:** Bug

**Severity:** Medium

**Difficulty:** Beginner

**Affected Files:** [rag_engine.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/core/rag_engine.py)

**Short Description:** The exception handler in `retrieve_with_scores` attempts to construct a retriever using `self.vector_store.as_retriever()`. Since `ShardedVectorStore` has no such method, it raises an `AttributeError` instead of gracefully resolving the fallback search.

---

## Issue 42

**Title:** Absence of Schema Version Locking for Critical JSON Fields

**Category:** Refactor

**Severity:** Low

**Difficulty:** Intermediate

**Affected Files:** [cases.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/db/models/cases.py)

**Short Description:** Case document remedies are stored as unstructured JSON columns without schemas or version metadata. As the extraction parser evolves, loading historical documents with older formats will result in parsing failures.

---

## Issue 43

**Title:** Lack of Timeout Parameters in External SSO API Requests

**Category:** Security

**Severity:** Medium

**Difficulty:** Beginner

**Affected Files:** [sso.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/api/sso.py)

**Short Description:** Requests fetching metadata from external identity providers do not specify network timeouts. If the identity provider experiences a slow outage, the server thread will block indefinitely, causing high resource utilization.

---

## Issue 44

**Title:** Streamlit File Uploader Lacks Content-Type Validation Check

**Category:** Security

**Severity:** Medium

**Difficulty:** Beginner

**Affected Files:** [2_Case_Details.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/pages/2_Case_Details.py)

**Short Description:** The file attachment uploader validates extensions but does not verify actual file mime headers. A malicious user can rename an executable file to a `.pdf` extension and upload it to the attachments directory.

---

## Issue 45

**Title:** Direct PDF Stream Extraction Risks High Memory Utilization

**Category:** Performance

**Severity:** Low

**Difficulty:** Intermediate

**Affected Files:** [2_Case_Details.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/pages/2_Case_Details.py)

**Short Description:** The page extraction utility reads entire large PDF files into memory synchronously. Concurrent uploads of several multi-megabyte PDFs can consume all available system RAM, resulting in server Out-Of-Memory termination.

---

## Issue 46

**Title:** Redundant DB Query Sessions checking for RLS and User Identity

**Category:** Performance

**Severity:** Low

**Difficulty:** Intermediate

**Affected Files:** [dependencies.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/api/dependencies.py)

**Short Description:** RLS context retrieval checks verify session credentials by opening separate DB sessions from the main request dependency injection cycle, increasing query latencies.

---

## Issue 47

**Title:** Lack of CSRF Protection Exempt Path Validation Sanitization

**Category:** Security

**Severity:** Low

**Difficulty:** Beginner

**Affected Files:** [csrf.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/api/csrf.py)

**Short Description:** Path exemptions list exact matching strings rather than structured path prefixes. If proxy headers append query params in unexpected cases, CSRF validation might fail or get bypassed irregularly.

---

## Issue 48

**Title:** SQLite Database Locks on Concurrent OTP Generation

**Category:** Performance

**Severity:** Low

**Difficulty:** Intermediate

**Affected Files:** [auth.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/auth.py)

**Short Description:** Concurrent OTP generation requests for multiple users can result in `database is locked` errors in SQLite because of session conflicts, preventing users from logging in during peak load spikes.

---

## Issue 49

**Title:** Missing RLS Initialization on Main Streamlit Entrypoint

**Category:** Security

**Severity:** Medium

**Difficulty:** Intermediate

**Affected Files:** [app.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/app.py)

**Short Description:** The Streamlit application coordinates RLS manually instead of systematically binding RLS session parameters at the connection pool bootstrap level. This creates a risk that custom widgets bypass row-level permissions under specific conditions.

---

## Issue 50

**Title:** Inefficient Full-Text Scan in Citation Engine Matching

**Category:** Performance

**Severity:** Low

**Difficulty:** Beginner

**Affected Files:** [citation_engine.py](file:///c:/Users/Sujal/PROJECTS/NSOC_OS_1/Legalassist-AI/core/citation_engine.py)

**Short Description:** The citation extraction engine relies on executing multiple sequential regular expressions over entire multi-page documents, which results in high CPU usage and long processing delays for large briefs.
