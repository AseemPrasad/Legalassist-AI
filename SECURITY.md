# Security Policy 🛡️

## 🔒 Working Principles of the Codebase

This section describes the "working principle" of each file in the **Legalassist-AI** repository to help researchers and contributors understand the internal logic and security boundaries.

### 🧩 Core Application & UI
| File | Principle |
| :--- | :--- |
| `app.py` 🚀 | **Main Entry Point**: Orchestrates the Streamlit UI, handles PDF uploads, and coordinates between the LLM engine and the database. |
| `auth.py` 🔑 | **Security Gatekeeper**: Manages User Authentication via JWT, session persistence, and password hashing logic. |
| `case_manager.py` 📂 | **Business Logic Layer**: Handles CRUD operations for legal cases, document mapping, and case history management. |
| `notifications_ui.py` 🔔 | **Alert Interface**: Dedicated UI components for managing user notification preferences and history. |

### 🧠 Backend Engines & Services
| File | Principle |
| :--- | :--- |
| `analytics_engine.py` 📊 | **Data Aggregator**: Processes case records to generate regional trends and statistical insights for the dashboard. |
| `notification_service.py` ✉️ | **Communication Hub**: Dispatches external notifications (Email/SMS) based on scheduled deadlines. |
| `pdf_exporter.py` 📄 | **Document Generator**: Converts analyzed judgments and legal drafts into professional PDF formats. |
| `scheduler.py` ⏰ | **Temporal Controller**: Manages periodic background jobs like deadline monitoring and automated reminders. |
| `celery_app.py` 🏗️ | **Distributed Tasks**: Configures Celery for handling resource-intensive operations asynchronously. |

### ⚙️ Configuration & Infrastructure
| File | Principle |
| :--- | :--- |
| `config.py` 🛠️ | **Global Settings**: Centralizes environment variables, API keys, and application-wide constants. |
| `database.py` 🗄️ | **Persistence Layer**: Defines SQLAlchemy models and manages the database connection lifecycle. |
| `logging_config.py` 📝 | **Observability**: Standardizes log formats and levels across all modules for easier debugging. |
| `Dockerfile` / `Dockerfile.api` 🐳 | **Environment Blueprint**: Defines the containerized environment for consistent deployment. |
| `docker-compose.yml` 🚢 | **Orchestration**: Defines multi-container setups for the app, database, and Redis. |

### 🛠️ Utilities & CLI
| File | Principle |
| :--- | :--- |
| `core.py` 🛠️ | **Core Utilities**: Contains low-level helper functions for PDF text extraction and text compression. |
| `cli.py` / `deadline_cli.py` 💻 | **Terminal Access**: Provides administrative tools for managing the system via the command line. |
| `modify_pdf.py` 🖋️ | **PDF Manipulation**: Specialized logic for editing or watermarking PDF documents. |
| `verify_otp_protection.py` 🛡️ | **Security Audit Tool**: Validates the integrity and effectiveness of the OTP protection mechanism. |
| `fix_secrets.py` 🤫 | **Secret Management**: Utility for rotating or repairing sensitive configuration secrets. |

---

## 🛡️ Reporting a Vulnerability

We take the security of our legal assistance platform seriously. If you discover a security vulnerability, please help us protect our users by reporting it responsibly.

### 🚩 Reporting Process
1. **Do not** open a public GitHub issue for security vulnerabilities.
2. Send a detailed report to **security@legalassist-ai.io** (placeholder).
3. Include a description of the vulnerability, steps to reproduce, and potential impact.

### 📜 Our Commitment
- We will acknowledge receipt of your report within **48 hours**.
- We will provide an estimated timeframe for a fix.
- We will notify you once the vulnerability is resolved.

### 🚫 Prohibited Actions
- Performing denial of service (DoS) attacks.
- Accessing or modifying data that does not belong to you.
- Social engineering against our team or users.

### 🔐 Production Controls
- JWT, Twilio, SendGrid, database, and Redis secrets must come from environment variables or a managed secret store.
- Rotate JWT secrets by setting `JWT_SECRET` to the new value while keeping `JWT_SECRET_PREVIOUS` for a grace period, then remove the previous value after the longest token lifetime expires.
- Production deployments must use HTTPS at the ingress or load balancer; application URLs must begin with `https://`.
- `DEBUG` and `TESTING` must remain disabled in production.
- CI should run `scripts/check_secrets_policy.py` before deployment to catch committed secrets or unsafe defaults.
- CI pipelines fail on HIGH severity Bandit findings. MEDIUM severity findings must be reviewed and ignored using `# nosec` only if justified.

---
*Thank you for helping keep Legalassist-AI secure!* 🙏
