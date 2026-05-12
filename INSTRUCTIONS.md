# ⚖️ LegalAssist AI - Codebase Instructions

Welcome to the **LegalAssist AI** codebase! This document describes the working principles and roles of all files present in the repository to help developers and maintainers navigate the system effectively.

---

## 🏗️ Core Architecture

### 🔐 `auth.py`
The heart of our security system. It manages **Email-based OTP authentication**, **JWT session handling**, and **Organization context**. It ensures that users are correctly scoped to their law firms and roles (Partner, Associate, Paralegal).

### 🗄️ `database.py`
Defines the **SQLAlchemy ORM models** and low-level database operations. It includes models for `Users`, `Organizations`, `Cases`, `Documents`, `Deadlines`, and `AuditLogs`. It uses SQLite by default for simplicity and speed.

### 💼 `case_manager.py`
The business logic layer for **Case Management**. It handles CRUD operations for cases, document uploads (linked to cases), and organizational permission checks to ensure team members can only see what they are authorized to see.

### 📄 `core.py`
The document processing engine. It handles **PDF text extraction**, **OCR** (via Tesseract) for scanned judgments, and **text compression** to ensure judgments fit within LLM token limits.

### 📊 `analytics_engine.py`
A statistical powerhouse that calculates **Case Similarity**, **Judge Win Rates**, and **Appeal Probabilities**. it aggregates historical data to provide actionable insights for legal practitioners.

---

## 🌐 Web Interface (Streamlit)

Our UI is organized into logical pages within the `pages/` directory:

- 🏠 **`pages/0_Home.py`**: The main landing page for uploading judgments and initiating analysis.
- 📈 **`pages/1_Analytics_Dashboard.py`**: Visualizes national legal trends and team-specific performance metrics.
- 🔍 **`pages/2_Case_Details.py`**: A deep-dive view into specific cases, allowing users to upload documents and view extracted remedies.
- 🗓️ **`pages/3_Deadline_Tracker.py`**: A collaborative calendar for tracking appeal deadlines across the entire organization.
- 👥 **`pages/4_Team_Management.py`**: The administrative hub for creating organizations, inviting team members, and viewing audit logs.
- 🔌 **`pages/5_REST_API.py`**: A developer-focused page demonstrating how to use the LegalAssist SDK and API.

---

## 🛠️ Services & Utilities

### 🔔 `notification_service.py`
Manages the dispatch of notifications via **Email (SendGrid)** and **SMS (Twilio)**. It is used to alert users about upcoming deadlines and account activity.

### ⏰ `scheduler.py`
A background worker (using APScheduler) that periodically checks for upcoming deadlines and triggers the `notification_service`.

### 💻 `cli.py`
A robust **Command Line Interface** for batch processing hundreds of judgments at once. It supports parallel workers, resume capability, and CSV/JSON exports.

### 🧪 `scripts/generate_sample_analytics_data.py`
A utility script to populate the database with realistic sample cases for testing and demonstration purposes.

---

## 📦 Project Configuration

- **`requirements.txt`**: Lists all Python dependencies required to run the application.
- **`pyproject.toml`**: Modern project metadata configuration, helping IDEs resolve dependencies and managing build settings.
- **`.env`**: (User-created) Stores sensitive environment variables like API keys.

---

## 🐍 SDK

- **`sdk/python/client.py`**: A standard Python client library that wraps the LegalAssist REST API, making it easy to integrate our analysis engine into other legal tech platforms.

---

*“Justice delayed is justice denied. We make justice understandable.”* ⚖️
