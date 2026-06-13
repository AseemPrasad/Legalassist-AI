# Project Instructions & Architecture

## API Reference
Visit `/docs` for Swagger UI. 📖

## 🏛️ System Overview

**Legalassist-AI** is a sophisticated legal technology platform designed to simplify complex judicial judgments using AI. This document provides a detailed breakdown of the project's architecture, file responsibilities, and operational principles.

---

## 📂 Codebase Navigation & Working Principles

Each file in the repository serves a specific role in the application lifecycle. Below is a comprehensive guide to the system's components:

### 🚀 Core Application Layer
| File | Working Principle |
| :--- | :--- |
| `app.py` 🖥️ | **The Nerve Center**: Acts as the main entry point for the Streamlit web interface. It handles user interactions, file uploads, and coordinates requests between the UI and backend logic. |
| `auth.py` 🔐 | **Identity Provider**: Implements JWT-based authentication. It handles user registration, secure login, password hashing, and session state management. |
| `case_manager.py` 💼 | **Case Workflow**: Encapsulates the logic for managing legal cases. It handles document linking, timeline generation, and historical data retrieval. |

### ⚙️ Backend Services & Automation
| File | Working Principle |
| :--- | :--- |
| `database.py` 🗄️ | **Persistence Engine**: Manages the SQLAlchemy ORM models and database connections. It defines the schema for Users, Cases, Documents, and Notifications. |
| `analytics_engine.py` 📊 | **Intelligence Module**: Aggregates raw data from the database to compute success rates, regional trends, and jurisdictional analytics. |
| `notification_service.py` 📢 | **Alert Dispatcher**: Logic for sending multi-channel notifications (Email/SMS). Integrates with external APIs to ensure users never miss a deadline. |
| `scheduler.py` ⏳ | **Job Orchestrator**: Uses APScheduler to run periodic tasks in the background, such as checking for upcoming legal deadlines. |
| `celery_app.py` 🐝 | **Task Queue**: Configures the Celery worker for handling high-latency tasks like deep PDF analysis without blocking the main UI. |

### 🛠️ Utilities & Configuration
| File | Working Principle |
| :--- | :--- |
| `config.py` ⚙️ | **Central Registry**: Loads and validates environment variables, ensuring all modules have access to the correct API keys and service URLs. |
| `pdf_exporter.py` 📑 | **Document Factory**: Specialized logic for generating high-quality PDF reports and legal drafts from AI-generated text. |
| `core.py` 🔧 | **Foundation Utils**: Contains reusable helper functions for text processing, LLM prompt engineering, and PDF extraction. |
| `logging_config.py` 📋 | **Diagnostics**: Sets up a unified logging bridge to capture system events and errors for debugging. |

### 🐳 Infrastructure & DevOps
| File | Working Principle |
| :--- | :--- |
| `Dockerfile` / `Dockerfile.api` | **Container Recipes**: Instructions for building isolated environments for the Streamlit app and the FastAPI service. |
| `docker-compose.yml` | **Service Stack**: Orchestrates the interaction between the application, PostgreSQL database, and Redis cache. |

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.9+
- Docker & Docker Compose (Optional but recommended)
- PostgreSQL

### Local Development Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/KGFCH2/Legalassist-AI.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment**:
   Copy `.env.example` to `.env` and fill in your API keys.
4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

---
*For contributing, please refer to [CONTRIBUTING.md](CONTRIBUTING.md). For security concerns, see [SECURITY.md](SECURITY.md).*
