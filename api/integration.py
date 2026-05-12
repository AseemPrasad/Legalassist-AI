"""
REST API Integration module for Flask/Streamlit applications
"""
from functools import wraps
from celery_app import celery_app
import structlog

logger = structlog.get_logger(__name__)


class StreamlitAPIAdapter:
    """Adapter for Streamlit application to use async API tasks"""
    
    @staticmethod
    def queue_document_analysis(text: str, document_type: str = "unknown"):
        """Queue document analysis and return job ID"""
        from celery_app import analyze_document_task
        
        task = analyze_document_task.delay(
            user_id="streamlit-user",
            document_id="doc_" + __import__('uuid').uuid4().hex[:8],
            text=text,
            document_type=document_type
        )
        
        return task.id
    
    @staticmethod
    def get_task_progress(task_id: str):
        """Get task progress for WebSocket streaming"""
        from celery_app import TaskStatus
        
        return TaskStatus.get_task_status(task_id)
    
    @staticmethod
    def get_task_result(task_id: str):
        """Get task result"""
        from celery_app import TaskStatus
        
        status = TaskStatus.get_task_status(task_id)
        if status["status"] == "completed":
            return status["info"]
        return None


class FlaskAPIAdapter:
    """Adapter for Flask application"""
    
    @staticmethod
    def create_flask_blueprint():
        """Create Flask blueprint for API"""
        from flask import Blueprint
        
        bp = Blueprint("api", __name__, url_prefix="/api-bridge")
        
        @bp.route("/task/<task_id>/status")
        def get_task_status(task_id):
            from celery_app import TaskStatus
            
            status = TaskStatus.get_task_status(task_id)
            return status
        
        return bp


# Integration with existing core.py
def integrate_api_with_core():
    """Integrate API with core functions.

    Imports the real document-processing entry points from core.py and
    registers a Celery task that wires them into the async task queue.

    Raises RuntimeError on import failure so broken wiring is surfaced
    immediately at startup rather than silently downgraded to a warning.
    """
    logger.info("Integrating REST API with core application")

    # Import the real core entry points.  Raise immediately if they are
    # missing so broken wiring is visible at startup rather than silently
    # degraded to a no-op fallback.
    try:
        from core import extract_text_from_pdf, build_summary_prompt, parse_remedies_response, compress_text
    except ImportError as exc:
        raise RuntimeError(
            "integrate_api_with_core: required functions could not be imported "
            f"from core.py — {exc}"
        ) from exc

    @celery_app.task(name="core_analyze_document")
    def analyze_document_task_integrated(user_id, document_id, text, document_type):
        """Analyze a document using the core.py pipeline.

        Compresses the input text, builds a summary prompt, and parses
        the remedies response — mirroring the workflow used by the
        Streamlit app.
        """
        safe_text = compress_text(text)
        summary_prompt = build_summary_prompt(safe_text, language="English")
        remedies = parse_remedies_response(text)
        return {
            "document_id": document_id,
            "summary_prompt": summary_prompt,
            "remedies": remedies,
        }


if __name__ == "__main__":
    integrate_api_with_core()
