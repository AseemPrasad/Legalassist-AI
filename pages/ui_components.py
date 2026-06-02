"""Shared Streamlit UI components for pages/ to reduce duplication."""
from contextlib import contextmanager
from typing import Optional
import streamlit as st
from sqlalchemy import text

from database import SessionLocal


# Centralized session state keys
SESSION_KEYS = {
    "selected_case_id": "selected_case_id",
    "judgment_language": "judgment_language",
    "authenticated": "authenticated",
    "user_id": "user_id",
}


@contextmanager
def db_context():
    """Context manager that yields a DB session and ensures close."""
    db = SessionLocal()
    try:
        yield db
    finally:
        try:
            db.close()
        except Exception:
            pass


def render_header(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.subheader(subtitle)


def require_authentication(message: str = "Please sign in to continue.") -> bool:
    """Return True if the current session is authenticated, otherwise render a notice."""
    if st.session_state.get(SESSION_KEYS["authenticated"]) or st.session_state.get(SESSION_KEYS["user_id"]):
        return True
    st.warning(message)
    return False


def section(title: str):
    """Return a Streamlit container for a titled section.

    Usage:
        with section('My Section'):
            st.write('content')
    """
    st.markdown("---")
    st.subheader(title)
    return st.container()


def case_select_dropdown(db, user_id: int, key: str = SESSION_KEYS["selected_case_id"]) -> Optional[int]:
    """Render a case selection dropdown and return selected case id (or None).

    Expects a `cases` table with `id` and `case_number` fields.
    """
    try:
        cases = db.execute(
            text("SELECT id, case_number FROM cases WHERE user_id = :uid ORDER BY created_at DESC"),
            {"uid": user_id},
        ).fetchall()
    except Exception:
        cases = []

    options = ["Select a case"] + [f"{r[1]} (#{r[0]})" for r in cases]
    selected = st.selectbox("Select Case", options, key=key)
    if selected == "Select a case":
        return None
    # Parse id from label ending with (#id)
    try:
        sid = int(selected.split("(#")[-1].rstrip(")"))
        return sid
    except Exception:
        return None
