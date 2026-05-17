"""
Accessibility (a11y) utilities for WCAG 2.1 AA compliance.

Provides helpers for semantic HTML, ARIA attributes, keyboard navigation,
and screen reader support across the application.
"""

import streamlit as st


def a11y_button_label(text: str, context: str = "") -> str:
    """Generate accessible button label with context for screen readers."""
    if context:
        return f"{text} - {context}"
    return text


def a11y_status_message(message: str, role: str = "status") -> None:
    """Announce status messages to screen readers via aria-live."""
    st.markdown(
        f'<div role="{role}" aria-live="polite" class="sr-only">{message}</div>',
        unsafe_allow_html=True
    )


def a11y_progress_indicator(current: int, total: int, label: str = "Progress") -> None:
    """Render accessible progress indicator with ARIA attributes."""
    percentage = int((current / total) * 100) if total > 0 else 0
    st.markdown(
        f"""
        <div role="progressbar" 
             aria-valuenow="{current}" 
             aria-valuemin="0" 
             aria-valuemax="{total}"
             aria-label="{label}">
            <span>{label}: {percentage}%</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def a11y_form_field(label: str, field_id: str, required: bool = False) -> str:
    """Generate accessible form field label with required indicator."""
    required_mark = " *" if required else ""
    return f'<label for="{field_id}">{label}{required_mark}</label>'


def a11y_alert(message: str, alert_type: str = "info") -> None:
    """Render accessible alert with appropriate ARIA role."""
    role_map = {
        "error": "alert",
        "warning": "alert",
        "info": "status",
        "success": "status"
    }
    role = role_map.get(alert_type, "status")
    st.markdown(
        f'<div role="{role}" aria-live="assertive">{message}</div>',
        unsafe_allow_html=True
    )


def a11y_expandable_section(title: str, content: str, expanded: bool = False) -> None:
    """Render accessible expandable section with proper ARIA attributes."""
    aria_expanded = "true" if expanded else "false"
    st.markdown(
        f"""
        <details {'open' if expanded else ''}>
            <summary role="button" aria-expanded="{aria_expanded}">{title}</summary>
            <div>{content}</div>
        </details>
        """,
        unsafe_allow_html=True
    )


def a11y_data_table(headers: list, rows: list) -> None:
    """Render accessible data table with proper semantic structure."""
    header_html = "<thead><tr>" + "".join(f"<th scope='col'>{h}</th>" for h in headers) + "</tr></thead>"
    body_html = "<tbody>" + "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" 
        for row in rows
    ) + "</tbody>"
    st.markdown(
        f'<table role="table" aria-label="Data table">{header_html}{body_html}</table>',
        unsafe_allow_html=True
    )


def a11y_skip_link(target_id: str, label: str = "Skip to main content") -> None:
    """Render accessible skip link for keyboard navigation."""
    st.markdown(
        f'<a href="#{target_id}" class="skip-link">{label}</a>',
        unsafe_allow_html=True
    )