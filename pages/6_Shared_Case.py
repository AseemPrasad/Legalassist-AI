"""Shared Anonymized Case Viewer.

Allows anyone with a valid anonymized share ID to view a redacted case
summary without logging in.  Owner identity and PII are never shown.

Usage:
    Navigate to this page and enter the share ID provided by the case owner,
    or open the direct link:
        /?page=6_Shared_Case&anon_id=<share_id>
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="View Shared Case – LegalAssist AI",
    page_icon="⚖️",
    layout="centered",
)


def _get_api_base() -> str:
    try:
        from core.api_client import get_api_base_url

        return get_api_base_url().rstrip("/")
    except Exception:
        return "http://localhost:8000"


def _fetch_anonymized_case(anon_id: str) -> dict | None:
    """Call the API and return the payload, or None on 404."""
    import requests  # stdlib-compatible; streamlit env always has it

    base = _get_api_base()
    url = f"{base}/api/v1/anonymized-cases/{anon_id}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"Could not reach the API: {exc}")
        return None


def _render_case(payload: dict) -> None:
    """Render the redacted case payload."""
    anon_id = payload.get("anonymized_id", "—")
    profile_label = payload.get("privacy_profile_label", payload.get("privacy_profile", "—"))

    st.success(f"✅ Anonymized case loaded  •  Share ID: `{anon_id}`")
    st.caption(f"Privacy profile: **{profile_label}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Case Type", payload.get("case_type", "—").replace("_", " ").title())
    col2.metric("Jurisdiction", payload.get("jurisdiction", "—"))
    col3.metric("Status", payload.get("status", "—").replace("_", " ").title())

    st.divider()

    # Documents
    documents = payload.get("documents") or []
    st.subheader(f"📄 Documents ({len(documents)})")
    if documents:
        for i, doc in enumerate(documents, 1):
            doc_type = doc.get("type", "Unknown")
            summary = doc.get("summary") or "_No summary available._"
            with st.expander(f"Document {i} – {doc_type}", expanded=(i == 1)):
                st.markdown(f"**Summary:** {summary}")
                remedies = doc.get("remedies")
                if remedies:
                    st.markdown("**Remedies:**")
                    if isinstance(remedies, list):
                        for r in remedies:
                            st.markdown(f"- {r}")
                    else:
                        st.markdown(str(remedies))
    else:
        st.info("No documents available in this anonymized view.")

    # Timeline
    timeline = payload.get("timeline") or []
    st.subheader(f"🕒 Timeline ({len(timeline)} events)")
    if timeline:
        for event in timeline:
            event_type = event.get("event_type", "Event")
            description = event.get("description") or "_No description._"
            st.markdown(f"**{event_type.replace('_', ' ').title()}** — {description}")
    else:
        st.info("No timeline events available in this anonymized view.")

    st.divider()
    st.caption(
        "This is an anonymized view. Owner identity and personal information "
        "have been redacted in accordance with the applied privacy profile."
    )


def main() -> None:
    st.title("⚖️ View Shared Case")
    st.markdown(
        "Enter the **Share ID** provided by the case owner to view an "
        "anonymized summary of their case."
    )

    # Pre-fill from query params if present (e.g. shared link).
    query_params = st.query_params
    default_id = query_params.get("anon_id", "")

    anon_id = st.text_input(
        "Share ID",
        value=default_id,
        placeholder="e.g. 3f9a1b2c4d5e",
        max_chars=64,
        help="The 12-character anonymized ID shown in the Case Details page.",
    )

    if st.button("🔍 Look Up Case", use_container_width=True, type="primary"):
        anon_id = (anon_id or "").strip()
        if not anon_id:
            st.warning("Please enter a Share ID.")
            return

        with st.spinner("Looking up anonymized case…"):
            payload = _fetch_anonymized_case(anon_id)

        if payload is None:
            st.error(
                "No anonymized case found for that Share ID. "
                "Please check the ID and try again."
            )
        else:
            _render_case(payload)


if __name__ == "__main__":
    main()
