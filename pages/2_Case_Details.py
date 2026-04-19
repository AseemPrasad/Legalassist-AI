"""
Case Detail Page - LegalAssist AI.
View case timeline, documents, deadlines, and remedies.
"""

import streamlit as st
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

from auth import require_auth, redirect_to_login, get_current_user_id
from case_manager import get_case_detail, upload_case_document, mark_deadline_completed, mark_deadline_incomplete, add_manual_deadline, mark_case_appealed, mark_case_closed, mark_case_active, generate_case_summary_text
from database import DocumentType, CaseStatus

# Page config
st.set_page_config(
    page_title="Case Details - LegalAssist AI",
    page_icon="📄",
    layout="wide",
)

# Using default Streamlit theme



def get_timeline_icon(event_type: str) -> str:
    """Get icon for timeline event type"""
    icons = {
        "case_created": "📁",
        "document_uploaded": "📄",
        "deadline_created": "⏰",
        "deadline_completed": "✅",
        "status_changed": "🔄",
        "appeal_filed": "📤",
    }
    return icons.get(event_type, "📌")


def render_timeline_section(timeline: list):
    """Render timeline visualization"""
    st.subheader("📅 Case Timeline")

    if not timeline:
        st.info("No timeline events yet. Upload a document to start tracking.")
        return

    # Sort by date descending (most recent first)
    sorted_timeline = sorted(timeline, key=lambda x: x["event_date"], reverse=True)

    for event in sorted_timeline:
        icon = get_timeline_icon(event["event_type"])
        event_date = datetime.fromisoformat(event["event_date"])
        date_str = event_date.strftime("%d %b %Y, %H:%M")

        with st.container():
            st.markdown(
                f"""
                <div class="timeline-item">
                    <div class="timeline-date">{date_str}</div>
                    <div>{icon} <strong>{event["event_type"].replace("_", " ").title()}</strong></div>
                    <div style="margin-top: 8px;">{event["description"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_documents_section(case_id: int, documents: list, user_id: int):
    """Render documents list and upload"""
    st.subheader("📄 Documents")

    if documents:
        for doc in documents:
            doc_date = datetime.fromisoformat(doc["uploaded_at"]).strftime("%d %b %Y")

            with st.container(border=True):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"### {doc['document_type']}")
                    st.caption(f"Uploaded: {doc_date}")

                    if doc.get("summary"):
                        with st.expander("📝 View Summary"):
                            st.write(doc["summary"])

                    if doc.get("has_remedies"):
                        st.success("✅ Legal remedies extracted")

                with col2:
                    if st.button("📄 View Full", key=f"doc_{doc['id']}"):
                        st.session_state.view_document_id = doc["id"]
                        st.rerun()

        st.markdown("---")

    # Upload new document
    with st.expander("📤 Upload New Document"):
        st.markdown("**Add document to this case**")

        doc_type = st.selectbox(
            "Document Type",
            ["FIR", "ChargeSheet", "Judgment", "Appeal", "Order", "Other"],
            key="new_doc_type",
        )

        # Option to paste text or upload file
        upload_method = st.radio(
            "Upload method",
            ["Paste text", "Upload PDF"],
            key="upload_method",
        )

        if upload_method == "Paste text":
            document_text = st.text_area(
                "Document Text",
                placeholder="Paste the full document text here...",
                height=200,
                key="new_doc_text",
            )
        else:
            uploaded_pdf = st.file_uploader("Upload Judgment PDF", type=["pdf"])
            document_text = None
            if uploaded_pdf:
                from pypdf import PdfReader
                try:
                    reader = PdfReader(uploaded_pdf)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    if text.strip():
                        document_text = text
                    else:
                        st.error("No extractable text found in PDF.")
                except Exception as e:
                    st.error(f"Error reading PDF: {str(e)}")

        if st.button("📤 Upload Document", use_container_width=True):
            if document_text:
                with st.spinner("Processing document..."):
                    success = upload_case_document(
                        user_id=user_id,
                        case_id=case_id,
                        document_type=DocumentType[doc_type.upper()],
                        document_content=document_text,
                    )

                    if success:
                        st.success("✅ Document uploaded successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to upload document.")


def render_deadlines_section(case_id: int, deadlines: list, user_id: int):
    """Render deadlines list and management"""
    st.subheader("⏰ Deadlines")

    if not deadlines:
        st.info("No deadlines yet. Deadlines are auto-created from remedies advice.")
    else:
        # Separate completed and pending
        pending = [d for d in deadlines if not d["is_completed"]]
        completed = [d for d in deadlines if d["is_completed"]]

        # Show pending first
        if pending:
            st.markdown("**Upcoming Deadlines**")
            for d in sorted(pending, key=lambda x: x["deadline_date"]):
                deadline_date = datetime.fromisoformat(d["deadline_date"])
                days = d.get("days_until")

                if days is not None:
                    if days <= 3:
                        urgency = "urgent"
                        emoji = "🔴"
                    elif days <= 10:
                        urgency = "soon"
                        emoji = "🟠"
                    else:
                        urgency = "normal"
                        emoji = "🟢"
                else:
                    urgency = "normal"
                    emoji = "🟢"

                date_str = deadline_date.strftime("%d %b %Y")

                with st.container():
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(
                            f"""
                            <div class="deadline-card deadline-{urgency}">
                                {emoji} <strong>{d["deadline_type"].title()}</strong> - {date_str}
                                {f"({days} days left)" if days else ""}
                                <br><small>{d.get("description", "")}</small>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col2:
                        if st.button("✓ Mark Done", key=f"complete_{d['id']}"):
                            mark_deadline_completed(user_id, d["id"])
                            st.rerun()

        # Show completed
        if completed:
            st.markdown("---")
            st.markdown("**Completed Deadlines**")
            for d in completed:
                deadline_date = datetime.fromisoformat(d["deadline_date"])
                date_str = deadline_date.strftime("%d %b %Y")

                with st.container():
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(
                            f"""
                            <div class="deadline-card deadline-completed">
                                ✅ <s><strong>{d["deadline_type"].title()}</strong> - {date_str}</s>
                                <br><small>{d.get("description", "")}</small>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col2:
                        if st.button("↩️ Undo", key=f"undo_{d['id']}"):
                            mark_deadline_incomplete(user_id, d["id"])
                            st.rerun()

    st.markdown("---")

    # Add manual deadline
    with st.expander("➕ Add Manual Deadline"):
        with st.form("add_deadline"):
            col1, col2 = st.columns(2)

            with col1:
                deadline_type = st.selectbox(
                    "Type",
                    ["Appeal", "Filing", "Submission", "Response", "Hearing", "Other"],
                )
                deadline_date = st.date_input(
                    "Date",
                    value=datetime.now() + timedelta(days=30),
                    min_value=datetime.now(),
                )

            with col2:
                description = st.text_input("Description", placeholder="Brief description...")

            submitted = st.form_submit_button("➕ Add Deadline", use_container_width=True)

            if submitted:
                deadline_datetime = datetime.combine(deadline_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                success = add_manual_deadline(
                    user_id=user_id,
                    case_id=case_id,
                    case_title=st.session_state.current_case_title or "Case",
                    deadline_date=deadline_datetime,
                    deadline_type=deadline_type.lower(),
                    description=description,
                )

                if success:
                    st.success("✅ Deadline added!")
                    st.rerun()
                else:
                    st.error("Failed to add deadline.")


def render_remedies_section(remedies: Optional[Dict]):
    """Render remedies advice section"""
    st.subheader("⚖️ Legal Remedies & Advice")

    if not remedies:
        st.info("No remedies advice available. Upload a judgment document to get advice.")
        return

    col1, col2 = st.columns(2)

    with col1:
        if remedies.get("what_happened"):
            st.markdown("**What Happened?**")
            st.write(remedies["what_happened"])

        if remedies.get("can_appeal"):
            st.markdown("**Can You Appeal?**")
            st.write(remedies["can_appeal"])

        if remedies.get("first_action"):
            st.markdown("**First Action**")
            st.success(f"✅ {remedies['first_action']}")

    with col2:
        if remedies.get("appeal_days"):
            st.metric("Days to Appeal", remedies["appeal_days"])

        if remedies.get("appeal_court"):
            st.markdown("**Appeal Court**")
            st.write(remedies["appeal_court"])

        if remedies.get("cost_estimate"):
            st.markdown("**Estimated Cost**")
            st.write(remedies["cost_estimate"])

        if remedies.get("deadline"):
            st.markdown("**Important Deadline**")
            st.warning(f"⏰ {remedies['deadline']}")


def render_case_actions(case: Dict, user_id: int):
    """Render case status actions"""
    st.subheader("🔧 Case Actions")

    col1, col2, col3 = st.columns(3)

    current_status = case.get("status", "active")

    with col1:
        if current_status != "appealed":
            if st.button("📤 Mark as Appealed", use_container_width=True, key="mark_appealed"):
                mark_case_appealed(user_id, case["id"])
                st.rerun()

    with col2:
        if current_status != "closed":
            if st.button("⚫ Mark as Closed", use_container_width=True, key="mark_closed"):
                mark_case_closed(user_id, case["id"])
                st.rerun()

    with col3:
        if current_status != "active":
            if st.button("🟢 Mark as Active", use_container_width=True, key="mark_active"):
                mark_case_active(user_id, case["id"])
                st.rerun()


def main():
    """Main case detail page logic"""
    # Require authentication
    if not require_auth():
        st.warning("🔐 Please log in to view case details")
        if st.button("Go to Login"):
            redirect_to_login()
        return

    user_id = get_current_user_id()

    # Get case ID from session or query params
    case_id = st.session_state.get("selected_case_id")

    if not case_id:
        st.warning("No case selected")
        if st.button("← Back to My Cases"):
            st.switch_page("pages/1_My_Cases.py")
        return

    # Get case details
    case_data = get_case_detail(user_id, case_id)

    if not case_data:
        st.error("Case not found or access denied")
        if st.button("← Back to My Cases"):
            st.switch_page("pages/1_My_Cases.py")
        return

    case = case_data["case"]
    documents = case_data["documents"]
    timeline = case_data["timeline"]
    deadlines = case_data["deadlines"]
    remedies = case_data.get("remedies")

    # Store case title in session for deadline creation
    st.session_state.current_case_title = case.get("title") or case.get("case_number")

    # Header
    col1, col2 = st.columns([3, 1])

    with col1:
        st.title(f"📄 {case.get('title') or case['case_number']}")
        st.caption(f"Case No: {case['case_number']}")

    with col2:
        status_class = f"status-{case['status']}"
        st.markdown(
            f'<span class="status-badge {status_class}">{case["status"]}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Case info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Case Type", case["case_type"].title())
    with col2:
        st.metric("Jurisdiction", case["jurisdiction"])
    with col3:
        created_date = datetime.fromisoformat(case["created_at"]).strftime("%d %b %Y")
        st.metric("Created", created_date)

    st.markdown("---")

    # Main content - tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Timeline", "📄 Documents", "⏰ Deadlines", "⚖️ Remedies"])

    with tab1:
        render_timeline_section(timeline)

    with tab2:
        render_documents_section(case_id, documents, user_id)

    with tab3:
        render_deadlines_section(case_id, deadlines, user_id)

    with tab4:
        render_remedies_section(remedies)

    st.markdown("---")

    # Case actions
    render_case_actions(case, user_id)

    # Export options
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        from pdf_exporter import generate_case_pdf
        pdf_bytes = generate_case_pdf(user_id, case_id)
        if pdf_bytes:
            st.download_button(
                label="📥 Download PDF Summary",
                data=pdf_bytes,
                file_name=f"case_summary_{case['case_number']}.pdf",
                mime="application/pdf",
                key="download_summary_pdf",
                use_container_width=True
            )

    with col2:
        from pdf_exporter import generate_anonymized_pdf
        from case_manager import generate_anonymized_case_data
        
        anon_data = generate_anonymized_case_data(case_id)
        if anon_data:
            anon_id = anon_data["anonymized_id"]
            anon_pdf_bytes = generate_anonymized_pdf(case_id, anon_id)
            if anon_pdf_bytes:
                st.download_button(
                    label="🔗 Download Anonymized PDF for Lawyer",
                    data=anon_pdf_bytes,
                    file_name=f"anonymized_case_{anon_id}.pdf",
                    mime="application/pdf",
                    key="download_anon_pdf",
                    use_container_width=True
                )
                
                if st.button("Show Share ID", use_container_width=True):
                    st.success(f"✅ Anonymized ID: `{anon_id}`")
                    st.info("Share this ID with lawyers to show anonymized case details (feature coming soon)")


if __name__ == "__main__":
    main()
