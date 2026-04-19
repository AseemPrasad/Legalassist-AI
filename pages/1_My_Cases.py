"""
My Cases Dashboard - LegalAssist AI.
View all cases, search, filter, and navigate to case details.
"""

import streamlit as st
from datetime import datetime, timezone
from typing import Optional

from auth import require_auth, redirect_to_login, get_current_user_id, get_current_user_email
from case_manager import get_user_cases_summary, get_or_create_case_for_document
from database import CaseStatus, DocumentType

# Page config
st.set_page_config(
    page_title="My Cases - LegalAssist AI",
    page_icon="📁",
    layout="wide",
)

@st.dialog("📥 Export Case Summary")
def export_dialog(user_id, case_id):
    from case_manager import generate_case_summary_text
    from pdf_exporter import generate_case_pdf
    
    st.write("Your case summary is ready. Choose your preferred format:")
    
    with st.spinner("Preparing files..."):
        txt_summary = generate_case_summary_text(user_id, case_id)
        pdf_bytes = generate_case_pdf(user_id, case_id)
        
    col1, col2 = st.columns(2)
    with col1:
        if txt_summary:
            st.download_button(
                label="📄 Download TXT",
                data=txt_summary,
                file_name=f"case_summary_{case_id}.txt",
                mime="text/plain",
                use_container_width=True
            )
    with col2:
        if pdf_bytes:
            st.download_button(
                label="📑 Download PDF",
                data=pdf_bytes,
                file_name=f"case_summary_{case_id}.pdf",
                mime="application/pdf",
                use_container_width=True
            )



def render_stats_bar(cases: list):
    """Render statistics bar"""
    total = len(cases)
    active = len([c for c in cases if c["status"] == "active"])
    appealed = len([c for c in cases if c["status"] == "appealed"])
    closed = len([c for c in cases if c["status"] == "closed"])

    # Count upcoming deadlines
    upcoming = sum(1 for c in cases if c.get("days_until_deadline") and c["days_until_deadline"] <= 30)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("📁 Total Cases", total)
    with col2:
        st.metric("🟢 Active", active)
    with col3:
        st.metric("🟠 Appealed", appealed)
    with col4:
        st.metric("⚫ Closed", closed)
    with col5:
        st.metric("⏰ Upcoming Deadlines", upcoming)


def render_case_card(case: dict):
    """Render a single case card"""
    case_id = case["id"]
    status = case["status"]

    # Status badge color
    status_class = f"status-{status}"

    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            # Case title and number
            st.markdown(f"### {case['title'] or case['case_number']}")
            st.caption(f"Case No: {case['case_number']}")

            # Case type and jurisdiction
            st.markdown(f"**Type:** {case['case_type'].title()} | **Jurisdiction:** {case['jurisdiction']}")

            # Document count
            st.caption(f"📄 {case['document_count']} document(s) uploaded")

        with col2:
            # Status badge
            st.markdown(
                f'<span class="status-badge {status_class}">{status}</span>',
                unsafe_allow_html=True,
            )

            # Latest document
            if case.get("latest_document_type"):
                doc_date = case.get("latest_document_date", "")
                if doc_date:
                    doc_date_str = datetime.fromisoformat(doc_date).strftime("%d %b %Y")
                else:
                    doc_date_str = "Unknown"
                st.caption(f"Latest: {case['latest_document_type']} ({doc_date_str})")

            # Next deadline
            if case.get("next_deadline_date"):
                deadline_date = datetime.fromisoformat(case["next_deadline_date"])
                days = case.get("days_until_deadline")

                if days is not None:
                    if days <= 3:
                        urgency_class = "deadline-urgent"
                        emoji = "🔴"
                    elif days <= 10:
                        urgency_class = "deadline-soon"
                        emoji = "🟠"
                    else:
                        urgency_class = "deadline-normal"
                        emoji = "🟢"

                    deadline_str = deadline_date.strftime("%d %b %Y")
                    st.markdown(
                        f'{emoji} **Next Deadline:** <span class="{urgency_class}">{case["next_deadline_type"]} in {days} days ({deadline_str})</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No upcoming deadlines")

        with col3:
            # Action buttons
            if st.button("📄 View", key=f"view_{case_id}", use_container_width=True):
                st.session_state.selected_case_id = case_id
                st.switch_page("pages/2_Case_Details.py")

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("📥 Export", key=f"export_{case_id}", use_container_width=True):
                export_dialog(get_current_user_id(), case_id)


def render_empty_state():
    """Render empty state when no cases exist"""
    st.markdown(
        """
        <div style="text-align: center; padding: 60px 20px;">
            <h2 style="color: #888;">📁 No cases yet</h2>
            <p style="color: #666;">Upload your first judgment to create a case</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        if st.button("📤 Upload Judgment", use_container_width=True):
            st.switch_page("app.py")
    with col3:
        pass


def render_create_case_modal():
    """Render modal for creating a new case"""
    with st.form("create_case_form"):
        st.subheader("📁 Create New Case")

        col1, col2 = st.columns(2)
        with col1:
            case_number = st.text_input("Case Number *", placeholder="e.g., CASE-2024-001")
            case_title = st.text_input("Case Title", placeholder="e.g., Property Dispute")
        with col2:
            case_type = st.selectbox(
                "Case Type *",
                ["Civil", "Criminal", "Family", "Labor", "Consumer", "Tax", "Other"],
            )
            jurisdiction = st.text_input("Jurisdiction *", placeholder="e.g., Delhi, Mumbai High Court")

        submitted = st.form_submit_button("🚀 Create Case", use_container_width=True)

        if submitted:
            if not case_number or not case_type or not jurisdiction:
                st.error("Please fill in all required fields (*)")
            else:
                user_id = get_current_user_id()
                case = get_or_create_case_for_document(
                    user_id=user_id,
                    new_case_number=case_number,
                    new_case_type=case_type.lower(),
                    new_jurisdiction=jurisdiction,
                    new_title=case_title,
                )

                if case:
                    st.success(f"✅ Case '{case_number}' created successfully!")
                    st.session_state.selected_case_id = case.id
                    st.rerun()
                else:
                    st.error("Failed to create case. Please try again.")


def main():
    """Main dashboard logic"""
    # Require authentication
    if not require_auth():
        st.warning("🔐 Please log in to view your cases")
        if st.button("Go to Login"):
            redirect_to_login()
        return

    user_id = get_current_user_id()
    user_email = get_current_user_email()

    # Header
    st.title("📁 My Cases")
    st.markdown(f"*Welcome back, {user_email}*")

    st.markdown("---")

    # Stats bar
    cases = get_user_cases_summary(user_id, include_closed=True)

    if cases:
        render_stats_bar(cases)
        st.markdown("---")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Active", "Appealed", "Closed", "Pending"],
            key="status_filter",
        )

    with col2:
        type_filter = st.selectbox(
            "Filter by Type",
            ["All"] + list(set(c["case_type"].title() for c in cases)),
            key="type_filter",
        )

    with col3:
        search_query = st.text_input("🔍 Search", placeholder="Case number or title...", key="search")

    # Apply filters
    filtered_cases = cases

    if status_filter != "All":
        filtered_cases = [c for c in filtered_cases if c["status"] == status_filter.lower()]

    if type_filter != "All":
        filtered_cases = [c for c in filtered_cases if c["case_type"].title() == type_filter]

    if search_query:
        query_lower = search_query.lower()
        filtered_cases = [
            c for c in filtered_cases
            if query_lower in c["case_number"].lower() or (c.get("title") and query_lower in c["title"].lower())
        ]

    # Display cases
    if filtered_cases:
        st.markdown(f"### 📋 Cases ({len(filtered_cases)} shown)")

        for case in filtered_cases:
            render_case_card(case)
    else:
        st.markdown("### 📋 Your Cases")
        if not cases:
            render_empty_state()
        else:
            st.info("No cases match your filters. Try adjusting your search.")

    st.markdown("---")

    # Create new case section
    with st.expander("📁 Create New Case Manually"):
        render_create_case_modal()

    # Handle export has been moved to export_dialog directly invoked by the case card


if __name__ == "__main__":
    main()
