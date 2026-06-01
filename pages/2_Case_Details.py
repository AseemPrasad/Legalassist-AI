"""
Case Detail Page - LegalAssist AI.
View case timeline, documents, deadlines, and remedies.
"""
import sys
import os
# Add parent directory to sys.path to resolve 'core' and other top-level modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import routes
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
import requests

from auth import require_auth, redirect_to_login, get_current_user_id
from case_manager import (
    get_case_detail,
    upload_case_document,
    mark_deadline_completed,
    mark_deadline_incomplete,
    add_manual_deadline,
    mark_case_appealed,
    mark_case_closed,
    mark_case_active,
    generate_case_summary_text,
    add_case_comment,
    update_case_presence,
    upload_case_attachment,
    get_user_cases_summary,
)
from core import extract_text_from_pdf
from api.feature_flags import is_feature_enabled_for_user, get_feature_flag_manager
from db.crud.knowledge import get_knowledge_freshness_summary, list_knowledge_invalidations
from db.crud.audit import list_audit_events
from database import DocumentType, CaseStatus, SessionLocal, UserPreference
import pytz
import html
from config import Config
from pypdf import PdfReader

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

    # Sort by date descending (most recent first). Support both old and new timeline formats.
    sorted_timeline = sorted(timeline, key=lambda x: x.get("timestamp") or x.get("event_date") or "", reverse=True)

    for event in sorted_timeline:
        ev_type = event.get("type") or event.get("event_type")
        icon = get_timeline_icon(ev_type)
        ts = event.get("timestamp") or event.get("event_date") or ""
        try:
            event_date = datetime.fromisoformat(ts)
            date_str = event_date.strftime("%d %b %Y, %H:%M")
        except Exception:
            date_str = ts

        desc = event.get("description", "")

        # Build HTML block; include message preview for notifications
        msg_preview_html = ""
        if event.get("source") == "notification":
            mp = event.get("message_preview")
            if mp:
                # If the preview looks like HTML, render as HTML inside an expander
                if bool(mp.strip().startswith("<")):
                    msg_preview_html = f"<div style=\"margin-top:8px;\"><details><summary>Rendered Message Preview</summary>{mp}</details></div>"
                else:
                    safe_text = html.escape(str(mp))
                    msg_preview_html = f"<div style=\"margin-top:8px;\"><details><summary>Rendered Message Preview</summary><pre style=\"white-space:pre-wrap;\">{safe_text}</pre></details></div>"

        with st.container():
            st.markdown(
                f"""
                <div class="timeline-item">
                    <div class="timeline-date">{date_str}</div>
                    <div>{icon} <strong>{ev_type.replace("_", " ").title()}</strong></div>
                    <div style="margin-top: 8px;">{html.escape(str(desc))}</div>
                    {msg_preview_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_collaboration_section(case_id: int, user_id: int, comments: list, presence: list):
    """Render collaborative discussion and presence state for a case."""
    st.subheader("🤝 Collaboration")
    st.caption("Threaded comments, replies, and recent collaborator activity for this case.")

    live_updates = st.checkbox("Live updates", value=True, key="case_live_updates")
    if live_updates:
        st.markdown('<meta http-equiv="refresh" content="10">', unsafe_allow_html=True)

    update_case_presence(user_id, case_id, active_view="collaboration")

    active_people = [p for p in presence if p.get("last_seen")]
    if active_people:
        names = [p.get("user_email") or f"User {p.get('user_id')}" for p in active_people]
        st.info("Active now: " + ", ".join(names))
    else:
        st.info("No recent collaborator activity yet.")

    comment_by_id = {comment["id"]: comment for comment in comments}
    children = {}
    for comment in comments:
        children.setdefault(comment.get("parent_comment_id"), []).append(comment)

    def render_comment_branch(parent_id, depth=0):
        for comment in sorted(children.get(parent_id, []), key=lambda item: item.get("created_at") or ""):
            author = comment.get("user_email") or f"User {comment.get('user_id')}"
            timestamp = comment.get("created_at", "")
            try:
                timestamp = datetime.fromisoformat(timestamp).strftime("%d %b %Y, %H:%M")
            except Exception:
                pass

            with st.container(border=True):
                indent = "&nbsp;" * (depth * 4)
                st.markdown(f"{indent}**{author}** • {timestamp}", unsafe_allow_html=True)
                st.write(comment.get("comment_text", ""))
                if comment.get("is_resolved"):
                    st.success("Resolved")
            render_comment_branch(comment.get("id"), depth + 1)

    if comments:
        render_comment_branch(None)
    else:
        st.info("No comments yet. Start the discussion below.")

    st.markdown("---")


def _get_api_base_url() -> str:
    return str(st.session_state.get("api_base_url") or Config.API_BASE_URL or "http://localhost:8000").rstrip("/")


def _create_anonymized_share_link(case_id: int, scope: str) -> Optional[str]:
    api_base = _get_api_base_url()
    token = st.session_state.get("user_token")
    if not token:
        st.error("Please sign in again to generate a share link.")
        return None

    try:
        response = requests.post(
            f"{api_base}/api/v1/cases/{case_id}/share-anonymized",
            headers={"Authorization": f"Bearer {token}"},
            json={"scope": scope},
            timeout=float(getattr(Config, "API_REQUEST_TIMEOUT_SECONDS", 5.0)),
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("share_url")
    except requests.RequestException as exc:
        st.error(f"Failed to create share link: {exc}")
        return None

    with st.form("case_comment_form", clear_on_submit=True):
        reply_options = ["Top-level comment"] + [
            f"Reply to #{comment['id']} - {(comment.get('comment_text') or '')[:40]}"
            for comment in comments
        ]
        reply_choice = st.selectbox("Reply target", reply_options)
        comment_text = st.text_area("Add a comment", height=120, placeholder="Share analysis, ask a question, or leave a review note...")
        submitted = st.form_submit_button("Post comment", use_container_width=True)

        if submitted:
            parent_comment_id = None
            if reply_choice != "Top-level comment":
                try:
                    parent_comment_id = int(reply_choice.split("#")[1].split(" ")[0])
                except Exception:
                    parent_comment_id = None

            result = add_case_comment(
                user_id=user_id,
                case_id=case_id,
                comment_text=comment_text,
                parent_comment_id=parent_comment_id,
                active_view="collaboration",
            )
            if result:
                st.success("Comment posted")
                st.rerun()
            else:
                st.error("Failed to post comment")


def render_documents_section(case_id: int, documents: list, user_id: int):
    """Render documents list and upload"""
    st.subheader("📄 Documents")

    if documents:
        for doc in documents:
            doc_date = datetime.fromisoformat(doc["uploaded_at"]).strftime("%d %b %Y")
            metadata = doc.get("extracted_metadata") or {}

            with st.container(border=True):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"### {doc['document_type']}")
                    st.caption(f"Uploaded: {doc_date}")
                    if doc.get("source_attachment_id"):
                        st.caption(f"Linked attachment: #{doc['source_attachment_id']}")

                    if doc.get("summary"):
                        with st.expander("📝 View Summary"):
                            st.write(doc["summary"])

                    if doc.get("has_remedies"):
                        st.success("✅ Legal remedies extracted")

                    if metadata:
                        with st.expander("🔎 Extracted Fields"):
                            if metadata.get("parties"):
                                st.markdown("**Parties**")
                                st.write(", ".join(metadata["parties"]))
                            if metadata.get("dates"):
                                st.markdown("**Dates**")
                                st.write(", ".join(metadata["dates"]))
                            if metadata.get("claims"):
                                st.markdown("**Claims**")
                                st.write("\n".join(f"- {item}" for item in metadata["claims"]))
                            if metadata.get("statutes"):
                                st.markdown("**Statutes**")
                                st.write(", ".join(metadata["statutes"]))
                            st.caption(f"Extraction method: {doc.get('extraction_method') or 'n/a'} · OCR used: {bool(doc.get('ocr_used'))}")

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
                # Validate file size
                file_size_mb = uploaded_pdf.size / (1024 * 1024)
                is_valid = True
                
                if file_size_mb > Config.MAX_FILE_SIZE_MB:
                    st.error(f"🛑 File too large. Maximum size is {Config.MAX_FILE_SIZE_MB}MB.")
                    is_valid = False
                elif file_size_mb > Config.WARN_FILE_SIZE_MB:
                    st.warning("⚠️ This file is quite large. Processing may take longer than usual.")
                
                if is_valid:
                    try:
                        # Check page count
                        pdf_reader = PdfReader(uploaded_pdf)
                        num_pages = len(pdf_reader.pages)
                        if num_pages > 100:
                            st.warning(f"⚠️ This document has {num_pages} pages. Analysis may be less precise.")
                        
                        document_text = extract_text_from_pdf(uploaded_pdf, enable_ocr=True)
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

        # Attachments / Evidence
        st.markdown("---")
        st.subheader("📎 Attachments / Evidence")

        # List existing attachments
        attachments = st.session_state.get("case_attachments")
        if attachments:
            for a in attachments:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{a['original_filename']}**")
                    st.caption(f"Uploaded: {a['uploaded_at']} • {a.get('size_bytes', 0)} bytes")
                    if a.get("document_id"):
                        st.caption(f"Linked document: #{a['document_id']}")
                with col2:
                    try:
                        from core.storage import get_attachment_path
                        path = get_attachment_path(a.get('stored_path') or a.get('stored_path'))
                        if path:
                            with open(path, 'rb') as f:
                                file_bytes = f.read()
                            st.download_button(label="Download", data=file_bytes, file_name=a['original_filename'], key=f"dl_att_{a['id']}")
                    except Exception:
                        st.button("Download", key=f"dl_att_{a['id']}")
        else:
            st.info("No attachments uploaded yet.")

        # Upload new attachment
        with st.expander("➕ Upload Attachment / Evidence"):
            uploaded = st.file_uploader("Select file (PDF / Image)", type=["pdf", "png", "jpg", "jpeg"], key="attachment_uploader")
            attach_deadline = None
            # Optionally link to a deadline
            if uploaded:
                # Show deadline select if deadlines exist
                deadlines = st.session_state.get("current_deadlines") or []
                if deadlines:
                    options = {f"{d['deadline_type']} - {d['deadline_date']}": d['id'] for d in deadlines}
                    sel = st.selectbox("Link to deadline (optional)", options=["None"] + list(options.keys()))
                    if sel != "None":
                        attach_deadline = options[sel]

            if st.button("Upload Attachment", use_container_width=True, key="upload_attachment_btn"):
                if not uploaded:
                    st.error("Please select a file to upload.")
                else:
                    try:
                        bytes_data = uploaded.read()
                        result = upload_case_attachment(
                            user_id=user_id,
                            case_id=case_id,
                            file_bytes=bytes_data,
                            filename=uploaded.name,
                            content_type=uploaded.type,
                            deadline_id=attach_deadline,
                        )
                        if result:
                            st.success("Attachment uploaded successfully")
                            # Refresh attachments in session so UI updates without full reload
                            if st.session_state.get("case_attachments") is None:
                                st.session_state["case_attachments"] = []
                            st.session_state["case_attachments"].insert(0, result)
                            st.rerun()
                        else:
                            st.error("Failed to upload attachment.")
                    except Exception as e:
                        st.error(f"Upload failed: {str(e)}")


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
                # Get user's timezone from preferences
                db = SessionLocal()
                user_tz_name = "UTC"
                try:
                    pref = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
                    if pref and pref.timezone:
                        user_tz_name = pref.timezone
                finally:
                    db.close()
                
                # Capture local date and convert to UTC
                user_tz = pytz.timezone(user_tz_name)
                local_dt = datetime.combine(deadline_date, datetime.min.time())
                localized_dt = user_tz.localize(local_dt)
                deadline_utc = localized_dt.astimezone(pytz.UTC)

                success = add_manual_deadline(
                    user_id=user_id,
                    case_id=case_id,
                    case_title=st.session_state.current_case_title or "Case",
                    deadline_date=deadline_utc,
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

    confidence_score = remedies.get("confidence_score")
    if confidence_score is not None:
        confidence_label = f"{float(confidence_score) * 100:.0f}%"
        if float(confidence_score) >= 0.75:
            st.success(f"Remedies confidence: {confidence_label}")
        elif float(confidence_score) >= 0.5:
            st.warning(f"Remedies confidence: {confidence_label}")
        else:
            st.error(f"Remedies confidence: {confidence_label}")

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

    evidence_spans = remedies.get("evidence_spans") or []
    if evidence_spans:
        with st.expander("Show evidence excerpts"):
            for span in evidence_spans:
                field = span.get("field", "unknown field")
                reason = span.get("snippet_reason", "Evidence extracted from remedies response.")
                span_text = span.get("span_text", "")
                st.markdown(f"**{field}**")
                st.caption(reason)
                st.write(span_text)


def render_knowledge_status_section(case_id: int, user_id: int):
    """Render the freshness dashboard for document-backed knowledge."""
    st.subheader("📡 Knowledge Status")

    current_user_email = st.session_state.get("user_email", "")
    current_user_role = st.session_state.get("user_role", "user")
    feature_enabled = is_feature_enabled_for_user(
        "knowledge_status_dashboard",
        str(user_id),
        attributes={"role": current_user_role, "email": current_user_email},
        surface="ui",
    )

    if not feature_enabled:
        st.info("This dashboard is being rolled out gradually. Check back soon.")
        return

    get_feature_flag_manager().mark_flag_used(
        "knowledge_status_dashboard",
        user_id=str(user_id),
        surface="ui",
    )

    db = SessionLocal()
    try:
        summary = get_knowledge_freshness_summary(db, user_id=user_id, case_id=case_id)
        invalidations = list_knowledge_invalidations(db, user_id=user_id, case_id=case_id, limit=20)
    finally:
        db.close()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fresh", summary["fresh"])
    with col2:
        st.metric("Stale", summary["stale"])
    with col3:
        next_run = summary.get("next_recompute_at")
        st.metric(
            "Next recompute",
            next_run.strftime("%d %b %H:%M UTC") if next_run else "queued on demand",
        )

    latest = summary.get("latest")
    if latest:
        st.info(f"Latest invalidation: {latest.reason} at {latest.invalidated_at.strftime('%d %b %Y %H:%M UTC')}")
    else:
        st.success("No invalidations recorded for this case yet.")

    if not invalidations:
        st.caption("Nothing stale right now.")
        return

    for invalidation in invalidations:
        stale_badge = "🟠" if invalidation.status != "completed" else "🟢"
        with st.container(border=True):
            st.markdown(f"{stale_badge} **{invalidation.scope_type.title()}** · {invalidation.scope_value}")
            st.caption(f"Reason: {invalidation.reason} · Status: {invalidation.status}")
            st.caption(
                f"Invalidated: {invalidation.invalidated_at.strftime('%d %b %Y %H:%M UTC')}"
                + (
                    f" · Recompute: {invalidation.scheduled_for.strftime('%d %b %Y %H:%M UTC')}"
                    if invalidation.scheduled_for
                    else ""
                )
            )
            details = invalidation.details or {}
            if details:
                changed_fields = details.get("changed_fields")
                if changed_fields:
                    st.write(f"Changed fields: {', '.join(changed_fields)}")
                elif details:
                    st.json(details)


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
            st.switch_page(routes.PAGE_MY_CASES)
        return

    # Get case details
    case_data = get_case_detail(user_id, case_id)

    if not case_data:
        st.error("Case not found or access denied")
        if st.button("← Back to My Cases"):
            st.switch_page(routes.PAGE_MY_CASES)
        return

    case = case_data["case"]
    documents = case_data["documents"]
    deadlines = case_data["deadlines"]
    remedies = case_data.get("remedies")
    comments = case_data.get("comments", [])
    presence = case_data.get("presence", [])
    timeline = case_data.get("timeline", [])

    update_case_presence(user_id, case_id, active_view="case_details")

    # Keep attachments and deadlines in session for uploader convenience
    st.session_state.setdefault("case_attachments", case_data.get("attachments", []))
    st.session_state.setdefault("current_deadlines", deadlines)

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📅 Timeline", "📄 Documents", "⏰ Deadlines", "⚖️ Remedies", "🤝 Collaboration"])

    with tab1:
        render_timeline_section(timeline)

    with tab2:
        render_documents_section(case_id, documents, user_id)

    with tab3:
        render_deadlines_section(case_id, deadlines, user_id)

    with tab4:
        render_remedies_section(remedies)

    with tab5:
        render_collaboration_section(case_id, user_id, comments, presence)

    st.markdown("---")

    # Case actions
    render_case_actions(case, user_id)

    # Export options
    st.markdown("---")
    st.subheader("📦 Export Builder")
    from services.export_builder import (
        build_case_export_artifact,
        build_case_export_bundle,
        get_case_export_preview,
        get_default_export_fields,
        get_export_field_options,
    )
    from services.privacy_redaction import get_default_privacy_profile, get_privacy_profile_options

    privacy_profile_options = get_privacy_profile_options()
    privacy_profile_labels = {item["name"]: item["label"] for item in privacy_profile_options}
    privacy_profile = st.selectbox(
        "Privacy profile",
        options=[item["name"] for item in privacy_profile_options],
        index=[item["name"] for item in privacy_profile_options].index(get_default_privacy_profile()),
        format_func=lambda profile_name: privacy_profile_labels.get(profile_name, profile_name.replace("_", " ").title()),
        key=f"privacy_profile_{case_id}",
    )

    export_field_options = get_export_field_options()
    export_fields = st.multiselect(
        "Fields to include",
        options=[item["field_id"] for item in export_field_options],
        default=get_default_export_fields(),
        format_func=lambda field_id: next(item["label"] for item in export_field_options if item["field_id"] == field_id),
        key=f"export_fields_{case_id}",
    )

    export_formats = st.multiselect(
        "Output formats",
        options=["json", "pdf", "docx"],
        default=["json", "pdf"],
        key=f"export_formats_{case_id}",
    )

    user_cases = get_user_cases_summary(user_id)
    case_options = [item["id"] for item in user_cases]
    case_labels = {item["id"]: f"{item['case_number']} - {item['title']}" for item in user_cases}
    selected_case_ids = st.multiselect(
        "Cases to export",
        options=case_options,
        default=[case_id],
        format_func=lambda case_option: case_labels.get(case_option, str(case_option)),
        key=f"export_cases_{case_id}",
    )

    preview_case_id = case_id if case_id in selected_case_ids else (selected_case_ids[0] if selected_case_ids else case_id)
    preview_payload = get_case_export_preview(
        user_id=user_id,
        case_id=preview_case_id,
        field_ids=export_fields,
        privacy_profile=privacy_profile,
    )

    with st.expander("Preview selected fields", expanded=False):
        if preview_payload:
            st.json(preview_payload)
        else:
            st.info("Select a case you own to preview the export payload.")

    export_col1, export_col2 = st.columns(2)
    with export_col1:
        if selected_case_ids and export_formats:
            if len(selected_case_ids) == 1 and len(export_formats) == 1:
                export_artifact = build_case_export_artifact(
                    user_id=user_id,
                    case_id=selected_case_ids[0],
                    format=export_formats[0],
                    field_ids=export_fields,
                    privacy_profile=privacy_profile,
                )
                if export_artifact:
                    st.download_button(
                        label=f"Download {export_formats[0].upper()} export",
                        data=export_artifact.data,
                        file_name=export_artifact.file_name,
                        mime=export_artifact.mime_type,
                        key=f"download_case_export_{case_id}",
                        use_container_width=True,
                    )
            else:
                export_artifact = build_case_export_bundle(
                    user_id=user_id,
                    case_ids=selected_case_ids,
                    field_ids=export_fields,
                    formats=export_formats,
                    privacy_profile=privacy_profile,
                )
                if export_artifact:
                    st.download_button(
                        label="Download export bundle",
                        data=export_artifact.data,
                        file_name=export_artifact.file_name,
                        mime=export_artifact.mime_type,
                        key=f"download_case_export_bundle_{case_id}",
                        use_container_width=True,
                    )

    with export_col2:
        from pdf_exporter import generate_anonymized_pdf
        from case_manager import generate_anonymized_case_data

        anon_data = generate_anonymized_case_data(case_id, profile_name=privacy_profile)
        if anon_data:
            anon_id = anon_data["anonymized_id"]
            anon_pdf_bytes = generate_anonymized_pdf(case_id, anon_id, user_id, profile_name=privacy_profile)
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
                    st.info(
                        "Share this ID with your lawyer. They can view the "
                        "anonymized case at the **View Shared Case** page, "
                        "or by entering the ID at `/6_Shared_Case`."
                    )


if __name__ == "__main__":
    main()
