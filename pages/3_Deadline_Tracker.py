"""
Deadline Tracker - LegalAssist AI.
Calendar view and management of all case deadlines.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

from auth import require_auth, redirect_to_login, get_current_user_id, get_current_user_email
from case_manager import get_user_cases_summary, mark_deadline_completed, mark_deadline_incomplete
from database import SessionLocal, CaseDeadline, Case

# Page config
st.set_page_config(
    page_title="Deadline Tracker - LegalAssist AI",
    page_icon="⏰",
    layout="wide",
)

# Using default Streamlit theme



def get_all_user_deadlines(user_id: int) -> List[Dict[str, Any]]:
    """Get all deadlines for a user across all cases"""
    db = SessionLocal()
    try:
        deadlines = db.query(CaseDeadline).filter(
            CaseDeadline.user_id == str(user_id)
        ).order_by(CaseDeadline.deadline_date).all()

        result = []
        for d in deadlines:
            # Get case info
            case = db.query(Case).filter(Case.id == d.case_id).first()

            result.append({
                "id": d.id,
                "case_id": d.case_id,
                "case_number": case.case_number if case else "Unknown",
                "case_title": d.case_title,
                "deadline_type": d.deadline_type,
                "deadline_date": d.deadline_date,
                "description": d.description,
                "is_completed": d.is_completed,
                "days_until": d.days_until_deadline(),
            })

        return result

    finally:
        db.close()


def render_summary_cards(deadlines: List[Dict]):
    """Render summary statistics cards"""
    total = len(deadlines)
    completed = len([d for d in deadlines if d["is_completed"]])
    pending = total - completed

    # Urgent: within 3 days
    urgent = len([d for d in deadlines if not d["is_completed"] and d["days_until"] is not None and d["days_until"] <= 3])

    # This week: within 7 days
    this_week = len([d for d in deadlines if not d["is_completed"] and d["days_until"] is not None and d["days_until"] <= 7])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 style="margin: 0; color: #888;">Total</h3>
                <h1 style="margin: 10px 0 0 0; color: #2d2dff;">{total}</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 style="margin: 0; color: #888;">🔴 Urgent</h3>
                <h1 style="margin: 10px 0 0 0; color: #ff5252;">{urgent}</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 style="margin: 0; color: #888;">📅 This Week</h3>
                <h1 style="margin: 10px 0 0 0; color: #ff9100;">{this_week}</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 style="margin: 0; color: #888;">✅ Completed</h3>
                <h1 style="margin: 10px 0 0 0; color: #00c853;">{completed}</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_calendar_view(deadlines: List[Dict]):
    """Render calendar-style view of deadlines"""
    st.subheader("📅 Calendar View")

    # Filter to pending only
    pending = [d for d in deadlines if not d["is_completed"]]

    if not pending:
        st.info("No upcoming deadlines!")
        return

    # Create dataframe for calendar
    df = pd.DataFrame(pending)
    df["date"] = df["deadline_date"].dt.date
    df["urgency"] = df["days_until"].apply(
        lambda x: "🔴 Urgent" if x <= 3 else ("🟠 Soon" if x <= 7 else "🟢 Normal")
    )

    # Group by date
    grouped = df.groupby("date")

    # Show next 30 days
    today = datetime.now(timezone.utc).date()
    end_date = today + timedelta(days=60)

    for date, group in sorted(grouped):
        if date > end_date:
            continue

        date_str = date.strftime("%A, %d %B %Y")
        days_from_now = (date - today).days

        if days_from_now == 0:
            date_label = "**TODAY**"
        elif days_from_now == 1:
            date_label = "**TOMORROW**"
        else:
            date_label = f"in {days_from_now} days"

        st.markdown(f"### {date_label} - {date_str}")

        for _, row in group.iterrows():
            urgency_emoji = "🔴" if row["days_until"] <= 3 else ("🟠" if row["days_until"] <= 7 else "🟢")

            with st.container():
                col1, col2 = st.columns([4, 1])

                with col1:
                    st.markdown(
                        f"""
                        <div class="deadline-card deadline-{'urgent' if row['days_until'] <= 3 else ('soon' if row['days_until'] <= 7 else 'normal')}">
                            {urgency_emoji} <strong>{row['deadline_type'].title()}</strong> - {row['case_title']}
                            <br><small>{row.get('description', '')}</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    if st.button("✓", key=f"cal_complete_{row['id']}", help="Mark as completed"):
                        mark_deadline_completed(st.session_state.user_id, row["id"])
                        st.rerun()


def render_list_view(deadlines: List[Dict], user_id: int):
    """Render list view of deadlines"""
    st.subheader("📋 List View")

    # Filter options
    col1, col2, col3 = st.columns(3)

    with col1:
        show_completed = st.checkbox("Show completed", value=False, key="show_completed_deadlines")

    with col2:
        type_filter = st.selectbox(
            "Type",
            ["All"] + list(set(d["deadline_type"].title() for d in deadlines)),
            key="deadline_type_filter",
        )

    with col3:
        search = st.text_input("Search", placeholder="Case or deadline...", key="deadline_search")

    # Apply filters
    filtered = deadlines

    if not show_completed:
        filtered = [d for d in filtered if not d["is_completed"]]

    if type_filter != "All":
        filtered = [d for d in filtered if d["deadline_type"].title() == type_filter]

    if search:
        query = search.lower()
        filtered = [
            d for d in filtered
            if query in d["case_title"].lower() or query in d["deadline_type"].lower() or (d.get("description") and query in d["description"].lower())
        ]

    if not filtered:
        st.info("No deadlines match your filters.")
        return

    # Sort by date
    filtered = sorted(filtered, key=lambda x: x["deadline_date"])

    for d in filtered:
        days = d["days_until"]

        if d["is_completed"]:
            urgency = "completed"
            emoji = "✅"
        elif days is not None and days <= 3:
            urgency = "urgent"
            emoji = "🔴"
        elif days is not None and days <= 7:
            urgency = "soon"
            emoji = "🟠"
        else:
            urgency = "normal"
            emoji = "🟢"

        date_str = d["deadline_date"].strftime("%d %b %Y")

        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                st.markdown(
                    f"""
                    <div class="deadline-card deadline-{urgency}">
                        {emoji} <strong>{d['deadline_type'].title()}</strong> - {d['case_title']}
                        <br><small>{date_str} {f'({days} days)' if days else ''}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                if d.get("description"):
                    st.caption(d["description"])

            with col3:
                if d["is_completed"]:
                    if st.button("↩️ Undo", key=f"list_undo_{d['id']}"):
                        mark_deadline_incomplete(user_id, d["id"])
                        st.rerun()
                else:
                    if st.button("✓ Done", key=f"list_done_{d['id']}"):
                        mark_deadline_completed(user_id, d["id"])
                        st.rerun()


def render_upcoming_section(deadlines: List[Dict]):
    """Render upcoming deadlines widget"""
    st.subheader("⏰ Upcoming Deadlines")

    pending = [d for d in deadlines if not d["is_completed"] and d["days_until"] is not None]
    upcoming = sorted(pending, key=lambda x: x["days_until"])[:5]

    if not upcoming:
        st.info("No upcoming deadlines!")
        return

    for i, d in enumerate(upcoming):
        days = d["days_until"]

        if days <= 3:
            color = "#ff5252"
        elif days <= 7:
            color = "#ff9100"
        else:
            color = "#00c853"

        col1, col2 = st.columns([1, 4])

        with col1:
            st.markdown(
                f"""
                <div style="background-color: {color}; color: white; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 1.2rem;">
                    {days}
                </div>
                <div style="text-align: center; color: #888; font-size: 0.8rem;">days</div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(f"**{d['deadline_type'].title()}** - {d['case_title']}")
            st.caption(f"{d['deadline_date'].strftime('%d %B %Y')}")
            if d.get("description"):
                st.caption(d["description"])

        if i < len(upcoming) - 1:
            st.markdown("---")


def main():
    """Main deadline tracker logic"""
    # Require authentication
    if not require_auth():
        st.warning("🔐 Please log in to view deadlines")
        if st.button("Go to Login"):
            redirect_to_login()
        return

    user_id = get_current_user_id()
    user_email = get_current_user_email()

    # Header
    st.title("⏰ Deadline Tracker")
    st.markdown(f"*Track all your case deadlines in one place*")

    st.markdown("---")

    # Get all deadlines
    deadlines = get_all_user_deadlines(user_id)

    if not deadlines:
        st.info("📭 No deadlines yet. Deadlines are created when you upload documents with remedies advice.")
        if st.button("📤 Upload a Judgment"):
            st.switch_page("app.py")
        return

    # Render summary cards
    render_summary_cards(deadlines)

    st.markdown("---")

    # Main content - tabs
    tab1, tab2, tab3 = st.tabs(["📅 Calendar", "📋 List", "⏰ Upcoming"])

    with tab1:
        render_calendar_view(deadlines)

    with tab2:
        render_list_view(deadlines, user_id)

    with tab3:
        render_upcoming_section(deadlines)

    st.markdown("---")

    # Export options
    st.subheader("📥 Export")

    col1, col2 = st.columns(2)

    with col1:
        # Export to CSV
        if st.button("📊 Export to CSV", use_container_width=True):
            df = pd.DataFrame(deadlines)
            df["deadline_date_str"] = df["deadline_date"].dt.strftime("%Y-%m-%d")
            df["days_until"] = df["days_until"].fillna(-1).astype(int)

            csv = df[["case_number", "case_title", "deadline_type", "deadline_date_str", "days_until", "is_completed", "description"]].to_csv(index=False)

            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"deadlines_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="export_csv",
            )

    with col2:
        # Export to ICS (iCalendar)
        if st.button("📆 Export to Calendar (ICS)", use_container_width=True):
            # Generate ICS content
            ics_lines = [
                "BEGIN:VCALENDAR",
                "VERSION:2.0",
                "PRODID:-//LegalAssist AI//Deadline Tracker//EN",
                "CALSCALE:GREGORIAN",
                "METHOD:PUBLISH",
            ]

            pending = [d for d in deadlines if not d["is_completed"]]

            for d in pending:
                deadline_dt = d["deadline_date"]
                ics_lines.extend([
                    "BEGIN:VEVENT",
                    f"UID:deadline-{d['id']}@legalassist.ai",
                    f"DTSTAMP:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
                    f"DTSTART;VALUE=DATE:{deadline_dt.strftime('%Y%m%d')}",
                    f"SUMMARY:[LegalAssist] {d['deadline_type'].title()} - {d['case_title']}",
                    f"DESCRIPTION:{d.get('description', '')}",
                    "STATUS:CONFIRMED",
                    "END:VEVENT",
                ])

            ics_lines.append("END:VCALENDAR")
            ics_content = "\n".join(ics_lines)

            st.download_button(
                label="📥 Download ICS",
                data=ics_content,
                file_name=f"legalassist_deadlines.ics",
                mime="text/calendar",
                key="export_ics",
            )


if __name__ == "__main__":
    main()
