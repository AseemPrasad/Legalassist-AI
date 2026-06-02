"""
Streamlit UI components for deadline management and notification preferences.
Integrate these into the main app.py or use as a separate page.
"""

import streamlit as st
from datetime import datetime, timezone, timedelta
from typing import Optional, List
import logging
import pandas as pd
import plotly.express as px
import io

# Initialize logger
logger = logging.getLogger(__name__)

def apply_custom_css():
    """Apply custom CSS to prevent long notification text from overflowing UI containers."""
    st.markdown(
        """
        <style>
        /* Fix for extremely long notification text messages overflowing UI containers */
        .stMarkdown, .stText, .stCaption, [data-testid="stVerticalBlock"], .stContainer {
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

import routes

from database import (
    SessionLocal,
    create_case_deadline,
    get_user_deadlines,
    create_or_update_user_preference,
    get_notification_history,
    NotificationChannel,
    CaseDeadline,
    UserPreference,
)
from notification_service import NotificationService

# Timezone list for user selection
TIMEZONES = [
    "UTC",
    "Asia/Kolkata",
    "Asia/Bangkok",
    "Asia/Singapore",
    "Asia/Dhaka",
    "Asia/Karachi",
    "Asia/Kathmandu",
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "Europe/London",
    "Europe/Paris",
    "Europe/Berlin",
    "Australia/Sydney",
]


def get_user_id() -> int:
    """Get authenticated user ID from session state"""
    user_id = st.session_state.get("user_id")
    if not user_id:
        st.error("Authentication required. Please log in.")
        st.stop()
    return int(user_id)


def page_notification_preferences():
    """Page: User Notification Preferences"""
    apply_custom_css()
    st.title("⚙️ Notification Preferences")

    db = SessionLocal()
    try:
        user_id = get_user_id()

        # Get existing preferences
        user_pref = db.query(UserPreference).filter(
            UserPreference.user_id == int(user_id)
        ).first()

        if not user_pref:
            # First time - create default preferences
            st.info("Setting up your notification preferences...")
            email = st.session_state.get("user_email", "")
        else:
            email = user_pref.email

        # Preferences form
        st.subheader("Contact Information")
        col1, col2 = st.columns(2)

        with col1:
            email_input = st.text_input(
                "Email Address",
                value=email,
                key="pref_email",
                help="We'll send deadline reminders to this email",
            )

        with col2:
            phone_input = st.text_input(
                "Phone Number (for SMS)",
                value=user_pref.phone_number if user_pref else "",
                key="pref_phone",
                placeholder="+91 9876543210",
                help="Enter with country code (e.g., +91 for India, +1 for USA)",
            )

        st.subheader("Notification Channels")
        channel_options = {
            "SMS Only": NotificationChannel.SMS,
            "Email Only": NotificationChannel.EMAIL,
            "Both SMS & Email": NotificationChannel.BOTH,
        }
        channel_labels = list(channel_options.keys())
        current_channel = user_pref.notification_channel if user_pref else NotificationChannel.BOTH
        channel_index = (
            list(channel_options.values()).index(current_channel)
            if current_channel in channel_options.values()
            else 2
        )

        selected_channel = st.radio(
            "How would you like to receive reminders?",
            channel_labels,
            index=channel_index,
        )

        st.subheader("Timezone")
        current_tz = user_pref.timezone if user_pref else "UTC"
        tz_index = TIMEZONES.index(current_tz) if current_tz in TIMEZONES else 0
        timezone = st.selectbox("Select your timezone", TIMEZONES, index=tz_index)

        st.subheader("Reminder Schedule")
        st.markdown(
            "Your reminders will be sent at **8 AM** in your local timezone on these days. "
            "Enter custom threshold days separated by commas (e.g., `45, 15, 7`):"
        )

        default_thresholds = "30, 10, 3, 1"
        if user_pref:
            thresholds_list = user_pref.get_reminder_thresholds()
            default_thresholds = ", ".join(str(t) for t in thresholds_list)

        reminder_days_input = st.text_input(
            "Reminder Days",
            value=default_thresholds,
            key="reminder_days_input",
            help="Comma-separated integers, e.g., 45, 15, 7",
        )

        # Save preferences
        if st.button("💾 Save Preferences", use_container_width=True):
            try:
                thresholds_parsed = [
                    int(x.strip())
                    for x in reminder_days_input.split(",")
                    if x.strip()
                ]
                if not all(t > 0 for t in thresholds_parsed):
                    st.error("❌ All reminder days must be positive integers.")
                    st.stop()
            except ValueError:
                st.error("❌ Please enter a valid comma-separated list of numbers (e.g., 45, 15, 7).")
                st.stop()

            try:
                create_or_update_user_preference(
                    db=db,
                    user_id=int(user_id),
                    email=email_input,
                    phone_number=phone_input if phone_input else None,
                    notification_channel=channel_options[selected_channel],
                    timezone=timezone,
                    reminder_thresholds=thresholds_parsed,
                )

                st.success("✅ Preferences saved successfully!")
                logger.info(f"Preferences updated for user {user_id}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error saving preferences: {str(e)}")
                logger.error(f"Error saving preferences: {str(e)}")

    finally:
        db.close()

    # --- Template Builder ---
    st.divider()
    st.subheader("✉️ Reminder Template Builder")
    st.markdown("Customize the SMS and Email templates used for reminders. Use only allowed variables listed below.")

    allowed = ["{case_title}", "{case_number}", "{deadline_date}", "{days_left}", "{court}", "{deadline_type}", "{deadline_description}", "{link}"]
    st.markdown("**Allowed variables:** " + ", ".join(allowed))

    db = SessionLocal()
    try:
        tmpl = db.query(__import__("database").NotificationTemplate).filter(__import__("database").NotificationTemplate.user_id == int(user_id)).first()

        sms_val = tmpl.sms_template if tmpl and tmpl.sms_template else "⚖️ Reminder: {case_title} has a deadline in {days_left} day(s). {link}"
        subj_val = tmpl.email_subject_template if tmpl and tmpl.email_subject_template else "⚖️ Reminder: {case_title} - {deadline_type} due"
        html_val = tmpl.email_html_template if tmpl and tmpl.email_html_template else ("<p>Dear user,</p><p>Your case <strong>{case_title}</strong> has a {deadline_type} deadline on {deadline_date} ({days_left} days left).</p><p><a href=\"{link}\">View case</a></p>")

        sms_input = st.text_area("SMS Template", value=sms_val, height=120, key="sms_template_input")
        subj_input = st.text_input("Email Subject Template", value=subj_val, key="email_subject_input")
        html_input = st.text_area("Email HTML Template", value=html_val, height=220, key="email_html_input")
        channel_scope = st.selectbox(
            "Template Scope",
            ["Default", "SMS", "Email"],
            index=0,
            help="Choose Default to keep the legacy single-template behavior, or target a specific channel.",
        )
        template_language = st.text_input(
            "Template Language (optional)",
            value="",
            help="Use a locale tag like en or hi. Leave blank to store the default template.",
        ).strip()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Preview Templates"):
                from core.template_renderer import validate_template, render_template, TemplateValidationError
                sample_values = {
                    "case_title": "Sharma vs State",
                    "case_number": "CA/123/2024",
                    "deadline_date": "12 May 2026",
                    "days_left": 3,
                    "court": "Delhi High Court",
                    "deadline_type": "appeal",
                    "deadline_description": "File appeal against lower court order",
                    "link": "https://legalassist.ai/cases/1",
                }

                ok_sms, unknown_sms = validate_template(sms_input)
                ok_subj, unknown_subj = validate_template(subj_input)
                ok_html, unknown_html = validate_template(html_input)

                if not ok_sms:
                    st.error(f"SMS template contains unknown variables: {unknown_sms}")
                else:
                    try:
                        st.markdown("**SMS Preview**")
                        st.write(render_template(sms_input, sample_values))
                    except TemplateValidationError as e:
                        st.error(str(e))

                if not ok_subj:
                    st.error(f"Email subject contains unknown variables: {unknown_subj}")
                else:
                    st.markdown("**Email Subject Preview**")
                    try:
                        st.write(render_template(subj_input, sample_values))
                    except TemplateValidationError as e:
                        st.error(str(e))

                if not ok_html:
                    st.error(f"Email HTML contains unknown variables: {unknown_html}")
                else:
                    st.markdown("**Email HTML Preview**")
                    try:
                        rendered_html = render_template(html_input, sample_values)
                        st.write(rendered_html, unsafe_allow_html=True)
                    except TemplateValidationError as e:
                        st.error(str(e))

        with col2:
            if st.button("Save Templates", use_container_width=True):
                try:
                    from database import create_or_update_notification_template
                    from db.models.notifications import NotificationChannel

                    selected_channel = None
                    if channel_scope == "SMS":
                        selected_channel = NotificationChannel.SMS
                    elif channel_scope == "Email":
                        selected_channel = NotificationChannel.EMAIL

                    selected_language = template_language or None
                    if selected_channel is None and selected_language is None:
                        create_or_update_notification_template(
                            db=db,
                            user_id=int(user_id),
                            sms_template=sms_input,
                            email_subject_template=subj_input,
                            email_html_template=html_input,
                        )
                    else:
                        create_or_update_notification_template(
                            db=db,
                            user_id=int(user_id),
                            sms_template=sms_input,
                            email_subject_template=subj_input,
                            email_html_template=html_input,
                            channel=selected_channel,
                            language=selected_language,
                        )
                    st.success("✅ Templates saved")
                except Exception as e:
                    st.error(f"Failed to save templates: {str(e)}")
    finally:
        db.close()

    # Info section
    st.divider()
    st.info(
        """
        ### How Deadline Reminders Work
        
        - **Custom reminder schedule**: Reminders are sent dynamically based on your configured threshold days.
        
        All reminders are sent at **8 AM** in your timezone to ensure you see them.
        """
    )


def page_manage_deadlines():
    """Page: Add and manage case deadlines"""
    apply_custom_css()
    st.title("📅 Case Deadlines")

    db = SessionLocal()
    try:
        user_id = get_user_id()

        # Check if user has preferences set up
        user_pref = db.query(UserPreference).filter(
            UserPreference.user_id == int(user_id)
        ).first()

        if not user_pref:
            st.warning("⚠️ Please set up your notification preferences first!")
            if st.button("Go to Preferences"):
                st.switch_page(routes.PAGE_SETTINGS)
            return

        # Add new deadline
        st.subheader("➕ Add New Deadline")
        with st.form("add_deadline_form"):
            col1, col2 = st.columns(2)

            # Load cases owned by current user for ownership validation in UI
            user_cases = db.query(getattr(__import__('database', fromlist=['Case']).Case, 'case_number')).all()


            with col2:
                deadline_date = st.date_input(
                    "Deadline Date",
                    value=datetime.now() + timedelta(days=90),
                    min_value=datetime.now(),
                )
                deadline_type = st.selectbox(
                    "Deadline Type",
                    ["Appeal", "Filing", "Submission", "Response", "Hearing", "Other"],
                )

            description = st.text_area(
                "Additional Details (optional)",
                placeholder="Any notes about this deadline...",
                height=80,
            )

            submitted = st.form_submit_button("📌 Add Deadline", use_container_width=True)

            if submitted:
                if not case_title:
                    st.error("❌ Case Title is required")
                else:
                    try:
                        # Convert date to datetime
                        deadline_datetime = datetime.combine(
                            deadline_date, datetime.min.time()
                        ).replace(tzinfo=timezone.utc)

                        create_case_deadline(
                            db=db,
                            user_id=int(user_id),
                            case_id=int(case_id),
                            case_title=case_title,
                            deadline_date=deadline_datetime,
                            deadline_type=deadline_type.lower(),
                            description=description if description else None,
                        )

                        st.success(
                            f"✅ Deadline added! Reminders will be sent on: 30, 10, 3, and 1 day(s) before."
                        )
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error adding deadline: {str(e)}")

        st.divider()

        # Display user's deadlines
        st.subheader("📋 Your Active Deadlines")
        deadlines = get_user_deadlines(db, int(user_id))

        if not deadlines:
            st.info("No active deadlines yet. Add one above!")
        else:
            for deadline in deadlines:
                days_left = deadline.days_until_deadline()
                
                # Color code based on urgency
                if days_left <= 3:
                    emoji = "🔴"  # Critical
                elif days_left <= 10:
                    emoji = "🟠"  # Urgent
                else:
                    emoji = "🟢"  # Normal

                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(
                            f"### {emoji} {deadline.case_title} ({deadline.deadline_type.title()})"
                        )
                        st.text(f"Case ID: {deadline.case_id}")

                        # Deadline info
                        formatted_date = deadline.deadline_date.strftime("%d %B %Y")
                        st.markdown(
                            f"**Deadline:** {formatted_date} | **Days Left:** {days_left}"
                        )

                        if deadline.description:
                            st.caption(deadline.description)

                    with col2:
                        st.metric("", f"{days_left} days")

                    # Mark as completed
                    if st.button(
                        "✓ Mark Complete",
                        key=f"complete_{deadline.id}",
                        use_container_width=True,
                    ):
                        deadline.is_completed = True
                        db.commit()
                        st.success("Deadline marked as completed!")
                        st.rerun()

    finally:
        db.close()


def page_notification_history():
    """Page: View notification delivery history"""
    apply_custom_css()
    st.title("📬 Notification History")

    db = SessionLocal()
    try:
        user_id = get_user_id()

        # Get notification history
        notifications = get_notification_history(db, int(user_id), limit=100)

        if not notifications:
            st.info("No notifications sent yet.")
            return

        # Summary statistics
        delivered_notifications = [n for n in notifications if n.status.value == "delivered"]
        failed_notifications = [n for n in notifications if n.status.value == "failed"]
        sms_notifications = [n for n in notifications if n.channel.value == "sms"]
        email_notifications = [n for n in notifications if n.channel.value == "email"]

        col1, col2, col3, col4 = st.columns(4)

        total = len(notifications)
        delivered = len(delivered_notifications)
        failed = len(failed_notifications)
        sms_count = len(sms_notifications)

        with col1:
            st.metric("Total Notifications", total)
        with col2:
            st.metric("Delivered", delivered)
        with col3:
            st.metric("Failed", failed)
        with col4:
            st.metric("Via SMS", sms_count)

        st.divider()

        channel_rows = []
        for channel_name, channel_notifications in (("SMS", sms_notifications), ("Email", email_notifications)):
            channel_rows.append({
                "Channel": channel_name,
                "Delivered": len([n for n in channel_notifications if n.status.value == "delivered"]),
                "Failed": len([n for n in channel_notifications if n.status.value == "failed"]),
                "Pending": len([n for n in channel_notifications if n.status.value == "pending"]),
                "Sent": len([n for n in channel_notifications if n.status.value == "sent"]),
            })

        st.subheader("Delivery by Channel")
        st.dataframe(pd.DataFrame(channel_rows), use_container_width=True, hide_index=True)

        # Notification table
        st.subheader("Recent Notifications")

        for notif in notifications[:20]:  # Show last 20
            status_emoji = {
                "delivered": "✅",
                "failed": "❌",
                "sent": "✅",
                "pending": "⏳",
                "bounced": "↩️",
                "opened": "👁️",
            }.get(notif.status.value, "❓")
            status_label = notif.status.value.replace("_", " ").title()
            timeline_value = notif.delivered_at or notif.failed_at or notif.sent_at or notif.created_at

            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

                with col1:
                    case_title = notif.deadline.case_title if notif.deadline else "Deleted Case/Deadline"
                    st.text(f"Case: {case_title}")
                    st.caption(notif.recipient)
                    if notif.message_id:
                        st.caption(f"Message ID: {notif.message_id}")

                with col2:
                    st.text(f"Channel: {notif.channel.value.upper()}")
                    st.caption(f"Reminder: {notif.days_before} day(s)")
                    st.caption(f"Status: {status_label}")

                with col3:
                    st.text(f"Created: {notif.created_at.strftime('%d %b %Y %H:%M')}")
                    if timeline_value:
                        st.caption(f"Updated: {timeline_value.strftime('%d %b %Y %H:%M')}")
                    if notif.error_message:
                        st.caption(f"Error: {notif.error_message}")

                with col4:
                    st.markdown(f"### {status_emoji}")


    finally:
        db.close()




def page_bulk_import():
    """Page: Bulk import deadlines from CSV"""
    apply_custom_css()
    st.title("📥 Bulk Import Deadlines")
    st.markdown("Upload a CSV file with your case deadlines. Required columns: `case_title`, `deadline_date`, `deadline_type` (optional: `description`, `case_id`)")

    # Sample CSV download
    sample_df = pd.DataFrame({
        "case_title": ["State vs. Smith", "Doe Estate"],
        "deadline_date": ["2026-06-15", "2026-07-20"],
        "deadline_type": ["Filing", "Hearing"],
        "description": ["Submit final brief", "Probate hearing"],
        "case_id": [101, 102]
    })
    
    csv = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Sample CSV Template",
        data=csv,
        file_name="deadline_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns for deadline import
            required_columns = ["case_title", "deadline_date"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"CSV is missing required columns: {', '.join(missing_columns)}. "
                         f"Required columns: case_title, deadline_date. Optional: case_number, deadline_type, description.")
                return
            
            st.write("### Data Preview")
            st.dataframe(df.head())

            if st.button("🚀 Process and Import Deadlines", use_container_width=True):
                db = SessionLocal()
                user_id = get_user_id()
                count = 0
                errors = []
                
                progress_bar = st.progress(0)
                for i, row in df.iterrows():
                    try:
                        # Basic validation
                        if pd.isna(row.get('case_title')) or pd.isna(row.get('deadline_date')):
                            errors.append(f"Row {i+1}: Missing required fields")
                            continue
                        
                        # Parse date
                        d_date = pd.to_datetime(row['deadline_date']).to_pydatetime()
                        d_date = d_date.replace(tzinfo=timezone.utc)
                        
                        create_case_deadline(
                            db=db,
                            user_id=int(user_id),
                            case_id=int(row.get('case_id', 0)),
                            case_title=str(row['case_title']),
                            deadline_date=d_date,
                            deadline_type=str(row.get('deadline_type', 'Other')).lower(),
                            description=str(row.get('description', '')) if not pd.isna(row.get('description')) else None
                        )
                        count += 1
                    except Exception as e:
                        errors.append(f"Row {i+1}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(df))
                
                db.commit()
                db.close()
                
                if count > 0:
                    st.success(f"✅ Successfully imported {count} deadlines!")
                if errors:
                    st.warning(f"⚠️ Encountered {len(errors)} errors during import.")
                    with st.expander("View Error Details"):
                        for err in errors:
                            st.write(f"- {err}")
                
                if count > 0:
                    st.balloons()

        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")


def page_analytics():
    """Page: Deadline and Notification Analytics"""
    apply_custom_css()
    st.title("📊 Analytics Dashboard")
    
    db = SessionLocal()
    try:
        user_id = get_user_id()
        
        # Get deadlines for charts
        deadlines = get_user_deadlines(db, int(user_id))
        
        if not deadlines:
            st.info("No data available yet. Add some deadlines to see analytics!")
            return

        # 1. Deadline Status Overview
        st.subheader("🗓️ Deadline Distribution")
        
        data = []
        for d in deadlines:
            days = d.days_until_deadline()
            status = "Completed" if d.is_completed else ("Critical (<=3d)" if days <= 3 else ("Upcoming" if days <= 10 else "Scheduled"))
            data.append({
                "Case": d.case_title,
                "Type": d.deadline_type.title(),
                "Days Left": days,
                "Status": status
            })
        
        df = pd.DataFrame(data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_status = px.pie(df, names='Status', title='Deadline Status', color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_status, use_container_width=True)
            
        with col2:
            fig_type = px.bar(df.groupby('Type').size().reset_index(name='Count'), x='Type', y='Count', title='Deadlines by Type', color='Type')
            st.plotly_chart(fig_type, use_container_width=True)

        # 2. Notification Delivery Performance
        st.divider()
        st.subheader("📬 Delivery Performance")
        
        notifications = get_notification_history(db, int(user_id), limit=200)
        if notifications:
            notif_data = []
            for n in notifications:
                notif_data.append({
                    "Date": n.created_at.date(),
                    "Status": n.status.value.capitalize(),
                    "Channel": n.channel.value.upper()
                })
            
            ndf = pd.DataFrame(notif_data)
            
            col3, col4 = st.columns(2)
            
            with col3:
                fig_notif_status = px.pie(ndf, names='Status', title='Notification Success Rate', 
                                         color='Status', color_discrete_map={'Sent': '#4CAF50', 'Failed': '#F44336', 'Pending': '#FFC107'})
                st.plotly_chart(fig_notif_status, use_container_width=True)
                
            with col4:
                daily_notifs = ndf.groupby(['Date', 'Status']).size().reset_index(name='Count')
                fig_timeline = px.line(daily_notifs, x='Date', y='Count', color='Status', title='Notification Timeline')
                st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("Notification history is empty.")

    finally:
        db.close()


# Export for use in main app
if __name__ == "__main__":
    st.set_page_config(page_title="Deadline Reminders", layout="wide")

    # Sidebar navigation
    page = st.sidebar.radio(
        "Select Page",
        ["Manage Deadlines", "Bulk Import", "Analytics", "Notification History", "Preferences"],
    )

    if page == "Manage Deadlines":
        page_manage_deadlines()
    elif page == "Bulk Import":
        page_bulk_import()
    elif page == "Analytics":
        page_analytics()
    elif page == "Notification History":
        page_notification_history()
    elif page == "Preferences":
        page_notification_preferences()


def is_valid_email_address(email: str) -> bool:
    """
    Performs quick syntax validation on raw input email addresses 
    prior to dispatcher queue insertion.
    """
    import re
    email_regex = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    return bool(email and email_regex.match(email))
