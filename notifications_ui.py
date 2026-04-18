"""
Streamlit UI components for deadline management and notification preferences.
Integrate these into the main app.py or use as a separate page.
"""

import streamlit as st
from datetime import datetime, timezone, timedelta
import pytz
from typing import Optional

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


def get_user_id() -> str:
    """Get user ID from session state (implement auth as needed)"""
    if "user_id" not in st.session_state:
        st.session_state.user_id = st.secrets.get("TEST_USER_ID", "user_default")
    return st.session_state.user_id


def page_notification_preferences():
    """Page: User Notification Preferences"""
    st.title("⚙️ Notification Preferences")

    db = SessionLocal()
    try:
        user_id = get_user_id()

        # Get existing preferences
        user_pref = db.query(UserPreference).filter(
            UserPreference.user_id == user_id
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
            "Your reminders will be sent at **8 AM** in your local timezone on these days:"
        )

        col1, col2 = st.columns(2)
        with col1:
            notify_30 = st.checkbox(
                "30 days before deadline",
                value=user_pref.notify_30_days if user_pref else True,
                key="notify_30",
            )
            notify_3 = st.checkbox(
                "3 days before deadline",
                value=user_pref.notify_3_days if user_pref else True,
                key="notify_3",
            )

        with col2:
            notify_10 = st.checkbox(
                "10 days before deadline",
                value=user_pref.notify_10_days if user_pref else True,
                key="notify_10",
            )
            notify_1 = st.checkbox(
                "1 day before deadline",
                value=user_pref.notify_1_day if user_pref else True,
                key="notify_1",
            )

        # Save preferences
        if st.button("💾 Save Preferences", use_container_width=True):
            try:
                create_or_update_user_preference(
                    db=db,
                    user_id=user_id,
                    email=email_input,
                    phone_number=phone_input if phone_input else None,
                    notification_channel=channel_options[selected_channel],
                    timezone=timezone,
                )

                # Update the preference object to reflect new values
                user_pref = db.query(UserPreference).filter(
                    UserPreference.user_id == user_id
                ).first()
                
                # Update boolean fields
                user_pref.notify_30_days = notify_30
                user_pref.notify_10_days = notify_10
                user_pref.notify_3_days = notify_3
                user_pref.notify_1_day = notify_1
                db.commit()

                st.success("✅ Preferences saved successfully!")
                logger.info(f"Preferences updated for user {user_id}")
            except Exception as e:
                st.error(f"❌ Error saving preferences: {str(e)}")
                logger.error(f"Error saving preferences: {str(e)}")

    finally:
        db.close()

    # Info section
    st.divider()
    st.info(
        """
        ### How Deadline Reminders Work
        
        - **30-day reminder**: Initial alert to prepare for the deadline
        - **10-day reminder**: Action required soon
        - **3-day reminder**: Critical - urgent action needed
        - **1-day reminder**: Last chance warning
        
        All reminders are sent at **8 AM** in your timezone to ensure you see them
        """
    )


def page_manage_deadlines():
    """Page: Add and manage case deadlines"""
    st.title("📅 Case Deadlines")

    db = SessionLocal()
    try:
        user_id = get_user_id()

        # Check if user has preferences set up
        user_pref = db.query(UserPreference).filter(
            UserPreference.user_id == user_id
        ).first()

        if not user_pref:
            st.warning("⚠️ Please set up your notification preferences first!")
            if st.button("Go to Preferences"):
                st.switch_page("pages/notifications.py")
            return

        # Add new deadline
        st.subheader("➕ Add New Deadline")
        with st.form("add_deadline_form"):
            col1, col2 = st.columns(2)

            with col1:
                case_id = st.text_input("Case ID", placeholder="e.g., CASE-2024-001")
                case_title = st.text_input("Case Title", placeholder="e.g., Property Dispute")

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
                if not case_id or not case_title:
                    st.error("❌ Case ID and Case Title are required")
                else:
                    try:
                        # Convert date to datetime
                        deadline_datetime = datetime.combine(
                            deadline_date, datetime.min.time()
                        ).replace(tzinfo=timezone.utc)

                        create_case_deadline(
                            db=db,
                            user_id=user_id,
                            case_id=case_id,
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
        deadlines = get_user_deadlines(db, user_id)

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
    st.title("📬 Notification History")

    db = SessionLocal()
    try:
        user_id = get_user_id()

        # Get notification history
        notifications = get_notification_history(db, user_id, limit=100)

        if not notifications:
            st.info("No notifications sent yet.")
            return

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)

        total = len(notifications)
        sent = len([n for n in notifications if n.status.value == "sent"])
        failed = len([n for n in notifications if n.status.value == "failed"])
        sms_count = len([n for n in notifications if n.channel.value == "sms"])

        with col1:
            st.metric("Total Notifications", total)
        with col2:
            st.metric("Successfully Sent", sent)
        with col3:
            st.metric("Failed", failed)
        with col4:
            st.metric("Via SMS", sms_count)

        st.divider()

        # Notification table
        st.subheader("Recent Notifications")

        for notif in notifications[:20]:  # Show last 20
            status_emoji = {
                "sent": "✅",
                "failed": "❌",
                "pending": "⏳",
                "bounced": "↩️",
                "opened": "👁️",
            }.get(notif.status.value, "❓")

            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

                with col1:
                    st.text(f"Case: {notif.deadline.case_title}")
                    st.caption(notif.recipient)

                with col2:
                    st.text(f"Channel: {notif.channel.value.upper()}")
                    st.caption(f"Reminder: {notif.days_before} day(s)")

                with col3:
                    st.text(f"Sent: {notif.created_at.strftime('%d %b %Y %H:%M')}")

                with col4:
                    st.markdown(f"### {status_emoji}")

                if notif.error_message:
                    st.error(f"Error: {notif.error_message}")

    finally:
        db.close()


# Logging for debugging
import logging

logger = logging.getLogger(__name__)


# Export for use in main app
if __name__ == "__main__":
    st.set_page_config(page_title="Deadline Reminders", layout="wide")

    # Sidebar navigation
    page = st.sidebar.radio(
        "Select Page",
        ["Manage Deadlines", "Notification History", "Preferences"],
    )

    if page == "Manage Deadlines":
        page_manage_deadlines()
    elif page == "Notification History":
        page_notification_history()
    elif page == "Preferences":
        page_notification_preferences()
