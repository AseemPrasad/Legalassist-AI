"""
Notification History page - View past notifications
"""

import streamlit as st

# Import the notification history UI
from notifications_ui import page_notification_history

st.set_page_config(
    page_title="Notification History",
    page_icon="📜",
    layout="wide"
)

if __name__ == "__main__":
    page_notification_history()
