"""
Case Deadlines page - Manage appeal deadlines
"""

import streamlit as st

# Import the deadlines management UI
from notifications_ui import page_manage_deadlines

st.set_page_config(
    page_title="Case Deadlines",
    page_icon="📅",
    layout="wide"
)

if __name__ == "__main__":
    page_manage_deadlines()
