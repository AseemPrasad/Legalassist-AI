"""
Settings/Preferences page - Notification preferences, Model Routing Rules, and Model Performance.
"""

import streamlit as st
import pandas as pd
import datetime as dt

from database import SessionLocal
from db.models.analytics import ModelRoutingRule, ModelPerformance
from notifications_ui import page_notification_preferences

st.set_page_config(
    page_title="Settings & Preferences",
    page_icon="⚙️",
    layout="wide"
)

# Sidebar tab routing
tab = st.sidebar.radio(
    "Select Settings Tab",
    ["Notification Preferences", "Model Routing Rules", "Model Performance Metrics"]
)

if tab == "Notification Preferences":
    page_notification_preferences()

elif tab == "Model Routing Rules":
    st.title("⚖️ Model Routing Rules")
    st.markdown(
        "Configure custom routing rules to map specific tasks (e.g. `summary`, `remedies`, `extraction`) "
        "or specific case attributes (case type, jurisdiction) to specific LLM models."
    )

    db = SessionLocal()
    try:
        # Form to add new rule
        st.subheader("➕ Add Routing Rule")
        with st.form("add_rule_form"):
            col1, col2 = st.columns(2)
            with col1:
                rule_name = st.text_input("Rule Name", placeholder="e.g. Llama-Remedies-Delhi")
                task_option = st.selectbox("LLM Task", ["remedies", "summary", "extraction", "drafting", "general"])
                preferred_model = st.text_input("Preferred Model", value="meta-llama/llama-3.1-8b-instruct")
            with col2:
                case_type = st.text_input("Case Type Filter (Optional)", placeholder="e.g. civil (leave empty for all)")
                jurisdiction = st.text_input("Jurisdiction Filter (Optional)", placeholder="e.g. Delhi (leave empty for all)")
                approved = st.checkbox("Approve & Enable Rule", value=True)

            submitted = st.form_submit_button("💾 Save Rule", use_container_width=True)
            if submitted:
                if not rule_name or not preferred_model:
                    st.error("❌ Rule Name and Preferred Model are required fields.")
                else:
                    new_rule = ModelRoutingRule(
                        name=rule_name,
                        task=task_option,
                        case_type=case_type.strip().lower() if case_type.strip() else None,
                        jurisdiction=jurisdiction.strip().lower() if jurisdiction.strip() else None,
                        preferred_model=preferred_model.strip(),
                        approved=approved
                    )
                    db.add(new_rule)
                    db.commit()
                    st.success(f"✅ Saved rule '{rule_name}' successfully!")
                    st.rerun()

        st.divider()

        # Display and manage existing rules
        st.subheader("📋 Configured Routing Rules")
        rules = db.query(ModelRoutingRule).all()
        if not rules:
            st.info("No routing rules configured. The system will fall back to DEFAULT_MODEL.")
        else:
            for rule in rules:
                status_emoji = "🟢 Active" if rule.approved else "🔴 Disabled"
                with st.container(border=True):
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.markdown(f"### {rule.name} ({status_emoji})")
                        st.markdown(f"**Target Task:** `{rule.task}` | **Model:** `{rule.preferred_model}`")
                        filters = []
                        if rule.case_type:
                            filters.append(f"Case Type: `{rule.case_type}`")
                        if rule.jurisdiction:
                            filters.append(f"Jurisdiction: `{rule.jurisdiction}`")
                        filter_str = " & ".join(filters) if filters else "Any (No Filters)"
                        st.caption(f"**Filters:** {filter_str}")
                    with col2:
                        # Toggle button
                        toggle_label = "Disable" if rule.approved else "Enable"
                        if st.button(toggle_label, key=f"toggle_{rule.id}"):
                            rule.approved = not rule.approved
                            db.commit()
                            st.rerun()
                    with col3:
                        # Delete button
                        if st.button("🗑️ Delete", key=f"delete_{rule.id}"):
                            db.delete(rule)
                            db.commit()
                            st.rerun()
    finally:
        db.close()

elif tab == "Model Performance Metrics":
    st.title("📊 Model Performance Metrics")
    st.markdown(
        "Monitor latency and token counts across different LLMs and legal tasks. "
        "Statistics are captured dynamically from live API requests."
    )

    db = SessionLocal()
    try:
        metrics = db.query(ModelPerformance).all()
        if not metrics:
            st.info("No performance metrics captured yet. Run summaries or remedies tasks to collect statistics!")
        else:
            # Display metrics table
            data = []
            for m in metrics:
                data.append({
                    "Model": m.model_name,
                    "Task": m.task,
                    "Case Type": m.case_type or "any",
                    "Jurisdiction": m.jurisdiction or "any",
                    "Requests (Samples)": m.samples,
                    "Avg Latency (ms)": m.average_latency_ms,
                    "Accuracy": m.accuracy,
                    "Last Updated": m.last_updated.strftime("%Y-%m-%d %H:%M:%S") if m.last_updated else "N/A"
                })
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            if st.button("🗑️ Reset All Metrics", use_container_width=True):
                db.query(ModelPerformance).delete()
                db.commit()
                st.success("✅ Performance metrics reset successfully!")
                st.rerun()
    finally:
        db.close()
