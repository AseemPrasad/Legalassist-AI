"""
Outcome Feedback Form

Allows users to report back on case outcomes and appeal results.
This data is crucial for training the analytics engine.
"""

import streamlit as st
from datetime import datetime, timezone
from database import SessionLocal, submit_user_feedback
import logging

logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Report Outcome - LegalEase AI",
    page_icon="📝",
    layout="centered",
)

# Styling
st.markdown("""
<style>
    body {
        background-color: #0d0d0f;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📝 Report Your Case Outcome")
st.markdown("*Help us improve by sharing your case results*")
st.markdown("---")

st.info("""
**Your feedback helps others!**

By sharing the outcome of your case and appeal, you help us build more accurate 
prediction models that help future users make better decisions.

All data is anonymized and kept private.
""")

# Get database session
db = SessionLocal()

try:
    # Create form
    with st.form("outcome_form", clear_on_submit=True):
        
        st.subheader("1️⃣ About Your Case")
        
        # User ID (optional, for tracking if user wants)
        user_id = st.text_input(
            "Your ID (optional)",
            help="Optional - helps us track follow-ups",
            value="anonymous"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            case_type = st.selectbox(
                "Case Type",
                options=["Civil", "Criminal", "Family", "Commercial", "Labor", "Other"]
            )
        
        with col2:
            jurisdiction = st.selectbox(
                "Jurisdiction",
                options=[
                    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
                    "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
                    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
                    "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
                    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
                    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Other"
                ]
            )
        
        st.markdown("---")
        st.subheader("2️⃣ First Judgment Outcome")
        
        col1, col2 = st.columns(2)
        
        with col1:
            your_role = st.radio(
                "Were you the plaintiff/complainant or defendant?",
                options=["Plaintiff/Complainant", "Defendant"]
            )
        
        with col2:
            first_outcome = st.radio(
                "What was the outcome for you?",
                options=["You won", "You lost", "Settlement/Compromise"]
            )
        
        st.markdown("---")
        st.subheader("3️⃣ Appeal Status")
        
        did_appeal = st.radio(
            "Did you appeal the judgment?",
            options=["Yes", "No", "Considering it"],
            index=2
        )
        
        if did_appeal == "Yes":
            st.markdown("**Appeal Details**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                appeal_outcome = st.selectbox(
                    "Appeal outcome",
                    options=["Appeal allowed (you won)", "Appeal rejected (you lost)", 
                             "Still pending", "Withdrawn"]
                )
            
            with col2:
                time_to_verdict = st.number_input(
                    "Days until appeal verdict (or expected)",
                    min_value=0,
                    max_value=3650,  # 10 years
                    step=30,
                    help="How many days from appeal filing to verdict?"
                )
            
            appeal_cost = st.number_input(
                "Appeal cost (approximate, in rupees)",
                min_value=0,
                max_value=500000,
                step=1000,
                help="Total cost including lawyer fees, court costs, etc."
            )
        else:
            appeal_outcome = None
            time_to_verdict = None
            appeal_cost = None
            
            if did_appeal == "Considering it":
                st.warning("We'd love to hear from you once you make a decision. Please come back and update!")
        
        st.markdown("---")
        st.subheader("4️⃣ Your Feedback")
        
        satisfaction = st.slider(
            "How satisfied are you with the legal system?",
            min_value=1,
            max_value=5,
            value=3,
            help="1 = Very unsatisfied, 5 = Very satisfied"
        )
        
        feedback_text = st.text_area(
            "Additional comments (optional)",
            placeholder="Share your experience, lessons learned, or suggestions...",
            height=150
        )
        
        st.markdown("---")
        
        # Submit button
        submitted = st.form_submit_button(
            "✅ Submit My Feedback",
            use_container_width=True
        )
        
        if submitted:
            try:
                # Process appeal outcome
                appeal_outcome_mapped = None
                if did_appeal == "Yes":
                    outcome_map = {
                        "Appeal allowed (you won)": "appeal_allowed",
                        "Appeal rejected (you lost)": "appeal_rejected",
                        "Still pending": "pending",
                        "Withdrawn": "withdrawn",
                    }
                    appeal_outcome_mapped = outcome_map.get(appeal_outcome)
                
                # Submit feedback
                feedback = submit_user_feedback(
                    db,
                    user_id=user_id,
                    did_appeal=did_appeal == "Yes" if did_appeal in ["Yes", "No"] else None,
                    appeal_outcome=appeal_outcome_mapped,
                    appeal_cost=int(appeal_cost) if appeal_cost and did_appeal == "Yes" else None,
                    time_to_verdict=int(time_to_verdict) if time_to_verdict and did_appeal == "Yes" else None,
                    case_type=case_type,
                    jurisdiction=jurisdiction,
                    satisfaction_rating=satisfaction,
                    feedback_text=feedback_text if feedback_text else None,
                )
                
                logger.info(f"Feedback submitted by {user_id}")
                
                st.success("""
                ✅ Thank you for your feedback!
                
                Your data has been recorded and will help us improve our estimates for other users.
                All information is kept anonymous and confidential.
                """)
                
                # Show what was recorded
                st.subheader("📋 What We Recorded")
                
                recorded_data = {
                    "Case Type": case_type,
                    "Jurisdiction": jurisdiction,
                    "Your Role": your_role,
                    "First Judgment": first_outcome,
                    "Appeal Filed": "Yes" if did_appeal == "Yes" else "No",
                }
                
                if did_appeal == "Yes":
                    recorded_data["Appeal Outcome"] = appeal_outcome
                    if time_to_verdict:
                        recorded_data["Days to Verdict"] = time_to_verdict
                    if appeal_cost:
                        recorded_data["Appeal Cost"] = f"₹{appeal_cost:,.0f}"
                
                recorded_data["Satisfaction"] = f"{satisfaction}/5"
                
                for key, value in recorded_data.items():
                    st.write(f"• **{key}:** {value}")
                
            except Exception as e:
                logger.error(f"Error submitting feedback: {str(e)}")
                st.error(f"Error submitting feedback: {str(e)}")
                st.info("Please try again or contact support if the problem persists.")

except Exception as e:
    logger.error(f"Form error: {str(e)}")
    st.error(f"Error: {str(e)}")

finally:
    db.close()

# ==================== INFORMATION ====================
st.markdown("---")
st.subheader("ℹ️ Why We Need This Data")

st.markdown("""
**Your feedback helps:**
- 📊 Create accurate statistics about case outcomes
- 🎯 Build better appeal success predictions
- 🏆 Show which judges and courts have better outcomes
- 💡 Help future users make informed decisions

**Your privacy is protected:**
- ✅ All data is anonymized
- ✅ No personally identifiable information is stored
- ✅ Your original case details are never shared
- ✅ Data is only used for aggregate statistics
""")

st.markdown("---")
st.subheader("📊 Recent Feedback Stats")

try:
    from database import CaseRecord
    all_cases = db.query(CaseRecord).all()
    
    if all_cases:
        st.write(f"✅ {len(all_cases)} cases tracked")
        st.write(f"📤 Appeal feedback received from multiple users")
    else:
        st.info("Be the first to share your case outcome!")
except Exception:
    st.info("Feedback statistics will appear here soon.")
