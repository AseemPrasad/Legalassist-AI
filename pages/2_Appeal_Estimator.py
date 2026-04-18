"""
Appeal Probability Estimator

Helps users estimate their appeal success chances based on similar cases.
"""

import streamlit as st
import pandas as pd
from database import SessionLocal, CaseRecord
from analytics_engine import (
    CaseSimilarityCalculator,
    AppealProbabilityEstimator,
)
import logging

logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Appeal Estimator - LegalEase AI",
    page_icon="🎯",
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

st.title("🎯 Appeal Success Estimator")
st.markdown("*Estimate your appeal chances based on similar cases in your jurisdiction*")
st.markdown("---")

st.info("""
**How it works:**
1. Answer questions about your case
2. We find similar cases from our database
3. Calculate your estimated appeal success rate based on real data
4. Get estimates for cost and time
""")

# Get database session
db = SessionLocal()

try:
    # Get all available options
    all_cases = db.query(CaseRecord).all()
    
    if not all_cases:
        st.warning("""
        ⚠️ Not enough case data yet to make estimates.
        
        As more cases are processed and outcomes are tracked, 
        this estimator will become more accurate.
        """)
    else:
        # ==================== CASE INFORMATION ====================
        st.subheader("📋 Tell us about your case")
        
        col1, col2 = st.columns(2)
        
        with col1:
            case_type = st.selectbox(
                "Case Type",
                options=["Civil", "Criminal", "Family", "Commercial", "Labor", "Other"],
                help="What type of case is this?"
            )
        
        with col2:
            jurisdiction = st.selectbox(
                "Jurisdiction",
                options=sorted(set(case.jurisdiction for case in all_cases if case.jurisdiction)),
                help="Which state/jurisdiction is the case in?"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            courts = sorted(set(
                case.court_name for case in all_cases
                if case.court_name and case.jurisdiction == jurisdiction
            ))
            
            if courts:
                court_name = st.selectbox(
                    "Court",
                    options=courts + ["Not specified"],
                    help="Which court is this case in?"
                )
                if court_name == "Not specified":
                    court_name = None
            else:
                court_name = None
                st.info(f"No specific court data for {jurisdiction}")
        
        with col2:
            judges = sorted(set(
                case.judge_name for case in all_cases
                if case.judge_name and case.jurisdiction == jurisdiction
            ))
            
            if judges:
                judge_name = st.selectbox(
                    "Judge (Optional)",
                    options=judges + ["Not specified"],
                    help="If you know the judge, select them"
                )
                if judge_name == "Not specified":
                    judge_name = None
            else:
                judge_name = None
        
        st.markdown("---")
        
        # ==================== CASE OUTCOME ====================
        st.subheader("⚖️ About the judgment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            outcome = st.radio(
                "How did the case go?",
                options=["You won (plaintiff)", "You lost (defendant)", "Settlement"],
                help="What was the outcome of the judgment?"
            )
            
            outcome_mapping = {
                "You won (plaintiff)": "plaintiff_won",
                "You lost (defendant)": "defendant_won",
                "Settlement": "settlement",
            }
            outcome_value = outcome_mapping[outcome]
        
        with col2:
            magnitude = st.radio(
                "Decision strength",
                options=["Marginal (close decision)", "Moderate", "Clear cut"],
                help="How clear-cut was the judgment?"
            )
            
            magnitude_mapping = {
                "Marginal (close decision)": "low",
                "Moderate": "moderate",
                "Clear cut": "high",
            }
            magnitude_value = magnitude_mapping[magnitude]
        
        st.markdown("---")
        
        # ==================== ESTIMATION ====================
        if st.button("🚀 Estimate Appeal Success Rate", key="estimate_btn"):
            
            with st.spinner("Analyzing similar cases..."):
                try:
                    # Get estimate
                    estimate = AppealProbabilityEstimator.estimate_appeal_success(
                        db,
                        case_type=case_type,
                        jurisdiction=jurisdiction,
                        court_name=court_name,
                        judge_name=judge_name,
                        outcome_magnitude=magnitude_value,
                        similar_cases_limit=100,
                    )
                    
                    # Get cost and time estimates
                    cost_time = AppealProbabilityEstimator.estimate_appeal_cost_and_time(
                        db,
                        case_type=case_type,
                        jurisdiction=jurisdiction,
                    )
                    
                    st.markdown("---")
                    st.subheader("📊 Your Appeal Estimate")
                    
                    # Main result
                    if estimate["estimated_success_rate"] is not None:
                        # Create visual representation
                        success_rate = estimate["estimated_success_rate"]
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col2:
                            # Progress bar visualization
                            st.metric(
                                "Estimated Appeal Success Rate",
                                f"{success_rate}%",
                                delta=f"Based on {estimate['similar_cases_found']} similar cases",
                            )
                            
                            # Confidence indicator
                            confidence_colors = {
                                "high": "🟢",
                                "medium": "🟡",
                                "low": "🟠",
                                "very_low": "🔴",
                            }
                            
                            confidence = estimate["confidence"]
                            st.write(f"**Confidence Level:** {confidence_colors.get(confidence, '❓')} {confidence.title()}")
                        
                        st.markdown("---")
                        
                        # Details
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("💰 Cost & Time Estimates")
                            st.write(f"**Estimated Cost:** {cost_time['avg_cost']}")
                            st.write(f"**Typical Duration:** {cost_time['avg_time']}")
                        
                        with col2:
                            st.subheader("📈 Success Rate Breakdown")
                            st.write(f"**Similar Cases Found:** {estimate['similar_cases_found']}")
                            st.write(f"**Success in Similar Cases:** {estimate['appeal_success_rate_from_similar']:.1f}%")
                        
                        st.markdown("---")
                        
                        st.subheader("📝 Analysis")
                        st.info(estimate["reasoning"])
                        
                        # ==================== RECOMMENDATIONS ====================
                        st.markdown("---")
                        st.subheader("💡 What Should You Do?")
                        
                        if success_rate >= 60:
                            st.success("""
                            ✅ **High Likelihood of Success**
                            
                            Your case has good chances on appeal based on similar cases.
                            Consider consulting with a lawyer to prepare your appeal.
                            """)
                        elif success_rate >= 40:
                            st.warning("""
                            ⚠️ **Moderate Likelihood**
                            
                            Your appeal has a reasonable chance, but it's not guaranteed.
                            Consult with a lawyer about the strengths and weaknesses.
                            """)
                        else:
                            st.error("""
                            ❌ **Low Likelihood of Success**
                            
                            Based on similar cases, appeals like yours are rarely successful.
                            Consider alternative remedies or seeking legal advice first.
                            """)
                        
                        # Legal resources
                        st.markdown("---")
                        st.subheader("📞 Free Legal Help")
                        st.info("""
                        **Need help with your appeal?**
                        - National Legal Services (NALSA): 1800-180-8111
                        - Bar Council of India: bci.org.in
                        - Find local legal clinics in your area
                        """)
                    
                    else:
                        st.warning("""
                        ⚠️ Not Enough Data
                        
                        We don't have enough similar cases to make an estimate yet.
                        Try selecting a different jurisdiction or check back later.
                        """)
                
                except Exception as e:
                    logger.error(f"Estimation error: {str(e)}")
                    st.error(f"Error calculating estimate: {str(e)}")
        
        # ==================== SIMILAR CASES ====================
        st.markdown("---")
        st.subheader("🔍 Find Similar Cases")
        
        if st.button("Show Similar Cases", key="similar_btn"):
            with st.spinner("Finding similar cases..."):
                try:
                    # Find similar cases
                    sample_case = db.query(CaseRecord).filter(
                        CaseRecord.case_type == case_type,
                        CaseRecord.jurisdiction == jurisdiction,
                    ).first()
                    
                    if sample_case:
                        similar_cases = CaseSimilarityCalculator.find_similar_cases(
                            db, sample_case, min_similarity=50, limit=20
                        )
                        
                        if similar_cases:
                            st.subheader(f"Found {len(similar_cases)} Similar Cases")
                            
                            # Create dataframe
                            similar_data = []
                            for case, score in similar_cases:
                                appeal_info = "No"
                                if case.outcome_data and case.outcome_data.appeal_filed:
                                    appeal_info = f"Yes - {case.outcome_data.appeal_outcome}"
                                
                                similar_data.append({
                                    "Case Type": case.case_type,
                                    "Court": case.court_name or "N/A",
                                    "Outcome": case.outcome,
                                    "Appeal": appeal_info,
                                    "Similarity": f"{score:.1f}%",
                                })
                            
                            df_similar = pd.DataFrame(similar_data)
                            st.dataframe(df_similar, use_container_width=True)
                        else:
                            st.info("No similar cases found.")
                    else:
                        st.info("No cases match your criteria yet.")
                
                except Exception as e:
                    logger.error(f"Similar cases error: {str(e)}")
                    st.error(f"Error finding similar cases: {str(e)}")

except Exception as e:
    logger.error(f"Estimator error: {str(e)}")
    st.error(f"Error: {str(e)}")

finally:
    db.close()
