"""
Sample Data Generator for Analytics Testing

Generates anonymized sample case data to seed the analytics database.
This allows testing of the analytics dashboard and estimator before real data arrives.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from datetime import datetime, timezone, timedelta
from database import (
    SessionLocal,
    init_db,
    CaseRecord,
    CaseOutcome,
    create_case_record,
    update_case_outcome,
)
from analytics_engine import generate_anonymous_case_id
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample data
JURISDICTIONS = [
    "Delhi", "Maharashtra", "Karnataka", "Tamil Nadu", "West Bengal",
    "Uttar Pradesh", "Bihar", "Rajasthan", "Gujarat", "Telangana"
]

CASE_TYPES = ["Civil", "Criminal", "Family", "Commercial", "Labor"]

COURTS = {
    "Delhi": ["Delhi District Court", "Delhi High Court"],
    "Maharashtra": ["Mumbai District Court", "Bombay High Court"],
    "Karnataka": ["Bangalore District Court", "Karnataka High Court"],
    "Tamil Nadu": ["Chennai District Court", "Madras High Court"],
    "West Bengal": ["Kolkata District Court", "Calcutta High Court"],
}

JUDGES = [
    "Justice Sharma", "Justice Patel", "Justice Singh", "Justice Kumar",
    "Justice Verma", "Justice Desai", "Justice Roy", "Justice Reddy",
    "Justice Banerjee", "Justice Iyer"
]

OUTCOMES = ["plaintiff_won", "defendant_won", "settlement", "dismissal"]

APPEAL_OUTCOMES = ["appeal_allowed", "appeal_rejected", "pending", "withdrawn"]


def generate_sample_cases(num_cases=200):
    """Generate sample case records"""
    db = SessionLocal()
    
    logger.info(f"Generating {num_cases} sample cases...")
    
    try:
        for i in range(num_cases):
            # Random selection
            case_type = random.choice(CASE_TYPES)
            jurisdiction = random.choice(JURISDICTIONS)
            
            court_options = COURTS.get(jurisdiction, ["District Court", "High Court"])
            court_name = random.choice(court_options)
            
            judge_name = random.choice(JUDGES) if random.random() > 0.3 else None
            outcome = random.choice(OUTCOMES)
            
            # Create case
            case_id = generate_anonymous_case_id(f"case_{jurisdiction}_{i}_{random.random()}")
            
            case = create_case_record(
                db,
                case_id=case_id,
                case_type=case_type,
                jurisdiction=jurisdiction,
                court_name=court_name,
                judge_name=judge_name,
                plaintiff_type=random.choice(["individual", "organization", "government"]),
                defendant_type=random.choice(["individual", "organization", "government"]),
                case_value=random.choice(["<1L", "1-5L", "5-10L", ">10L"]),
                outcome=outcome,
                judgment_summary=f"Sample {case_type} case outcome: {outcome}",
            )
            
            # Add appeal data (60% of cases)
            if random.random() < 0.6:
                appeal_filed = True
                appeal_outcome = random.choice(APPEAL_OUTCOMES)
                appeal_success = True if appeal_outcome == "appeal_allowed" else (False if appeal_outcome == "appeal_rejected" else None)
                
                # Appeal duration: 300-1200 days (10-40 months)
                time_to_verdict = random.randint(300, 1200) if appeal_success is not None else None
                
                # Cost: varies by case type
                cost_multipliers = {
                    "Civil": (10000, 30000),
                    "Criminal": (5000, 15000),
                    "Family": (8000, 20000),
                    "Commercial": (20000, 50000),
                    "Labor": (5000, 12000),
                }
                
                min_cost, max_cost = cost_multipliers.get(case_type, (10000, 30000))
                appeal_cost = random.randint(min_cost, max_cost)
                
                update_case_outcome(
                    db,
                    case_id=case_id,
                    appeal_filed=appeal_filed,
                    appeal_date=datetime.now(timezone.utc) - timedelta(days=random.randint(100, 800)),
                    appeal_outcome=appeal_outcome,
                    appeal_success=appeal_success,
                    time_to_appeal_verdict=time_to_verdict,
                    appeal_cost=f"₹{appeal_cost:,.0f}",
                )
            
            if (i + 1) % 50 == 0:
                logger.info(f"Generated {i + 1} cases...")
        
        logger.info(f"✅ Successfully generated {num_cases} sample cases!")
        
        # Print summary
        from analytics_engine import AnalyticsAggregator
        summary = AnalyticsAggregator.get_dashboard_summary(db)
        
        logger.info(f"Summary:")
        logger.info(f"  - Total cases: {summary['total_cases_processed']}")
        logger.info(f"  - Appeals filed: {summary['appeals_filed']} ({summary['appeal_rate_percent']:.1f}%)")
        logger.info(f"  - Plaintiff wins: {summary['plaintiff_wins']}")
        logger.info(f"  - Defendant wins: {summary['defendant_wins']}")
        
    except Exception as e:
        logger.error(f"Error generating sample cases: {str(e)}")
        raise
    finally:
        db.close()


def clear_sample_data():
    """Clear all sample data (careful!)"""
    db = SessionLocal()
    try:
        # Clear all records
        db.query(CaseOutcome).delete()
        db.query(CaseRecord).delete()
        db.commit()
        logger.info("✅ Cleared all case records")
    except Exception as e:
        logger.error(f"Error clearing data: {str(e)}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    
    # Initialize database
    init_db()
    
    if len(sys.argv) > 1 and sys.argv[1] == "clear":
        logger.warning("Clearing all sample data...")
        clear_sample_data()
    else:
        # Generate sample data
        num_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 200
        generate_sample_cases(num_cases)
        
        logger.info("""
        ✅ Sample data generation complete!
        
        You can now:
        1. Run the app with: streamlit run app.py
        2. Navigate to the Analytics Dashboard page
        3. Try the Appeal Estimator
        4. Submit feedback via Report Outcome
        
        To clear this sample data later, run:
        python scripts/generate_sample_analytics_data.py clear
        """)
