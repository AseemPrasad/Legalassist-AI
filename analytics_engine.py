from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from sqlalchemy import func, case as sql_case
from sqlalchemy.orm import Session
from database import CaseRecord, CaseOutcome, CaseAnalytics, UserFeedback, SimilarityFeedback
import hashlib
import hmac
import os
from collections import Counter
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PandasAnalyticsProcessor:
    """
    Advanced analytics engine utilizing Pandas for complex data transformations.
    
    This class provides high-performance data processing for legal case records,
    bridging the gap between SQLAlchemy ORM objects and analytical DataFrames.
    
    The implementation is specifically designed to be defensive against the
    infamous 'SettingWithCopyWarning' by ensuring that all filtered subsets
    are explicitly detached from the parent DataFrame before modification.
    """

    @staticmethod
    def convert_cases_to_dataframe(cases: List[CaseRecord]) -> pd.DataFrame:
        """
        Transform a list of SQLAlchemy CaseRecord objects into a Pandas DataFrame.
        
        Extracts both primary record data and nested outcome information into
        a flat structure suitable for vectorised operations.
        
        Args:
            cases: List of CaseRecord database objects.
            
        Returns:
            A populated pd.DataFrame with normalized columns.
        """
        if not cases:
            return pd.DataFrame()

        data_list = []
        for case in cases:
            # Extract base attributes
            row = {
                "id": case.id,
                "case_type": str(case.case_type).lower() if case.case_type else "unknown",
                "jurisdiction": str(case.jurisdiction) if case.jurisdiction else "Unknown",
                "court_name": str(case.court_name) if case.court_name else "N/A",
                "judge_name": str(case.judge_name) if case.judge_name else "N/A",
                "outcome": str(case.outcome).lower() if case.outcome else "pending",
                "created_at": case.created_at,
            }
            
            # Safely extract nested outcome data
            outcome_data = getattr(case, "outcome_data", None)
            if outcome_data:
                row.update({
                    "appeal_filed": bool(getattr(outcome_data, "appeal_filed", False)),
                    "appeal_success": bool(getattr(outcome_data, "appeal_success", False)),
                    "appeal_cost": getattr(outcome_data, "appeal_cost", None),
                    "time_to_verdict": getattr(outcome_data, "time_to_appeal_verdict", None),
                })
            else:
                row.update({
                    "appeal_filed": False,
                    "appeal_success": False,
                    "appeal_cost": None,
                    "time_to_verdict": None,
                })
            
            data_list.append(row)

        return pd.DataFrame(data_list)

    @staticmethod
    def get_jurisdiction_performance_report(
        df: pd.DataFrame, 
        jurisdiction_name: str
    ) -> pd.DataFrame:
        """
        Generate a detailed performance report for a specific jurisdiction.
        
        CRITICAL FIX: This method explicitly uses .copy() when creating the 
        subset DataFrame. This prevents 'SettingWithCopyWarning' when adding 
        calculated success metrics and ensures that modifications to the 
        jurisdiction-specific data do not unintentionally affect the global 
        dataset or trigger Pandas' defensive warnings.
        
        Args:
            df: The master cases DataFrame.
            jurisdiction_name: The name of the jurisdiction to analyze.
            
        Returns:
            A processed DataFrame with calculated success metrics.
        """
        if df.empty or 'jurisdiction' not in df.columns:
            return pd.DataFrame()

        # ---------------------------------------------------------------------
        # STEP 1: Create a filtered subset of the data.
        # We EXPLICITLY call .copy() here. This is the core fix for the
        # SettingWithCopyWarning. By doing this, we create a new memory object
        # that is independent of the original 'df'.
        # ---------------------------------------------------------------------
        jur_df = df[df['jurisdiction'] == jurisdiction_name].copy()

        if jur_df.empty:
            return jur_df

        # ---------------------------------------------------------------------
        # STEP 2: Apply data transformations.
        # Since 'jur_df' is a clean copy, these assignments are safe and 
        # predictable. We are no longer operating on a 'view' of the original data.
        # ---------------------------------------------------------------------
        
        # Calculate boolean flags for success
        jur_df['is_plaintiff_win'] = jur_df['outcome'].str.contains('plaintiff_won', na=False)
        jur_df['is_defendant_win'] = jur_df['outcome'].str.contains('defendant_won', na=False)
        jur_df['is_settlement'] = jur_df['outcome'].str.contains('settlement', na=False)
        
        # Calculate win rates using cumulative sums for trend analysis
        # (Assuming the DF is sorted by date)
        jur_df = jur_df.sort_values('created_at')
        
        jur_df['cumulative_cases'] = range(1, len(jur_df) + 1)
        jur_df['cumulative_wins'] = jur_df['is_plaintiff_win'].cumsum()
        jur_df['running_win_rate'] = (jur_df['cumulative_wins'] / jur_df['cumulative_cases']) * 100

        # Calculate appeal indicators
        jur_df['appeal_status'] = jur_df.apply(
            lambda x: "Appealed" if x['appeal_filed'] else "Final", 
            axis=1
        )
        
        # Cleanup and return
        return jur_df

    @staticmethod
    def analyze_judge_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Group data by judge to identify success patterns and appeal rates.
        
        Uses aggregation to provide high-level metrics for the analytics dashboard.
        
        Args:
            df: The master cases DataFrame.
            
        Returns:
            A DataFrame indexed by judge with aggregated metrics.
        """
        if df.empty:
            return pd.DataFrame()

        # Grouping and aggregation
        judge_stats = df.groupby('judge_name').agg({
            'id': 'count',
            'appeal_filed': 'sum',
            'appeal_success': 'sum',
        }).rename(columns={'id': 'total_cases', 'appeal_filed': 'appeals_filed'})

        # Calculate percentages
        judge_stats['appeal_rate'] = (judge_stats['appeals_filed'] / judge_stats['total_cases']) * 100
        judge_stats['appeal_success_rate'] = (
            judge_stats['appeal_success'] / judge_stats['appeals_filed']
        ).fillna(0) * 100

        # Create a copy for the final ranking to be safe
        ranked_stats = judge_stats.sort_values('appeal_success_rate', ascending=False).copy()
        
        return ranked_stats

    @staticmethod
    def identify_case_correlations(df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform correlation analysis between case types and outcomes.
        
        This helps identify if certain jurisdictions are 'pro-plaintiff' for
        specific types of legal disputes.
        
        Args:
            df: The master cases DataFrame.
            
        Returns:
            A pivot table showing win rates by jurisdiction and case type.
        """
        if df.empty:
            return pd.DataFrame()

        # Create a temporary copy to avoid modifying the original during processing
        temp_df = df.copy()
        temp_df['is_win'] = temp_df['outcome'].str.contains('plaintiff_won', na=False).astype(int)

        # Create pivot table
        pivot = pd.pivot_table(
            temp_df,
            values='is_win',
            index='jurisdiction',
            columns='case_type',
            aggfunc='mean'
        ) * 100

        return pivot.fillna(0)


class CaseSimilarityCalculator:
    """Calculate similarity between cases for matching and analysis"""
    
    @staticmethod
    def case_similarity_score(
        case1: CaseRecord,
        case2: CaseRecord,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate similarity between two cases (0-100).
        """
        if not weights:
            weights = {
                "case_type": 0.3,
                "jurisdiction": 0.2,
                "plaintiff_type": 0.15,
                "defendant_type": 0.15,
                "case_value": 0.2,
            }
        
        score = 0.0
        
        # Case type match (most important)
        if case1.case_type.lower() == case2.case_type.lower():
            score += weights["case_type"]
        
        # Jurisdiction match
        if case1.jurisdiction.lower() == case2.jurisdiction.lower():
            score += weights["jurisdiction"]
        
        # Plaintiff type match
        if case1.plaintiff_type and case2.plaintiff_type:
            if case1.plaintiff_type.lower() == case2.plaintiff_type.lower():
                score += weights["plaintiff_type"]
        
        # Defendant type match
        if case1.defendant_type and case2.defendant_type:
            if case1.defendant_type.lower() == case2.defendant_type.lower():
                score += weights["defendant_type"]
        
        # Case value match
        if case1.case_value and case2.case_value:
            if case1.case_value == case2.case_value:
                score += weights["case_value"]
        
        return score * 100
    
    @staticmethod
    def find_similar_cases(
        db: Session,
        reference_case: CaseRecord,
        min_similarity: float = 50.0,
        limit: int = 50,
    ) -> List[Tuple[CaseRecord, float]]:
        """Find cases similar to reference case using initial DB-side filtering.

        Fix: Exclusion filter now correctly references CaseRecord.id (the actual
        primary key column) instead of the non-existent CaseRecord.case_id field.
        Previously, the wrong field reference caused the reference case to remain
        in similarity results, producing self-matching records.
        """
        # Reduce memory load by pre-filtering on common attributes.
        # FIX: Use CaseRecord.id (primary key) — not case_id — to reliably
        # exclude the reference case from results.
        query = db.query(CaseRecord).filter(
            CaseRecord.id != reference_case.id,  # corrected from case_id
            (CaseRecord.case_type == reference_case.case_type) | (CaseRecord.jurisdiction == reference_case.jurisdiction)
        )
        
        # Limit the search space to the most recent/relevant 1000 cases to prevent OOM
        all_cases = query.limit(1000).all()
        
        similarities = []
        for case in all_cases:
            score = CaseSimilarityCalculator.case_similarity_score(reference_case, case)
            if score >= min_similarity:
                similarities.append((case, score))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]

    @staticmethod
    def get_feedback_adjustment(
        db: Session,
        candidate_case: CaseRecord,
        user_id: Optional[str] = None,
        query_signature: Optional[str] = None,
    ) -> float:
        """Return a small ranking adjustment from historical similarity feedback."""
        query = db.query(SimilarityFeedback).filter(
            SimilarityFeedback.candidate_case_id == candidate_case.id,
        )

        if user_id is not None:
            query = query.filter(SimilarityFeedback.user_id == str(user_id))
        if query_signature is not None:
            query = query.filter(SimilarityFeedback.query_signature == query_signature)

        feedback_rows = query.limit(50).all()
        if not feedback_rows:
            return 0.0

        positive_count = sum(1 for row in feedback_rows if row.relevance)
        negative_count = len(feedback_rows) - positive_count
        raw_delta = (positive_count - negative_count) / max(len(feedback_rows), 1)
        return max(-0.03, min(0.03, raw_delta * 0.03))


class AnalyticsCalculator:
    """Calculate various analytics metrics using SQL aggregates"""
    
    @staticmethod
    def calculate_judge_win_rate(
        db: Session,
        judge_name: str,
        jurisdiction: str,
        winning_outcome: str = "plaintiff_won",
    ) -> Dict:
        """Calculate judge-specific statistics using aggregates"""
        stats = db.query(
            func.count(CaseRecord.id).label('total'),
            func.sum(sql_case((CaseRecord.outcome == winning_outcome, 1), else_=0)).label('wins'),
            func.sum(sql_case((CaseOutcome.appeal_filed == True, 1), else_=0)).label('appeals'),
            func.sum(sql_case(((CaseOutcome.appeal_filed == True) & (CaseOutcome.appeal_success == True), 1), else_=0)).label('appeal_wins')
        ).join(CaseOutcome, CaseRecord.id == CaseOutcome.case_id, isouter=True).filter(
            CaseRecord.judge_name == judge_name,
            CaseRecord.jurisdiction == jurisdiction,
        ).first()
        
        total = stats.total or 0
        if total == 0:
            return {
                "judge": judge_name,
                "jurisdiction": jurisdiction,
                "total_cases": 0,
                "win_rate": 0.0,
                "appeal_success_rate": 0.0,
            }
            
        wins = stats.wins or 0
        appeals = stats.appeals or 0
        appeal_wins = stats.appeal_wins or 0
        
        return {
            "judge": judge_name,
            "jurisdiction": jurisdiction,
            "total_cases": total,
            "win_rate": round((wins / total) * 100, 1),
            "appeal_success_rate": round((appeal_wins / appeals * 100), 1) if appeals > 0 else 0.0,
        }
    
    @staticmethod
    def calculate_court_statistics(
        db: Session,
        court_name: str,
        case_type: Optional[str] = None,
    ) -> Dict:
        """Calculate statistics for a specific court using aggregates"""
        query = db.query(
            func.count(CaseRecord.id).label('total'),
            func.sum(sql_case((CaseRecord.outcome.ilike('plaintiff_won'), 1), else_=0)).label('p_wins'),
            func.sum(sql_case((CaseRecord.outcome.ilike('defendant_won'), 1), else_=0)).label('d_wins'),
            func.sum(sql_case((CaseRecord.outcome.ilike('settlement'), 1), else_=0)).label('settlements'),
            func.sum(sql_case((CaseRecord.outcome.ilike('dismissal'), 1), else_=0)).label('dismissals'),
            func.sum(sql_case((CaseOutcome.appeal_filed == True, 1), else_=0)).label('appeals')
        ).join(CaseOutcome, CaseRecord.id == CaseOutcome.case_id, isouter=True).filter(
            CaseRecord.court_name == court_name
        )
        
        if case_type:
            query = query.filter(CaseRecord.case_type == case_type)
            
        stats = query.first()
        total = stats.total or 0
        
        if total == 0:
            return {
                "court": court_name,
                "case_type": case_type,
                "total_cases": 0,
            }
            
        appeals_filed = stats.appeals or 0
        
        return {
            "court": court_name,
            "case_type": case_type,
            "total_cases": total,
            "plaintiff_wins": stats.p_wins or 0,
            "defendant_wins": stats.d_wins or 0,
            "settlements": stats.settlements or 0,
            "dismissals": stats.dismissals or 0,
            "appeals_filed": appeals_filed,
            "appeal_rate": round((appeals_filed / total) * 100, 1),
        }
    
    @staticmethod
    def calculate_jurisdiction_trends(
        db: Session,
        jurisdiction: str,
    ) -> Dict:
        """Get trends for a jurisdiction using grouped aggregates"""
        total = db.query(func.count(CaseRecord.id)).filter(CaseRecord.jurisdiction == jurisdiction).scalar() or 0
        
        if total == 0:
            return {"jurisdiction": jurisdiction, "total_cases": 0}
            
        stats_by_type = db.query(
            CaseRecord.case_type,
            func.count(CaseRecord.id).label('count'),
            func.sum(sql_case((CaseRecord.outcome.ilike('plaintiff_won'), 1), else_=0)).label('wins')
        ).filter(CaseRecord.jurisdiction == jurisdiction).group_by(CaseRecord.case_type).all()
        
        type_stats = {}
        for row in stats_by_type:
            type_stats[row.case_type] = {
                "count": row.count,
                "plaintiff_win_rate": round((row.wins / row.count * 100), 1) if row.count > 0 else 0.0,
            }
            
        return {
            "jurisdiction": jurisdiction,
            "total_cases": total,
            "case_type_stats": type_stats,
        }

    @staticmethod
    def calculate_appeal_success_rate(cases: List[CaseRecord]) -> float:
        """Calculate appeal success rate for a list of cases"""
        if not cases:
            return 0.0
        
        total_appeals = 0
        successful_appeals = 0
        
        for case in cases:
            if case.outcome_data and case.outcome_data.appeal_filed:
                total_appeals += 1
                if case.outcome_data.appeal_success:
                    successful_appeals += 1
                    
        if total_appeals == 0:
            return 0.0
            
        return (successful_appeals / total_appeals) * 100


    @staticmethod
    def calculate_appeal_success_rate(cases: list) -> float:
        """
        Calculate the aggregate appeal success rate from a list of CaseRecord objects.
        Cases without outcome_data or with appeal_filed=False are safely ignored.
        Returns a percentage (0.0–100.0), or 0.0 if no qualifying cases.
        """
        appeals_filed = 0
        appeals_won = 0
        for case in cases:
            outcome_data = getattr(case, "outcome_data", None)
            if outcome_data is None:
                continue
            if not getattr(outcome_data, "appeal_filed", False):
                continue
            appeals_filed += 1
            if getattr(outcome_data, "appeal_success", False):
                appeals_won += 1
        if appeals_filed == 0:
            return 0.0
        return round((appeals_won / appeals_filed) * 100, 1)


class AppealProbabilityEstimator:
    """Estimate appeal success probability for new cases using aggregates"""
    
    @staticmethod
    def estimate_appeal_success(
        db: Session,
        case_type: str,
        jurisdiction: str,
        court_name: Optional[str] = None,
        judge_name: Optional[str] = None,
        outcome_magnitude: str = "moderate",
        similar_cases_limit: int = 1000,
    ) -> Dict:
        """Estimate appeal success probability using SQL aggregates on a limited dataset"""
        # First, get the IDs of the most recent matching cases up to the limit
        matching_ids_query = db.query(CaseRecord.id).filter(
            CaseRecord.case_type == case_type,
            CaseRecord.jurisdiction == jurisdiction,
        )
        
        if court_name:
            matching_ids_query = matching_ids_query.filter(CaseRecord.court_name == court_name)
        if judge_name:
            matching_ids_query = matching_ids_query.filter(CaseRecord.judge_name == judge_name)
            
        # Apply the workload limit
        subquery = matching_ids_query.order_by(CaseRecord.created_at.desc()).limit(similar_cases_limit).subquery()
        
        # Aggregate over this limited subset
        query = db.query(
            func.count(CaseRecord.id).label('total'),
            func.sum(sql_case(((CaseOutcome.appeal_filed == True) & (CaseOutcome.appeal_success == True), 1), else_=0)).label('appeal_wins'),
            func.sum(sql_case((CaseOutcome.appeal_filed == True, 1), else_=0)).label('appeals')
        ).join(CaseOutcome, CaseRecord.id == CaseOutcome.case_id, isouter=True).filter(
            CaseRecord.id.in_(subquery)
        )
        
        stats = query.first()
        total_similar = stats.total or 0
        
        if total_similar == 0:
            return {
                "estimated_success_rate": None,
                "confidence": "very_low",
                "similar_cases_found": 0,
                "reasoning": f"No similar cases found in {jurisdiction} for {case_type} cases.",
            }
            
        appeals = stats.appeals or 0
        appeal_wins = stats.appeal_wins or 0
        appeal_success_rate = (appeal_wins / appeals * 100) if appeals > 0 else 0.0
        
        # Adjust based on outcome magnitude
        adjustment = {"low": 0.95, "moderate": 1.0, "high": 1.05}.get(outcome_magnitude.lower(), 1.0)
        adjusted_rate = min(100, max(0, appeal_success_rate * adjustment))
        
        # Confidence logic
        if total_similar >= 50: confidence = "high"
        elif total_similar >= 20: confidence = "medium"
        elif total_similar >= 10: confidence = "low"
        else: confidence = "very_low"
        
        reasoning = f"Based on {total_similar} similar {case_type} cases in {jurisdiction}. "
        reasoning += f"Appeal success rate in similar cases: {appeal_success_rate:.1f}%"
        
        return {
            "estimated_success_rate": round(adjusted_rate, 1),
            "confidence": confidence,
            "similar_cases_found": total_similar,
            "appeal_success_rate_from_similar": round(appeal_success_rate, 1),
            "reasoning": reasoning,
        }

    @staticmethod
    def estimate_appeal_cost_and_time(
        db: Session,
        case_type: str,
        jurisdiction: str,
    ) -> Dict:
        """Estimate typical appeal cost and time using aggregates"""
        # Fetch only necessary columns
        res = db.query(
            CaseOutcome.appeal_cost,
            CaseOutcome.time_to_appeal_verdict
        ).join(CaseRecord, CaseRecord.id == CaseOutcome.case_id).filter(
            CaseRecord.case_type == case_type,
            CaseRecord.jurisdiction == jurisdiction,
            CaseOutcome.appeal_filed == True
        ).all()
        
        if not res:
            # Fallback
            default_costs = {"civil": "₹12,000 - ₹25,000", "criminal": "₹5,000 - ₹15,000", "family": "₹8,000 - ₹20,000", "commercial": "₹20,000 - ₹50,000"}
            default_time = {"civil": "12-24 months", "criminal": "12-30 months", "family": "12-24 months", "commercial": "18-36 months"}
            return {
                "avg_cost": default_costs.get(case_type.lower(), "₹10,000 - ₹30,000"),
                "avg_time": default_time.get(case_type.lower(), "12-24 months"),
                "note": "Generic estimates - not based on local data",
            }
            
        costs = []
        times = []
        for row in res:
            if row.appeal_cost:
                import re
                numbers = re.findall(r'\d+', row.appeal_cost)
                if numbers: costs.append(int(numbers[0]))
            if row.time_to_appeal_verdict:
                times.append(row.time_to_appeal_verdict)
                
        avg_cost = sum(costs) / len(costs) if costs else None
        avg_time_days = sum(times) / len(times) if times else None
        
        cost_str = f"₹{int(avg_cost * 0.8):.0f} - ₹{int(avg_cost * 1.2):.0f}" if avg_cost else "Unknown"
        time_str = f"{int(avg_time_days/30 * 0.8)}-{int(avg_time_days/30 * 1.2)} months" if avg_time_days else "12-24 months"
        
        return {"avg_cost": cost_str, "avg_time": time_str, "similar_cases": len(res)}


class AnalyticsAggregator:
    """Generate aggregated analytics for dashboard using SQL aggregates"""
    
    @staticmethod
    def get_dashboard_summary(db: Session) -> Dict:
        """Get overall dashboard summary using aggregates"""
        stats = db.query(
            func.count(CaseRecord.id).label('total'),
            func.sum(sql_case((CaseOutcome.appeal_filed == True, 1), else_=0)).label('appeals'),
            func.sum(sql_case((CaseRecord.outcome.ilike('plaintiff_won'), 1), else_=0)).label('p_wins'),
            func.sum(sql_case((CaseRecord.outcome.ilike('defendant_won'), 1), else_=0)).label('d_wins'),
            func.sum(sql_case((CaseRecord.outcome.ilike('settlement'), 1), else_=0)).label('settlements'),
            func.sum(sql_case((CaseRecord.outcome.ilike('dismissal'), 1), else_=0)).label('dismissals')
        ).join(CaseOutcome, CaseRecord.id == CaseOutcome.case_id, isouter=True).first()
        
        total = stats.total or 0
        appeals = stats.appeals or 0
        
        return {
            "total_cases_processed": total,
            "appeals_filed": appeals,
            "appeal_rate_percent": round((appeals / total * 100), 1) if total > 0 else 0,
            "plaintiff_wins": stats.p_wins or 0,
            "defendant_wins": stats.d_wins or 0,
            "settlements": stats.settlements or 0,
            "dismissals": stats.dismissals or 0,
        }
    
    @staticmethod
    def get_top_judges(db: Session, jurisdiction: str, limit: int = 10) -> List[Dict]:
        """Get top judges by appeal success rate using grouped aggregates"""
        judge_stats = db.query(
            CaseRecord.judge_name,
            func.count(CaseRecord.id).label('total'),
            func.sum(sql_case((CaseRecord.outcome.ilike('plaintiff_won'), 1), else_=0)).label('wins'),
            func.sum(sql_case(((CaseOutcome.appeal_filed == True) & (CaseOutcome.appeal_success == True), 1), else_=0)).label('appeal_wins'),
            func.sum(sql_case((CaseOutcome.appeal_filed == True, 1), else_=0)).label('appeals')
        ).join(CaseOutcome, CaseRecord.id == CaseOutcome.case_id, isouter=True).filter(
            CaseRecord.jurisdiction == jurisdiction,
            CaseRecord.judge_name != None
        ).group_by(CaseRecord.judge_name).having(func.count(CaseRecord.id) >= 5).all()
        
        results = []
        for row in judge_stats:
            results.append({
                "judge": row.judge_name,
                "jurisdiction": jurisdiction,
                "total_cases": row.total,
                "win_rate": round((row.wins / row.total * 100), 1),
                "appeal_success_rate": round((row.appeal_wins / row.appeals * 100), 1) if row.appeals > 0 else 0.0
            })
            
        results.sort(key=lambda x: x["appeal_success_rate"], reverse=True)
        return results[:limit]
    
    @staticmethod
    def get_regional_trends(db: Session) -> List[Dict]:
        """Get trends by jurisdiction using grouped aggregates"""
        trends_stats = db.query(
            CaseRecord.jurisdiction,
            func.count(CaseRecord.id).label('total'),
            func.sum(sql_case(((CaseOutcome.appeal_filed == True) & (CaseOutcome.appeal_success == True), 1), else_=0)).label('appeal_wins'),
            func.sum(sql_case((CaseOutcome.appeal_filed == True, 1), else_=0)).label('appeals')
        ).join(CaseOutcome, CaseRecord.id == CaseOutcome.case_id, isouter=True).filter(
            CaseRecord.jurisdiction != None
        ).group_by(CaseRecord.jurisdiction).all()
        
        results = []
        for row in trends_stats:
            results.append({
                "jurisdiction": row.jurisdiction,
                "total_cases": row.total,
                "appeal_success_rate": round((row.appeal_wins / row.appeals * 100), 1) if row.appeals > 0 else 0.0
            })
            
        results.sort(key=lambda x: x["total_cases"], reverse=True)
        return results


# Utility function to anonymize case ID
def generate_anonymous_case_id(case_data: str) -> str:
    """Generate an anonymous case ID from case data using HMAC-SHA256.

    Raw SHA-256 without a secret key is deterministic and vulnerable to
    precomputation and correlation attacks.  HMAC-SHA256 binds the output
    to a server-side secret so identical inputs produce unpredictable
    identifiers across environments.

    The secret is read from the CASE_ANONYMIZATION_SECRET environment
    variable (same source used by case_manager._get_case_anonymization_secret).
    Raises RuntimeError if the secret is not configured, consistent with the
    project-wide policy of failing loudly rather than silently degrading
    anonymization strength.
    """
    secret = os.getenv("CASE_ANONYMIZATION_SECRET", "").strip()
    if not secret:
        raise RuntimeError(
            "CASE_ANONYMIZATION_SECRET is not configured. "
            "Set this environment variable to a strong random value before "
            "generating anonymous case identifiers."
        )
    return hmac.new(
        secret.encode("utf-8"),
        case_data.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:16]
