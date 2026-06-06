from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone, timedelta
from sqlalchemy import func, case as sql_case
from sqlalchemy.orm import Session, joinedload
from database import CaseRecord, CaseOutcome, CaseAnalytics, UserFeedback, SimilarityFeedback, Case
import hashlib
import hmac
import os
import re
from collections import Counter
import logging
import pandas as pd
import numpy as np
import gc
import sys
import time
from sqlalchemy.orm import Query

logger = logging.getLogger(__name__)


# ── Helper functions for fuzzy / search-based text comparison ──────────────

def _normalize_text(text: Optional[str]) -> str:
    """Normalize text by lowercasing and stripping whitespace."""
    if not text:
        return ""
    return text.strip().lower()


def _summary_overlap(text1: Optional[str], text2: Optional[str]) -> float:
    """Token-level Jaccard similarity between two free-form text fields.

    Treats each whitespace-separated token as a feature and returns
    the fraction of shared tokens.  This replaces exact string equality
    with search-based retrieval for free-form legal arguments.
    """
    if not text1 or not text2:
        return 0.0
    tokens1 = set(_normalize_text(text1).split())
    tokens2 = set(_normalize_text(text2).split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)


_cost_range_re = re.compile(r"(\d[\d,.]*)")


def _parse_cost_value(cost_str: Optional[str]) -> Optional[float]:
    """Extract a numeric cost value from a free-form cost string."""
    if not cost_str:
        return None
    match = _cost_range_re.search(cost_str)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except (ValueError, TypeError):
        return None


def _confidence_from_samples(sample_count: int, min_samples: int = 5) -> float:
    """Return a confidence score (0..1) based on sample size.

    Fewer than *min_samples* yields low confidence; 50+ samples
    approaches 1.0.  Mirrors the function referenced (but never
    defined) across the analytics engine.
    """
    if sample_count >= 50:
        return 1.0
    if sample_count < min_samples:
        return 0.0
    # Smooth ramp from 0 → 1 between min_samples and 50
    return (sample_count - min_samples) / (50 - min_samples)


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

        # This method is suitable for small lists (< 5000 cases)
        data_list = []
        for case in cases:
            row = PandasAnalyticsProcessor._extract_case_row(case)
            data_list.append(row)

        return pd.DataFrame(data_list)

    @staticmethod
    def convert_query_to_dataframe_batched(
        db: Session,
        query: Query, 
        batch_size: int = 2000
    ) -> pd.DataFrame:
        """
        Memory-efficient conversion of a large query into a DataFrame.

        Uses LIMIT/OFFSET pagination and periodic session clearing to handle 
        10,000+ records without leaking memory or causing OOM crashes.
        
        Args:
            db: The active database session.
            query: The SQLAlchemy query object to execute.
            batch_size: Number of records to process per memory cycle.
            
        Returns:
            A single concatenated pd.DataFrame containing all results.
        """
        all_chunks = []
        current_chunk_data = []
        total_processed = 0

        logger.info(f"Converting large query to DataFrame using batch size {batch_size}")
        
        # Use LIMIT/OFFSET pagination for SQLite compatibility.
        offset = 0
        while True:
            batch = query.limit(batch_size).offset(offset).all()
            if not batch:
                break

            for case in batch:
                # Flatten DB object into a simple dict immediately
                current_chunk_data.append(PandasAnalyticsProcessor._extract_case_row(case))
                total_processed += 1

            if total_processed % batch_size == 0 and current_chunk_data:
                chunk_df = pd.DataFrame(current_chunk_data)
                all_chunks.append(chunk_df)
                current_chunk_data = []
                db.expunge_all()
                gc.collect()
                logger.debug(f"Memory optimization checkpoint: {total_processed} records batched.")

            offset += batch_size

        # Handle the final partial batch
        if current_chunk_data:
            all_chunks.append(pd.DataFrame(current_chunk_data))

        if not all_chunks:
            return pd.DataFrame()

        # Concatenate all chunks into the final result set
        final_df = pd.concat(all_chunks, ignore_index=True)
        
        # Release the chunk list as early as possible
        del all_chunks
        gc.collect()
        
        logger.info(f"Batched conversion complete. Final DataFrame size: {len(final_df)} rows.")
        return final_df

    @staticmethod
    def _extract_case_row(case: CaseRecord) -> Dict:
        """
        Internal utility to transform a CaseRecord into a flat dictionary.
        
        Decouples the analytical data from the SQLAlchemy ORM layer.
        """
        row = {
            "id": case.id,
            "case_type": str(case.case_type).lower() if case.case_type else "unknown",
            "jurisdiction": str(case.jurisdiction) if case.jurisdiction else "Unknown",
            "court_name": str(case.court_name) if case.court_name else "N/A",
            "judge_name": str(case.judge_name) if case.judge_name else "N/A",
            "outcome": str(case.outcome).lower() if case.outcome else "pending",
            "created_at": case.created_at,
        }
        
        # Safely handle the one-to-one relationship with CaseOutcome
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
        return row

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
    """Calculate similarity between cases for matching and analysis.

    The calculator scores shared case attributes such as type, jurisdiction,
    party roles, case value, and judgment-summary overlap to support ranked
    similar-case retrieval.
    """
    
    @staticmethod
    def case_similarity_score(
        case1: CaseRecord,
        case2: CaseRecord,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Calculate a weighted similarity score between two cases.

        The score is normalized to a 0-100 range and combines exact matches on
        case type, jurisdiction, plaintiff type, defendant type, case value,
        and optional judgment-summary overlap.

        Args:
            case1: The reference case to compare.
            case2: The candidate case to score against the reference.
            weights: Optional field weights keyed by attribute name. When not
                provided, a default weight distribution is used.

        Returns:
            A similarity score between 0.0 and 100.0.
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

        # Judgment summary overlap helps surface cases with similar reasoning.
        if case1.judgment_summary and case2.judgment_summary:
            summary_overlap = _summary_overlap(case1.judgment_summary, case2.judgment_summary)
            score += weights.get("judgment_summary", 0.0) * summary_overlap
        
        return score * 100
    
    @staticmethod
    def find_similar_cases(
        db: Session,
        reference_case: CaseRecord,
        min_similarity: float = 50.0,
        limit: int = 50,
        user_id: Optional[str] = None,
    ) -> List[Tuple[CaseRecord, float]]:
        """Find cases similar to a reference case using DB pre-filtering.

        The query excludes the reference case itself and narrows candidates to
        cases that share either the same case type or jurisdiction before
        applying the in-memory similarity score.

        Ownership scope: when user_id is provided, candidates are restricted to
        case types and jurisdictions the user owns (from the Case table).
        """
        query = db.query(CaseRecord).filter(
            CaseRecord.id != reference_case.id,
            (CaseRecord.case_type == reference_case.case_type) | (CaseRecord.jurisdiction == reference_case.jurisdiction)
        )

        # Scope candidates to user's case types / jurisdictions
        if user_id is not None:
            user_scope = (
                db.query(Case.case_type, Case.jurisdiction)
                .filter(Case.user_id == int(user_id))
                .distinct()
                .all()
            )
            if user_scope:
                query = query.filter(
                    CaseRecord.case_type.in_([row.case_type for row in user_scope]),
                    CaseRecord.jurisdiction.in_([row.jurisdiction for row in user_scope]),
                )
        
        # Limit the search space to prevent OOM. Configurable via SIMILARITY_QUERY_LIMIT env var.
        # Default 1000 for memory efficiency, increase for larger jurisdictions.
        max_cases = int(os.getenv("SIMILARITY_QUERY_LIMIT", "1000"))
        all_cases = query.limit(max_cases).all()
        
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
        prefetched_feedback: Optional[List[SimilarityFeedback]] = None,
    ) -> float:
        """Return a small ranking adjustment from similarity feedback.

        Historical relevance feedback is aggregated for the candidate case and
        converted into a bounded ranking delta so prior user judgments can
        nudge future similarity ordering without dominating the base score.

        Args:
            db: Active SQLAlchemy session.
            candidate_case: Case being ranked.
            user_id: Optional user scope for feedback rows.
            query_signature: Optional query signature scope for feedback rows.
            prefetched_feedback: Optional pre-fetched feedback list for this candidate.

        Returns:
            A bounded adjustment in the range ``[-0.03, 0.03]``.
        """
        if prefetched_feedback is not None:
            feedback_rows = prefetched_feedback
            if user_id is not None:
                feedback_rows = [row for row in feedback_rows if row.user_id == str(user_id)]
            if query_signature is not None:
                feedback_rows = [row for row in feedback_rows if row.query_signature == query_signature]
            feedback_rows = feedback_rows[:50]
        else:
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
            func.sum(sql_case((CaseRecord.outcome == 'plaintiff_won', 1), else_=0)).label('p_wins'),
            func.sum(sql_case((CaseRecord.outcome == 'defendant_won', 1), else_=0)).label('d_wins'),
            func.sum(sql_case((CaseRecord.outcome == 'settlement', 1), else_=0)).label('settlements'),
            func.sum(sql_case((CaseRecord.outcome == 'dismissal', 1), else_=0)).label('dismissals'),
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
            func.sum(sql_case((CaseRecord.outcome == 'plaintiff_won', 1), else_=0)).label('wins')
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
    def calculate_appeal_success_rate(cases: list) -> float:
        """
        Calculate the aggregate appeal success rate from a list of CaseRecord objects.
        Cases without outcome_data or with appeal_filed=False are safely ignored.
        Returns a percentage (0.0–100.0), or 0.0 if no qualifying cases.
        """
        if not cases:
            return 0.0
        
        appeals_filed = 0
        appeals_won = 0
        for case in cases:
            if case is None:
                continue
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


class PredictiveAnalyticsEngine:
    """Generate actionable predictions from historical case data."""

    DEFAULT_APPEAL_WINDOW_DAYS = {
        "civil": 30,
        "criminal": 30,
        "family": 45,
        "commercial": 45,
        "labor": 30,
        "general": 30,
    }

    DEFAULT_COST_RANGE = {
        "civil": (12000, 25000),
        "criminal": (5000, 15000),
        "family": (8000, 20000),
        "commercial": (20000, 50000),
        "labor": (7000, 18000),
        "general": (10000, 30000),
    }

    DEFAULT_TIMELINE_DAYS = {
        "civil": 180,
        "criminal": 120,
        "family": 150,
        "commercial": 240,
        "labor": 120,
        "general": 180,
    }

    @staticmethod
    def _normalized_case_type(case_type: Optional[str]) -> str:
        normalized = _normalize_text(case_type)
        return normalized if normalized else "general"

    @staticmethod
    def _candidate_query(
        db: Session,
        case_type: Optional[str],
        jurisdiction: Optional[str],
        court_name: Optional[str] = None,
        judge_name: Optional[str] = None,
        plaintiff_type: Optional[str] = None,
        defendant_type: Optional[str] = None,
        limit: int = 1000,
        streaming_mode: bool = False
    ):
        """
        Internal utility to build a filtered query for candidate cases.
        
        Refactored to support either direct result fetching or returning 
        the Query object for memory-efficient batch processing (streaming_mode).
        """
        # We always joinedload outcome_data as it's frequently accessed in predictions
        query = db.query(CaseRecord).options(joinedload(CaseRecord.outcome_data))

        normalized_case_type = _normalize_text(case_type)
        if normalized_case_type and normalized_case_type != "general":
            query = query.filter(func.lower(CaseRecord.case_type) == normalized_case_type)

        # Apply specific filters if provided
        if jurisdiction:
            query = query.filter(func.lower(CaseRecord.jurisdiction) == _normalize_text(jurisdiction))
        if court_name:
            query = query.filter(func.lower(CaseRecord.court_name) == _normalize_text(court_name))
        if judge_name:
            query = query.filter(func.lower(CaseRecord.judge_name) == _normalize_text(judge_name))
        if plaintiff_type:
            query = query.filter(func.lower(CaseRecord.plaintiff_type) == _normalize_text(plaintiff_type))
        if defendant_type:
            query = query.filter(func.lower(CaseRecord.defendant_type) == _normalize_text(defendant_type))

        ordered_query = query.order_by(CaseRecord.created_at.desc())
        
        if streaming_mode:
            # Return the Query object so caller can use yield_per()
            return ordered_query
        
        # Return a limited list for immediate consumption
        return ordered_query.limit(limit).all()

    @staticmethod
    def _score_case_profile(
        reference_case: CaseRecord,
        candidate_case: CaseRecord,
        case_summary: Optional[str] = None,
    ) -> float:
        """Score a candidate case against a reference profile.

        The method combines the core case similarity score with an additional
        judgment-summary overlap bonus when a summary is available.

        Args:
            reference_case: Synthetic or persisted case used as the anchor.
            candidate_case: Case being evaluated.
            case_summary: Optional summary text to use instead of the
                reference case summary.

        Returns:
            A similarity score capped at 100.0.
        """
        base_score = CaseSimilarityCalculator.case_similarity_score(reference_case, candidate_case)
        summary_source = case_summary or reference_case.judgment_summary
        summary_bonus = 0.0
        if summary_source:
            summary_bonus = _summary_overlap(summary_source, candidate_case.judgment_summary) * 15.0
        return min(100.0, base_score + summary_bonus)

    @staticmethod
    def find_similar_cases_for_profile(
        db: Session,
        case_type: str,
        jurisdiction: str,
        court_name: Optional[str] = None,
        judge_name: Optional[str] = None,
        plaintiff_type: Optional[str] = None,
        defendant_type: Optional[str] = None,
        case_value: Optional[str] = None,
        case_summary: Optional[str] = None,
        min_similarity: float = 50.0,
        limit: int = 10,
    ) -> List[Tuple[CaseRecord, float]]:
        """Find similar cases for a user-provided case profile.

        Args:
            db: Active SQLAlchemy session.
            case_type: Case type for the synthetic reference profile.
            jurisdiction: Jurisdiction for the synthetic reference profile.
            court_name: Optional court name for filtering and scoring.
            judge_name: Optional judge name for filtering and scoring.
            plaintiff_type: Optional plaintiff type for filtering and scoring.
            defendant_type: Optional defendant type for filtering and scoring.
            case_value: Optional case value for the synthetic profile.
            case_summary: Optional narrative summary used in scoring.
            min_similarity: Minimum score required for inclusion in results.
            limit: Maximum number of results to return.

        Returns:
            A list of ``(CaseRecord, score)`` tuples sorted by score descending.
        """
        reference_case = CaseRecord(
            hashed_case_id="prediction-profile",
            case_type=case_type,
            jurisdiction=jurisdiction,
            court_name=court_name,
            judge_name=judge_name,
            plaintiff_type=plaintiff_type,
            defendant_type=defendant_type,
            case_value=case_value,
            outcome="pending",
            judgment_summary=case_summary,
        )

        candidates = PredictiveAnalyticsEngine._candidate_query(
            db,
            case_type=case_type,
            jurisdiction=jurisdiction,
            court_name=court_name,
            judge_name=judge_name,
            plaintiff_type=plaintiff_type,
            defendant_type=defendant_type,
        )

        scored: List[Tuple[CaseRecord, float]] = []
        for candidate in candidates:
            score = PredictiveAnalyticsEngine._score_case_profile(reference_case, candidate, case_summary)
            if score >= min_similarity:
                scored.append((candidate, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    @staticmethod
    def _scope_metrics(cases: List[CaseRecord]) -> Dict[str, Optional[float]]:
        total_cases = len(cases)
        appealed_cases = 0
        appeal_successes = 0
        plaintiff_wins = 0
        defendant_wins = 0
        cost_values: List[float] = []
        duration_values: List[int] = []

        for case in cases:
            if _normalize_text(case.outcome) == "plaintiff_won":
                plaintiff_wins += 1
            elif _normalize_text(case.outcome) == "defendant_won":
                defendant_wins += 1

            outcome = case.outcome_data
            if not outcome or not outcome.appeal_filed:
                continue

            appealed_cases += 1
            if outcome.appeal_success:
                appeal_successes += 1

            parsed_cost = _parse_cost_value(outcome.appeal_cost)
            if parsed_cost is not None:
                cost_values.append(parsed_cost)
            if outcome.time_to_appeal_verdict:
                duration_values.append(outcome.time_to_appeal_verdict)

        appeal_success_rate = None
        if appealed_cases:
            appeal_success_rate = round((appeal_successes / appealed_cases) * 100, 1)

        plaintiff_win_rate = round((plaintiff_wins / total_cases) * 100, 1) if total_cases else 0.0
        avg_cost = round(sum(cost_values) / len(cost_values), 0) if cost_values else None
        avg_duration = round(sum(duration_values) / len(duration_values), 0) if duration_values else None

        return {
            "total_cases": total_cases,
            "appealed_cases": appealed_cases,
            "appeal_success_rate": appeal_success_rate,
            "plaintiff_win_rate": plaintiff_win_rate,
            "avg_cost": avg_cost,
            "avg_duration": avg_duration,
        }

    @staticmethod
    def _scope_rate(
        db: Session,
        case_type: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        court_name: Optional[str] = None,
        judge_name: Optional[str] = None,
        plaintiff_type: Optional[str] = None,
        defendant_type: Optional[str] = None,
    ) -> Dict[str, Optional[float]]:
        cases = PredictiveAnalyticsEngine._candidate_query(
            db,
            case_type=case_type,
            jurisdiction=jurisdiction,
            court_name=court_name,
            judge_name=judge_name,
            plaintiff_type=plaintiff_type,
            defendant_type=defendant_type,
        )
        return PredictiveAnalyticsEngine._scope_metrics(cases)

    @staticmethod
    def predict_appeal_success(
        db: Session,
        case_type: str,
        jurisdiction: str,
        court_name: Optional[str] = None,
        judge_name: Optional[str] = None,
        plaintiff_type: Optional[str] = None,
        defendant_type: Optional[str] = None,
        case_value: Optional[str] = None,
        case_summary: Optional[str] = None,
        similar_cases_limit: int = 10,
    ) -> Dict:
        similar_cases = PredictiveAnalyticsEngine.find_similar_cases_for_profile(
            db,
            case_type=case_type,
            jurisdiction=jurisdiction,
            court_name=court_name,
            judge_name=judge_name,
            plaintiff_type=plaintiff_type,
            defendant_type=defendant_type,
            case_value=case_value,
            case_summary=case_summary,
            min_similarity=45.0,
            limit=similar_cases_limit,
        )

        similar_with_appeals = [case for case, _ in similar_cases if case.outcome_data and case.outcome_data.appeal_filed]
        similar_stats = PredictiveAnalyticsEngine._scope_metrics(similar_with_appeals)

        judge_stats = PredictiveAnalyticsEngine._scope_rate(
            db,
            case_type=case_type,
            jurisdiction=jurisdiction,
            court_name=court_name,
            judge_name=judge_name,
            plaintiff_type=plaintiff_type,
            defendant_type=defendant_type,
        ) if judge_name else {"total_cases": 0, "appeal_success_rate": None}

        court_stats = PredictiveAnalyticsEngine._scope_rate(
            db,
            case_type=case_type,
            jurisdiction=jurisdiction,
            court_name=court_name,
            plaintiff_type=plaintiff_type,
            defendant_type=defendant_type,
        ) if court_name else {"total_cases": 0, "appeal_success_rate": None}

        jurisdiction_stats = PredictiveAnalyticsEngine._scope_rate(
            db,
            case_type=case_type,
            jurisdiction=jurisdiction,
            plaintiff_type=plaintiff_type,
            defendant_type=defendant_type,
        )

        global_case_stats = PredictiveAnalyticsEngine._scope_rate(
            db,
            case_type=case_type,
            plaintiff_type=plaintiff_type,
            defendant_type=defendant_type,
        )

        components: List[Tuple[float, float, int, str]] = []

        if similar_stats["appeal_success_rate"] is not None:
            components.append((similar_stats["appeal_success_rate"], 0.45, int(similar_stats["appealed_cases"] or 0), "similar_cases"))
        if judge_stats.get("appeal_success_rate") is not None:
            components.append((judge_stats["appeal_success_rate"], 0.2, int(judge_stats.get("appealed_cases") or 0), "judge_history"))
        if court_stats.get("appeal_success_rate") is not None:
            components.append((court_stats["appeal_success_rate"], 0.15, int(court_stats.get("appealed_cases") or 0), "court_history"))
        if jurisdiction_stats.get("appeal_success_rate") is not None:
            components.append((jurisdiction_stats["appeal_success_rate"], 0.12, int(jurisdiction_stats.get("appealed_cases") or 0), "jurisdiction_history"))
        if global_case_stats.get("appeal_success_rate") is not None:
            components.append((global_case_stats["appeal_success_rate"], 0.08, int(global_case_stats.get("appealed_cases") or 0), "case_type_history"))

        if not components:
            predicted_rate = 50.0
            dominant_source = "fallback"
        else:
            weighted_total = 0.0
            weight_sum = 0.0
            dominant_source = components[0][3]

            for rate, weight, sample_count, source_name in components:
                adjusted_weight = weight * min(1.0, max(sample_count, 1) / 15.0)
                weighted_total += rate * adjusted_weight
                weight_sum += adjusted_weight
                if adjusted_weight > 0 and source_name == "similar_cases":
                    dominant_source = source_name

            predicted_rate = round(weighted_total / weight_sum, 1) if weight_sum else 50.0

        sample_count = sum(
            int(scope.get("appealed_cases") or 0)
            for scope in [similar_stats, judge_stats, court_stats, jurisdiction_stats, global_case_stats]
        )

        confidence = _confidence_from_samples(sample_count)

        reasoning_parts = []
        if similar_stats["appealed_cases"]:
            reasoning_parts.append(
                f"Similar cases: {similar_stats['appeal_success_rate']:.1f}% success across {similar_stats['appealed_cases']} appealed cases"
            )
        if judge_stats.get("appealed_cases"):
            reasoning_parts.append(
                f"Judge history: {judge_stats['appeal_success_rate']:.1f}% success across {int(judge_stats['appealed_cases'])} appealed cases"
            )
        if court_stats.get("appealed_cases"):
            reasoning_parts.append(
                f"Court history: {court_stats['appeal_success_rate']:.1f}% success across {int(court_stats['appealed_cases'])} appealed cases"
            )
        if not reasoning_parts:
            reasoning_parts.append("Historical appeal data is sparse, so the estimate falls back to case-type baselines.")

        return {
            "predicted_success_rate": predicted_rate,
            "confidence": confidence,
            "source": dominant_source,
            "sample_count": sample_count,
            "similar_cases_found": len(similar_cases),
            "similar_cases_success_rate": similar_stats["appeal_success_rate"],
            "judge_success_rate": judge_stats.get("appeal_success_rate"),
            "court_success_rate": court_stats.get("appeal_success_rate"),
            "jurisdiction_success_rate": jurisdiction_stats.get("appeal_success_rate"),
            "reasoning": "; ".join(reasoning_parts),
            "similar_cases": [
                {
                    "case_id": case.id,
                    "case_number": case.hashed_case_id,
                    "title": case.judge_name or case.court_name or "Precedent",
                    "jurisdiction": case.jurisdiction,
                    "case_type": case.case_type,
                    "relevance_score": round(score / 100.0, 4),
                    "appeal_filed": bool(case.outcome_data.appeal_filed) if case.outcome_data else False,
                    "appeal_success": case.outcome_data.appeal_success if case.outcome_data else None,
                }
                for case, score in similar_cases
            ],
        }

    @staticmethod
    def estimate_judgment_timeline(
        db: Session,
        case_type: str,
        jurisdiction: str,
        court_name: Optional[str] = None,
        judge_name: Optional[str] = None,
        plaintiff_type: Optional[str] = None,
        defendant_type: Optional[str] = None,
        case_summary: Optional[str] = None,
    ) -> Dict:
        similar_cases = PredictiveAnalyticsEngine.find_similar_cases_for_profile(
            db,
            case_type=case_type,
            jurisdiction=jurisdiction,
            court_name=court_name,
            judge_name=judge_name,
            plaintiff_type=plaintiff_type,
            defendant_type=defendant_type,
            case_summary=case_summary,
            min_similarity=40.0,
            limit=20,
        )

        durations = []
        for case, _ in similar_cases:
            outcome = case.outcome_data
            if not outcome or not outcome.appeal_filed:
                continue
            if outcome.time_to_appeal_verdict:
                durations.append(outcome.time_to_appeal_verdict)

        average_duration = round(sum(durations) / len(durations), 0) if durations else None
        if average_duration is None:
            average_duration = float(PredictiveAnalyticsEngine.DEFAULT_TIMELINE_DAYS.get(PredictiveAnalyticsEngine._normalized_case_type(case_type), 180))

        appeal_window_days = PredictiveAnalyticsEngine.DEFAULT_APPEAL_WINDOW_DAYS.get(
            PredictiveAnalyticsEngine._normalized_case_type(case_type),
            30,
        )

        filing_stage = max(7, int(round(average_duration * 0.15)))
        admission_stage = max(10, int(round(average_duration * 0.25)))
        hearing_stage = max(14, int(round(average_duration * 0.35)))
        decision_stage = max(7, int(round(max(average_duration - (filing_stage + admission_stage + hearing_stage), 0))))
        estimated_total_days = filing_stage + admission_stage + hearing_stage + decision_stage

        if estimated_total_days >= appeal_window_days:
            deadline_risk = "high"
        elif estimated_total_days >= appeal_window_days * 0.8:
            deadline_risk = "medium"
        elif estimated_total_days >= appeal_window_days * 0.6:
            deadline_risk = "low"
        else:
            deadline_risk = "very_low"

        return {
            "estimated_total_days": int(estimated_total_days),
            "average_duration_days": int(round(average_duration, 0)),
            "deadline_window_days": appeal_window_days,
            "deadline_risk": deadline_risk,
            "stages": {
                "filing_preparation_days": filing_stage,
                "admission_days": admission_stage,
                "hearing_days": hearing_stage,
                "decision_days": decision_stage,
            },
            "sample_count": len(durations),
            "confidence": _confidence_from_samples(len(durations)),
            "reasoning": (
                f"Estimated from {len(durations)} similar appeal timelines. "
                if durations
                else "No matching duration data found, so this uses case-type baselines."
            ),
        }

    @staticmethod
    def predict_cost(
        db: Session,
        case_type: str,
        jurisdiction: str,
        court_name: Optional[str] = None,
        judge_name: Optional[str] = None,
        plaintiff_type: Optional[str] = None,
        defendant_type: Optional[str] = None,
        case_summary: Optional[str] = None,
    ) -> Dict:
        similar_cases = PredictiveAnalyticsEngine.find_similar_cases_for_profile(
            db,
            case_type=case_type,
            jurisdiction=jurisdiction,
            court_name=court_name,
            judge_name=judge_name,
            plaintiff_type=plaintiff_type,
            defendant_type=defendant_type,
            case_summary=case_summary,
            min_similarity=40.0,
            limit=20,
        )

        costs: List[float] = []
        for case, _ in similar_cases:
            outcome = case.outcome_data
            if not outcome or not outcome.appeal_filed:
                continue
            parsed_cost = _parse_cost_value(outcome.appeal_cost)
            if parsed_cost is not None:
                costs.append(parsed_cost)

        if costs:
            average_cost = sum(costs) / len(costs)
            lower = int(round(average_cost * 0.8, 0))
            upper = int(round(average_cost * 1.2, 0))
            source = "historical_cases"
        else:
            default_low, default_high = PredictiveAnalyticsEngine.DEFAULT_COST_RANGE.get(
                PredictiveAnalyticsEngine._normalized_case_type(case_type),
                PredictiveAnalyticsEngine.DEFAULT_COST_RANGE["general"],
            )
            lower = default_low
            upper = default_high
            average_cost = (lower + upper) / 2
            source = "case_type_baseline"

        return {
            "estimated_cost_range": f"₹{lower:,.0f} - ₹{upper:,.0f}",
            "average_cost": int(round(average_cost, 0)),
            "sample_count": len(costs),
            "confidence": _confidence_from_samples(len(costs)),
            "source": source,
            "reasoning": (
                f"Based on {len(costs)} similar appeal cost records."
                if costs
                else "No local cost records were found, so this uses a case-type baseline."
            ),
        }

    @staticmethod
    def recommend_judge_and_court(
        db: Session,
        case_type: str,
        jurisdiction: str,
        limit: int = 5,
    ) -> Dict:
        judge_rows = db.query(
            CaseRecord.judge_name,
            func.count(CaseRecord.id).label("total_cases"),
            func.sum(sql_case((CaseRecord.outcome == "plaintiff_won", 1), else_=0)).label("plaintiff_wins"),
            func.sum(sql_case(((CaseOutcome.appeal_filed == True) & (CaseOutcome.appeal_success == True), 1), else_=0)).label("appeal_successes"),
            func.sum(sql_case((CaseOutcome.appeal_filed == True, 1), else_=0)).label("appeals"),
        ).join(CaseOutcome, CaseRecord.id == CaseOutcome.case_id, isouter=True).filter(
            func.lower(CaseRecord.case_type) == _normalize_text(case_type),
            func.lower(CaseRecord.jurisdiction) == _normalize_text(jurisdiction),
            CaseRecord.judge_name.isnot(None),
        ).group_by(CaseRecord.judge_name).having(func.count(CaseRecord.id) >= 3).all()

        court_rows = db.query(
            CaseRecord.court_name,
            func.count(CaseRecord.id).label("total_cases"),
            func.sum(sql_case((CaseRecord.outcome == "plaintiff_won", 1), else_=0)).label("plaintiff_wins"),
            func.sum(sql_case(((CaseOutcome.appeal_filed == True) & (CaseOutcome.appeal_success == True), 1), else_=0)).label("appeal_successes"),
            func.sum(sql_case((CaseOutcome.appeal_filed == True, 1), else_=0)).label("appeals"),
        ).join(CaseOutcome, CaseRecord.id == CaseOutcome.case_id, isouter=True).filter(
            func.lower(CaseRecord.case_type) == _normalize_text(case_type),
            func.lower(CaseRecord.jurisdiction) == _normalize_text(jurisdiction),
            CaseRecord.court_name.isnot(None),
        ).group_by(CaseRecord.court_name).having(func.count(CaseRecord.id) >= 3).all()

        def _build_rankings(rows, label_key: str) -> List[Dict]:
            rankings = []
            for row in rows:
                appeals = row.appeals or 0
                appeal_rate = round((row.appeal_successes / appeals) * 100, 1) if appeals else 0.0
                total_cases = row.total_cases or 0
                win_rate = round((row.plaintiff_wins / total_cases) * 100, 1) if total_cases else 0.0
                rankings.append(
                    {
                        label_key: getattr(row, label_key),
                        "total_cases": total_cases,
                        "win_rate": win_rate,
                        "appeal_success_rate": appeal_rate,
                        "appeals": appeals,
                    }
                )

            rankings.sort(key=lambda item: (item["appeal_success_rate"], item["total_cases"]), reverse=True)
            return rankings[:limit]

        judge_rankings = _build_rankings(judge_rows, "judge_name")
        court_rankings = _build_rankings(court_rows, "court_name")

        return {
            "top_judges": judge_rankings,
            "top_courts": court_rankings,
            "recommended_judge": judge_rankings[0]["judge_name"] if judge_rankings else None,
            "recommended_court": court_rankings[0]["court_name"] if court_rankings else None,
        }

    @staticmethod
    def build_case_prediction_pack(
        db: Session,
        case_type: str,
        jurisdiction: str,
        court_name: Optional[str] = None,
        judge_name: Optional[str] = None,
        plaintiff_type: Optional[str] = None,
        defendant_type: Optional[str] = None,
        case_value: Optional[str] = None,
        case_summary: Optional[str] = None,
    ) -> Dict:
        """
        Orchestrates the generation of a multi-faceted analytical prediction pack.
        
        This method combines success probability, timeline estimation, and cost 
        analysis. It handles large datasets by checking regional case volume 
        and triggering manual memory reclamation after heavy processing.
        """
        # Pre-check volume for memory safety optimization
        jurisdiction_normalized = _normalize_text(jurisdiction)
        vol_count = db.query(func.count(CaseRecord.id)).filter(
            func.lower(CaseRecord.jurisdiction) == jurisdiction_normalized
        ).scalar() or 0

        is_high_volume = vol_count > 5000
        if is_high_volume:
            logger.warning(f"High-volume dataset detected ({vol_count} records). Implementing memory guardrails.")

        # Step 1: Predict appeal success probability
        appeal_success = PredictiveAnalyticsEngine.predict_appeal_success(
            db,
            case_type=case_type,
            jurisdiction=jurisdiction,
            court_name=court_name,
            judge_name=judge_name,
            plaintiff_type=plaintiff_type,
            defendant_type=defendant_type,
            case_value=case_value,
            case_summary=case_summary,
        )
        
        # Step 2: Estimate judgment timelines
        timeline = PredictiveAnalyticsEngine.estimate_judgment_timeline(
            db,
            case_type=case_type,
            jurisdiction=jurisdiction,
            court_name=court_name,
            judge_name=judge_name,
            plaintiff_type=plaintiff_type,
            defendant_type=defendant_type,
            case_summary=case_summary,
        )
        
        # Step 3: Predict associated legal costs
        cost = PredictiveAnalyticsEngine.predict_cost(
            db,
            case_type=case_type,
            jurisdiction=jurisdiction,
            court_name=court_name,
            judge_name=judge_name,
            plaintiff_type=plaintiff_type,
            defendant_type=defendant_type,
            case_summary=case_summary,
        )
        
        # Step 4: Recommend judges and courts based on historical success
        recommendations = PredictiveAnalyticsEngine.recommend_judge_and_court(
            db,
            case_type=case_type,
            jurisdiction=jurisdiction,
        )

        # CRITICAL FIX: Reclaim memory after building large prediction packs
        # especially important in jurisdictions with thousands of precedents.
        gc.collect()

        return {
            "appeal_success": appeal_success,
            "timeline": timeline,
            "cost": cost,
            "recommendations": recommendations,
            "similar_cases": appeal_success["similar_cases"],
            "processing_info": {
                "memory_optimized": is_high_volume,
                "dataset_volume": vol_count,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }


class AnalyticsAggregator:
    """Generate aggregated analytics for dashboard using SQL aggregates"""
    
    @staticmethod
    def get_dashboard_summary(db: Session) -> Dict:
        """Get overall dashboard summary using aggregates"""
        stats = db.query(
            func.count(CaseRecord.id).label('total'),
            func.sum(sql_case((CaseOutcome.appeal_filed == True, 1), else_=0)).label('appeals'),
            func.sum(sql_case((CaseRecord.outcome == 'plaintiff_won', 1), else_=0)).label('p_wins'),
            func.sum(sql_case((CaseRecord.outcome == 'defendant_won', 1), else_=0)).label('d_wins'),
            func.sum(sql_case((CaseRecord.outcome == 'settlement', 1), else_=0)).label('settlements'),
            func.sum(sql_case((CaseRecord.outcome == 'dismissal', 1), else_=0)).label('dismissals')
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
            func.sum(sql_case((CaseRecord.outcome == 'plaintiff_won', 1), else_=0)).label('wins'),
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


class ReminderInsightsEngine:
    """Compute reminder effectiveness and drop-off metrics from deadlines and notification logs."""

    ELIGIBLE_STATUSES = {
        NotificationStatus.SENT,
        NotificationStatus.DELIVERED,
        NotificationStatus.OPENED,
    }

    @staticmethod
    def _completion_map(db: Session) -> Dict[int, datetime]:
        completion_map: Dict[int, datetime] = {}
        events = db.query(CaseTimeline).filter(CaseTimeline.event_type == "deadline_completed").all()
        for event in events:
            metadata = event.event_metadata if isinstance(event.event_metadata, dict) else {}
            deadline_id = metadata.get("deadline_id")
            try:
                deadline_id_int = int(deadline_id)
            except (TypeError, ValueError):
                continue

            completed_at = _utc_datetime(event.event_date)
            if completed_at is None:
                continue

            current = completion_map.get(deadline_id_int)
            if current is None or completed_at < current:
                completion_map[deadline_id_int] = completed_at

        return completion_map

    @staticmethod
    def build_attribution_frame(
        db: Session,
        attribution_window_days: int = 14,
        user_id: Optional[int] = None,
        jurisdiction: Optional[str] = None,
        court_name: Optional[str] = None,
        deadline_type: Optional[str] = None,
        channel: Optional[NotificationChannel | str] = None,
    ) -> pd.DataFrame:
        now = datetime.now(timezone.utc)
        completion_map = ReminderInsightsEngine._completion_map(db)

        query = db.query(NotificationLog, CaseDeadline, Case).join(
            CaseDeadline,
            NotificationLog.deadline_id == CaseDeadline.id,
        ).join(
            Case,
            CaseDeadline.case_id == Case.id,
        )

        if user_id is not None:
            query = query.filter(Case.user_id == user_id)

        if jurisdiction:
            query = query.filter(func.lower(Case.jurisdiction) == _normalize_text(jurisdiction))
        if court_name:
            query = query.filter(func.lower(func.coalesce(CaseDeadline.court_name, "")) == _normalize_text(court_name))
        if deadline_type:
            query = query.filter(func.lower(CaseDeadline.deadline_type) == _normalize_text(deadline_type))
        if channel:
            channel_enum = channel if isinstance(channel, NotificationChannel) else NotificationChannel(str(channel).strip().lower())
            query = query.filter(NotificationLog.channel == channel_enum)

        deadline_buckets: Dict[int, Dict[str, object]] = {}
        window = timedelta(days=max(1, int(attribution_window_days)))

        for log, deadline, case in query.all():
            if log.status not in ReminderInsightsEngine.ELIGIBLE_STATUSES:
                continue

            sent_at = _utc_datetime(log.sent_at or log.created_at)
            deadline_date = _utc_datetime(deadline.deadline_date)
            completion_at = completion_map.get(deadline.id)
            if sent_at is None or deadline_date is None:
                continue

            bucket = deadline_buckets.setdefault(
                deadline.id,
                {
                    "deadline": deadline,
                    "case": case,
                    "completion_at": completion_at,
                    "candidates": [],
                },
            )
            candidates = bucket["candidates"]
            if isinstance(candidates, list):
                candidates.append(
                    {
                        "notification_log_id": log.id,
                        "channel": log.channel.value if hasattr(log.channel, "value") else str(log.channel),
                        "sent_at": sent_at,
                    }
                )

        rows = []
        for deadline_id, bucket in deadline_buckets.items():
            deadline = bucket["deadline"]
            case = bucket["case"]
            completion_at = bucket["completion_at"]
            candidates = bucket["candidates"] if isinstance(bucket["candidates"], list) else []
            deadline_date = _utc_datetime(deadline.deadline_date)
            if deadline_date is None:
                continue

            cutoff = completion_at or min(deadline_date, now)
            eligible_candidates = [candidate for candidate in candidates if candidate["sent_at"] <= cutoff]
            if not eligible_candidates:
                continue

            chosen = max(eligible_candidates, key=lambda candidate: candidate["sent_at"])
            sent_at = chosen["sent_at"]

            matured = completion_at is not None or deadline_date <= now
            effective = bool(completion_at and sent_at <= completion_at <= sent_at + window)
            drop_off = bool(matured and not effective)
            if not (effective or drop_off):
                continue

            days_to_completion = None
            if effective and completion_at is not None:
                days_to_completion = round((completion_at - sent_at).total_seconds() / 86400.0, 2)

            rows.append(
                {
                    "deadline_id": deadline_id,
                    "notification_log_id": chosen["notification_log_id"],
                    "jurisdiction": _normalize_group_value(case.jurisdiction),
                    "court_name": _normalize_group_value(getattr(deadline, "court_name", None)),
                    "deadline_type": _normalize_group_value(deadline.deadline_type),
                    "channel": chosen["channel"],
                    "sent_at": sent_at,
                    "completion_at": completion_at,
                    "deadline_date": deadline_date,
                    "effective": effective,
                    "drop_off": drop_off,
                    "days_to_completion": days_to_completion,
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def summarize(frame: pd.DataFrame, attribution_window_days: int = 14) -> Dict[str, object]:
        if frame.empty:
            return {
                "attribution_window_days": attribution_window_days,
                "reminder_count": 0,
                "effective_reminders": 0,
                "drop_off_reminders": 0,
                "effectiveness_rate": 0.0,
                "drop_off_rate": 0.0,
                "avg_days_to_completion": None,
            }

        reminder_count = int(len(frame))
        effective_reminders = int(frame["effective"].sum())
        drop_off_reminders = int(frame["drop_off"].sum())
        completed_days = frame.loc[frame["effective"], "days_to_completion"].dropna()

        return {
            "attribution_window_days": attribution_window_days,
            "reminder_count": reminder_count,
            "effective_reminders": effective_reminders,
            "drop_off_reminders": drop_off_reminders,
            "effectiveness_rate": round((effective_reminders / reminder_count) * 100, 1) if reminder_count else 0.0,
            "drop_off_rate": round((drop_off_reminders / reminder_count) * 100, 1) if reminder_count else 0.0,
            "avg_days_to_completion": round(float(completed_days.mean()), 2) if not completed_days.empty else None,
        }

    @staticmethod
    def breakdown(frame: pd.DataFrame, group_by: str) -> pd.DataFrame:
        if frame.empty or group_by not in frame.columns:
            return pd.DataFrame(columns=[group_by, "reminders", "effective_reminders", "drop_off_reminders", "effectiveness_rate", "drop_off_rate", "avg_days_to_completion"])

        grouped = frame.groupby(group_by, dropna=False).agg(
            reminders=("notification_log_id", "count"),
            effective_reminders=("effective", "sum"),
            drop_off_reminders=("drop_off", "sum"),
            avg_days_to_completion=("days_to_completion", "mean"),
        ).reset_index()

        grouped["effectiveness_rate"] = grouped.apply(
            lambda row: round((row["effective_reminders"] / row["reminders"]) * 100, 1) if row["reminders"] else 0.0,
            axis=1,
        )
        grouped["drop_off_rate"] = grouped.apply(
            lambda row: round((row["drop_off_reminders"] / row["reminders"]) * 100, 1) if row["reminders"] else 0.0,
            axis=1,
        )
        grouped["avg_days_to_completion"] = grouped["avg_days_to_completion"].round(2)
        grouped = grouped.sort_values(["effectiveness_rate", "reminders"], ascending=[False, False]).reset_index(drop=True)
        return grouped

    @staticmethod
    def build_insights(
        db: Session,
        attribution_window_days: int = 14,
        user_id: Optional[int] = None,
        jurisdiction: Optional[str] = None,
        court_name: Optional[str] = None,
        deadline_type: Optional[str] = None,
        channel: Optional[NotificationChannel | str] = None,
    ) -> Dict[str, object]:
        frame = ReminderInsightsEngine.build_attribution_frame(
            db=db,
            attribution_window_days=attribution_window_days,
            user_id=user_id,
            jurisdiction=jurisdiction,
            court_name=court_name,
            deadline_type=deadline_type,
            channel=channel,
        )

        return {
            "summary": ReminderInsightsEngine.summarize(frame, attribution_window_days=attribution_window_days),
            "frame": frame,
            "by_jurisdiction": ReminderInsightsEngine.breakdown(frame, "jurisdiction"),
            "by_court": ReminderInsightsEngine.breakdown(frame, "court_name"),
            "by_deadline_type": ReminderInsightsEngine.breakdown(frame, "deadline_type"),
            "by_channel": ReminderInsightsEngine.breakdown(frame, "channel"),
        }


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


def calculate_statistical_metrics(values: list[float]) -> dict[str, float]:
    """
    Calculates essential descriptive statistics for analytical telemetry 
    purposes including mean, median, and variance.
    """
    if not values:
        return {"mean": 0.0, "median": 0.0, "variance": 0.0}
    sorted_vals = sorted(values)
    n = len(values)
    mean = sum(values) / n
    median = sorted_vals[n // 2] if n % 2 != 0 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    variance = sum((x - mean) ** 2 for x in values) / n
    return {"mean": mean, "median": median, "variance": variance}
