"""
Analytics Engine for LegalEase AI

Provides case similarity calculations, success rate estimations, 
judge analytics, and trend analysis.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from database import CaseRecord, CaseOutcome, CaseAnalytics, UserFeedback
import hashlib
from collections import Counter
import logging

logger = logging.getLogger(__name__)


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
        
        Weights:
        - case_type: 0.3
        - jurisdiction: 0.2
        - plaintiff_type: 0.15
        - defendant_type: 0.15
        - case_value: 0.2
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
        
        # Case value match (broad ranges)
        if case1.case_value and case2.case_value:
            if case1.case_value == case2.case_value:
                score += weights["case_value"]
        
        return score * 100  # Return as percentage (0-100)
    
    @staticmethod
    def find_similar_cases(
        db: Session,
        reference_case: CaseRecord,
        min_similarity: float = 50.0,
        limit: int = 50,
    ) -> List[Tuple[CaseRecord, float]]:
        """Find cases similar to reference case with similarity scores"""
        all_cases = db.query(CaseRecord).filter(
            CaseRecord.case_id != reference_case.case_id
        ).all()
        
        similarities = []
        for case in all_cases:
            score = CaseSimilarityCalculator.case_similarity_score(reference_case, case)
            if score >= min_similarity:
                similarities.append((case, score))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]


class AnalyticsCalculator:
    """Calculate various analytics metrics"""
    
    @staticmethod
    def calculate_success_rate(cases: List[CaseRecord], winning_outcome: str = "plaintiff_won") -> float:
        """Calculate success rate for given cases"""
        if not cases:
            return 0.0
        
        wins = sum(1 for case in cases if case.outcome.lower() == winning_outcome.lower())
        return (wins / len(cases)) * 100
    
    @staticmethod
    def calculate_appeal_success_rate(cases: List[CaseRecord]) -> float:
        """Calculate appeal success rate for cases with appeal data"""
        cases_with_appeals = [
            case for case in cases
            if case.outcome_data and case.outcome_data.appeal_filed
        ]
        
        if not cases_with_appeals:
            return 0.0
        
        successful = sum(
            1 for case in cases_with_appeals
            if case.outcome_data.appeal_success is True
        )
        
        return (successful / len(cases_with_appeals)) * 100
    
    @staticmethod
    def calculate_judge_win_rate(
        db: Session,
        judge_name: str,
        jurisdiction: str,
        winning_outcome: str = "plaintiff_won",
    ) -> Dict:
        """Calculate judge-specific statistics"""
        cases = db.query(CaseRecord).filter(
            CaseRecord.judge_name == judge_name,
            CaseRecord.jurisdiction == jurisdiction,
        ).all()
        
        if not cases:
            return {
                "judge": judge_name,
                "jurisdiction": jurisdiction,
                "total_cases": 0,
                "win_rate": 0.0,
                "appeal_success_rate": 0.0,
            }
        
        win_rate = AnalyticsCalculator.calculate_success_rate(cases, winning_outcome)
        appeal_rate = AnalyticsCalculator.calculate_appeal_success_rate(cases)
        
        return {
            "judge": judge_name,
            "jurisdiction": jurisdiction,
            "total_cases": len(cases),
            "win_rate": round(win_rate, 1),
            "appeal_success_rate": round(appeal_rate, 1),
        }
    
    @staticmethod
    def calculate_court_statistics(
        db: Session,
        court_name: str,
        case_type: Optional[str] = None,
    ) -> Dict:
        """Calculate statistics for a specific court"""
        query = db.query(CaseRecord).filter(CaseRecord.court_name == court_name)
        
        if case_type:
            query = query.filter(CaseRecord.case_type == case_type)
        
        cases = query.all()
        
        if not cases:
            return {
                "court": court_name,
                "case_type": case_type,
                "total_cases": 0,
            }
        
        # Count outcomes
        outcomes = Counter(case.outcome for case in cases)
        
        appeals_filed = sum(
            1 for case in cases
            if case.outcome_data and case.outcome_data.appeal_filed
        )
        
        return {
            "court": court_name,
            "case_type": case_type,
            "total_cases": len(cases),
            "plaintiff_wins": outcomes.get("plaintiff_won", 0),
            "defendant_wins": outcomes.get("defendant_won", 0),
            "settlements": outcomes.get("settlement", 0),
            "dismissals": outcomes.get("dismissal", 0),
            "appeals_filed": appeals_filed,
            "appeal_rate": round((appeals_filed / len(cases)) * 100, 1) if cases else 0,
        }
    
    @staticmethod
    def calculate_jurisdiction_trends(
        db: Session,
        jurisdiction: str,
    ) -> Dict:
        """Get trends for a jurisdiction"""
        cases = db.query(CaseRecord).filter(
            CaseRecord.jurisdiction == jurisdiction
        ).all()
        
        if not cases:
            return {"jurisdiction": jurisdiction, "total_cases": 0}
        
        # Group by case type
        by_type = {}
        for case in cases:
            if case.case_type not in by_type:
                by_type[case.case_type] = []
            by_type[case.case_type].append(case)
        
        type_stats = {}
        for case_type, type_cases in by_type.items():
            win_rate = AnalyticsCalculator.calculate_success_rate(type_cases, "plaintiff_won")
            type_stats[case_type] = {
                "count": len(type_cases),
                "plaintiff_win_rate": round(win_rate, 1),
            }
        
        return {
            "jurisdiction": jurisdiction,
            "total_cases": len(cases),
            "case_type_stats": type_stats,
        }


class AppealProbabilityEstimator:
    """Estimate appeal success probability for new cases"""
    
    @staticmethod
    def estimate_appeal_success(
        db: Session,
        case_type: str,
        jurisdiction: str,
        court_name: Optional[str] = None,
        judge_name: Optional[str] = None,
        outcome_magnitude: str = "moderate",  # low, moderate, high
        similar_cases_limit: int = 50,
    ) -> Dict:
        """
        Estimate appeal success probability for a case.
        
        Returns:
        {
            "estimated_success_rate": 22.5,
            "confidence": "medium",  # low, medium, high
            "similar_cases_found": 23,
            "reasoning": "Based on 23 similar cases in Delhi District Court..."
        }
        """
        # Get similar cases
        similar_query = db.query(CaseRecord).filter(
            CaseRecord.case_type == case_type,
            CaseRecord.jurisdiction == jurisdiction,
        )
        
        if court_name:
            similar_query = similar_query.filter(CaseRecord.court_name == court_name)
        
        if judge_name:
            similar_query = similar_query.filter(CaseRecord.judge_name == judge_name)
        
        similar_cases = similar_query.all()
        
        if not similar_cases:
            return {
                "estimated_success_rate": None,
                "confidence": "very_low",
                "similar_cases_found": 0,
                "reasoning": f"No similar cases found in {jurisdiction} for {case_type} cases.",
            }
        
        # Calculate appeal success rate for similar cases
        appeal_success_rate = AnalyticsCalculator.calculate_appeal_success_rate(similar_cases)
        
        # Adjust based on outcome magnitude
        adjustment = {
            "low": 0.95,      # Lower success if outcome is marginal
            "moderate": 1.0,  # No adjustment for moderate cases
            "high": 1.05,     # Slightly higher if decision is clear
        }.get(outcome_magnitude.lower(), 1.0)
        
        adjusted_rate = min(100, max(0, appeal_success_rate * adjustment))
        
        # Determine confidence based on number of similar cases
        if len(similar_cases) >= 50:
            confidence = "high"
        elif len(similar_cases) >= 20:
            confidence = "medium"
        elif len(similar_cases) >= 10:
            confidence = "low"
        else:
            confidence = "very_low"
        
        reasoning = f"Based on {len(similar_cases)} similar {case_type} cases in {jurisdiction}. "
        if court_name:
            reasoning += f"Court: {court_name}. "
        if judge_name:
            reasoning += f"Judge: {judge_name}. "
        reasoning += f"Appeal success rate in similar cases: {appeal_success_rate:.1f}%"
        
        return {
            "estimated_success_rate": round(adjusted_rate, 1),
            "confidence": confidence,
            "similar_cases_found": len(similar_cases),
            "appeal_success_rate_from_similar": round(appeal_success_rate, 1),
            "reasoning": reasoning,
        }
    
    @staticmethod
    def estimate_appeal_cost_and_time(
        db: Session,
        case_type: str,
        jurisdiction: str,
    ) -> Dict:
        """Estimate typical appeal cost and time"""
        cases_with_appeals = db.query(CaseRecord).filter(
            CaseRecord.case_type == case_type,
            CaseRecord.jurisdiction == jurisdiction,
        ).all()
        
        cases_with_outcome = [
            case for case in cases_with_appeals
            if case.outcome_data and case.outcome_data.appeal_filed
        ]
        
        if not cases_with_outcome:
            # Return generic estimates based on case type
            default_costs = {
                "civil": "₹12,000 - ₹25,000",
                "criminal": "₹5,000 - ₹15,000",
                "family": "₹8,000 - ₹20,000",
                "commercial": "₹20,000 - ₹50,000",
            }
            default_time = {
                "civil": "12-24 months",
                "criminal": "12-30 months",
                "family": "12-24 months",
                "commercial": "18-36 months",
            }
            return {
                "avg_cost": default_costs.get(case_type.lower(), "₹10,000 - ₹30,000"),
                "avg_time": default_time.get(case_type.lower(), "12-24 months"),
                "note": "Generic estimates - not based on local data",
            }
        
        # Extract costs from case data
        costs = []
        times = []
        
        for case in cases_with_outcome:
            if case.outcome_data.appeal_cost:
                # Try to extract numeric value
                cost_str = case.outcome_data.appeal_cost
                try:
                    # Very basic extraction - assumes format like "12000" or "12000-15000"
                    import re
                    numbers = re.findall(r'\d+', cost_str)
                    if numbers:
                        costs.append(int(numbers[0]))
                except:
                    pass
            
            if case.outcome_data.time_to_appeal_verdict:
                times.append(case.outcome_data.time_to_appeal_verdict)
        
        # Calculate averages
        avg_cost = sum(costs) / len(costs) if costs else None
        avg_time_days = sum(times) / len(times) if times else None
        
        cost_str = f"₹{int(avg_cost)}" if avg_cost else "Unknown"
        if avg_cost:
            cost_str = f"₹{int(avg_cost * 0.8):.0f} - ₹{int(avg_cost * 1.2):.0f}"
        
        time_str = "12-24 months"
        if avg_time_days:
            months = avg_time_days / 30
            time_str = f"{int(months * 0.8)}-{int(months * 1.2)} months"
        
        return {
            "avg_cost": cost_str,
            "avg_time": time_str,
            "similar_cases": len(cases_with_outcome),
        }


class AnalyticsAggregator:
    """Generate aggregated analytics for dashboard"""
    
    @staticmethod
    def get_dashboard_summary(db: Session) -> Dict:
        """Get overall dashboard summary"""
        all_cases = db.query(CaseRecord).all()
        
        total_cases = len(all_cases)
        appeals_filed = sum(1 for case in all_cases if case.outcome_data and case.outcome_data.appeal_filed)
        
        outcomes = Counter(case.outcome for case in all_cases)
        
        return {
            "total_cases_processed": total_cases,
            "appeals_filed": appeals_filed,
            "appeal_rate_percent": round((appeals_filed / total_cases * 100), 1) if total_cases else 0,
            "plaintiff_wins": outcomes.get("plaintiff_won", 0),
            "defendant_wins": outcomes.get("defendant_won", 0),
            "settlements": outcomes.get("settlement", 0),
            "dismissals": outcomes.get("dismissal", 0),
        }
    
    @staticmethod
    def get_top_judges(db: Session, jurisdiction: str, limit: int = 10) -> List[Dict]:
        """Get top judges by appeal success rate"""
        all_cases = db.query(CaseRecord).filter(
            CaseRecord.jurisdiction == jurisdiction
        ).all()
        
        judges = set(case.judge_name for case in all_cases if case.judge_name)
        
        judge_stats = []
        for judge in judges:
            stats = AnalyticsCalculator.calculate_judge_win_rate(db, judge, jurisdiction)
            if stats["total_cases"] >= 5:  # Only include judges with 5+ cases
                judge_stats.append(stats)
        
        # Sort by appeal success rate
        judge_stats.sort(key=lambda x: x["appeal_success_rate"], reverse=True)
        return judge_stats[:limit]
    
    @staticmethod
    def get_regional_trends(db: Session) -> List[Dict]:
        """Get trends by region/jurisdiction"""
        all_cases = db.query(CaseRecord).all()
        jurisdictions = set(case.jurisdiction for case in all_cases if case.jurisdiction)
        
        trends = []
        for jurisdiction in jurisdictions:
            jur_cases = [case for case in all_cases if case.jurisdiction == jurisdiction]
            appeal_success = AnalyticsCalculator.calculate_appeal_success_rate(jur_cases)
            
            trends.append({
                "jurisdiction": jurisdiction,
                "total_cases": len(jur_cases),
                "appeal_success_rate": round(appeal_success, 1),
            })
        
        trends.sort(key=lambda x: x["total_cases"], reverse=True)
        return trends


# Utility function to anonymize case ID
def generate_anonymous_case_id(case_data: str) -> str:
    """Generate anonymous case ID from case data"""
    return hashlib.sha256(case_data.encode()).hexdigest()[:16]
