"""
Case Comparison Engine
Compare user's case with precedent cases to highlight similarities and differences.
"""

import logging
from typing import List, Optional, Dict, Any
from difflib import SequenceMatcher

from sqlalchemy.orm import Session

from database import (
    Case,
    CaseDocument,
    CaseIssue,
    CaseArgument,
)

logger = logging.getLogger(__name__)


class CaseComparison:
    """Compare cases to highlight patterns and suggest arguments"""

    @staticmethod
    def calculate_length_penalized_similarity(text1: str, text2: str) -> float:
        """Calculate SequenceMatcher ratio penalized by length difference.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Normalized and penalized float similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0
            
        raw_ratio = SequenceMatcher(None, text1, text2).ratio()
        
        len1 = len(text1)
        len2 = len(text2)
        
        if len1 == 0 or len2 == 0:
            return 0.0
            
        # Multiplicative penalty based on the length ratio
        # Raising ratio to 0.25 scales down similarity if length mismatch is severe,
        # but keeps it reasonable for minor differences.
        len_ratio = min(len1, len2) / max(len1, len2)
        penalty = len_ratio ** 0.25
        
        return raw_ratio * penalty

    @staticmethod
    def compare_cases(
        db: Session,
        user_case_id: int,
        precedent_case_id: int,
    ) -> Dict[str, Any]:
        """Compare user's case with precedent case
        
        Args:
            db: Database session
            user_case_id: User's case ID
            precedent_case_id: Precedent case ID to compare with
            
        Returns:
            Dict with comparison details
        """
        try:
            user_case = db.query(Case).filter(Case.id == user_case_id).first()
            precedent_case = db.query(Case).filter(Case.id == precedent_case_id).first()
            
            if not user_case or not precedent_case:
                logger.error("One or both cases not found")
                return {}
            
            # Get issues for both cases
            user_issues = db.query(CaseIssue).filter(
                CaseIssue.case_id == user_case_id
            ).all()
            
            precedent_issues = db.query(CaseIssue).filter(
                CaseIssue.case_id == precedent_case_id
            ).all()
            
            user_issue_names = {issue.issue_name for issue in user_issues}
            precedent_issue_names = {issue.issue_name for issue in precedent_issues}
            
            # Find similarities and differences
            shared_issues = user_issue_names & precedent_issue_names
            user_only_issues = user_issue_names - precedent_issue_names
            precedent_only_issues = precedent_issue_names - user_issue_names
            
            # Get arguments for user case
            user_arguments = []
            for issue in user_issues:
                args = db.query(CaseArgument).filter(
                    CaseArgument.issue_id == issue.id
                ).all()
                for arg in args:
                    user_arguments.append({
                        "issue": issue.issue_name,
                        "text": arg.argument_text,
                        "type": arg.argument_type,
                    })
            
            # Get arguments for precedent case
            precedent_arguments = []
            for issue in precedent_issues:
                args = db.query(CaseArgument).filter(
                    CaseArgument.issue_id == issue.id
                ).all()
                for arg in args:
                    precedent_arguments.append({
                        "issue": issue.issue_name,
                        "text": arg.argument_text,
                        "type": arg.argument_type,
                        "succeeded": arg.argument_succeeded,
                    })
            
            # Find similar arguments
            similar_arguments = []
            for user_arg in user_arguments:
                for prec_arg in precedent_arguments:
                    if user_arg["issue"] not in shared_issues:
                        continue
                    
                    # Calculate text similarity with length penalty normalization
                    similarity = CaseComparison.calculate_length_penalized_similarity(
                        user_arg["text"],
                        prec_arg["text"]
                    )
                    
                    if similarity > 0.5:  # Threshold for similarity
                        similar_arguments.append({
                            "issue": user_arg["issue"],
                            "user_argument": user_arg["text"][:150],
                            "precedent_argument": prec_arg["text"][:150],
                            "precedent_succeeded": prec_arg["succeeded"],
                            "similarity": round(similarity, 2),
                        })
            
            # Get case documents for summary
            user_summary = None
            precedent_summary = None
            
            user_doc = db.query(CaseDocument).filter(
                CaseDocument.case_id == user_case_id,
                CaseDocument.document_type == "Judgment"
            ).first()
            if user_doc:
                user_summary = user_doc.summary
            
            precedent_doc = db.query(CaseDocument).filter(
                CaseDocument.case_id == precedent_case_id,
                CaseDocument.document_type == "Judgment"
            ).first()
            if precedent_doc:
                precedent_summary = precedent_doc.summary
            
            # Build comparison result
            comparison = {
                "user_case": {
                    "id": user_case.id,
                    "number": user_case.case_number,
                    "type": user_case.case_type,
                    "jurisdiction": user_case.jurisdiction,
                    "status": user_case.status.value,
                    "summary": user_summary,
                    "issues": list(user_issue_names),
                    "arguments_count": len(user_arguments),
                },
                "precedent_case": {
                    "id": precedent_case.id,
                    "number": precedent_case.case_number,
                    "type": precedent_case.case_type,
                    "jurisdiction": precedent_case.jurisdiction,
                    "status": precedent_case.status.value,
                    "summary": precedent_summary,
                    "issues": list(precedent_issue_names),
                    "arguments_count": len(precedent_arguments),
                },
                "similarities": {
                    "shared_issues": list(shared_issues),
                    "shared_issues_count": len(shared_issues),
                    "similar_arguments": similar_arguments,
                    "similar_arguments_count": len(similar_arguments),
                },
                "differences": {
                    "user_unique_issues": list(user_only_issues),
                    "precedent_unique_issues": list(precedent_only_issues),
                    "case_type_match": user_case.case_type == precedent_case.case_type,
                    "jurisdiction_match": user_case.jurisdiction == precedent_case.jurisdiction,
                },
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing cases: {str(e)}")
            return {}

    @staticmethod
    def suggest_arguments(
        db: Session,
        user_case_id: int,
        precedent_case_id: int,
    ) -> List[Dict[str, Any]]:
        """Suggest legal arguments based on winning precedents
        
        Args:
            db: Database session
            user_case_id: User's case ID
            precedent_case_id: Precedent case ID
            
        Returns:
            List of suggested arguments
        """
        try:
            # Get comparison
            comparison = CaseComparison.compare_cases(db, user_case_id, precedent_case_id)
            if not comparison:
                return []
            
            suggestions = []
            
            # Suggest arguments that succeeded in precedent but not used in user's case
            for shared_issue in comparison["similarities"]["shared_issues"]:
                # Get all successful arguments from precedent for this issue
                precedent_case = db.query(Case).filter(
                    Case.id == precedent_case_id
                ).first()
                
                precedent_issue = db.query(CaseIssue).filter(
                    CaseIssue.case_id == precedent_case_id,
                    CaseIssue.issue_name == shared_issue
                ).first()
                
                if not precedent_issue:
                    continue
                
                successful_args = db.query(CaseArgument).filter(
                    CaseArgument.issue_id == precedent_issue.id,
                    CaseArgument.argument_succeeded == True
                ).all()
                
                for arg in successful_args:
                    suggestion = {
                        "issue": shared_issue,
                        "argument": arg.argument_text,
                        "argument_type": arg.argument_type,
                        "reason": f"This argument succeeded in a similar case ({precedent_case.case_number})",
                        "supporting_evidence": arg.supporting_evidence,
                        "confidence": "high" if arg.argument_succeeded is True else "medium",
                        "precedent_case_number": precedent_case.case_number,
                    }
                    suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {str(e)}")
            return []

    @staticmethod
    def highlight_differences(
        db: Session,
        user_case_id: int,
        precedent_case_id: int,
    ) -> Dict[str, Any]:
        """Highlight key differences between cases
        
        Args:
            db: Database session
            user_case_id: User's case ID
            precedent_case_id: Precedent case ID
            
        Returns:
            Dict highlighting key differences
        """
        try:
            comparison = CaseComparison.compare_cases(db, user_case_id, precedent_case_id)
            if not comparison:
                return {}
            
            diffs = comparison["differences"]
            
            highlights = {
                "case_type_different": not diffs["case_type_match"],
                "case_type_detail": f"{comparison['user_case']['type']} vs {comparison['precedent_case']['type']}" if not diffs["case_type_match"] else None,
                "jurisdiction_different": not diffs["jurisdiction_match"],
                "jurisdiction_detail": f"{comparison['user_case']['jurisdiction']} vs {comparison['precedent_case']['jurisdiction']}" if not diffs["jurisdiction_match"] else None,
                "unique_issues_in_user_case": diffs["user_unique_issues"],
                "unique_issues_in_precedent": diffs["precedent_unique_issues"],
                "warning_different_case_type": "Case types differ - precedent may have limited applicability" if not diffs["case_type_match"] else None,
                "warning_different_jurisdiction": "Different jurisdictions - local laws may differ" if not diffs["jurisdiction_match"] else None,
            }
            
            # Clean up None values
            highlights = {k: v for k, v in highlights.items() if v is not None}
            
            return highlights
            
        except Exception as e:
            logger.error(f"Failed to highlight differences: {str(e)}")
            return {}
