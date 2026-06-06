"""
Precedent Matcher
Finds cases where similar arguments won/lost to guide legal strategy.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from database import (
    Case,
    CaseIssue,
    CaseArgument,
    KnowledgeGraphEdge,
    CaseEmbedding,
)
from core.precedent_ranker import rank_precedents

logger = logging.getLogger(__name__)


class PrecedentMatcher:
    """Find precedent cases with winning arguments"""

    @staticmethod
    def find_winning_precedents(
        db: Session,
        case_id: int,
        issue_name: Optional[str] = None,
        argument_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find precedent cases where similar arguments led to winning outcomes
        
        Args:
            db: Database session
            case_id: Case ID to find precedents for
            issue_name: Filter by specific issue (optional)
            argument_type: Filter by argument type (optional)
            limit: Maximum number of results
            
        Returns:
            List of precedent cases with successful arguments
        """
        try:
            # Get query case
            query_case = db.query(Case).filter(Case.id == case_id).first()
            if not query_case:
                logger.error(f"Case {case_id} not found")
                return []
            
            # Get issues for query case
            case_issues = db.query(CaseIssue).filter(
                CaseIssue.case_id == case_id
            ).all()
            
            if not case_issues and not issue_name:
                logger.warning(f"No issues found for case {case_id}")
                return []
            
            # Build query for knowledge graph edges
            results = []
            
            # Search by specific issue if provided
            if issue_name:
                case_issues = db.query(CaseIssue).filter(
                    CaseIssue.issue_name == issue_name
                ).all()
            
            for issue in case_issues:
                # Get knowledge graph edges for this issue
                edges = db.query(KnowledgeGraphEdge).filter(
                    KnowledgeGraphEdge.issue_id == issue.id,
                    # Only success cases
                    KnowledgeGraphEdge.outcome.in_(["plaintiff_won", "defendant_won", "settled_favorably"])
                ).all()
                
                for edge in edges:
                    precedent_case = edge.case
                    argument = edge.argument
                    
                    # Get embedding info
                    embedding = db.query(CaseEmbedding).filter(
                        CaseEmbedding.case_id == precedent_case.id
                    ).first()
                    
                    result = {
                        "case_id": precedent_case.id,
                        "case_number": precedent_case.case_number,
                        "case_type": precedent_case.case_type,
                        "jurisdiction": precedent_case.jurisdiction,
                        "title": precedent_case.title,
                        "outcome": edge.outcome,
                        "issue": issue.issue_name,
                        "argument": argument.argument_text[:200],  # Truncate long args
                        "argument_succeeded": argument.argument_succeeded,
                        "weight": float(edge.weight) if edge.weight else 1.0,
                        "supporting_evidence": argument.supporting_evidence,
                        "created_at": precedent_case.created_at.isoformat(),
                    }
                    
                    results.append(result)
            
            # Sort by weight (relevance)
            results.sort(key=lambda x: float(x.get("weight", 1.0)), reverse=True)
            
            # Remove duplicates (same precedent case)
            seen = set()
            unique_results = []
            for r in results:
                key = r["case_id"]
                if key not in seen:
                    seen.add(key)
                    unique_results.append(r)

            # Rank using multi-factor ranking
            ranked = rank_precedents(db, unique_results)
            return ranked[:limit]
            
        except Exception as e:
            logger.error(f"Error finding winning precedents: {str(e)}")
            return []

    @staticmethod
    def find_losing_precedents(
        db: Session,
        case_id: int,
        issue_name: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find precedent cases where similar arguments failed
        
        Args:
            db: Database session
            case_id: Case ID to find precedents for
            issue_name: Filter by specific issue (optional)
            limit: Maximum number of results
            
        Returns:
            List of precedent cases with failed arguments (to avoid)
        """
        try:
            query_case = db.query(Case).filter(Case.id == case_id).first()
            if not query_case:
                return []
            
            case_issues = db.query(CaseIssue).filter(
                CaseIssue.case_id == case_id
            ).all()
            
            if issue_name:
                case_issues = db.query(CaseIssue).filter(
                    CaseIssue.issue_name == issue_name
                ).all()
            
            results = []
            
            for issue in case_issues:
                # Get edges where arguments failed
                edges = db.query(KnowledgeGraphEdge).filter(
                    KnowledgeGraphEdge.issue_id == issue.id,
                    KnowledgeGraphEdge.outcome.in_(["defendant_won", "plaintiff_lost", "dismissed"])
                ).all()
                
                for edge in edges:
                    precedent_case = edge.case
                    argument = edge.argument
                    
                    if not argument.argument_succeeded:
                        result = {
                            "case_id": precedent_case.id,
                            "case_number": precedent_case.case_number,
                            "case_type": precedent_case.case_type,
                            "jurisdiction": precedent_case.jurisdiction,
                            "title": precedent_case.title,
                            "outcome": edge.outcome,
                            "issue": issue.issue_name,
                            "failed_argument": argument.argument_text[:200],
                            "why_it_failed": argument.supporting_evidence,
                            "created_at": precedent_case.created_at.isoformat(),
                        }
                        results.append(result)
            
            results.sort(key=lambda x: x["case_id"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error finding losing precedents: {str(e)}")
            return []

    @staticmethod
    def get_argument_success_rate(
        db: Session,
        argument_text: str,
        issue_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get success rate of an argument across all cases
        
        Args:
            db: Database session
            argument_text: The argument to analyze
            issue_name: Filter by issue (optional)
            
        Returns:
            Dict with success statistics
        """
        try:
            from difflib import SequenceMatcher
            query = db.query(CaseArgument)
            
            if issue_name:
                issue = db.query(CaseIssue).filter(
                    CaseIssue.issue_name == issue_name
                ).first()
                if issue:
                    query = query.filter(CaseArgument.issue_id == issue.id)
            
            all_arguments = query.all()
            if not all_arguments:
                return {"success_rate": 0, "total_uses": 0, "successful": 0, "failed": 0}

            # Filter candidates using a fuzzy similarity match threshold
            matched_arguments = []
            for arg in all_arguments:
                if not arg.argument_text:
                    continue
                ratio = SequenceMatcher(None, arg.argument_text.lower(), argument_text.lower()).ratio()
                if ratio >= 0.65:
                    matched_arguments.append(arg)

            if not matched_arguments:
                return {"success_rate": 0, "total_uses": 0, "successful": 0, "failed": 0}
            
            successful = sum(1 for arg in matched_arguments if arg.argument_succeeded is True)
            failed = sum(1 for arg in matched_arguments if arg.argument_succeeded is False)
            total = len(matched_arguments)
            
            success_rate = (successful / total * 100) if total > 0 else 0
            
            return {
                "argument": argument_text[:100],
                "success_rate": round(success_rate, 1),
                "total_uses": total,
                "successful": successful,
                "failed": failed,
                "unknown": total - successful - failed,
            }
            
        except Exception as e:
            logger.error(f"Failed to get argument success rate: {str(e)}")
            return {}

    @staticmethod
    def find_arguments_by_issue(
        db: Session,
        issue_name: str,
        outcome: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find all arguments used for a specific issue
        
        Args:
            db: Database session
            issue_name: Issue name to search for
            outcome: Filter by outcome (optional)
            
        Returns:
            List of arguments with success rate
        """
        try:
            issue = db.query(CaseIssue).filter(
                CaseIssue.issue_name == issue_name
            ).first()
            
            if not issue:
                logger.warning(f"Issue {issue_name} not found")
                return []
            
            query = db.query(CaseArgument).filter(
                CaseArgument.issue_id == issue.id
            )
            
            arguments = query.all()
            
            # Group by argument text and calculate success rate
            arg_groups = {}
            for arg in arguments:
                key = arg.argument_text[:100]  # Group by first 100 chars
                if key not in arg_groups:
                    arg_groups[key] = {
                        "text": arg.argument_text,
                        "succeeded": 0,
                        "failed": 0,
                        "unknown": 0,
                        "cases": [],
                    }
                
                if arg.argument_succeeded is True:
                    arg_groups[key]["succeeded"] += 1
                elif arg.argument_succeeded is False:
                    arg_groups[key]["failed"] += 1
                else:
                    arg_groups[key]["unknown"] += 1
                
                arg_groups[key]["cases"].append(arg.case_id)
            
            # Convert to results
            results = []
            for key, group in arg_groups.items():
                total = group["succeeded"] + group["failed"] + group["unknown"]
                success_rate = (group["succeeded"] / total * 100) if total > 0 else 0
                
                results.append({
                    "argument": group["text"][:200],
                    "issue": issue_name,
                    "success_rate": round(success_rate, 1),
                    "total_uses": total,
                    "succeeded": group["succeeded"],
                    "failed": group["failed"],
                    "related_cases": group["cases"][:5],
                })
            
            # Sort by success rate
            results.sort(key=lambda x: x["success_rate"], reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Failed to find arguments by issue: {str(e)}")
            return []
