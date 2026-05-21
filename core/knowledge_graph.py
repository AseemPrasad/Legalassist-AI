"""
Knowledge Graph Builder
Build and query a graph of Issues → Arguments → Outcomes.
"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from database import (
    Case,
    CaseIssue,
    CaseArgument,
    KnowledgeGraphEdge,
    CaseDocument,
)
from core.argument_extraction_engine import ArgumentExtractionEngine
from core.argument_extraction_schema import ArgumentExtractionResult

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Build and query the case knowledge graph"""

    @staticmethod
    def extract_issues_from_case(
        db: Session,
        case_id: int,
        override_existing: bool = False,
    ) -> List[CaseIssue]:
        """Extract issues from a case document
        
        Args:
            db: Database session
            case_id: Case ID
            override_existing: Replace existing issues
            
        Returns:
            List of created CaseIssue objects
        """
        try:
            # Get case
            case = db.query(Case).filter(Case.id == case_id).first()
            if not case:
                logger.error(f"Case {case_id} not found")
                return []
            
            # Remove existing issues if requested
            if override_existing:
                db.query(CaseIssue).filter(CaseIssue.case_id == case_id).delete()
                db.commit()
            
            # Get judgment document
            doc = db.query(CaseDocument).filter(
                CaseDocument.case_id == case_id,
                CaseDocument.document_type == "Judgment"
            ).first()
            
            if not doc or not doc.document_content:
                logger.warning(f"No judgment document found for case {case_id}")
                return []
            
            # Extract issues from document using LLM
            # For now, use simple keyword extraction
            # In production, would use GPT-4 for semantic extraction
            issues = KnowledgeGraphBuilder._extract_issues_from_text(
                doc.document_content,
                case_id,
            )
            
            # Store in database
            issue_objects = []
            for issue_data in issues:
                # Check if issue already exists
                existing = db.query(CaseIssue).filter(
                    CaseIssue.case_id == case_id,
                    CaseIssue.issue_name == issue_data["name"]
                ).first()
                
                if existing:
                    issue_objects.append(existing)
                    continue
                
                issue_obj = CaseIssue(
                    case_id=case_id,
                    issue_name=issue_data["name"],
                    issue_description=issue_data.get("description"),
                    issue_category=case.case_type,
                    confidence_score=str(issue_data.get("confidence", 1.0)),
                    extracted_from_document=doc.id,
                    extraction_method="llm",
                )
                
                db.add(issue_obj)
                issue_objects.append(issue_obj)
            
            db.commit()
            
            logger.info(f"Extracted {len(issue_objects)} issues for case {case_id}")
            return issue_objects
            
        except Exception as e:
            logger.error(f"Failed to extract issues: {str(e)}")
            db.rollback()
            return []

    @staticmethod
    def extract_arguments_from_case(
        db: Session,
        case_id: int,
        override_existing: bool = False,
    ) -> List[CaseArgument]:
        """Extract legal arguments from a case using LLM-powered engine
        
        Args:
            db: Database session
            case_id: Case ID
            override_existing: Replace existing arguments
            
        Returns:
            List of created CaseArgument objects
        """
        try:
            case = db.query(Case).filter(Case.id == case_id).first()
            if not case:
                logger.error(f"Case {case_id} not found")
                return []
            
            if override_existing:
                db.query(CaseArgument).filter(CaseArgument.case_id == case_id).delete()
                db.commit()
            
            # Get judgment document
            doc = db.query(CaseDocument).filter(
                CaseDocument.case_id == case_id,
                CaseDocument.document_type == "Judgment"
            ).first()
            
            if not doc or not doc.document_content:
                logger.warning(f"No judgment document found for case {case_id}")
                return []
            
            # Get case issues
            issues = db.query(CaseIssue).filter(
                CaseIssue.case_id == case_id
            ).all()
            
            if not issues:
                logger.warning(f"No issues found for case {case_id}")
                return []
            
            # Initialize extraction engine
            engine = ArgumentExtractionEngine(use_llm=True)
            
            # Extract arguments for each issue
            argument_objects = []
            
            for issue in issues:
                # Use LLM-powered extraction
                result: ArgumentExtractionResult = engine.extract_arguments(
                    text=doc.document_content,
                    issue_name=issue.issue_name,
                    case_id=case_id,
                )
                
                # Log extraction result
                logger.info(
                    f"Issue '{issue.issue_name}': extracted {result.total_extracted} arguments "
                    f"(method={result.extraction_method}, quality={result.extraction_quality})"
                )
                
                # Store arguments in database
                for arg_data in result.arguments:
                    try:
                        # Check if argument already exists
                        existing = db.query(CaseArgument).filter(
                            CaseArgument.case_id == case_id,
                            CaseArgument.issue_id == issue.id,
                            CaseArgument.argument_text == arg_data.argument_text
                        ).first()
                        
                        if existing:
                            argument_objects.append(existing)
                            continue
                        
                        # Create CaseArgument object
                        arg_obj = CaseArgument(
                            case_id=case_id,
                            issue_id=issue.id,
                            argument_text=arg_data.argument_text,
                            argument_type=arg_data.argument_type,
                            argument_succeeded=None,  # Determined from outcome
                            supporting_evidence=arg_data.supporting_evidence,
                            citation_references=arg_data.citation_references or [],
                        )
                        
                        db.add(arg_obj)
                        argument_objects.append(arg_obj)
                        
                    except Exception as e:
                        logger.error(f"Failed to store argument: {str(e)}")
            
            db.commit()
            
            logger.info(f"Extracted {len(argument_objects)} total arguments for case {case_id}")
            return argument_objects
            
        except Exception as e:
            logger.error(f"Failed to extract arguments: {str(e)}")
            db.rollback()
            return []

    @staticmethod
    def build_graph_edges(
        db: Session,
        case_id: int,
    ) -> List[KnowledgeGraphEdge]:
        """Build knowledge graph edges for a case
        
        Args:
            db: Database session
            case_id: Case ID
            
        Returns:
            List of created edges
        """
        try:
            case = db.query(Case).filter(Case.id == case_id).first()
            if not case:
                return []
            
            # Get issues and arguments
            issues = db.query(CaseIssue).filter(CaseIssue.case_id == case_id).all()
            arguments = db.query(CaseArgument).filter(
                CaseArgument.case_id == case_id
            ).all()
            
            if not issues or not arguments:
                logger.warning(f"Missing issues or arguments for case {case_id}")
                return []
            
            # Determine outcome
            outcome = KnowledgeGraphBuilder._extract_outcome_from_case(db, case_id)
            
            # Create edges for each argument
            edges = []
            for issue in issues:
                for arg in arguments:
                    if arg.issue_id != issue.id:
                        continue
                    
                    # Check if edge already exists
                    existing = db.query(KnowledgeGraphEdge).filter(
                        KnowledgeGraphEdge.issue_id == issue.id,
                        KnowledgeGraphEdge.argument_id == arg.id,
                        KnowledgeGraphEdge.case_id == case_id
                    ).first()
                    
                    if existing:
                        edges.append(existing)
                        continue
                    
                    # Calculate weight (frequency + confidence)
                    weight = 1.0
                    if arg.argument_succeeded is True:
                        weight = 2.0  # Boost successful arguments
                    
                    edge = KnowledgeGraphEdge(
                        issue_id=issue.id,
                        argument_id=arg.id,
                        case_id=case_id,
                        outcome=outcome,
                        weight=str(weight),
                    )
                    
                    db.add(edge)
                    edges.append(edge)
            
            db.commit()
            
            logger.info(f"Created {len(edges)} graph edges for case {case_id}")
            return edges
            
        except Exception as e:
            logger.error(f"Failed to build graph edges: {str(e)}")
            db.rollback()
            return []

    @staticmethod
    def query_graph(
        db: Session,
        issue_name: str,
        desired_outcome: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Query the knowledge graph for cases matching criteria
        
        Args:
            db: Database session
            issue_name: Issue to search for
            desired_outcome: Desired outcome (optional filter)
            limit: Maximum results
            
        Returns:
            List of matching cases with arguments and outcomes
        """
        try:
            # Find issues
            issues = db.query(CaseIssue).filter(
                CaseIssue.issue_name.ilike(f"%{issue_name}%")
            ).all()
            
            if not issues:
                logger.warning(f"No issues found for '{issue_name}'")
                return []
            
            results = []
            
            for issue in issues:
                # Get graph edges for this issue
                query = db.query(KnowledgeGraphEdge).filter(
                    KnowledgeGraphEdge.issue_id == issue.id
                )
                
                if desired_outcome:
                    query = query.filter(KnowledgeGraphEdge.outcome == desired_outcome)
                
                edges = query.all()
                
                for edge in edges:
                    case = edge.case
                    argument = edge.argument
                    
                    result = {
                        "case_id": case.id,
                        "case_number": case.case_number,
                        "case_type": case.case_type,
                        "jurisdiction": case.jurisdiction,
                        "issue": issue.issue_name,
                        "argument": argument.argument_text[:200],
                        "argument_succeeded": argument.argument_succeeded,
                        "outcome": edge.outcome,
                        "weight": float(edge.weight) if edge.weight else 1.0,
                        "case_created_at": case.created_at.isoformat(),
                    }
                    
                    results.append(result)
            
            # Sort by weight (relevance)
            results.sort(key=lambda x: x["weight"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to query graph: {str(e)}")
            return []

    @staticmethod
    def get_graph_statistics(db: Session) -> Dict[str, Any]:
        """Get statistics about the knowledge graph
        
        Args:
            db: Database session
            
        Returns:
            Dict with graph statistics
        """
        try:
            total_issues = db.query(CaseIssue).count()
            total_arguments = db.query(CaseArgument).count()
            total_edges = db.query(KnowledgeGraphEdge).count()
            
            # Count successful paths
            successful_edges = db.query(KnowledgeGraphEdge).filter(
                KnowledgeGraphEdge.outcome.in_(["plaintiff_won", "defendant_won"])
            ).count()
            
            # Count issues with most arguments
            issue_arg_counts = {}
            issues = db.query(CaseIssue).all()
            for issue in issues:
                arg_count = db.query(CaseArgument).filter(
                    CaseArgument.issue_id == issue.id
                ).count()
                if arg_count > 0:
                    issue_arg_counts[issue.issue_name] = arg_count
            
            return {
                "total_issues": total_issues,
                "total_arguments": total_arguments,
                "total_edges": total_edges,
                "successful_paths": successful_edges,
                "top_issues": sorted(
                    issue_arg_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10],
            }
            
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {str(e)}")
            return {}

    @staticmethod
    def _extract_issues_from_text(text: str, case_id: int) -> List[Dict[str, Any]]:
        """Extract issues from case text (simplified version)
        
        Args:
            text: Case document text
            case_id: Case ID
            
        Returns:
            List of extracted issues
        """
        # In production, use GPT-4 for semantic extraction
        # For now, use simple keyword matching
        
        common_issues = {
            "property dispute": ["property", "land", "ownership", "title"],
            "contract dispute": ["contract", "agreement", "terms", "breach"],
            "employment": ["employment", "termination", "wages", "labor"],
            "family law": ["marriage", "divorce", "custody", "alimony"],
            "criminal": ["criminal", "prosecution", "offense", "guilty"],
            "civil rights": ["rights", "discrimination", "harassment"],
            "personal injury": ["injury", "damages", "negligence", "accident"],
        }
        
        text_lower = text.lower()
        found_issues = []
        
        for issue_name, keywords in common_issues.items():
            if any(keyword in text_lower for keyword in keywords):
                found_issues.append({
                    "name": issue_name,
                    "description": f"Detected based on document keywords",
                    "confidence": 0.7,
                })
        
        return found_issues

    @staticmethod
    def _extract_arguments_from_text(
        text: str,
        issue_name: str,
        case_id: int,
    ) -> List[Dict[str, Any]]:
        """DEPRECATED: Use ArgumentExtractionEngine instead.
        
        This method is kept for backward compatibility but should not be used.
        All argument extraction now goes through ArgumentExtractionEngine which
        provides LLM-powered extraction with structured schema validation.
        
        Args:
            text: Case document text
            issue_name: Issue name for context
            case_id: Case ID
            
        Returns:
            Empty list (deprecated)
        """
        logger.warning(
            "KnowledgeGraphBuilder._extract_arguments_from_text() is deprecated. "
            "Use ArgumentExtractionEngine instead."
        )
        return []

    @staticmethod
    def _extract_outcome_from_case(db: Session, case_id: int) -> str:
        """Extract outcome from case
        
        Args:
            db: Database session
            case_id: Case ID
            
        Returns:
            Outcome string
        """
        try:
            doc = db.query(CaseDocument).filter(
                CaseDocument.case_id == case_id,
                CaseDocument.document_type == "Judgment"
            ).first()
            
            if doc and doc.remedies:
                remedies = doc.remedies
                if isinstance(remedies, str):
                    remedies = json.loads(remedies)
                if isinstance(remedies, dict) and "outcome" in remedies:
                    return remedies["outcome"]
            
            return "unknown"
            
        except Exception as e:
            logger.error(f"Failed to extract outcome: {str(e)}")
            return "unknown"

