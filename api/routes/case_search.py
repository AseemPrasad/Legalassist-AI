"""
API routes for Case Search and Precedent Matching
Endpoints for finding similar cases, precedents, comparisons, and knowledge graph queries.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List, Optional


from api.auth import get_current_user
from database import User, Case
from api.dependencies import get_db_rls

# Import case search engines
from core.embedding_engine import EmbeddingEngine
from core.case_search_engine import SemanticCaseSearch
from core.precedent_matcher import PrecedentMatcher
from core.case_comparison import CaseComparison
from core.knowledge_graph import KnowledgeGraphBuilder

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cases", tags=["case-search"])

# Initialize engines
embedding_engine = EmbeddingEngine()


def _user_case_ids(current_user: User, db: Session) -> set[int]:
    return {row[0] for row in db.query(Case.id).filter(Case.user_id == current_user.user_id).all()}


def _require_owned_case(case_id: int, current_user: User, db: Session) -> Case:
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    if current_user.role != "admin" and case.user_id != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")

    return case


# ==================== Case Search Endpoints ====================

@router.get("/{case_id}/search-similar")
def search_similar_cases(
    case_id: int,
    limit: int = Query(10, ge=1, le=50),
    min_similarity: float = Query(0.5, ge=0, le=1),
    case_type: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    outcome: Optional[str] = None,
    exclude_same_user: bool = Query(True),
    cross_jurisdiction: bool = Query(False),
    jurisdiction_weight: float = Query(0.2, ge=0, le=1),
    db: Session = Depends(get_db_rls),
    current_user: User = Depends(get_current_user),
):
    """
    Find similar cases based on semantic similarity
    
    Args:
        case_id: Case ID to search for similar cases
        limit: Maximum number of results (1-50)
        min_similarity: Minimum similarity score (0-1)
        case_type: Filter by case type (optional)
        jurisdiction: Filter by jurisdiction (optional)
        outcome: Filter by outcome (optional)
        exclude_same_user: Exclude cases from same user
        
    Returns:
        List of similar cases with similarity scores
    """
    _require_owned_case(case_id, current_user, db)

    try:
        search_engine = SemanticCaseSearch(embedding_engine)
        results = search_engine.search_similar_cases(
            db=db,
            case_id=case_id,
            limit=limit,
            min_similarity=min_similarity,
            filter_case_type=case_type,
            filter_jurisdiction=jurisdiction,
            filter_outcome=outcome,
            exclude_same_user=exclude_same_user,
            cross_jurisdiction=cross_jurisdiction,
            jurisdiction_weight=jurisdiction_weight,
        )
        
        return {
            "query_case_id": case_id,
            "similar_cases": results,
            "count": len(results),
        }
        
    except Exception as e:
        logger.error(f"Error searching similar cases: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search similar cases")


@router.get("/search/text")
def search_by_text(
    query: str = Query(..., min_length=10),
    limit: int = Query(10, ge=1, le=50),
    min_similarity: float = Query(0.5, ge=0, le=1),
    case_type: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    db: Session = Depends(get_db_rls),
    current_user: User = Depends(get_current_user),
):
    """
    Search for cases by free text
    
    Args:
        query: Search text (minimum 10 characters)
        limit: Maximum results
        min_similarity: Minimum similarity threshold
        case_type: Filter by case type (optional)
        jurisdiction: Filter by jurisdiction (optional)
        
    Returns:
        List of matching cases
    """
    try:
        search_engine = SemanticCaseSearch(embedding_engine)
        results = search_engine.search_by_text(
            db=db,
            search_text=query,
            limit=limit,
            min_similarity=min_similarity,
            filter_case_type=case_type,
            filter_jurisdiction=jurisdiction,
        )
        
        owned = _user_case_ids(current_user, db)
        results = [r for r in results if r.get("case_id") in owned]
        
        return {
            "query": query,
            "results": results,
            "count": len(results),
        }
        
    except Exception as e:
        logger.error(f"Error in text search: {str(e)}")
        raise HTTPException(status_code=500, detail="Search failed")


@router.get("/search/statistics")
def get_search_statistics(
    db: Session = Depends(get_db_rls),
    current_user: User = Depends(get_current_user),
):
    """Get statistics about the authenticated user's indexed cases"""
    try:
        from database import CaseEmbedding
        owned = _user_case_ids(current_user, db)
        if not owned:
            return {"total_indexed_cases": 0, "case_types": {}, "jurisdictions": {}}

        base = db.query(CaseEmbedding).filter(CaseEmbedding.case_id.in_(owned))
        total = base.count()

        case_type_counts = {}
        for row in base.with_entities(CaseEmbedding.case_type).distinct().all():
            ct = row[0]
            case_type_counts[ct] = base.filter(CaseEmbedding.case_type == ct).count()

        jurisdiction_counts = {}
        for row in base.with_entities(CaseEmbedding.jurisdiction).distinct().all():
            j = row[0]
            jurisdiction_counts[j] = base.filter(CaseEmbedding.jurisdiction == j).count()

        return {
            "total_indexed_cases": total,
            "case_types": case_type_counts,
            "jurisdictions": jurisdiction_counts,
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


# ==================== Precedent Matching Endpoints ====================

@router.get("/{case_id}/precedents/winning")
def get_winning_precedents(
    case_id: int,
    issue: Optional[str] = None,
    argument_type: Optional[str] = None,
    limit: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db_rls),
    current_user: User = Depends(get_current_user),
):
    """
    Find precedent cases where similar arguments won
    
    Args:
        case_id: Case ID to find precedents for
        issue: Filter by specific issue (optional)
        argument_type: Filter by argument type (optional)
        limit: Maximum results
        
    Returns:
        List of precedent cases with winning arguments
    """
    _require_owned_case(case_id, current_user, db)

    try:
        results = PrecedentMatcher.find_winning_precedents(
            db=db,
            case_id=case_id,
            issue_name=issue,
            argument_type=argument_type,
            limit=limit,
        )
        
        return {
            "case_id": case_id,
            "winning_precedents": results,
            "count": len(results),
        }
        
    except Exception as e:
        logger.error(f"Error finding winning precedents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to find precedents")


@router.get("/{case_id}/precedents/losing")
def get_losing_precedents(
    case_id: int,
    issue: Optional[str] = None,
    limit: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db_rls),
    current_user: User = Depends(get_current_user),
):
    """
    Find precedent cases where similar arguments failed
    
    Args:
        case_id: Case ID
        issue: Filter by issue (optional)
        limit: Maximum results
        
    Returns:
        List of cases to avoid based on failed arguments
    """
    _require_owned_case(case_id, current_user, db)

    try:
        results = PrecedentMatcher.find_losing_precedents(
            db=db,
            case_id=case_id,
            issue_name=issue,
            limit=limit,
        )
        
        return {
            "case_id": case_id,
            "losing_precedents": results,
            "count": len(results),
        }
        
    except Exception as e:
        logger.error(f"Error finding losing precedents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to find precedents")


@router.get("/argument-analysis/success-rate")
def get_argument_success_rate(
    argument: str = Query(..., min_length=10),
    issue: Optional[str] = None,
    db: Session = Depends(get_db_rls),
    current_user: User = Depends(get_current_user),
):
    """
    Get success rate of a specific argument
    
    Args:
        argument: The argument text to analyze
        issue: Filter by issue (optional)
        
    Returns:
        Success statistics for the argument
    """
    try:
        from database import CaseArgument, CaseIssue
        owned = _user_case_ids(current_user, db)
        if not owned:
            return {"success_rate": 0, "total_uses": 0, "successful": 0, "failed": 0}

        query = db.query(CaseArgument).filter(
            CaseArgument.argument_text == argument,
            CaseArgument.case_id.in_(owned),
        )
        if issue:
            issue_obj = db.query(CaseIssue).filter(CaseIssue.issue_name == issue).first()
            if issue_obj:
                query = query.filter(CaseArgument.issue_id == issue_obj.id)

        all_arguments = query.all()
        if not all_arguments:
            return {"success_rate": 0, "total_uses": 0, "successful": 0, "failed": 0}

        successful = sum(1 for a in all_arguments if a.argument_succeeded is True)
        failed = sum(1 for a in all_arguments if a.argument_succeeded is False)
        total = len(all_arguments)
        success_rate = (successful / total * 100) if total > 0 else 0
        return {
            "argument": argument[:100],
            "success_rate": round(success_rate, 1),
            "total_uses": total,
            "successful": successful,
            "failed": failed,
            "unknown": total - successful - failed,
        }
    except Exception as e:
        logger.error(f"Error analyzing argument: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze argument")


@router.get("/issue-analysis/arguments")
def get_arguments_by_issue(
    issue: str = Query(..., min_length=3),
    outcome: Optional[str] = None,
    db: Session = Depends(get_db_rls),
    current_user: User = Depends(get_current_user),
):
    """
    Find all arguments used for a specific issue
    
    Args:
        issue: Issue name to search for
        outcome: Filter by outcome (optional)
        
    Returns:
        List of arguments with success rates
    """
    try:
        from database import CaseArgument, CaseIssue
        owned = _user_case_ids(current_user, db)

        issue_obj = db.query(CaseIssue).filter(CaseIssue.issue_name == issue).first()
        if not issue_obj:
            return {"issue": issue, "arguments": [], "count": 0}

        query = db.query(CaseArgument).filter(
            CaseArgument.issue_id == issue_obj.id,
            CaseArgument.case_id.in_(owned),
        )
        arguments = query.all()

        result_list = []
        total = len(arguments)
        for arg in arguments:
            result_list.append({
                "argument_id": arg.id,
                "argument_text": arg.argument_text[:200],
                "argument_type": arg.argument_type,
                "succeeded": arg.argument_succeeded,
                "case_id": arg.case_id,
            })

        successful = sum(1 for a in arguments if a.argument_succeeded is True)
        failed = sum(1 for a in arguments if a.argument_succeeded is False)

        return {
            "issue": issue,
            "arguments": result_list,
            "count": len(result_list),
            "success_rate": round((successful / total * 100), 1) if total > 0 else 0,
        }
    except Exception as e:
        logger.error(f"Error finding arguments: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to find arguments")


# ==================== Case Comparison Endpoints ====================

@router.get("/{case_id}/compare/{precedent_id}")
def compare_cases(
    case_id: int,
    precedent_id: int,
    db: Session = Depends(get_db_rls),
    current_user: User = Depends(get_current_user),
):
    """
    Compare user's case with a precedent case
    
    Args:
        case_id: User's case ID
        precedent_id: Precedent case ID to compare with
        
    Returns:
        Detailed comparison including issues, arguments, and differences
    """
    _require_owned_case(case_id, current_user, db)
    _require_owned_case(precedent_id, current_user, db)

    try:
        comparison = CaseComparison.compare_cases(db, case_id, precedent_id)
        return comparison
    except Exception as e:
        logger.error(f"Error comparing cases: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to compare cases")


@router.get("/{case_id}/comparison/{precedent_id}/suggestions")
def get_comparison_suggestions(
    case_id: int,
    precedent_id: int,
    db: Session = Depends(get_db_rls),
    current_user: User = Depends(get_current_user),
):
    """
    Get suggested legal arguments based on precedent comparison
    
    Args:
        case_id: User's case ID
        precedent_id: Precedent case ID
        
    Returns:
        List of suggested arguments based on winning precedents
    """
    _require_owned_case(case_id, current_user, db)
    _require_owned_case(precedent_id, current_user, db)

    try:
        suggestions = CaseComparison.suggest_arguments(db, case_id, precedent_id)
        return {
            "case_id": case_id,
            "precedent_id": precedent_id,
            "suggestions": suggestions,
            "count": len(suggestions),
        }
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate suggestions")


@router.get("/{case_id}/comparison/{precedent_id}/differences")
def get_comparison_differences(
    case_id: int,
    precedent_id: int,
    db: Session = Depends(get_db_rls),
    current_user: User = Depends(get_current_user),
):
    """
    Highlight key differences between cases
    
    Args:
        case_id: User's case ID
        precedent_id: Precedent case ID
        
    Returns:
        Highlighted differences and warnings
    """
    _require_owned_case(case_id, current_user, db)
    _require_owned_case(precedent_id, current_user, db)

    try:
        differences = CaseComparison.highlight_differences(db, case_id, precedent_id)
        return differences
    except Exception as e:
        logger.error(f"Error highlighting differences: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to highlight differences")


# ==================== Knowledge Graph Endpoints ====================

@router.get("/knowledge-graph/query")
def query_knowledge_graph(
    issue: str = Query(..., min_length=3),
    outcome: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db_rls),
    current_user: User = Depends(get_current_user),
):
    """
    Query the knowledge graph for cases matching criteria
    
    Args:
        issue: Issue name to search for
        outcome: Desired outcome (optional filter)
        limit: Maximum results
        
    Returns:
        List of cases matching the query
    """
    try:
        from database import CaseIssue, KnowledgeGraphEdge

        owned = _user_case_ids(current_user, db)
        issues = db.query(CaseIssue).filter(CaseIssue.issue_name.ilike(f"%{issue}%")).all()
        results = []

        for iss in issues:
            query = db.query(KnowledgeGraphEdge).filter(
                KnowledgeGraphEdge.issue_id == iss.id,
                KnowledgeGraphEdge.case_id.in_(owned),
            )
            if outcome:
                query = query.filter(KnowledgeGraphEdge.outcome == outcome)
            edges = query.all()

            for edge in edges:
                case = edge.case
                argument = edge.argument
                results.append({
                    "case_id": case.id,
                    "case_number": case.case_number,
                    "case_type": case.case_type,
                    "jurisdiction": case.jurisdiction,
                    "issue": iss.issue_name,
                    "argument": argument.argument_text[:200] if argument else None,
                    "argument_succeeded": argument.argument_succeeded if argument else None,
                    "outcome": edge.outcome,
                    "weight": float(edge.weight) if edge.weight else 1.0,
                    "case_created_at": case.created_at.isoformat(),
                })

        results.sort(key=lambda x: x["weight"], reverse=True)
        return {
            "query": {
                "issue": issue,
                "outcome": outcome,
            },
            "results": results[:limit],
            "count": min(len(results), limit),
        }
    except Exception as e:
        logger.error(f"Error querying knowledge graph: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to query knowledge graph")


@router.get("/knowledge-graph/statistics")
def get_knowledge_graph_statistics(
    db: Session = Depends(get_db_rls),
    current_user: User = Depends(get_current_user),
):
    """Get statistics about the knowledge graph scoped to the authenticated user"""
    try:
        from database import CaseIssue, CaseArgument, KnowledgeGraphEdge
        owned = _user_case_ids(current_user, db)
        if not owned:
            return {
                "total_issues": 0, "total_arguments": 0, "total_edges": 0,
                "successful_edges": 0, "issue_argument_counts": {},
            }

        total_issues = db.query(CaseIssue).filter(
            CaseIssue.id.in_(
                db.query(KnowledgeGraphEdge.issue_id).filter(
                    KnowledgeGraphEdge.case_id.in_(owned)
                ).distinct().subquery()
            )
        ).count()

        total_arguments = db.query(CaseArgument).filter(
            CaseArgument.case_id.in_(owned)
        ).count()

        edge_query = db.query(KnowledgeGraphEdge).filter(
            KnowledgeGraphEdge.case_id.in_(owned)
        )
        total_edges = edge_query.count()
        successful_edges = edge_query.filter(
            KnowledgeGraphEdge.outcome.in_(["plaintiff_won", "defendant_won"])
        ).count()

        issue_arg_counts = {}
        issues = db.query(CaseIssue).filter(
            CaseIssue.id.in_(
                db.query(KnowledgeGraphEdge.issue_id).filter(
                    KnowledgeGraphEdge.case_id.in_(owned)
                ).distinct().subquery()
            )
        ).all()
        for iss in issues:
            arg_count = db.query(CaseArgument).filter(
                CaseArgument.issue_id == iss.id,
                CaseArgument.case_id.in_(owned),
            ).count()
            if arg_count > 0:
                issue_arg_counts[iss.issue_name] = arg_count

        return {
            "total_issues": total_issues,
            "total_arguments": total_arguments,
            "total_edges": total_edges,
            "successful_edges": successful_edges,
            "issue_argument_counts": issue_arg_counts,
        }
    except Exception as e:
        logger.error(f"Error getting graph statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


# ==================== Indexing/Management Endpoints ====================

@router.post("/{case_id}/index")
def index_case(
    case_id: int,
    force_regenerate: bool = Query(False),
    db: Session = Depends(get_db_rls),
    current_user: User = Depends(get_current_user),
):
    """
    Index a case for semantic search
    
    Args:
        case_id: Case ID to index
        force_regenerate: Force regeneration of embedding
        
    Returns:
        Indexing status
    """
    _require_owned_case(case_id, current_user, db)

    try:
        embedding_obj = embedding_engine.embed_case(
            db=db,
            case_id=case_id,
            force_regenerate=force_regenerate,
        )
        
        if not embedding_obj:
            raise HTTPException(status_code=400, detail="Failed to index case")
        
        return {
            "case_id": case_id,
            "indexed": True,
            "model": embedding_obj.embedding_model,
        }
        
    except Exception as e:
        logger.error(f"Error indexing case: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to index case")


@router.post("/{case_id}/extract-issues")
def extract_case_issues(
    case_id: int,
    override_existing: bool = Query(False),
    db: Session = Depends(get_db_rls),
    current_user: User = Depends(get_current_user),
):
    """
    Extract issues from a case document
    
    Args:
        case_id: Case ID
        override_existing: Replace existing issues
        
    Returns:
        List of extracted issues
    """
    _require_owned_case(case_id, current_user, db)

    try:
        issues = KnowledgeGraphBuilder.extract_issues_from_case(
            db=db,
            case_id=case_id,
            override_existing=override_existing,
        )
        
        return {
            "case_id": case_id,
            "issues_extracted": len(issues),
            "issues": [
                {"id": i.id, "name": i.issue_name, "category": i.issue_category}
                for i in issues
            ],
        }
        
    except Exception as e:
        logger.error(f"Error extracting issues: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to extract issues")


@router.post("/{case_id}/extract-arguments")
def extract_case_arguments(
    case_id: int,
    override_existing: bool = Query(False),
    db: Session = Depends(get_db_rls),
    current_user: User = Depends(get_current_user),
):
    """
    Extract arguments from a case document
    
    Args:
        case_id: Case ID
        override_existing: Replace existing arguments
        
    Returns:
        List of extracted arguments
    """
    _require_owned_case(case_id, current_user, db)

    try:
        arguments = KnowledgeGraphBuilder.extract_arguments_from_case(
            db=db,
            case_id=case_id,
            override_existing=override_existing,
        )
        
        return {
            "case_id": case_id,
            "arguments_extracted": len(arguments),
        }
        
    except Exception as e:
        logger.error(f"Error extracting arguments: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to extract arguments")



