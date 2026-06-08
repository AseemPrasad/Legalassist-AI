"""
Knowledge Graph Builder
Build and query a graph of Issues -> Arguments -> Outcomes.
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

logger = logging.getLogger(__name__)


SUCCESSFUL_OUTCOMES = frozenset({
    "plaintiff_won",
    "defendant_won",
    "settled",
    "settled_favorably",
    "dismissed_with_prejudice",
    "favorable_ruling",
})


class KnowledgeGraphBuilder:
    """Build and query the case knowledge graph"""

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

            successful_edges = db.query(KnowledgeGraphEdge).filter(
                KnowledgeGraphEdge.outcome.in_(SUCCESSFUL_OUTCOMES)
            ).count()

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
