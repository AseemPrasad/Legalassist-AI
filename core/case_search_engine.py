"""
Semantic Case Search Engine
Finds similar cases based on semantic similarity of embeddings.
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np

from sqlalchemy.orm import Session

from database import (
    CaseEmbedding,
    Case,
)
from core.embedding_engine import EmbeddingEngine

logger = logging.getLogger(__name__)


class SemanticCaseSearch:
    """Search for similar cases using semantic embeddings"""

    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None):
        """Initialize search engine
        
        Args:
            embedding_engine: Embedding engine instance (created if not provided)
        """
        self.embedding_engine = embedding_engine or EmbeddingEngine()

    def search_similar_cases(
        self,
        db: Session,
        case_id: int,
        limit: int = 10,
        min_similarity: float = 0.5,
        filter_case_type: Optional[str] = None,
        filter_jurisdiction: Optional[str] = None,
        filter_outcome: Optional[str] = None,
        exclude_same_user: bool = True,
    ) -> List[Dict[str, Any]]:
        """Find similar cases to the given case
        
        Args:
            db: Database session
            case_id: Case ID to search for similar cases
            limit: Maximum number of results
            min_similarity: Minimum similarity score (0-1)
            filter_case_type: Filter by case type (optional)
            filter_jurisdiction: Filter by jurisdiction (optional)
            filter_outcome: Filter by outcome (optional)
            exclude_same_user: Exclude cases from same user
            
        Returns:
            List of dicts with case info and similarity score
        """
        try:
            # Get query case
            query_case = db.query(Case).filter(Case.id == case_id).first()
            if not query_case:
                logger.error(f"Case {case_id} not found")
                return []
            
            # Get query case embedding
            query_embedding_obj = db.query(CaseEmbedding).filter(
                CaseEmbedding.case_id == case_id
            ).first()
            
            if not query_embedding_obj:
                logger.warning(f"No embedding found for case {case_id}. Generating...")
                query_embedding_obj = self.embedding_engine.embed_case(db, case_id)
            
            if not query_embedding_obj:
                logger.error(f"Failed to get embedding for case {case_id}")
                return []
            
            # Parse query embedding
            query_vector = self.embedding_engine.embedding_to_array(
                query_embedding_obj.embedding_vector
            )
            if query_vector is None:
                return []
            
            # Check if database dialect is PostgreSQL
            if db.bind.dialect.name == "postgresql":
                # For PostgreSQL, we perform a database-level similarity search using pgvector's cosine_distance
                # similarity = 1 - cosine_distance / 2
                max_distance = 2.0 * (1.0 - min_similarity)
                query = db.query(
                    CaseEmbedding,
                    CaseEmbedding.embedding_vector.cosine_distance(query_vector.tolist()).label("distance")
                ).filter(
                    CaseEmbedding.case_id != case_id,
                    CaseEmbedding.embedding_vector.cosine_distance(query_vector.tolist()) <= max_distance
                )
                
                # Apply filters
                if filter_case_type:
                    query = query.filter(CaseEmbedding.case_type == filter_case_type)
                else:
                    query = query.filter(CaseEmbedding.case_type == query_case.case_type)
                
                if filter_jurisdiction:
                    query = query.filter(CaseEmbedding.jurisdiction == filter_jurisdiction)
                
                if filter_outcome:
                    query = query.filter(CaseEmbedding.outcome == filter_outcome)
                
                # Order by distance (similarity descending)
                query = query.order_by("distance")
                
                # Skip same user if requested
                if exclude_same_user:
                    query = query.join(Case, CaseEmbedding.case_id == Case.id).filter(Case.user_id != query_case.user_id)
                
                embeddings_with_dist = query.limit(limit).all()
                
                results = []
                for embedding_obj, distance in embeddings_with_dist:
                    similarity = 1.0 - (float(distance) / 2.0)
                    candidate_case = embedding_obj.case
                    result = {
                        "case_id": candidate_case.id,
                        "case_number": candidate_case.case_number,
                        "case_type": candidate_case.case_type,
                        "jurisdiction": candidate_case.jurisdiction,
                        "title": candidate_case.title,
                        "status": candidate_case.status.value if hasattr(candidate_case.status, 'value') else str(candidate_case.status),
                        "outcome": embedding_obj.outcome,
                        "similarity_score": round(similarity, 3),
                        "created_at": candidate_case.created_at.isoformat(),
                    }
                    
                    doc = embedding_obj.document
                    if doc and doc.summary:
                        result["summary"] = doc.summary
                    
                    results.append(result)
                return results

            # Fallback path for SQLite or other dialects:
            # Get all case embeddings (excluding query case)
            query = db.query(CaseEmbedding).filter(CaseEmbedding.case_id != case_id)
            
            # Apply filters
            if filter_case_type:
                query = query.filter(CaseEmbedding.case_type == filter_case_type)
            else:
                # Default: same case type
                query = query.filter(CaseEmbedding.case_type == query_case.case_type)
            
            if filter_jurisdiction:
                query = query.filter(CaseEmbedding.jurisdiction == filter_jurisdiction)
            
            if filter_outcome:
                query = query.filter(CaseEmbedding.outcome == filter_outcome)
            
            embeddings = query.all()
            
            # Calculate similarity for each embedding
            results = []
            for embedding_obj in embeddings:
                # Parse embedding
                candidate_vector = self.embedding_engine.embedding_to_array(
                    embedding_obj.embedding_vector
                )
                if candidate_vector is None:
                    continue
                
                # Calculate similarity
                similarity = self.embedding_engine.cosine_similarity(
                    query_vector, candidate_vector
                )
                
                if similarity < min_similarity:
                    continue
                
                # Get case details
                candidate_case = embedding_obj.case
                
                # Skip same user if requested
                if exclude_same_user and candidate_case.user_id == query_case.user_id:
                    continue
                
                # Build result
                result = {
                    "case_id": candidate_case.id,
                    "case_number": candidate_case.case_number,
                    "case_type": candidate_case.case_type,
                    "jurisdiction": candidate_case.jurisdiction,
                    "title": candidate_case.title,
                    "status": candidate_case.status.value if hasattr(candidate_case.status, 'value') else str(candidate_case.status),
                    "outcome": embedding_obj.outcome,
                    "similarity_score": round(similarity, 3),
                    "created_at": candidate_case.created_at.isoformat(),
                }
                
                # Add summary if available
                doc = embedding_obj.document
                if doc and doc.summary:
                    result["summary"] = doc.summary
                
                results.append(result)
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Return top N
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching similar cases: {str(e)}")
            return []

    def search_by_text(
        self,
        db: Session,
        search_text: str,
        limit: int = 10,
        min_similarity: float = 0.5,
        filter_case_type: Optional[str] = None,
        filter_jurisdiction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar cases by free text
        
        Args:
            db: Database session
            search_text: Text to search for
            limit: Maximum number of results
            min_similarity: Minimum similarity score
            filter_case_type: Filter by case type (optional)
            filter_jurisdiction: Filter by jurisdiction (optional)
            
        Returns:
            List of similar cases
        """
        try:
            # Generate embedding for search text
            search_vector = self.embedding_engine.generate_embedding(search_text)
            if search_vector is None:
                logger.error("Failed to generate embedding for search text")
                return []
            
            # Check if database dialect is PostgreSQL
            if db.bind.dialect.name == "postgresql":
                # For PostgreSQL, we perform a database-level similarity search using pgvector's cosine_distance
                # similarity = 1 - cosine_distance / 2
                max_distance = 2.0 * (1.0 - min_similarity)
                query = db.query(
                    CaseEmbedding,
                    CaseEmbedding.embedding_vector.cosine_distance(search_vector).label("distance")
                ).filter(
                    CaseEmbedding.embedding_vector.cosine_distance(search_vector) <= max_distance
                )
                
                if filter_case_type:
                    query = query.filter(CaseEmbedding.case_type == filter_case_type)
                
                if filter_jurisdiction:
                    query = query.filter(CaseEmbedding.jurisdiction == filter_jurisdiction)
                
                # Order by distance (similarity descending)
                query = query.order_by("distance")
                
                embeddings_with_dist = query.limit(limit).all()
                
                results = []
                for embedding_obj, distance in embeddings_with_dist:
                    similarity = 1.0 - (float(distance) / 2.0)
                    case = embedding_obj.case
                    result = {
                        "case_id": case.id,
                        "case_number": case.case_number,
                        "case_type": case.case_type,
                        "jurisdiction": case.jurisdiction,
                        "title": case.title,
                        "status": case.status.value if hasattr(case.status, 'value') else str(case.status),
                        "outcome": embedding_obj.outcome,
                        "similarity_score": round(similarity, 3),
                        "created_at": case.created_at.isoformat(),
                    }
                    
                    doc = embedding_obj.document
                    if doc and doc.summary:
                        result["summary"] = doc.summary
                    
                    results.append(result)
                return results

            # Fallback path for SQLite or other dialects:
            search_array = np.array(search_vector, dtype=np.float32)
            
            # Get all case embeddings
            query = db.query(CaseEmbedding)
            
            if filter_case_type:
                query = query.filter(CaseEmbedding.case_type == filter_case_type)
            
            if filter_jurisdiction:
                query = query.filter(CaseEmbedding.jurisdiction == filter_jurisdiction)
            
            embeddings = query.all()
            
            # Calculate similarity for each embedding
            results = []
            for embedding_obj in embeddings:
                candidate_vector = self.embedding_engine.embedding_to_array(
                    embedding_obj.embedding_vector
                )
                if candidate_vector is None:
                    continue
                
                similarity = self.embedding_engine.cosine_similarity(
                    search_array, candidate_vector
                )
                
                if similarity < min_similarity:
                    continue
                
                case = embedding_obj.case
                result = {
                    "case_id": case.id,
                    "case_number": case.case_number,
                    "case_type": case.case_type,
                    "jurisdiction": case.jurisdiction,
                    "title": case.title,
                    "status": case.status.value if hasattr(case.status, 'value') else str(case.status),
                    "outcome": embedding_obj.outcome,
                    "similarity_score": round(similarity, 3),
                    "created_at": case.created_at.isoformat(),
                }
                
                doc = embedding_obj.document
                if doc and doc.summary:
                    result["summary"] = doc.summary
                
                results.append(result)
            
            # Sort by similarity
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in text search: {str(e)}")
            return []

    def get_statistics(self, db: Session) -> Dict[str, Any]:
        """Get statistics about indexed cases
        
        Args:
            db: Database session
            
        Returns:
            Dict with statistics
        """
        try:
            total_embeddings = db.query(CaseEmbedding).count()
            
            # Group by case type
            case_types = db.query(CaseEmbedding.case_type).distinct().all()
            case_type_counts = {}
            for ct in case_types:
                count = db.query(CaseEmbedding).filter(
                    CaseEmbedding.case_type == ct[0]
                ).count()
                case_type_counts[ct[0]] = count
            
            # Group by jurisdiction
            jurisdictions = db.query(CaseEmbedding.jurisdiction).distinct().all()
            jurisdiction_counts = {}
            for j in jurisdictions:
                count = db.query(CaseEmbedding).filter(
                    CaseEmbedding.jurisdiction == j[0]
                ).count()
                jurisdiction_counts[j[0]] = count
            
            return {
                "total_indexed_cases": total_embeddings,
                "case_types": case_type_counts,
                "jurisdictions": jurisdiction_counts,
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {}


def get_search_engine(
    db: Session,
    embedding_engine: Optional[EmbeddingEngine] = None,
) -> SemanticCaseSearch:
    """Factory function to get search engine instance
    
    Args:
        db: Database session
        embedding_engine: Embedding engine instance (optional)
        
    Returns:
        SemanticCaseSearch instance
    """
    return SemanticCaseSearch(embedding_engine=embedding_engine)
