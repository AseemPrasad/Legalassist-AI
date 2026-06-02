"""
Embedding Engine for Case Search
Generates semantic embeddings for case documents using OpenAI or other models.
"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from datetime import datetime, timezone

import openai
from sqlalchemy.orm import Session

from database import (
    CaseEmbedding,
    CaseDocument,
    Case,
    SessionLocal,
)
from config import Config
from core.vector_store import ShardedVectorStore
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Generate and manage embeddings for case documents"""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
    ):
        self.model = model
        self.dimension = dimension
        self._client = openai.OpenAI(
            api_key=getattr(Config, "OPENAI_API_KEY", None) or getattr(Config, "OPENROUTER_API_KEY", "")
        )

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats, or None if failed
        """
        try:
            # Clean and prepare text
            text = text.strip()
            if not text:
                logger.warning("Empty text provided for embedding")
                return None
            
            # Truncate to max tokens (roughly 8000 tokens for small model)
            # Estimate: ~4 chars per token
            max_chars = 8000 * 4
            if len(text) > max_chars:
                logger.warning(f"Text truncated from {len(text)} to {max_chars} chars")
                text = text[:max_chars]
            
            response = self._client.embeddings.create(
                model=self.model,
                input=text,
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return None

    def embed_case(
        self,
        db: Session,
        case_id: int,
        document_id: Optional[int] = None,
        force_regenerate: bool = False,
    ) -> Optional[CaseEmbedding]:
        """Generate and store embedding for a case
        
        Args:
            db: Database session
            case_id: Case ID to embed
            document_id: Specific document to embed (default: use first judgment)
            force_regenerate: Regenerate even if exists
            
        Returns:
            CaseEmbedding object or None if failed
        """
        try:
            # Get case
            case = db.query(Case).filter(Case.id == case_id).first()
            if not case:
                logger.error(f"Case {case_id} not found")
                return None
            
            # Check if embedding already exists
            existing = db.query(CaseEmbedding).filter(CaseEmbedding.case_id == case_id).first()
            if existing and not force_regenerate:
                logger.info(f"Embedding for case {case_id} already exists")
                return existing
            
            # Get document to embed
            if document_id:
                doc = db.query(CaseDocument).filter(CaseDocument.id == document_id).first()
            else:
                # Use first judgment document
                doc = db.query(CaseDocument).filter(
                    CaseDocument.case_id == case_id,
                    CaseDocument.document_type == "Judgment"
                ).first()
            
            if not doc or not doc.document_content:
                logger.warning(f"No judgment document found for case {case_id}")
                return None
            
            # Generate embedding
            embedding_vector = self.generate_embedding(doc.document_content)
            if embedding_vector is None:
                logger.error(f"Failed to generate embedding for case {case_id}")
                return None
            
            # Determine outcome from case status or metadata
            outcome = self._extract_outcome(case, doc)
            
            # Create or update embedding record
            if existing and force_regenerate:
                existing.embedding_vector = json.dumps(embedding_vector)
                existing.embedding_model = self.model
                existing.embedding_dimension = self.dimension
                existing.outcome = outcome
                existing.updated_at = datetime.now(timezone.utc)
                db.commit()
                db.refresh(existing)
                return existing
            
            embedding_obj = CaseEmbedding(
                case_id=case_id,
                document_id=document_id or doc.id,
                embedding_vector=json.dumps(embedding_vector),
                embedding_model=self.model,
                embedding_dimension=self.dimension,
                case_type=case.case_type,
                jurisdiction=case.jurisdiction,
                outcome=outcome,
            )
            
            db.add(embedding_obj)
            db.commit()
            db.refresh(embedding_obj)
            
            logger.info(f"Created embedding for case {case_id}")
            return embedding_obj
            
        except Exception as e:
            logger.error(f"Failed to embed case {case_id}: {str(e)}")
            db.rollback()
            return None

    def embed_multiple_cases(
        self,
        db: Session,
        case_ids: List[int],
        force_regenerate: bool = False,
    ) -> Dict[int, Optional[CaseEmbedding]]:
        """Generate embeddings for multiple cases
        
        Args:
            db: Database session
            case_ids: List of case IDs
            force_regenerate: Regenerate existing embeddings
            
        Returns:
            Dict mapping case_id -> CaseEmbedding
        """
        results = {}
        # Batch configuration
        batch_size = int(getattr(Config, "EMBEDDING_BATCH_SIZE", 16))
        concurrency = int(getattr(Config, "EMBEDDING_CONCURRENCY", 4))

        vector_store = get_vector_store(num_shards=int(getattr(Config, "VECTOR_SHARDS", 4)), dimension=self.dimension)

        # Helper to process a single case id (used by threads)
        def process_case(cid: int) -> Tuple[int, Optional[CaseEmbedding], Optional[List[float]]]:
            emb_obj = self.embed_case(db, cid, force_regenerate=force_regenerate)
            if emb_obj:
                vec = None
                try:
                    vec = json.loads(emb_obj.embedding_vector)
                except Exception:
                    vec = None
                return cid, emb_obj, vec
            return cid, None, None

        # Use ThreadPoolExecutor to parallelize embedding generation within batches
        for i in range(0, len(case_ids), batch_size):
            batch = case_ids[i : i + batch_size]
            futures = []
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                for cid in batch:
                    futures.append(ex.submit(process_case, cid))

                to_add = []  # items for vector store
                for fut in as_completed(futures):
                    try:
                        cid, emb_obj, vec = fut.result()
                        results[cid] = emb_obj
                        if vec is not None:
                            to_add.append((cid, vec))
                    except Exception as e:
                        logger.error(f"Embedding generation failed in batch: {e}")

                # Push batch to vector store
                if to_add:
                    try:
                        vector_store.add_batch(to_add)
                    except Exception as e:
                        logger.exception("Failed to add batch to vector store: %s", e)

        return results

    @staticmethod
    def _extract_outcome(case: Case, doc: CaseDocument) -> Optional[str]:
        """Extract outcome from case or document metadata
        
        Args:
            case: Case object
            doc: CaseDocument object
            
        Returns:
            Outcome string or None
        """
        try:
            # Try to extract from remedies JSON
            if doc.remedies:
                if isinstance(doc.remedies, str):
                    remedies = json.loads(doc.remedies)
                else:
                    remedies = doc.remedies
                
                if isinstance(remedies, dict):
                    if "outcome" in remedies:
                        return remedies["outcome"]
                    if "case_result" in remedies:
                        return remedies["case_result"]
            
            # Fallback: infer from case status
            if case.status.value == "closed":
                return "closed"
            elif case.status.value == "appealed":
                return "appealed"
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not extract outcome: {str(e)}")
            return None

    @staticmethod
    def embedding_to_array(embedding_json: str) -> np.ndarray:
        """Convert JSON embedding to numpy array
        
        Args:
            embedding_json: JSON-encoded embedding vector
            
        Returns:
            Numpy array
        """
        try:
            vector = json.loads(embedding_json)
            return np.array(vector, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to parse embedding: {str(e)}")
            return None

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors
        
        Args:
            vec1, vec2: Numpy arrays
            
        Returns:
            Similarity score (0-1)
        """
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            # Normalize to 0-1 range (cosine similarity is -1 to 1)
            return max(0.0, min(1.0, (similarity + 1) / 2))
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {str(e)}")
            return 0.0


def get_embedding_engine(
    model: str = "text-embedding-3-small",
    dimension: int = 1536,
) -> EmbeddingEngine:
    """Factory function to get embedding engine instance
    
    Args:
        model: Embedding model
        dimension: Embedding dimension
        
    Returns:
        EmbeddingEngine instance
    """
    return EmbeddingEngine(model=model, dimension=dimension)


# Default singleton vector store for the application
_vector_store: Optional[ShardedVectorStore] = None


def get_vector_store(num_shards: int = 4, dimension: int = 1536) -> ShardedVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = ShardedVectorStore(num_shards=num_shards, dimension=dimension)
    return _vector_store
