import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from pathlib import Path

from db.base import Base
from db.models.cases import Case, CaseDocument
from db.models.analytics import CaseEmbedding

from core.vector_store import ShardedVectorStore


def test_document_pipeline_ocr_to_embedding_and_vector_persist(tmp_path):
    # Setup in-memory DB
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        # Create a case and an empty document (simulating an uploaded scanned PDF)
        c = Case(user_id=1, case_number="INT-1", case_type="civil", jurisdiction="X")
        db.add(c)
        db.commit()
        db.refresh(c)

        doc = CaseDocument(case_id=c.id, document_type="Judgment", document_content="")
        db.add(doc)
        db.commit()
        db.refresh(doc)

        # Simulate OCR output
        ocr_text = "This is the extracted OCR text for testing the integration pipeline."
        doc.document_content = ocr_text
        db.add(doc)
        db.commit()

        # Use a fake embedding engine that writes CaseEmbedding and pushes to vector store
        dimension = 8
        vs = ShardedVectorStore(num_shards=2, dimension=dimension)

        class FakeEmbeddingEngine:
            def __init__(self, model=None, dimension=dimension):
                self.dimension = dimension

            def generate_embedding(self, text: str):
                # deterministic fake embedding
                base = sum(ord(c) for c in text) % 10
                return [float((i + base) % 7 + 0.1) for i in range(self.dimension)]

            def embed_case(self, db_sess, case_id, document_id=None, force_regenerate=False):
                case = db_sess.query(Case).filter(Case.id == case_id).first()
                if not case:
                    return None
                doc = db_sess.query(CaseDocument).filter(CaseDocument.case_id == case_id).first()
                if not doc or not doc.document_content:
                    return None
                vec = self.generate_embedding(doc.document_content)
                emb = CaseEmbedding(
                    case_id=case_id,
                    document_id=doc.id,
                    embedding_vector=json.dumps(vec),
                    embedding_model="fake",
                    embedding_dimension=self.dimension,
                    case_type=case.case_type,
                    jurisdiction=case.jurisdiction,
                )
                db_sess.add(emb)
                db_sess.commit()
                db_sess.refresh(emb)
                # push to vector store
                vs.add_batch([(case_id, vec)])
                return emb

        engine = FakeEmbeddingEngine()
        emb_obj = engine.embed_case(db, c.id, document_id=doc.id, force_regenerate=True)
        assert emb_obj is not None

        # Verify CaseEmbedding persisted
        stored = db.query(CaseEmbedding).filter(CaseEmbedding.case_id == c.id).first()
        assert stored is not None
        vec = json.loads(stored.embedding_vector)
        assert len(vec) == dimension

        # Verify vector store has the vector (by checking ids in shard)
        shard = vs.shard_for_id(c.id)
        assert c.id in vs._shards[shard]['ids']
    finally:
        db.close()
