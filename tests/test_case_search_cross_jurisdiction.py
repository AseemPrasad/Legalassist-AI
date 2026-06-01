from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.case_search_engine import SemanticCaseSearch
from core.embedding_engine import EmbeddingEngine
from database import Base, Case, CaseEmbedding, CaseStatus, User


@pytest.fixture()
def search_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)
    db = session_factory()

    user = User(email="search@example.com")
    db.add(user)
    db.commit()
    db.refresh(user)

    now = datetime.now(timezone.utc)
    query_case = Case(
        user_id=user.id,
        case_number="CASE-Q",
        case_type="civil",
        jurisdiction="Delhi High Court",
        status=CaseStatus.ACTIVE,
        title="Query case",
        created_at=now,
    )
    candidate_same = Case(
        user_id=user.id + 1,
        case_number="CASE-SAME",
        case_type="civil",
        jurisdiction="Delhi",
        status=CaseStatus.ACTIVE,
        title="Same family case",
        created_at=now,
    )
    candidate_cross = Case(
        user_id=user.id + 2,
        case_number="CASE-CROSS",
        case_type="civil",
        jurisdiction="California",
        status=CaseStatus.ACTIVE,
        title="Cross jurisdiction case",
        created_at=now,
    )

    db.add_all([query_case, candidate_same, candidate_cross])
    db.commit()
    db.refresh(query_case)
    db.refresh(candidate_same)
    db.refresh(candidate_cross)

    db.add_all(
        [
            CaseEmbedding(
                case_id=query_case.id,
                document_id=None,
                embedding_vector="[1.0, 0.0]",
                embedding_model="test",
                embedding_dimension=2,
                case_type="civil",
                jurisdiction="Delhi High Court",
                outcome="pending",
            ),
            CaseEmbedding(
                case_id=candidate_same.id,
                document_id=None,
                embedding_vector="[0.99, 0.01]",
                embedding_model="test",
                embedding_dimension=2,
                case_type="civil",
                jurisdiction="Delhi",
                outcome="pending",
            ),
            CaseEmbedding(
                case_id=candidate_cross.id,
                document_id=None,
                embedding_vector="[0.98, 0.02]",
                embedding_model="test",
                embedding_dimension=2,
                case_type="civil",
                jurisdiction="California",
                outcome="pending",
            ),
        ]
    )
    db.commit()
    return db, query_case.id


def test_cross_jurisdiction_search_returns_related_cases(search_db):
    db, query_case_id = search_db
    search_engine = SemanticCaseSearch(EmbeddingEngine(model="test", dimension=2))

    strict_results = search_engine.search_similar_cases(
        db=db,
        case_id=query_case_id,
        limit=10,
        min_similarity=0.1,
        cross_jurisdiction=False,
    )
    assert strict_results == []

    cross_results = search_engine.search_similar_cases(
        db=db,
        case_id=query_case_id,
        limit=10,
        min_similarity=0.1,
        cross_jurisdiction=True,
        jurisdiction_weight=0.3,
    )

    assert len(cross_results) == 2
    assert cross_results[0]["confidence_score"] >= cross_results[1]["confidence_score"]
    assert all("confidence_score" in item for item in cross_results)
    assert any(item["jurisdiction"] == "California" for item in cross_results)
