import json
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.base import Base
from db.models.cases import Case
from db.models.analytics import CaseEmbedding
from core.precedent_drift import compute_embedding_stats, detect_drift, retrain_embeddings


def make_vec(dim, val):
    return [float(val) for _ in range(dim)]


def test_detects_drift_and_retrains(monkeypatch, tmp_path):
    # in-memory DB
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    # create cases and embeddings (baseline cluster around 1.0)
    dim = 8
    cases = []
    for i in range(10):
        c = Case(user_id=1, case_number=f"C{i}", case_type="civil", jurisdiction="X")
        db.add(c)
        db.commit()
        db.refresh(c)
        cases.append(c)
        emb = CaseEmbedding(
            case_id=c.id,
            document_id=None,
            embedding_vector=json.dumps(make_vec(dim, 1.0)),
            embedding_model="test",
            embedding_dimension=dim,
            case_type=c.case_type,
            jurisdiction=c.jurisdiction,
        )
        db.add(emb)
    db.commit()

    stats = compute_embedding_stats(db, sample_size=50)
    assert stats and stats.get("mean_sim") is not None

    stats_file = str(tmp_path / "baseline_stats.json")

    # initialize baseline by calling detect_drift (it will save baseline)
    res = detect_drift(db, stats_path=stats_file)
    assert res.get("reason") == "baseline_initialized"

    # Now simulate drift: replace embeddings with opposite vectors
    all_embeddings = db.query(CaseEmbedding).all()
    for e in all_embeddings:
        e.embedding_vector = json.dumps(make_vec(dim, 0.0))
        db.add(e)
    db.commit()

    drift_res = detect_drift(db, threshold_drop=0.01, stats_path=stats_file)
    assert drift_res.get("drift") is True

    # Monkeypatch EmbeddingEngine to avoid external API calls during retrain
    class FakeEngine:
        def __init__(self, model=None):
            pass

        def embed_multiple_cases(self, db_sess, case_ids, force_regenerate=False):
            # restore embeddings to neutral 0.5 vectors to simulate retrain
            for cid in case_ids:
                emb = db_sess.query(CaseEmbedding).filter(CaseEmbedding.case_id == cid).first()
                if emb:
                    emb.embedding_vector = json.dumps(make_vec(dim, 0.5))
                    db_sess.add(emb)
            db_sess.commit()

    # Inject a fake core.embedding_engine module to avoid importing the real one
    import types, sys

    fake_mod = types.ModuleType("core.embedding_engine")
    fake_mod.EmbeddingEngine = FakeEngine
    sys.modules["core.embedding_engine"] = fake_mod

    retrain_res = retrain_embeddings(db, model="fake", batch_size=4)
    assert retrain_res.get("retrained") is True
    assert retrain_res.get("stats") and retrain_res["stats"]["count"] > 0
