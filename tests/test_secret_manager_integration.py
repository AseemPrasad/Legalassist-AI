import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from db.base import Base
from core.secrets import SecretStore, LocalKeyManager
def test_secret_manager_prefers_env_then_db(tmp_path, monkeypatch):
    # import utils.secret_manager after we set up a test DB to avoid importing large top-level models
    # prepare env
    monkeypatch.setenv("JWT_SECRET", "env-jwt-secret")

    import importlib

    sm = importlib.import_module("utils.secret_manager")
    assert sm.get_secret("jwt_secret") == "env-jwt-secret"

    # remove env and use DB
    monkeypatch.delenv("JWT_SECRET", raising=False)

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    key_path = tmp_path / "master.key"
    mgr = LocalKeyManager(str(key_path))
    master = mgr.get_or_create_key()

    store = SecretStore(db, master_key=master)
    store.set_secret("jwt_secret", "db-jwt-secret", rotated_by="tester", reason="init")

    # monkeypatch SessionLocal used by secret_manager to use our DB
    import types, sys
    fake_db_mod = types.SimpleNamespace(SessionLocal=Session)
    sys.modules["database"] = fake_db_mod
    # ensure secret manager uses same master key file
    monkeypatch.setenv("SECRETS_MASTER_KEY_PATH", str(key_path))

    try:
        # reload module to pick up patched SessionLocal
        importlib.reload(sm)
        assert sm.get_secret("jwt_secret") == "db-jwt-secret"
    finally:
        db.close()
