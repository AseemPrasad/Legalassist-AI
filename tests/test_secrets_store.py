from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from db.base import Base
from core.secrets import SecretStore, LocalKeyManager


def test_secrets_set_get_rotate(tmp_path):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    # use separate master key file in tmp_path
    key_path = tmp_path / "master.key"
    mgr = LocalKeyManager(str(key_path))
    master = mgr.get_or_create_key()

    store = SecretStore(db, master_key=master)
    name = "TEST_API_KEY"
    val = "secret-123"
    entry = store.set_secret(name, val, rotated_by="tester", reason="initial")
    assert entry.name == name

    got = store.get_secret(name)
    assert got == val

    # rotate
    new_val = "secret-456"
    store.rotate_secret(name, new_val, rotated_by="rotator", reason="rotate for test")
    got2 = store.get_secret(name)
    assert got2 == new_val

    # delete
    assert store.delete_secret(name)
    assert store.get_secret(name) is None
