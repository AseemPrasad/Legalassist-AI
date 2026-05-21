import os
import pytest
from datetime import datetime, timedelta, timezone

# Setup environment variables before imports to satisfy configuration requirements
os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ["CELERY_BROKER_URL"] = "redis://localhost:6379"
os.environ["CELERY_RESULT_BACKEND"] = "redis://localhost:6379"
os.environ["APP_ALLOWED_HOSTS"] = "localhost"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-that-is-long-enough"

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.base import Base
from db.models import APIKey, User
from api.auth import (
    get_current_user,
    create_api_key_record,
    CurrentUser,
    verify_api_key,
)
import api.auth
import database
import db.session as db_session_module

# In-memory SQLite for isolated testing (sharing connection via StaticPool)
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(autouse=True)
def setup_database(monkeypatch):
    # Re-create all tables for each test to ensure isolation
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    
    # Patch SessionLocal to direct DB queries in our test database
    monkeypatch.setattr(api.auth, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(database, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(db_session_module, "SessionLocal", TestingSessionLocal)
    
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Test application wrapping get_current_user dependency
app = FastAPI()


@app.get("/protected")
def protected_route(current_user: CurrentUser = Depends(get_current_user)):
    return {
        "user_id": current_user.user_id,
        "email": current_user.email,
        "role": current_user.role,
    }


client = TestClient(app)


def test_api_key_valid_without_user(setup_database):
    db = setup_database
    combined_key, record = create_api_key_record(db, name="Test Key No User")
    
    response = client.get("/protected", headers={"Authorization": f"Bearer {combined_key}"})
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 0
    assert data["email"] == "api_user"
    assert data["role"] == "api"


def test_api_key_valid_linked_to_user(setup_database):
    db = setup_database
    # Create test user in DB
    user = User(email="test_user@example.com", is_verified=True, is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    
    combined_key, record = create_api_key_record(db, name="User Key", user_id=user.id)
    
    response = client.get("/protected", headers={"Authorization": f"Bearer {combined_key}"})
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == user.id
    assert data["email"] == "test_user@example.com"
    assert data["role"] == "user"


def test_api_key_valid_linked_to_admin(setup_database):
    db = setup_database
    # Create admin user in DB
    admin = User(email="admin@example.com", is_verified=True, is_admin=True)
    db.add(admin)
    db.commit()
    db.refresh(admin)
    
    combined_key, record = create_api_key_record(db, name="Admin Key", user_id=admin.id)
    
    response = client.get("/protected", headers={"Authorization": f"Bearer {combined_key}"})
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == admin.id
    assert data["email"] == "admin@example.com"
    assert data["role"] == "admin"


def test_api_key_invalid_format(setup_database):
    # Missing period delimiter format
    response = client.get("/protected", headers={"Authorization": "Bearer key_invalidformat"})
    assert response.status_code == 401
    assert "Invalid API key format" in response.json()["detail"]


def test_api_key_nonexistent_key_id(setup_database):
    # Prefix key_id is not in DB
    response = client.get("/protected", headers={"Authorization": "Bearer key_nonexistent.secretstuff"})
    assert response.status_code == 401
    assert "Invalid API key" in response.json()["detail"]


def test_api_key_incorrect_secret(setup_database):
    db = setup_database
    combined_key, record = create_api_key_record(db, name="Test Key")
    key_id, _ = combined_key.split(".", 1)
    
    # Valid key_id but wrong secret material
    bad_key = f"{key_id}.wrongsecret"
    response = client.get("/protected", headers={"Authorization": f"Bearer {bad_key}"})
    assert response.status_code == 401
    assert "Invalid API key" in response.json()["detail"]


def test_api_key_expired(setup_database):
    db = setup_database
    secret = api.auth.generate_api_key()
    salt = "expiredsalt"
    key_hash = api.auth.hash_api_key(secret, salt)
    expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    
    key_id = "key_expired"
    key_record = APIKey(
        key_id=key_id,
        name="Expired Key",
        key_hash=key_hash,
        key_salt=salt,
        expires_at=expires_at,
        created_at=datetime.now(timezone.utc) - timedelta(days=1)
    )
    db.add(key_record)
    db.commit()
    
    combined_key = f"{key_id}.{secret}"
    response = client.get("/protected", headers={"Authorization": f"Bearer {combined_key}"})
    assert response.status_code == 401
    assert "API key has expired" in response.json()["detail"]
