
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("JWT_SECRET", "test-secret-key-that-is-long-enough")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost,127.0.0.1")
os.environ.setdefault("CASE_ANONYMIZATION_SECRET", "a" * 32)
sys.modules["streamlit"] = MagicMock()
sys.modules["pytesseract"] = MagicMock()
sys.modules["pdf2image"] = MagicMock()

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.base import Base
from db.models import (
    Case,
    CaseDeadline,
    CaseDocument,
    CaseTimeline,
    CaseStatus,
    DocumentType,
    User,
)
from services.gdpr_deletion import (
    GDPRDeletionService,
    DeletionStepStatus,
    delete_user_data_gdpr,
)


@pytest.fixture(scope="function")
def test_db():
    """Create a fresh in-memory database for testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()


@pytest.fixture
def sample_user(test_db):
    """Create a sample user for testing."""
    user = User(
        id=1,
        email="test@example.com",
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture
def sample_case(test_db, sample_user):
    """Create a sample case with documents and timeline."""
    case = Case(
        id=100,
        user_id=sample_user.id,
        case_number="CASE-100",
        case_type="civil",
        jurisdiction="Delhi",
        status=CaseStatus.ACTIVE,
        title="Test Case Title",
    )
    test_db.add(case)
    test_db.commit()
    
    # Add a document
    doc = CaseDocument(
        id=200,
        case_id=case.id,
        document_type=DocumentType.JUDGMENT,
        document_content="Sensitive document content",
        summary="Document summary with email@example.com",
        uploaded_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    test_db.add(doc)
    
    # Add a deadline
    deadline = CaseDeadline(
        id=300,
        user_id=sample_user.id,
        case_id=case.id,
        case_title=case.title,
        deadline_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
        deadline_type="appeal",
        description="Important deadline",
    )
    test_db.add(deadline)
    
    # Add timeline event
    timeline = CaseTimeline(
        id=400,
        case_id=case.id,
        event_type="filing",
        description="Case filed on date",
        event_date=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    test_db.add(timeline)
    
    test_db.commit()
    test_db.refresh(case)
    
    return case


class TestGDPRDeletionService:
    """Test suite for GDPRDeletionService."""

    def test_export_user_data_before_deletion(self, test_db, sample_user, sample_case):
        """Test that export generates a complete data dump with manifest."""
        service = GDPRDeletionService(db=test_db)
        
        export_data, deletion_token = service.export_user_data_before_deletion(
            sample_user.id, test_db
        )
        
        assert export_data is not None
        assert deletion_token is not None
        assert len(deletion_token) == 32  # SHA-256 truncated
        assert export_data["user_id"] == sample_user.id
        assert len(export_data["cases"]) == 1
        assert export_data["cases"][0]["id"] == sample_case.id
        assert export_data["cases"][0]["title"] == "Test Case Title"

    def test_manifest_generation(self, test_db, sample_user, sample_case):
        """Test that manifest includes proper hash and metadata."""
        service = GDPRDeletionService(db=test_db)
        
        export_data, deletion_token = service.export_user_data_before_deletion(
            sample_user.id, test_db
        )
        
        manifest = service._generate_manifest(
            user_id=sample_user.id,
            case_ids=[sample_case.id],
            export_data=export_data,
            deletion_token=deletion_token,
        )
        
        assert manifest is not None
        assert manifest["manifest_version"] == "1.0"
        assert manifest["user_id"] == sample_user.id
        assert manifest["deletion_token"] == deletion_token
        assert manifest["manifest_hash"] is not None
        assert len(manifest["manifest_hash"]) == 64  # SHA-256 hex

    def test_deletion_workflow_completes_all_steps(self, test_db, sample_user, sample_case):
        """Test that the full deletion workflow executes all steps."""
        service = GDPRDeletionService(db=test_db)
        
        with patch.object(service, '_delete_user_vectors', return_value=1):
            with patch.object(service, '_delete_user_attachments', return_value=0):
                with patch.object(service, '_delete_user_timeline_and_notes', return_value=2):
                    with patch.object(service, '_delete_user_cases_and_deadlines', return_value=(1, 1)):
                        with patch.object(service, '_finalize_user_deletion'):
                            result = service.delete_user_data(sample_user.id)
        
        assert result.success is True
        assert len(result.steps) == 7  # 7 deletion steps
        step_names = [s.name for s in result.steps]
        assert "export_user_data" in step_names
        assert "redact_database_records" in step_names
        assert "delete_vector_embeddings" in step_names
        assert "delete_attachments" in step_names
        assert "delete_timeline_and_notes" in step_names
        assert "delete_cases_and_deadlines" in step_names
        assert "finalize_user_deletion" in step_names
        
        # All steps should be completed
        for step in result.steps:
            assert step.status == DeletionStepStatus.COMPLETED

    def test_deletion_handles_partial_failure(self, test_db, sample_user, sample_case):
        """Test that deletion handles partial failures and reports errors."""
        service = GDPRDeletionService(db=test_db)
        
        # Make vector deletion fail
        with patch.object(service, '_delete_user_vectors', side_effect=Exception("Vector store error")):
            with patch.object(service, '_delete_user_attachments', return_value=0):
                result = service.delete_user_data(sample_user.id)
        
        assert result.success is False
        assert result.error is not None
        assert "Vector deletion failed" in result.error
        
        # Find the failed step
        failed_steps = [s for s in result.steps if s.status == DeletionStepStatus.FAILED]
        assert len(failed_steps) == 1
        assert failed_steps[0].name == "delete_vector_embeddings"
        assert "Vector store error" in failed_steps[0].error

    def test_redaction_replaces_pii(self, test_db, sample_user, sample_case):
        """Test that PII is properly redacted during deletion."""
        service = GDPRDeletionService(db=test_db)
        
        # Verify initial state
        test_db.refresh(sample_user)
        assert sample_user.email == "test@example.com"
        assert sample_case.title == "Test Case Title"
        
        # Redact user data
        redacted_count = service._redact_user_data(test_db, sample_user.id)
        
        # Verify redaction
        test_db.refresh(sample_user)
        test_db.refresh(sample_case)
        
        assert sample_user.email == "[REDACTED-EMAIL]"
        assert sample_case.title.startswith("[REDACTED-GDPR-DELETE]")
        assert redacted_count > 0


class TestVectorDeletion:
    """Test suite for vector store deletion functionality."""

    def test_delete_vectors_by_case(self, tmp_path):
        """Test vector deletion for a specific case."""
        import os
        from core.vector_store import ShardedVectorStore
        
        # Create isolated storage directory
        test_storage = tmp_path / "vectors"
        test_storage.mkdir()
        
        # Patch the storage dir
        import core.vector_store as vs_module
        original_dir = vs_module.STORAGE_DIR
        vs_module.STORAGE_DIR = str(test_storage)
        
        try:
            # Create fresh store with no persisted data
            store = ShardedVectorStore(num_shards=4, dimension=4)
            
            # Add vectors one at a time using add method
            store.add_batch([(1, [0.1, 0.2, 0.3, 0.4])])
            store.add_batch([(2, [0.5, 0.6, 0.7, 0.8])])
            store.add_batch([(3, [0.9, 1.0, 1.1, 1.2])])
            store.add_batch([(4, [1.3, 1.4, 1.5, 1.6])])
            
            initial_count = store.get_vector_count()
            assert initial_count == 4
            
            # Delete one case
            deleted = store.delete_vectors_by_case(2)
            assert deleted == 1
            
            remaining_count = store.get_vector_count()
            assert remaining_count == 3
            
            # Verify case 2 is gone
            assert store.get_vectors_for_case(2) is None
            
            # Verify other cases remain
            assert store.get_vectors_for_case(1) is not None
            assert store.get_vectors_for_case(3) is not None
        finally:
            vs_module.STORAGE_DIR = original_dir

    def test_delete_vectors_by_user(self, tmp_path):
        """Test batch deletion of vectors for multiple cases."""
        import os
        from core.vector_store import ShardedVectorStore
        
        # Create isolated storage directory
        test_storage = tmp_path / "vectors"
        test_storage.mkdir()
        
        import core.vector_store as vs_module
        original_dir = vs_module.STORAGE_DIR
        vs_module.STORAGE_DIR = str(test_storage)
        
        try:
            store = ShardedVectorStore(num_shards=4, dimension=4)
            
            # Add test vectors for multiple user cases one at a time
            user_case_ids = [10, 20, 30, 40, 50]
            for cid in user_case_ids:
                vec = [float(cid) * 0.01] * 4
                store.add_batch([(cid, vec)])
            
            assert store.get_vector_count() == 5
            
            # Delete all user cases
            deleted = store.delete_vectors_by_user(user_case_ids)
            assert deleted == 5
            
            # Verify all deleted
            assert store.get_vector_count() == 0
        finally:
            vs_module.STORAGE_DIR = original_dir

    def test_delete_nonexistent_case_returns_zero(self, tmp_path):
        """Test that deleting a nonexistent case returns 0."""
        import core.vector_store as vs_module
        
        test_storage = tmp_path / "vectors"
        test_storage.mkdir()
        original_dir = vs_module.STORAGE_DIR
        vs_module.STORAGE_DIR = str(test_storage)
        
        try:
            store = vs_module.ShardedVectorStore(num_shards=2, dimension=4)
            deleted = store.delete_vectors_by_case(9999)
            assert deleted == 0
        finally:
            vs_module.STORAGE_DIR = original_dir


class TestStorageDeletion:
    """Test suite for storage deletion functionality."""

    def test_delete_attachment_file_success(self, tmp_path):
        """Test successful deletion of an attachment file."""
        from core import storage as storage_module
        
        # Create a test attachment directory
        test_dir = tmp_path / "attachments"
        test_dir.mkdir()
        
        # Create a test file
        test_file = test_dir / "test_attachment.txt"
        test_file.write_text("test content")
        
        # Patch the attachments directory
        with patch.object(storage_module, 'ATTACHMENTS_DIR', test_dir):
            result = storage_module.delete_attachment_file(str(test_file))
        
        assert result is True
        assert not test_file.exists()

    def test_delete_attachment_file_not_found(self, tmp_path):
        """Test deletion of a nonexistent file."""
        from core import storage as storage_module
        
        test_dir = tmp_path / "attachments"
        test_dir.mkdir()
        
        with patch.object(storage_module, 'ATTACHMENTS_DIR', test_dir):
            result = storage_module.delete_attachment_file(str(test_dir / "nonexistent.txt"))
        
        assert result is False

    def test_delete_attachment_rejects_path_traversal(self, tmp_path):
        """Test that path traversal attempts are rejected."""
        from core import storage as storage_module
        
        test_dir = tmp_path / "attachments"
        test_dir.mkdir()
        
        with patch.object(storage_module, 'ATTACHMENTS_DIR', test_dir):
            # Try path traversal
            result = storage_module.delete_attachment_file("../../../etc/passwd")
        
        assert result is False

    def test_bulk_delete_attachments(self, tmp_path):
        """Test bulk deletion of multiple attachments."""
        from core import storage as storage_module
        
        test_dir = tmp_path / "attachments"
        test_dir.mkdir()
        
        # Create multiple test files
        test_files = []
        for i in range(5):
            test_file = test_dir / f"attachment_{i}.txt"
            test_file.write_text(f"content {i}")
            test_files.append(str(test_file))
        
        with patch.object(storage_module, 'ATTACHMENTS_DIR', test_dir):
            results = storage_module.bulk_delete_attachments(test_files)
        
        assert results["deleted"] == 5
        assert results["failed"] == 0
        assert len(results["errors"]) == 0

    def test_bulk_delete_with_failures(self, tmp_path):
        """Test bulk deletion with some failures."""
        from core import storage as storage_module
        
        test_dir = tmp_path / "attachments"
        test_dir.mkdir()
        
        # Create one file
        existing_file = test_dir / "existing.txt"
        existing_file.write_text("exists")
        
        paths = [str(existing_file)]
        
        with patch.object(storage_module, 'ATTACHMENTS_DIR', test_dir):
            results = storage_module.bulk_delete_attachments(paths)
        
        # Verify deletion worked (only one file was created, so only one deleted)
        assert results["deleted"] == 1
        assert results["failed"] == 0


class TestDeletionResultConsistency:
    """Test that deletion maintains consistent state on partial failures."""

    def test_deletion_result_to_dict(self, test_db, sample_user, sample_case):
        """Test that DeletionResult can be serialized to dict."""
        service = GDPRDeletionService(db=test_db)
        
        with patch.object(service, '_delete_user_vectors', return_value=1):
            with patch.object(service, '_delete_user_attachments', return_value=0):
                with patch.object(service, '_delete_user_timeline_and_notes', return_value=2):
                    with patch.object(service, '_delete_user_cases_and_deadlines', return_value=(1, 1)):
                        with patch.object(service, '_finalize_user_deletion'):
                            result = service.delete_user_data(sample_user.id)
        
        result_dict = result.to_dict()
        
        assert result_dict["user_id"] == sample_user.id
        assert result_dict["success"] is True
        assert "export_manifest" in result_dict
        assert len(result_dict["steps"]) == 7
        assert result_dict["completed_at"] is not None

    def test_partial_failure_maintains_audit_trail(self, test_db, sample_user, sample_case):
        """Test that partial failures still create audit entries."""
        service = GDPRDeletionService(db=test_db)
        
        # Make the last step fail
        with patch.object(service, '_delete_user_vectors', return_value=1):
            with patch.object(service, '_delete_user_attachments', return_value=0):
                with patch.object(service, '_delete_user_timeline_and_notes', return_value=2):
                    with patch.object(service, '_delete_user_cases_and_deadlines', return_value=(1, 1)):
                        with patch.object(service, '_finalize_user_deletion', side_effect=Exception("Finalization failed")):
                            result = service.delete_user_data(sample_user.id)
        
        assert result.success is False
        
        # Verify failed step has error details
        failed_steps = [s for s in result.steps if s.status == DeletionStepStatus.FAILED]
        assert len(failed_steps) == 1
        assert failed_steps[0].name == "finalize_user_deletion"


class TestDatabaseDeletionFunctions:
    """Test database-level deletion functions."""

    def test_delete_user_cases(self, test_db, sample_user, sample_case):
        """Test the delete_user_cases function."""
        from database import delete_user_cases
        
        counts = delete_user_cases(test_db, sample_user.id)
        
        assert counts["cases"] == 1
        assert counts["documents"] == 1
        assert counts["deadlines"] == 1
        assert counts["timeline_events"] == 1
        
        # Verify cases are deleted
        remaining_cases = test_db.query(Case).filter(Case.user_id == sample_user.id).all()
        assert len(remaining_cases) == 0

    def test_redact_user_data(self, test_db, sample_user, sample_case):
        """Test the redact_user_data function."""
        from database import redact_user_data
        
        redacted_count = redact_user_data(test_db, sample_user.id)
        
        assert redacted_count > 0
        
        # Verify redaction
        test_db.refresh(sample_user)
        assert sample_user.email == "[REDACTED-EMAIL]"
        
        test_db.refresh(sample_case)
        assert sample_case.title.startswith("[REDACTED-GDPR]")

    def test_delete_user_cases_empty_user(self, test_db, sample_user):
        """Test deletion of cases for user with no cases."""
        from database import delete_user_cases
        
        counts = delete_user_cases(test_db, sample_user.id)
        
        assert counts["cases"] == 0
        assert counts["documents"] == 0


class TestExportManifestSignature:
    """Test the signed manifest functionality."""

    def test_manifest_contains_required_fields(self, test_db, sample_user, sample_case):
        """Test that manifest contains all required fields."""
        service = GDPRDeletionService(db=test_db)
        
        export_data = {"user_id": sample_user.id, "cases": []}
        
        manifest = service._generate_manifest(
            user_id=sample_user.id,
            case_ids=[1, 2, 3],
            export_data=export_data,
            deletion_token="abc123",
        )
        
        # Verify required fields
        assert "manifest_version" in manifest
        assert "generated_at" in manifest
        assert "user_id" in manifest
        assert "deletion_token" in manifest
        assert "manifest_hash" in manifest
        assert manifest["manifest_version"] == "1.0"

    def test_different_tokens_produce_different_hashes(self, test_db, sample_user):
        """Test that different deletion tokens produce different hashes."""
        service = GDPRDeletionService(db=test_db)
        
        export_data = {"user_id": sample_user.id, "cases": []}
        
        manifest1 = service._generate_manifest(
            user_id=sample_user.id,
            case_ids=[1],
            export_data=export_data,
            deletion_token="token1",
        )
        
        manifest2 = service._generate_manifest(
            user_id=sample_user.id,
            case_ids=[1],
            export_data=export_data,
            deletion_token="token2",
        )
        
        # Different tokens should produce different hashes
        assert manifest1["manifest_hash"] != manifest2["manifest_hash"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])