import json
import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.base import Base
from db.models.exports import ExportJob, ExportChunk
from core.export_service import create_export_job, process_export_job


def make_records(n):
    return [{"id": i, "value": f"item-{i}"} for i in range(n)]


def test_resumable_export(tmp_path):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        records = make_records(50)
        out_file = tmp_path / "out.json"

        job = create_export_job(db, records, str(out_file), export_format="json", chunk_size=10)
        assert job.total_chunks == 5

        # simulate failure after processing chunk index 1
        try:
            process_export_job(db, job.id, fail_after=1)
        except Exception:
            pass

        job = db.query(ExportJob).filter(ExportJob.id == job.id).one()
        assert job.status == "failed"
        assert job.last_completed_chunk >= 1

        # resume normally
        job = process_export_job(db, job.id)
        assert job.status == "completed"
        assert out_file.exists()

        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert isinstance(data, list) and len(data) == 50
    finally:
        db.close()
