import json
import math
import os
import shutil
import csv
import logging
from pathlib import Path
from typing import List, Dict, Optional
from sqlalchemy.orm import Session

from db.models.exports import ExportJob, ExportChunk

LOGGER = logging.getLogger(__name__)


def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def create_export_job(db: Session, records: List[Dict], output_path: str, export_format: str = "json", chunk_size: int = 500) -> ExportJob:
    # create job record
    job = ExportJob(output_path=output_path, export_format=export_format, status="pending")
    db.add(job)
    db.commit()
    db.refresh(job)

    total_chunks = math.ceil(len(records) / chunk_size) if records else 0
    job.total_chunks = total_chunks
    db.add(job)
    db.commit()

    base = Path(".exports") / str(job.id)
    for i in range(total_chunks):
        chunk_records = records[i * chunk_size : (i + 1) * chunk_size]
        chunk_dir = base / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = chunk_dir / f"chunk_{i}.json"
        # write chunk as JSON array
        with chunk_path.open("w", encoding="utf-8") as f:
            json.dump(chunk_records, f, ensure_ascii=False)

        chunk = ExportChunk(job_id=job.id, index=i, path=str(chunk_path))
        db.add(chunk)
    db.commit()
    return job


def process_export_job(db: Session, job_id: int, fail_after: Optional[int] = None) -> ExportJob:
    job = db.query(ExportJob).filter(ExportJob.id == job_id).one()
    job.status = "processing"
    db.add(job)
    db.commit()

    try:
        chunks = sorted(job.chunks, key=lambda c: c.index)
        out_path = Path(job.output_path)
        temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        _ensure_dir(temp_path)

        if job.export_format == "json":
            first = True
            with temp_path.open("w", encoding="utf-8") as outf:
                outf.write("[")
                for c in chunks:
                    # always read chunk files in-order to rebuild output deterministically
                    with open(c.path, "r", encoding="utf-8") as cf:
                        arr = json.load(cf)
                    for item in arr:
                        if not first:
                            outf.write(",")
                        json.dump(item, outf, ensure_ascii=False)
                        first = False

                    c.processed = True
                    db.add(c)
                    job.last_completed_chunk = c.index
                    db.add(job)
                    db.commit()

                    if fail_after is not None and c.index >= fail_after:
                        raise RuntimeError("simulated failure")

                outf.write("]")

        elif job.export_format == "csv":
            wrote_header = False
            with temp_path.open("w", encoding="utf-8", newline="") as outf:
                writer = None
                for c in chunks:
                    with open(c.path, "r", encoding="utf-8") as cf:
                        arr = json.load(cf)
                    if not arr:
                        c.processed = True
                        db.add(c)
                        job.last_completed_chunk = c.index
                        db.add(job)
                        db.commit()
                        continue
                    if not wrote_header:
                        fieldnames = list(arr[0].keys())
                        writer = csv.DictWriter(outf, fieldnames=fieldnames)
                        writer.writeheader()
                        wrote_header = True
                    for row in arr:
                        writer.writerow(row)

                    c.processed = True
                    db.add(c)
                    job.last_completed_chunk = c.index
                    db.add(job)
                    db.commit()

                    if fail_after is not None and c.index >= fail_after:
                        raise RuntimeError("simulated failure")

        # atomic move
        os.replace(str(temp_path), str(out_path))
        job.status = "completed"
        job.error = None
        db.add(job)
        db.commit()
        LOGGER.info("export_completed", job_id=job.id, path=job.output_path)
        return job
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        db.add(job)
        db.commit()
        LOGGER.exception("export_failed", job_id=job.id, error=str(e))
        raise
