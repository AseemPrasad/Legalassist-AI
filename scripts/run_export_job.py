"""CLI to create and run resumable export jobs for given records (JSON).

Usage examples:
  python scripts/run_export_job.py --output out/results.json --from-file data.json
  python scripts/run_export_job.py --resume job_id
"""
import argparse
import json
from database import SessionLocal
from core.export_service import create_export_job, process_export_job


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from-file", help="JSON file with records to export")
    p.add_argument("--output", help="Output path for final export")
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--resume", type=int, help="Job id to resume/process")
    args = p.parse_args()

    db = SessionLocal()
    try:
        if args.resume:
            job = process_export_job(db, args.resume)
            print("Processed job", job.id, job.status)
            return

        if not args.from_file or not args.output:
            p.print_help()
            return

        with open(args.from_file, "r", encoding="utf-8") as f:
            records = json.load(f)

        job = create_export_job(db, records, args.output, export_format="json", chunk_size=args.chunk_size)
        print("Created job", job.id)
        job = process_export_job(db, job.id)
        print("Job finished", job.id, job.status)
    finally:
        db.close()


if __name__ == "__main__":
    main()
