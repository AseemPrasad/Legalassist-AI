"""Script to check precedent matcher drift and optionally trigger retraining."""
import sys
import logging
from database import SessionLocal
from core.precedent_drift import detect_drift, retrain_embeddings

logger = logging.getLogger("check_precedent_drift")
logging.basicConfig(level=logging.INFO)


def main():
    db = SessionLocal()
    try:
        result = detect_drift(db)
        logger.info("Drift check result: %s", result)
        if result.get("drift"):
            logger.warning("Drift detected (relative_drop=%s) - starting retrain", result.get("relative_drop"))
            r = retrain_embeddings(db)
            logger.info("Retrain result: %s", r)
        return 0
    except Exception as e:
        logger.exception("Drift check failed: %s", e)
        return 2
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
