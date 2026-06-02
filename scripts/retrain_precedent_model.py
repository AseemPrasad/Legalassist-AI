"""Script to force retrain precedent embeddings and rebuild baseline stats."""
import sys
import logging
from database import SessionLocal
from core.precedent_drift import retrain_embeddings

logger = logging.getLogger("retrain_precedent_model")
logging.basicConfig(level=logging.INFO)


def main():
    db = SessionLocal()
    try:
        res = retrain_embeddings(db)
        logger.info("Retrain finished: %s", res)
        return 0 if res.get("retrained") else 1
    except Exception as e:
        logger.exception("Retrain failed: %s", e)
        return 2
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
