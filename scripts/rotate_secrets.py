"""CLI to rotate or set secrets in the secret store."""
import argparse
from database import SessionLocal
from core.secrets import SecretStore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--value", help="New secret value")
    p.add_argument("--delete", action="store_true")
    p.add_argument("--who", help="Rotated by")
    p.add_argument("--reason", help="Reason for rotation")
    args = p.parse_args()

    db = SessionLocal()
    try:
        store = SecretStore(db)
        if args.delete:
            ok = store.delete_secret(args.name)
            print("deleted" if ok else "not_found")
            return
        if not args.value:
            print(store.get_secret(args.name) or "")
            return
        entry = store.rotate_secret(args.name, args.value, rotated_by=args.who, reason=args.reason)
        print("rotated", entry.name, entry.version)
    finally:
        db.close()


if __name__ == "__main__":
    main()
