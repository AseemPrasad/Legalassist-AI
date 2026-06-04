"""Rotate JWT secrets with minimal downtime.

Usage:
    python scripts/rotate_jwt_secret.py --current NEW_SECRET --previous OLD_SECRET

The script prints shell-safe export commands for current and previous secrets.
Deploy the new secret first, keep the previous secret in place for token
verification during the grace period, then remove the previous secret after
the longest token lifetime has elapsed.
"""

from __future__ import annotations

import argparse
import secrets


def build_exports(current: str, previous: str | None) -> list[str]:
    exports = [f'export JWT_SECRET="{current}"']
    if previous:
        exports.append(f'export JWT_SECRET_PREVIOUS="{previous}"')
    return exports


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate JWT secret rotation exports")
    parser.add_argument("--current", help="New active JWT secret")
    parser.add_argument("--previous", default="", help="Previous JWT secret kept for verification")
    parser.add_argument("--generate", action="store_true", help="Generate a random current secret")
    args = parser.parse_args()

    current = args.current or (secrets.token_urlsafe(48) if args.generate else "")
    if not current:
        raise SystemExit("Provide --current or --generate")

    for line in build_exports(current, args.previous.strip() or None):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())