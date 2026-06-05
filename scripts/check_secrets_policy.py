"""Operational security checks for secrets and production settings.

This script is intended for CI and pre-deploy validation. It fails when it
detects plaintext JWT fallbacks or obviously unsafe production defaults.
"""

from __future__ import annotations

import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]

PATTERNS = [
    re.compile(r'JWT_SECRET_KEY\s*=\s*os\.getenv\([^\n]*"your-secret-key-change-in-production"'),
    re.compile(r'JWT_SECRET\s*=\s*"test_secret_key_do_not_use_in_production"'),
    re.compile(r'JWT_SECRET_KEY\s*=\s*"your-secret-key-change-in-production"'),
    re.compile(r'POSTGRES_PASSWORD\s*=\s*(?:password|secure_password)'),
]


def main() -> int:
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if path.name == "check_secrets_policy.py":
            continue
        if path.suffix.lower() not in {".py", ".yml", ".yaml", ".toml", ".md", ".sh"}:
            continue
        if any(part in {".git", ".venv", "venv", "htmlcov", "__pycache__"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for pattern in PATTERNS:
            if pattern.search(text):
                offenders.append(str(path.relative_to(ROOT)))
                break

    if offenders:
        print("Potential secret-policy violations found:")
        for item in sorted(set(offenders)):
            print(f"- {item}")
        return 1

    print("Secrets policy check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())