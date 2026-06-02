"""Command-line rotation tool for wrapped master keys manifest.

Usage:
    python scripts/rotate_e2ee_keys.py <manifest.json> [--new-root-file path]

This will use the current local KMS (default `.e2ee_root_key`) and rotate all
wrapped master keys to a new root key located at `--new-root-file`. If
`--new-root-file` does not exist it will be created.
"""
import argparse
import os
from core.kms import LocalFileKMS
from core.e2ee import rotate_wrapped_master_keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest", help="Path to JSON manifest mapping id->wrapped_master_key")
    p.add_argument("--new-root-file", help="Path for new root key file", default=".e2ee_root_key.new")
    args = p.parse_args()

    # Use existing KMS (old) and new KMS pointed at the new root file.
    old_kms = LocalFileKMS()

    # create new root file if missing
    new_kms = LocalFileKMS(path=args.new_root_file)

    rotate_wrapped_master_keys(args.manifest, old_kms, new_kms)
    print(f"Rotation completed; backup saved to {args.manifest}.bak")


if __name__ == "__main__":
    main()
