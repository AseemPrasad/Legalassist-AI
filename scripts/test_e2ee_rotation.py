"""Simple sanity test for E2EE KMS wrapping and rotation.

This script is intended to be runnable locally to validate the new helpers.
"""
import json
import os
from core.kms import LocalFileKMS
from core.e2ee import generate_master_key, wrap_master_key_with_kms, unwrap_master_key_with_kms, rotate_wrapped_master_keys


def run():
    manifest = {
        "user:alice": None,
        "user:bob": None,
    }

    old_kms = LocalFileKMS(path=".e2ee_root_key.old")
    new_kms = LocalFileKMS(path=".e2ee_root_key.new")

    # generate master keys and wrap with old KMS
    for uid in list(manifest.keys()):
        mk = generate_master_key()
        wrapped = wrap_master_key_with_kms(mk, old_kms)
        manifest[uid] = wrapped

    path = "e2ee_manifest_test.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # rotate
    rotate_wrapped_master_keys(path, old_kms, new_kms)

    # verify new manifest unwraps with new KMS
    with open(path, "r", encoding="utf-8") as f:
        rotated = json.load(f)

    for uid, wrapped in rotated.items():
        mk_b64 = unwrap_master_key_with_kms(wrapped, new_kms)
        assert mk_b64 is not None and len(mk_b64) > 0

    print("Rotation test passed")


if __name__ == "__main__":
    run()
