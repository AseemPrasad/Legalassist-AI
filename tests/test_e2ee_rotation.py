import json

from core.e2ee import (
    generate_master_key,
    rotate_wrapped_master_keys,
    unwrap_master_key_with_kms,
    verify_wrapped_master_keys,
    wrap_master_key_with_kms,
)
from core.kms import LocalFileKMS


def test_rotation_keeps_legacy_compatibility_and_verifies(tmp_path, monkeypatch):
    old_kms = LocalFileKMS(path=str(tmp_path / "root.old"))
    new_kms = LocalFileKMS(path=str(tmp_path / "root.new"))

    manifest = {
        "user:alice": wrap_master_key_with_kms(generate_master_key(), old_kms),
        "user:bob": wrap_master_key_with_kms(generate_master_key(), old_kms),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    audit_calls = []
    monkeypatch.setattr("core.e2ee._append_rotation_audit", lambda **kwargs: audit_calls.append(kwargs))

    rotate_wrapped_master_keys(str(manifest_path), old_kms, new_kms)

    rotated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(rotated.keys()) == {"user:alice", "user:bob"}
    assert isinstance(rotated["user:alice"], dict)
    assert rotated["user:alice"]["version"] >= 2
    assert rotated["user:alice"]["kms_key_id"] == new_kms.key_id

    for entry in rotated.values():
        assert unwrap_master_key_with_kms(entry, new_kms)

    report = verify_wrapped_master_keys(str(manifest_path), new_kms)
    assert report["valid"] is True
    assert report["decryptable"] == 2

    assert any(call["outcome"] == "verified" for call in audit_calls)