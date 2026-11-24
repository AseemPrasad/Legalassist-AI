import base64

import pytest

from utils import upload_validation as uv

PDF = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n" + b"0" * 100
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAHGx0YQAAAABJRU5ErkJggg=="
)
JPG = b"\xff\xd8\xff\xe0" + b"0" * 100
EXE = b"MZ\x90\x00\x03\x00\x00\x00" + b"0" * 100


@pytest.fixture(params=["magic", "signatures"])
def backend(request, monkeypatch):
    """Run every case with libmagic and with the built-in signature fallback."""
    if request.param == "signatures":
        import builtins

        real_import = builtins.__import__

        def no_magic(name, *a, **kw):
            if name == "magic":
                raise ImportError("magic disabled for test")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", no_magic)
    else:
        pytest.importorskip("magic")
    return request.param


@pytest.mark.parametrize(
    "name, head",
    [("a.pdf", PDF), ("a.PDF", PDF), ("p.png", PNG), ("p.jpg", JPG), ("p.jpeg", JPG)],
)
def test_genuine_files_pass(backend, name, head):
    assert uv.validate_upload(name, head) is None


@pytest.mark.parametrize("name", ["evil.pdf", "evil.png", "evil.jpg"])
def test_renamed_executable_rejected(backend, name):
    msg = uv.validate_upload(name, EXE)
    assert msg and "does not look like" in msg


def test_wrong_image_type_rejected(backend):
    assert uv.validate_upload("photo.png", JPG) is not None
    assert uv.validate_upload("doc.pdf", PNG) is not None


def test_unknown_extension_rejected(backend):
    assert uv.validate_upload("run.exe", EXE) is not None
    assert uv.validate_upload("noext", PDF) is not None
