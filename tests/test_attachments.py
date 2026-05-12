import os
from core.storage import save_attachment, get_attachment_path


def test_save_and_get_attachment(tmp_path):
    # Create sample bytes
    data = b"hello world"
    filename = "test.txt"

    # Save
    stored_path, size = save_attachment(data, filename)

    assert os.path.exists(stored_path)
    assert size == len(data)

    # get path
    p = get_attachment_path(stored_path)
    assert p == stored_path

    # Cleanup
    try:
        os.remove(stored_path)
    except Exception:
        pass
