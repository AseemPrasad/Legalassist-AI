from api.middleware import is_safe_to_cache


def test_documents_path_not_safe_to_cache():
    assert not is_safe_to_cache("/api/v1/documents/123")


def test_cases_path_safe_to_cache():
    assert is_safe_to_cache("/api/v1/cases/1/events")
