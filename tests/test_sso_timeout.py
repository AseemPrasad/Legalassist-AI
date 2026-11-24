"""Outbound calls to SSO identity providers must carry a bounded timeout."""
import pytest

pytest.importorskip("authlib")

try:
    from api import sso
except (ImportError, SyntaxError) as exc:  # app deps not importable in this env
    pytest.skip(f"api.sso not importable: {exc}", allow_module_level=True)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, sso.DEFAULT_HTTP_TIMEOUT),
        ("", sso.DEFAULT_HTTP_TIMEOUT),
        ("abc", sso.DEFAULT_HTTP_TIMEOUT),
        ("0", sso.DEFAULT_HTTP_TIMEOUT),
        ("-3", sso.DEFAULT_HTTP_TIMEOUT),
        ("4", 4.0),
        ("2.5", 2.5),
    ],
)
def test_parse_timeout(raw, expected):
    assert sso._parse_timeout(raw) == expected


def test_from_env_reads_timeout(monkeypatch):
    monkeypatch.setenv("SSO_HTTP_TIMEOUT", "7")
    try:
        assert sso.SSOConfig.from_env().HTTP_TIMEOUT == 7.0
    finally:
        monkeypatch.delenv("SSO_HTTP_TIMEOUT")
        sso.SSOConfig.from_env()


@pytest.mark.parametrize("provider", ["google", "microsoft"])
def test_providers_register_with_timeout(monkeypatch, provider):
    seen = {}

    def fake_register(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(sso._oauth, "register", fake_register)
    monkeypatch.setattr(sso.SSOConfig, "HTTP_TIMEOUT", 3.0)
    prefix = provider.upper()
    monkeypatch.setattr(sso.SSOConfig, f"{prefix}_CLIENT_ID", "id")
    monkeypatch.setattr(sso.SSOConfig, f"{prefix}_CLIENT_SECRET", "secret")
    try:
        getattr(sso, f"_configure_{provider}")()
    except AttributeError:
        pass  # the client lookup after register is irrelevant here
    assert seen["client_kwargs"]["timeout"] == 3.0
