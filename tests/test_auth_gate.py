import os
import sys
from types import SimpleNamespace

import pytest

os.environ.setdefault("JWT_SECRET", "test_secret_key")

import auth


class _PageSwitch(Exception):
    pass


class _FakeSessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _FakeStreamlit(SimpleNamespace):
    def __init__(self):
        super().__init__(session_state=_FakeSessionState())
        self.redirects = []

    def switch_page(self, page):
        self.redirects.append(page)
        raise _PageSwitch(page)


def test_require_auth_redirects_revoked_token_to_login(monkeypatch):
    fake_st = _FakeStreamlit()
    fake_st.session_state.is_authenticated = True
    fake_st.session_state.user_token = "revoked-token"

    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.setattr(auth, "init_auth_session", lambda: None)
    monkeypatch.setattr(auth, "verify_jwt_token", lambda token: None)

    logout_calls = []

    def fake_logout_user():
        logout_calls.append(True)

    monkeypatch.setattr(auth, "logout_user", fake_logout_user)

    with pytest.raises(_PageSwitch):
        auth.require_auth()

    assert logout_calls == [True]
    assert fake_st.redirects == [auth.PAGE_LOGIN]
