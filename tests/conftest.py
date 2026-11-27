"""Shared test setup.

Every test gets its own empty user store and audit trail. Without this the
suite would read the project's real users.json: the day someone creates an
admin account, every API test would start failing with 401, for a reason
that has nothing to do with the code under test.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_auth(tmp_path, monkeypatch):
    from skyshine import auth, server

    # 600,000 PBKDF2 rounds per hash is right for production and turns the
    # auth tests into minutes. Hashes record their own round count, so
    # verification is unaffected; TestPasswords checks the production value.
    monkeypatch.setattr(auth, "PBKDF2_ITERATIONS", 2_000)
    monkeypatch.setattr(auth, "_DUMMY_HASH", None)
    monkeypatch.setattr(server, "USERS_FILE", tmp_path / "users.json")
    monkeypatch.setattr(server, "AUDIT_FILE", tmp_path / "audit.jsonl")
    # Same reason for the signing key: tests must neither read the real one
    # nor create one in the project folder.
    monkeypatch.setattr(server, "KEY_FILE", tmp_path / "mission.key")
    monkeypatch.setattr(server, "LINK_KEY_FILE", tmp_path / "link.key")
    from skyshine import assistant

    monkeypatch.setattr(assistant, "KEY_FILE", tmp_path / "claude.key")
    server._auth_cache.clear()
    yield
    server._auth_cache.clear()
