"""Tests for authentication, authorisation and the audit trail.

Before this layer, anything that could reach the port could arm a fleet.
The tests that matter are TestRoleBoundaries - each role can do what it
should and is refused one step beyond - and TestVoiceCannotBypassRoles,
because the voice endpoint calls handlers directly, past the route checks,
and was the obvious way around them.
"""

from __future__ import annotations

import io
import json
import time

import pytest
from fastapi.testclient import TestClient

from skyshine import auth, server

PW = {
    "root": "admin password 01",
    "olive": "operator pass 02",
    "dina": "designer pass 03",
    "vic": "viewer password 4",
}
ROLE_OF = {"root": "admin", "olive": "operator", "dina": "designer", "vic": "viewer"}


@pytest.fixture()
def store(tmp_path):
    return auth.AuthStore(tmp_path / "users.json")


@pytest.fixture()
def accounts(monkeypatch, tmp_path):
    """A server with one account per role, and isolated show storage."""
    monkeypatch.setattr(server, "SHOW_DIR", tmp_path / "shows")
    monkeypatch.setattr(server, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(server, "FLEET_FILE", tmp_path / "fleet.json")
    monkeypatch.setattr(server, "_fleet", None)
    (tmp_path / "shows").mkdir()
    store = server._auth()
    for name, pw in PW.items():
        store.add_user(name, pw, ROLE_OF[name])
    return store


def as_user(name):
    c = TestClient(server.app)
    r = c.post("/api/auth/login", json={"username": name, "password": PW[name]})
    assert r.status_code == 200, r.text
    return c


# ==========================================================================
# Passwords
# ==========================================================================


class TestPasswords:
    def test_round_trip(self):
        h = auth.hash_password("a long password 1")
        assert auth.verify_password("a long password 1", h)
        assert not auth.verify_password("a long password 2", h)

    def test_salted(self):
        """Same password, different hashes: no rainbow tables, no matching."""
        assert auth.hash_password("same password!") != auth.hash_password("same password!")

    def test_production_work_factor(self, monkeypatch):
        """Tests run with a low work factor; production must not."""
        assert auth.PRODUCTION_ITERATIONS >= 600_000
        monkeypatch.setattr(auth, "PBKDF2_ITERATIONS", auth.PRODUCTION_ITERATIONS)
        scheme, iterations, _salt, _digest = auth.hash_password("pw pw pw pw").split("$")
        assert scheme == "pbkdf2_sha256"
        assert int(iterations) == auth.PRODUCTION_ITERATIONS

    def test_old_hashes_still_verify_after_the_factor_changes(self, monkeypatch):
        """Each hash records its own round count, so raising it later is safe."""
        old = auth.hash_password("a long password 1", iterations=1_000)
        monkeypatch.setattr(auth, "PBKDF2_ITERATIONS", 50_000)
        assert auth.verify_password("a long password 1", old)

    @pytest.mark.parametrize("stored", ["", "garbage", "md5$1$aa$bb",
                                        "pbkdf2_sha256$x$zz$yy"])
    def test_malformed_hashes_never_verify(self, stored):
        assert auth.verify_password("anything", stored) is False

    @pytest.mark.parametrize("pw,user,why", [
        ("short", "", "at least"),
        ("olive password 1", "olive", "username"),
        ("aaaaaaaaaaaaaa", "", "repetitive"),
    ])
    def test_weak_passwords_are_refused(self, pw, user, why):
        with pytest.raises(auth.AuthError, match=why):
            auth.check_password_strength(pw, user)


# ==========================================================================
# Store
# ==========================================================================


class TestStore:
    def test_empty_store_is_unconfigured(self, store):
        assert not store.configured

    def test_add_and_reload(self, store, tmp_path):
        store.add_user("root", PW["root"], "admin")
        assert store.configured
        again = auth.AuthStore(tmp_path / "users.json")
        assert again.users["root"].role == "admin"

    def test_no_plaintext_on_disk(self, store, tmp_path):
        store.add_user("root", PW["root"], "admin")
        assert PW["root"] not in (tmp_path / "users.json").read_text(encoding="utf-8")

    @pytest.mark.parametrize("name", ["", "  ", "a b", "x;rm -rf", "../root"])
    def test_bad_usernames(self, store, name):
        with pytest.raises(auth.AuthError):
            store.add_user(name, "a decent password", "viewer")

    def test_duplicate_user(self, store):
        store.add_user("root", PW["root"], "admin")
        with pytest.raises(auth.AuthError, match="already exists"):
            store.add_user("root", "another password 9", "viewer")

    def test_unknown_role(self, store):
        with pytest.raises(auth.AuthError, match="role must be"):
            store.add_user("x", "a decent password", "pilot")

    def test_corrupt_file_fails_closed(self, tmp_path):
        """A broken user file must not read as 'no users' and open bootstrap."""
        p = tmp_path / "users.json"
        p.write_text("{ not json", encoding="utf-8")
        with pytest.raises(auth.AuthError, match="unreadable"):
            auth.AuthStore(p)

    def test_last_admin_cannot_be_demoted_or_removed(self, store):
        store.add_user("root", PW["root"], "admin")
        with pytest.raises(auth.AuthError, match="last admin"):
            store.set_role("root", "viewer")
        with pytest.raises(auth.AuthError, match="last admin"):
            store.remove_user("root")

    def test_a_second_admin_unblocks_it(self, store):
        store.add_user("root", PW["root"], "admin")
        store.add_user("root2", "second admin pass 7", "admin")
        store.set_role("root", "viewer")
        assert store.users["root"].role == "viewer"


class TestSessions:
    def test_login_issues_a_session(self, store):
        store.add_user("root", PW["root"], "admin")
        s = store.login("root", PW["root"])
        assert store.session(s.token).user == "root"
        assert len(s.token) >= 40

    def test_wrong_password_and_unknown_user_look_identical(self, store):
        """Different messages would tell an attacker which names are real."""
        store.add_user("root", PW["root"], "admin")
        with pytest.raises(auth.AuthError) as a:
            store.login("root", "not the password")
        with pytest.raises(auth.AuthError) as b:
            store.login("nobody", "not the password")
        assert str(a.value) == str(b.value)

    def test_lockout_after_repeated_failures(self, store):
        store.add_user("root", PW["root"], "admin")
        now = 1_000_000.0
        for _ in range(auth.LOCKOUT_FAILURES):
            with pytest.raises(auth.AuthError):
                store.login("root", "wrong wrong wrong", now=now)
        with pytest.raises(auth.AuthError, match="too many"):
            store.login("root", PW["root"], now=now + 1)

    def test_lockout_expires(self, store):
        store.add_user("root", PW["root"], "admin")
        now = 1_000_000.0
        for _ in range(auth.LOCKOUT_FAILURES):
            with pytest.raises(auth.AuthError):
                store.login("root", "wrong wrong wrong", now=now)
        assert store.login("root", PW["root"], now=now + auth.LOCKOUT_S + 1)

    def test_session_expiry(self, store):
        store.add_user("root", PW["root"], "admin")
        s = store.login("root", PW["root"], now=1000.0)
        assert store.session(s.token, now=1000.0 + auth.SESSION_TTL_S - 1)
        assert store.session(s.token, now=1000.0 + auth.SESSION_TTL_S + 1) is None

    def test_logout(self, store):
        store.add_user("root", PW["root"], "admin")
        s = store.login("root", PW["root"])
        store.logout(s.token)
        assert store.session(s.token) is None

    def test_password_change_ends_existing_sessions(self, store):
        """A stolen laptop stays signed in until the password changes."""
        store.add_user("root", PW["root"], "admin")
        s = store.login("root", PW["root"])
        store.set_password("root", "a brand new password")
        assert store.session(s.token) is None

    def test_role_change_takes_effect_immediately(self, store):
        store.add_user("root", PW["root"], "admin")
        store.add_user("olive", PW["olive"], "operator")
        s = store.login("olive", PW["olive"])
        store.set_role("olive", "viewer")
        assert store.session(s.token) is None, "an old session kept the old role"

    def test_disabled_user_cannot_sign_in(self, store):
        store.add_user("root", PW["root"], "admin")
        store.add_user("olive", PW["olive"], "operator")
        store.users["olive"].disabled = True
        with pytest.raises(auth.AuthError):
            store.login("olive", PW["olive"])

    @pytest.mark.parametrize("token", [None, "", "forged-token", "a" * 43])
    def test_unknown_tokens_have_no_session(self, store, token):
        assert store.session(token) is None


class TestRequiredRoles:
    @pytest.mark.parametrize("method,path,role", [
        ("GET", "/", None),
        ("GET", "/static/x.js", None),
        ("POST", "/api/auth/login", None),
        ("GET", "/api/auth/status", None),
        ("GET", "/api/design/shows", "viewer"),
        ("GET", "/api/ops/state", "viewer"),
        ("POST", "/api/design/shows", "designer"),
        ("POST", "/api/design/assets", "designer"),
        ("POST", "/api/voice/compose", "designer"),
        ("POST", "/api/ops/arm", "operator"),
        ("POST", "/api/ops/start", "operator"),
        ("POST", "/api/ops/emergency", "operator"),
        ("POST", "/api/fleet/drones", "operator"),
        ("GET", "/api/users", "admin"),
        ("POST", "/api/users", "admin"),
        ("GET", "/api/audit", "admin"),
    ])
    def test_mapping(self, method, path, role):
        assert auth.required_role(method, path) == role

    def test_role_order(self):
        assert auth.allows("admin", "operator")
        assert auth.allows("operator", "designer")
        assert not auth.allows("designer", "operator")
        assert not auth.allows("viewer", "designer")
        assert not auth.allows(None, "viewer")
        assert not auth.allows("pilot", "viewer")


# ==========================================================================
# The server
# ==========================================================================


class TestBootstrap:
    def test_no_accounts_local_caller_is_let_in(self):
        """Existing single-machine use keeps working with no setup."""
        c = TestClient(server.app)
        assert c.get("/api/design/shows").status_code == 200
        status = c.get("/api/auth/status").json()
        assert status == {"configured": False, "local": True,
                          "user": None, "role": "admin"}

    def test_no_accounts_remote_caller_is_refused(self):
        remote = TestClient(server.app, client=("10.1.2.3", 50000))
        r = remote.get("/api/design/shows")
        assert r.status_code == 403
        assert "user add" in r.json()["detail"]
        assert remote.post("/api/ops/arm", json={}).status_code == 403

    def test_the_page_itself_is_always_served(self):
        remote = TestClient(server.app, client=("10.1.2.3", 50000))
        assert remote.get("/").status_code in (200, 404)  # 404 only if web/ is absent

    def test_the_page_is_revalidated_so_upgrades_reach_the_browser(self):
        r = TestClient(server.app).get("/")
        if r.status_code == 200:
            assert r.headers["cache-control"] == "no-cache"

    def test_corrupt_user_file_refuses_everything(self, tmp_path):
        server.USERS_FILE.write_text("{ broken", encoding="utf-8")
        server._auth_cache.clear()
        c = TestClient(server.app)
        r = c.get("/api/design/shows")
        assert r.status_code == 503, "a corrupt user file must fail closed"


class TestSignIn:
    def test_unauthenticated_is_401(self, accounts):
        assert TestClient(server.app).get("/api/design/shows").status_code == 401

    def test_login_sets_a_hardened_cookie(self, accounts):
        c = TestClient(server.app)
        r = c.post("/api/auth/login", json={"username": "root", "password": PW["root"]})
        cookie = r.headers["set-cookie"].lower()
        assert server.SESSION_COOKIE in cookie
        assert "httponly" in cookie, "script must not be able to read the session"
        assert "samesite=strict" in cookie, "cross-site requests must not carry it"

    def test_wrong_password_is_401_with_no_cookie(self, accounts):
        r = TestClient(server.app).post(
            "/api/auth/login", json={"username": "root", "password": "nope nope nope"})
        assert r.status_code == 401
        assert "set-cookie" not in r.headers

    def test_bearer_token_works_for_scripts(self, accounts):
        token = TestClient(server.app).post(
            "/api/auth/login", json={"username": "vic", "password": PW["vic"]}
        ).json()["token"]
        c = TestClient(server.app)
        assert c.get("/api/design/shows").status_code == 401
        assert c.get("/api/design/shows",
                     headers={"Authorization": f"Bearer {token}"}).status_code == 200

    def test_logout_ends_the_session(self, accounts):
        c = as_user("vic")
        assert c.get("/api/design/shows").status_code == 200
        c.post("/api/auth/logout")
        assert c.get("/api/design/shows").status_code == 401

    def test_me(self, accounts):
        assert as_user("dina").get("/api/auth/me").json() == {
            "user": "dina", "role": "designer"}


SHOW = {"name": "secure", "drone_count": 6,
        "limits": {"max_altitude": 200.0, "battery_endurance_s": 1800.0}}


class TestRoleBoundaries:
    """Each role can do its job, and is stopped one step beyond it."""

    def test_viewer_reads_but_cannot_write(self, accounts):
        c = as_user("vic")
        assert c.get("/api/design/shows").status_code == 200
        assert c.post("/api/design/shows", json=SHOW).status_code == 403

    def test_designer_designs_but_cannot_fly(self, accounts):
        c = as_user("dina")
        assert c.post("/api/design/shows", json=SHOW).status_code == 200
        for path in ("/api/ops/load", "/api/ops/arm", "/api/ops/start",
                     "/api/ops/emergency", "/api/fleet/drones/bulk"):
            r = c.post(path, json={})
            assert r.status_code == 403, f"designer reached {path}"
            assert "needs operator" in r.json()["detail"]

    def test_operator_flies_but_cannot_manage_accounts(self, accounts):
        c = as_user("olive")
        assert c.post("/api/design/shows", json=SHOW).status_code == 200
        assert c.post("/api/ops/load", json={"show": "secure"}).status_code != 403
        assert c.get("/api/users").status_code == 403
        assert c.get("/api/audit").status_code == 403

    def test_admin_manages_accounts(self, accounts):
        c = as_user("root")
        names = {u["name"] for u in c.get("/api/users").json()["users"]}
        assert names == set(PW)
        r = c.post("/api/users", json={"name": "new1", "password": "fresh password 5",
                                       "role": "viewer"})
        assert r.status_code == 200
        assert c.patch("/api/users/new1", json={"role": "designer"}).status_code == 200

    def test_user_listing_never_exposes_hashes(self, accounts):
        body = as_user("root").get("/api/users").text
        assert "pbkdf2" not in body and "password" not in body

    def test_admin_cannot_delete_their_own_account(self, accounts):
        r = as_user("root").delete("/api/users/root")
        assert r.status_code == 400 and "signed in" in r.text

    def test_weak_password_via_api_is_refused(self, accounts):
        r = as_user("root").post("/api/users", json={
            "name": "weak", "password": "short", "role": "viewer"})
        assert r.status_code == 400


class TestVoiceCannotBypassRoles:
    """Voice calls handlers directly, so it must enforce roles itself."""

    def say(self, client, text, show=None, confirmed=False):
        return client.post("/api/voice", json={
            "text": text, "context": {"show": show}, "confirmed": confirmed,
        }).json()

    @pytest.mark.parametrize("phrase", [
        "upload the plans", "suspend", "collective return home",
        "land now", "load the fleet on the simulator", "run preflight",
    ])
    def test_viewer_cannot_operate_by_voice(self, accounts, phrase):
        out = self.say(as_user("vic"), phrase, confirmed=True)
        assert out.get("forbidden") is True, f"viewer got through with {phrase!r}"
        assert out["executed"] is False

    def test_viewer_cannot_design_by_voice(self, accounts):
        out = self.say(as_user("vic"), "create a show called sneaky with ten drones")
        assert out.get("forbidden") is True
        assert TestClient(server.app).get(
            "/api/design/shows/sneaky",
            cookies=as_user("root").cookies).status_code == 404

    def test_designer_cannot_operate_by_voice(self, accounts):
        out = self.say(as_user("dina"), "suspend")
        assert out.get("forbidden") is True
        assert "operator" in out["speech"]

    def test_designer_can_design_by_voice(self, accounts):
        out = self.say(as_user("dina"), "create a show called fine with ten drones")
        assert out["executed"], out

    def test_viewer_can_still_ask_for_status_and_help(self, accounts):
        assert not self.say(as_user("vic"), "status").get("forbidden")
        assert not self.say(as_user("vic"), "help").get("forbidden")

    def test_refusals_still_win_for_admins(self, accounts):
        """Role never unlocks arming by voice - that is not a permission."""
        out = self.say(as_user("root"), "arm the fleet", confirmed=True)
        assert out.get("refused") is True


class TestAudit:
    def test_mutations_are_recorded_with_identity(self, accounts):
        as_user("dina").post("/api/design/shows", json=SHOW)
        entries = server._audit().tail()
        hit = [e for e in entries if e.get("path") == "/api/design/shows"
               and e.get("method") == "POST"]
        assert hit and hit[-1]["user"] == "dina" and hit[-1]["status"] == 200

    def test_denials_are_recorded(self, accounts):
        as_user("vic").post("/api/ops/arm", json={})
        assert any(e.get("denied") and e.get("user") == "vic"
                   for e in server._audit().tail())

    def test_failed_logins_are_recorded_without_the_password(self, accounts):
        TestClient(server.app).post("/api/auth/login", json={
            "username": "root", "password": "the guess was this"})
        raw = server.AUDIT_FILE.read_text(encoding="utf-8")
        assert '"login": "failed"' in raw
        assert "the guess was this" not in raw

    def test_reads_are_not_recorded(self, accounts):
        before = len(server._audit().tail(10_000))
        as_user("vic").get("/api/design/shows")
        after = [e for e in server._audit().tail(10_000)][before:]
        assert not [e for e in after if e.get("method") == "GET"]

    def test_admin_can_read_the_trail(self, accounts):
        as_user("dina").post("/api/design/shows", json=SHOW)
        entries = as_user("root").get("/api/audit").json()["entries"]
        assert any(e.get("user") == "dina" for e in entries)

    def test_the_trail_is_append_only(self, accounts):
        audit = server._audit()
        audit.write(user="a", method="POST", path="/x", status=200)
        first = server.AUDIT_FILE.read_text(encoding="utf-8")
        audit.write(user="b", method="POST", path="/y", status=200)
        assert server.AUDIT_FILE.read_text(encoding="utf-8").startswith(first)


class TestWebSocket:
    def test_signed_out_socket_is_closed(self, accounts):
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect) as exc:
            with TestClient(server.app).websocket_connect("/ws") as ws:
                ws.receive_json()
        assert exc.value.code == 4401

    def test_signed_in_socket_streams(self, accounts):
        with as_user("vic").websocket_connect("/ws") as ws:
            assert "phase" in ws.receive_json()

    def test_remote_socket_with_no_accounts_is_closed(self):
        from starlette.websockets import WebSocketDisconnect

        remote = TestClient(server.app, client=("10.1.2.3", 50000))
        with pytest.raises(WebSocketDisconnect) as exc:
            with remote.websocket_connect("/ws") as ws:
                ws.receive_json()
        assert exc.value.code == 4403


class TestCli:
    def test_add_and_list_users_from_stdin(self, monkeypatch, capsys):
        from skyshine import cli

        monkeypatch.setattr("sys.stdin", io.StringIO("cli password 123\n"))
        assert cli.main(["user", "add", "ops1", "--role", "operator",
                         "--password-stdin"]) == 0
        assert cli.main(["user", "list"]) == 0
        out = capsys.readouterr().out
        assert "ops1" in out and "operator" in out
        assert "cli password 123" not in out

    def test_weak_password_from_cli_is_refused(self, monkeypatch, capsys):
        from skyshine import cli

        monkeypatch.setattr("sys.stdin", io.StringIO("short\n"))
        assert cli.main(["user", "add", "x", "--password-stdin"]) == 2
        assert "at least" in capsys.readouterr().err


def test_users_file_is_written_atomically(tmp_path):
    """A crash mid-write must not leave a half file that fails everything."""
    store = auth.AuthStore(tmp_path / "users.json")
    store.add_user("root", PW["root"], "admin")
    assert not (tmp_path / "users.tmp").exists()
    json.loads((tmp_path / "users.json").read_text(encoding="utf-8"))


def test_login_does_not_block_other_requests(tmp_path):
    """The deliberately slow hash must run outside the store lock."""
    import threading

    store = auth.AuthStore(tmp_path / "users.json")
    store.add_user("root", PW["root"], "admin")
    session = store.login("root", PW["root"])

    t = threading.Thread(target=lambda: store.login("root", PW["root"]))
    t.start()
    started = time.perf_counter()
    for _ in range(50):
        store.session(session.token)
    elapsed = time.perf_counter() - started
    t.join()
    assert elapsed < 0.15, f"session lookups stalled {elapsed:.2f}s behind a login"
