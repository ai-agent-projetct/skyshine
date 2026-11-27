"""Who is allowed to do what.

Before this module the server had no authentication at all: anything that
could reach the port could upload a plan and arm a fleet. Bound to
localhost that was contained. Served to a field network so a second laptop
or a tablet can watch telemetry, it was an open launch button.

Design choices, each for a reason:

**Server-side sessions, not JWTs.** A token here is a random string that
maps to a session held in memory. It can be revoked instantly - which is
exactly what you want the moment an operator's laptop walks off - and it
needs no signing library. The cost is that a server restart logs everyone
out, which on a show night is a feature.

**PBKDF2 from the standard library.** 600,000 iterations of SHA-256, a
random salt per user, and a constant-time comparison. No new dependency in
the code path that guards the arm button.

**Four ordered roles.** viewer < designer < operator < admin. An operator
can also edit shows, because the person flying the show is the one who has
to adapt it when two airframes fail preflight an hour before; a designer
cannot fly. Organisations and multi-tenant separation are left out: this
runs one fleet at one site.

**Bootstrap is local-only, not open.** With no users configured the server
still works - otherwise nobody would ever turn this on - but only for
requests from the machine it runs on. Anything arriving over the network
is refused until an admin account exists.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROLES = ("viewer", "designer", "operator", "admin")
RANK = {role: i for i, role in enumerate(ROLES)}

# OWASP's 2023 figure for PBKDF2-HMAC-SHA256. Tests lower the working
# value for speed; PRODUCTION_ITERATIONS is what a test checks against.
PRODUCTION_ITERATIONS = 600_000
PBKDF2_ITERATIONS = PRODUCTION_ITERATIONS
MIN_PASSWORD = 10
SESSION_TTL_S = 12 * 3600          # one show night
LOCKOUT_FAILURES = 5
LOCKOUT_S = 60.0

# A dummy hash, verified against when the username does not exist, so a
# failed login costs one PBKDF2 run either way and response timing does not
# reveal which usernames are real. Computed once, lazily: computing it per
# request would make the unknown-user path cost two runs instead of one,
# which is the very difference it is meant to hide.
_DUMMY_HASH: str | None = None


def _dummy_hash() -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password(secrets.token_urlsafe(16), bytes(16))
    return _DUMMY_HASH


class AuthError(ValueError):
    """Credentials, permissions or account changes that were refused."""


def hash_password(password: str, salt: bytes | None = None,
                  iterations: int | None = None) -> str:
    # Read at call time, not bound as a default, so the work factor can be
    # raised without every caller having to know.
    iterations = iterations or PBKDF2_ITERATIONS
    salt = salt if salt is not None else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt_hex, digest_hex = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(candidate.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def check_password_strength(password: str, username: str = "") -> None:
    if len(password) < MIN_PASSWORD:
        raise AuthError(f"password must be at least {MIN_PASSWORD} characters")
    if username and username.lower() in password.lower():
        raise AuthError("password must not contain the username")
    if len(set(password)) < 4:
        raise AuthError("password is too repetitive")


@dataclass
class User:
    name: str
    role: str
    password_hash: str
    created: float = field(default_factory=time.time)
    disabled: bool = False

    def to_public(self) -> dict:
        return {"name": self.name, "role": self.role, "created": self.created,
                "disabled": self.disabled}


@dataclass
class Session:
    token: str
    user: str
    role: str
    expires: float
    created: float = field(default_factory=time.time)


class AuthStore:
    """Users on disk, sessions in memory, failed logins counted per name."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._failures: dict[str, list[float]] = {}
        self.users: dict[str, User] = {}
        self.reload()
        # Warm the dummy hash now, so the first unknown-user login does not
        # cost an extra PBKDF2 run and stand out by its timing.
        _dummy_hash()

    # -- users ------------------------------------------------------------

    def reload(self) -> None:
        if not self.path.exists():
            self.users = {}
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Fail closed: a corrupt user file must not quietly turn into
            # "no users", which would put the server back in open bootstrap.
            raise AuthError(f"user file {self.path} is unreadable: {exc}") from None
        self.users = {u["name"]: User(**u) for u in data.get("users", [])}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"format": "skyshine.users", "version": 1,
             "users": [u.__dict__ for u in self.users.values()]},
            indent=1), encoding="utf-8")
        tmp.replace(self.path)    # atomic on the same filesystem

    @property
    def configured(self) -> bool:
        return any(not u.disabled for u in self.users.values())

    def add_user(self, name: str, password: str, role: str) -> User:
        name = (name or "").strip()
        if not name or not name.replace("_", "").replace("-", "").replace(".", "").isalnum():
            raise AuthError("username must be letters, digits, '.', '-' or '_'")
        if role not in ROLES:
            raise AuthError(f"role must be one of {', '.join(ROLES)}")
        with self._lock:
            if name in self.users:
                raise AuthError(f"user {name!r} already exists")
            check_password_strength(password, name)
            user = User(name=name, role=role, password_hash=hash_password(password))
            self.users[name] = user
            self._save()
        return user

    def set_password(self, name: str, password: str) -> None:
        with self._lock:
            user = self._get(name)
            check_password_strength(password, name)
            user.password_hash = hash_password(password)
            self._save()
            self._revoke_user(name)

    def set_role(self, name: str, role: str) -> None:
        if role not in ROLES:
            raise AuthError(f"role must be one of {', '.join(ROLES)}")
        with self._lock:
            user = self._get(name)
            if user.role == "admin" and role != "admin" and self._last_admin(name):
                raise AuthError("cannot demote the last admin; nobody could manage users")
            user.role = role
            self._save()
            self._revoke_user(name)   # role changes take effect immediately

    def remove_user(self, name: str) -> None:
        with self._lock:
            user = self._get(name)
            if user.role == "admin" and self._last_admin(name):
                raise AuthError("cannot remove the last admin")
            del self.users[name]
            self._save()
            self._revoke_user(name)

    def _get(self, name: str) -> User:
        try:
            return self.users[name]
        except KeyError:
            raise AuthError(f"no user {name!r}") from None

    def _last_admin(self, name: str) -> bool:
        others = [u for u in self.users.values()
                  if u.role == "admin" and not u.disabled and u.name != name]
        return not others

    # -- sessions ---------------------------------------------------------

    def login(self, name: str, password: str, now: float | None = None) -> Session:
        now = time.time() if now is None else now
        with self._lock:
            recent = [t for t in self._failures.get(name, []) if now - t < LOCKOUT_S]
            self._failures[name] = recent
            if len(recent) >= LOCKOUT_FAILURES:
                wait = int(LOCKOUT_S - (now - recent[0])) + 1
                raise AuthError(f"too many failed attempts; try again in {wait}s")
            user = self.users.get(name)
            stored = user.password_hash if user else _dummy_hash()

        # The hash runs outside the lock. It takes a large fraction of a
        # second by design, and holding the lock would stall every other
        # request's session lookup - including telemetry - for each login.
        ok = verify_password(password, stored) and user is not None

        with self._lock:
            if not ok or user.disabled:
                self._failures.setdefault(name, []).append(now)
                # Same message for unknown user and wrong password: which
                # one it was is exactly what an attacker wants to learn.
                raise AuthError("wrong username or password")

            self._failures.pop(name, None)
            token = secrets.token_urlsafe(32)
            session = Session(token=token, user=user.name, role=user.role,
                              expires=now + SESSION_TTL_S)
            self._sessions[token] = session
            return session

    def session(self, token: str | None, now: float | None = None) -> Session | None:
        if not token:
            return None
        now = time.time() if now is None else now
        with self._lock:
            s = self._sessions.get(token)
            if s is None:
                return None
            if s.expires <= now:
                del self._sessions[token]
                return None
            return s

    def logout(self, token: str | None) -> None:
        with self._lock:
            self._sessions.pop(token or "", None)

    def _revoke_user(self, name: str) -> None:
        for token in [t for t, s in self._sessions.items() if s.user == name]:
            del self._sessions[token]


def allows(role: str | None, required: str) -> bool:
    return role is not None and RANK.get(role, -1) >= RANK[required]


def required_role(method: str, path: str) -> str | None:
    """Minimum role for a request, or None when no login is needed.

    Reads are open to every signed-in user. Writes are graded by what they
    can do to an aircraft: editing a show cannot launch anything, so a
    designer may; anything under /api/ops or /api/fleet can, so it needs an
    operator; managing accounts and reading the audit trail is admin only.
    """
    m = method.upper()
    if path in ("/", "/simple") or path.startswith("/static") or path in (
        "/api/auth/login", "/api/auth/status", "/favicon.ico",
    ):
        return None
    if path.startswith("/api/users") or path.startswith("/api/audit") or (
        path == "/api/ai/key" and m not in ("GET", "HEAD", "OPTIONS")
    ):
        return "admin"
    if path in ("/api/auth/logout", "/api/auth/me"):
        return "viewer"
    if m in ("GET", "HEAD", "OPTIONS"):
        return "viewer"
    if path.startswith("/api/ops") or path.startswith("/api/fleet"):
        return "operator"
    if path == "/api/voice":
        # The voice endpoint is a router: the role it needs depends on the
        # intent, and is checked again once the phrase has been parsed.
        return "viewer"
    return "designer"


# Voice intents mapped onto the role their action needs. The voice
# endpoint calls the underlying handlers directly, so without this a viewer
# could say "upload the plans" and bypass the route checks entirely.
INTENT_ROLES = {
    "create_show": "designer", "add_formation": "designer",
    "apply_effect": "designer", "set_music": "designer", "compose": "designer",
    "prepare": "designer", "validate": "designer", "adapt": "designer",
    "auto_geofence": "designer", "export": "designer",
    "ops_load": "operator", "preflight": "operator", "upload": "operator",
    "suspend": "operator", "resume": "operator", "collective_rth": "operator",
    "land": "operator", "save_log": "operator",
}


class Audit:
    """Append-only record of who changed what.

    JSON lines, one per state-changing request plus every login attempt.
    Append-only on purpose: the audit trail is evidence, and evidence that
    can be edited in place is not evidence.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, **entry) -> None:
        entry.setdefault("ts", time.time())
        line = json.dumps(entry, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def tail(self, limit: int = 200) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out


LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def is_local(host: str | None) -> bool:
    return (host or "") in LOCAL_HOSTS
