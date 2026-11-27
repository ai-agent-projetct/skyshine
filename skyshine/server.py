"""HTTP and WebSocket API, plus the static designer and operations UI.

Two surfaces that deliberately do not share state beyond the show library:

  /api/design/*   create and edit shows, generate formations, validate
  /api/ops/*      load one validated show onto a fleet and fly it

The design side is stateless per request and reads and writes the show
library on disk. The operations side holds exactly one FleetManager,
because a ground station commands one swarm and a second concurrent
session would be a second chain of command.

Previews do not ship sampled trajectories. The browser receives the
formations and their timings and runs the same easing locally, which is a
few thousand numbers rather than the hundred megabytes a 500 drone
sampled timeline would be, and keeps scrubbing smooth.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import (
    FastAPI, File, Form, HTTPException, Request, Response, UploadFile, WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import geometry, planning, safety
from . import __version__
from .drivers import DRIVERS, SimDriver, SimFaults
from .fleet import FleetManager, Phase, PreflightPolicy
from .show import Formation, Motion, Show, ShowLimits, ShowOrigin, load_show
from .show import save_show as _write_show
from .trajectory import build_mission, build_timeline, mission_key, verify_mission

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
SHOW_DIR = ROOT / "shows"
LOG_DIR = ROOT / "logs"
ASSET_DIR = ROOT / "assets"
FLEET_FILE = ROOT / "fleet.json"
KEY_FILE = ROOT / "mission.key"
# MAVLink 2 link signing key. Separate from KEY_FILE: this one is loaded onto
# every aircraft, so it leaves the ground station; the mission key never does.
LINK_KEY_FILE = ROOT / "link.key"
USERS_FILE = ROOT / "users.json"
AUDIT_FILE = ROOT / "logs" / "audit.jsonl"
SESSION_COOKIE = "skyshine_session"

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    yield
    # A virtual fleet owns sockets and a thread; leaving them behind would
    # hold the ports and make the next run look like a dead link.
    _stop_sitl()


app = FastAPI(title="Sky Shine", version="1.2.0", lifespan=_lifespan)


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------

_auth_cache: dict[str, Any] = {}


def _auth():
    """The user store for the current USERS_FILE.

    Keyed on the path so that tests, which point USERS_FILE somewhere
    temporary, get their own store rather than whichever was built first.
    """
    from . import auth

    key = str(USERS_FILE)
    store = _auth_cache.get(key)
    if store is None:
        store = auth.AuthStore(USERS_FILE)
        _auth_cache.clear()
        _auth_cache[key] = store
    return store


def _audit():
    from . import auth

    return auth.Audit(AUDIT_FILE)


def _token(request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.cookies.get(SESSION_COOKIE)


@app.middleware("http")
async def _guard(request: Request, call_next):
    """Every request passes here: who is it, and may they do this?

    Unauthenticated requests are refused before they reach a handler,
    rather than each handler remembering to check - the one that forgets is
    the one that arms a fleet. State-changing requests are written to the
    audit trail with the identity behind them and the outcome.
    """
    from . import auth

    path = request.url.path
    method = request.method.upper()
    need = auth.required_role(method, path)
    host = request.client.host if request.client else None

    if need is None:
        return await call_next(request)

    try:
        store = _auth()
    except auth.AuthError as exc:
        # A corrupt user file fails closed: refusing everything is safer
        # than treating it as "no users" and opening bootstrap mode.
        return JSONResponse({"detail": str(exc)}, status_code=503)

    if not store.configured:
        if not auth.is_local(host):
            return JSONResponse({
                "detail": (
                    "no user accounts exist yet, so this server only accepts "
                    "requests from the machine it runs on. Create an admin with "
                    "`python -m skyshine user add <name> --role admin`."
                )}, status_code=403)
        request.state.user = None
        request.state.role = "admin"
    else:
        session = store.session(_token(request))
        if session is None:
            return JSONResponse({"detail": "sign in required"}, status_code=401)
        if not auth.allows(session.role, need):
            _audit().write(user=session.user, role=session.role, method=method,
                           path=path, status=403, ip=host, denied=True)
            return JSONResponse({
                "detail": f"{session.role} cannot do this; it needs {need}"},
                status_code=403)
        request.state.user = session.user
        request.state.role = session.role

    response = await call_next(request)
    if method not in ("GET", "HEAD", "OPTIONS"):
        _audit().write(
            user=getattr(request.state, "user", None),
            role=getattr(request.state, "role", None),
            method=method, path=path, status=response.status_code, ip=host,
        )
    return response


class LoginIn(BaseModel):
    username: str
    password: str


@app.get("/api/auth/status")
def auth_status(request: Request) -> dict:
    """Public: does this server need a login, and is this caller signed in?"""
    from . import auth

    store = _auth()
    session = store.session(_token(request))
    host = request.client.host if request.client else None
    return {
        "configured": store.configured,
        "local": auth.is_local(host),
        "user": session.user if session else None,
        "role": session.role if session else ("admin" if not store.configured
                                              and auth.is_local(host) else None),
    }


@app.post("/api/auth/login")
def auth_login(body: LoginIn, request: Request, response: Response) -> dict:
    from . import auth

    store = _auth()
    host = request.client.host if request.client else None
    try:
        session = store.login(body.username, body.password)
    except auth.AuthError as exc:
        _audit().write(user=body.username, method="POST", path="/api/auth/login",
                       status=401, ip=host, login="failed")
        raise HTTPException(401, str(exc)) from None
    _audit().write(user=session.user, role=session.role, method="POST",
                   path="/api/auth/login", status=200, ip=host, login="ok")
    response.set_cookie(
        SESSION_COOKIE, session.token, max_age=auth.SESSION_TTL_S,
        httponly=True, samesite="strict",
        secure=request.url.scheme == "https",
    )
    return {"user": session.user, "role": session.role, "token": session.token,
            "expires": session.expires}


@app.post("/api/auth/logout")
def auth_logout(request: Request, response: Response) -> dict:
    _auth().logout(_token(request))
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


@app.get("/api/auth/me")
def auth_me(request: Request) -> dict:
    return {"user": getattr(request.state, "user", None),
            "role": getattr(request.state, "role", None)}


class UserIn(BaseModel):
    name: str
    password: str
    role: str = "viewer"


class UserPatch(BaseModel):
    role: str | None = None
    password: str | None = None


@app.get("/api/users")
def users_list() -> dict:
    return {"users": [u.to_public() for u in _auth().users.values()]}


@app.post("/api/users")
def users_add(body: UserIn) -> dict:
    from . import auth

    try:
        user = _auth().add_user(body.name, body.password, body.role)
    except auth.AuthError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"ok": True, "user": user.to_public()}


@app.patch("/api/users/{name}")
def users_patch(name: str, body: UserPatch) -> dict:
    from . import auth

    store = _auth()
    try:
        if body.role is not None:
            store.set_role(name, body.role)
        if body.password is not None:
            store.set_password(name, body.password)
    except auth.AuthError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"ok": True, "user": store.users[name].to_public()}


@app.delete("/api/users/{name}")
def users_remove(name: str, request: Request) -> dict:
    from . import auth

    if name == getattr(request.state, "user", None):
        raise HTTPException(400, "you cannot remove the account you are signed in with")
    try:
        _auth().remove_user(name)
    except auth.AuthError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"ok": True}


@app.get("/api/audit")
def audit_tail(limit: int = 200) -> dict:
    return {"entries": _audit().tail(max(1, min(limit, 5000)))}

# The single live fleet, and a lock so two API calls cannot both decide to
# abort. Operations are short; contention is not a concern, correctness is.
_fleet: FleetManager | None = None
_fleet_lock = asyncio.Lock()
_ops_task: asyncio.Task | None = None
# A virtual MAVLink fleet, when the operator chose software in the loop.
_sitl = None


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------


class LimitsIn(BaseModel):
    min_separation: float = 3.0
    formation_separation: float = 0.0
    takeoff_grid_spacing: float = 3.0
    takeoff_layer_gap: float = 3.0
    takeoff_layers: int = 4
    min_nav_altitude: float = 2.5
    max_altitude: float = 120.0
    max_velocity_xy: float = 8.0
    max_velocity_up: float = 4.0
    max_velocity_down: float = 3.0
    max_acceleration: float = 4.0
    motion_speed_share: float = 0.5
    geofence: list[list[float]] = Field(default_factory=list)
    keep_out: list[dict] = Field(default_factory=list)
    battery_endurance_s: float = 900.0
    battery_reserve_s: float = 180.0


class OriginIn(BaseModel):
    latitude: float = 0.0
    longitude: float = 0.0
    amsl: float | None = None
    orientation_deg: float = 0.0


class ShowIn(BaseModel):
    name: str
    drone_count: int = Field(ge=1, le=2000)
    limits: LimitsIn = Field(default_factory=LimitsIn)
    origin: OriginIn = Field(default_factory=OriginIn)
    fps: float = 10.0
    notes: str = ""


class FormationIn(BaseModel):
    name: str
    shape: str
    params: dict[str, Any] = Field(default_factory=dict)
    color: list[int] = Field(default_factory=lambda: [255, 255, 255])
    altitude: float = 50.0
    transition: float = 8.0
    hold: float = 4.0
    easing: str = "minjerk"
    # Degrees about the vertical axis, for aiming a billboard at the crowd.
    yaw: float = 0.0
    motion: "MotionIn | None" = None
    # Use only this many drones; the rest wait unlit behind. None uses all.
    drones: int | None = None


class MotionIn(BaseModel):
    spin: float = 0.0
    move: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    grow: float = 1.0


class LoadIn(BaseModel):
    show: str
    driver: str = "simulator"
    endpoints: list[str] = Field(default_factory=list)
    policy: dict[str, Any] = Field(default_factory=dict)
    faults: dict[str, Any] = Field(default_factory=dict)
    # Sign every MAVLink message with link.key and drop any that are not.
    signed: bool = False


class EmergencyIn(BaseModel):
    action: str
    drones: list[int] | None = None
    reason: str = "operator"


class OverrideIn(BaseModel):
    override_reason: str | None = None


class StartIn(BaseModel):
    lead_in: float = 5.0
    at_unix_time: float | None = None


# --------------------------------------------------------------------------
# Show library
# --------------------------------------------------------------------------


def _show_path(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
    if not safe:
        raise HTTPException(400, "show name must contain usable characters")
    return SHOW_DIR / f"{safe}.show.json"


def _require_show(name: str) -> Show:
    p = _show_path(name)
    if not p.exists():
        raise HTTPException(404, f"no show named {name!r}")
    return load_show(p)


# Undo history: the show file as it was before each save.
# ponytail: held in memory, so a server restart forgets it. Write the
# snapshots to disk if history ever has to survive one.
UNDO_DEPTH = 30
_undo: dict[Path, list[str]] = {}
_redo: dict[Path, list[str]] = {}


def save_show(show: Show, path: str | Path) -> Path:
    """Every design write in this module goes through here, so every one of
    them - formations, effects, music, pyro, reorder, voice - is undoable."""
    path = Path(path)
    if path.exists():
        stack = _undo.setdefault(path, [])
        stack.append(path.read_text(encoding="utf-8"))
        del stack[:-UNDO_DEPTH]
    _redo.pop(path, None)
    return _write_show(show, path)


def _step_history(name: str, take: dict, give: dict, what: str) -> dict:
    path = _show_path(name)
    if not path.exists():
        raise HTTPException(404, f"no show named {name!r}")
    if not take.get(path):
        raise HTTPException(409, f"nothing to {what}")
    give.setdefault(path, []).append(path.read_text(encoding="utf-8"))
    path.write_text(take[path].pop(), encoding="utf-8")
    return {"ok": True, "undo": len(_undo.get(path, [])),
            "redo": len(_redo.get(path, []))}


@app.get("/api/shapes")
def list_shapes() -> dict:
    """The shape catalogue, with the arguments each one accepts."""
    import inspect

    out = []
    for name, fn in sorted(geometry.SHAPES.items()):
        sig = inspect.signature(fn)
        params = [
            {
                "name": p.name,
                "default": None if p.default is inspect.Parameter.empty else p.default,
            }
            for p in sig.parameters.values()
            if p.name != "n"
        ]
        out.append({"name": name, "params": params, "doc": (fn.__doc__ or "").strip()})
    return {"shapes": out}


@app.get("/api/design/shows")
def list_shows() -> dict:
    SHOW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(SHOW_DIR.glob("*.show.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "name": data.get("name"),
                "drone_count": data.get("drone_count"),
                "formations": len(data.get("formations", [])),
                "modified": p.stat().st_mtime,
            }
        )
    return {"shows": rows}


@app.post("/api/design/shows")
def create_show(body: ShowIn) -> dict:
    show = Show(
        name=body.name,
        drone_count=body.drone_count,
        limits=ShowLimits(**body.limits.model_dump()),
        origin=ShowOrigin(**body.origin.model_dump()),
        fps=body.fps,
        notes=body.notes,
    )
    save_show(show, _show_path(show.name))
    return {"ok": True, "show": show.name, "capacity_hint": _capacity_hint(show)}


def _capacity_hint(show: Show) -> dict:
    """How big a billboard this drone count needs, before anyone designs one."""
    w, h = geometry.billboard_for(
        show.drone_count, show.limits.formation_separation
    )
    return {
        "formation_separation_m": round(show.limits.formation_separation, 3),
        "billboard_width_m": round(w, 1),
        "billboard_height_m": round(h, 1),
        "fits_under_ceiling": bool(h + show.limits.min_nav_altitude <= show.limits.max_altitude),
    }


@app.get("/api/design/shows/{name}")
def get_show(name: str) -> dict:
    from .show import show_to_dict

    show = _require_show(name)
    body = show_to_dict(show)
    body["duration"] = round(show.duration, 3)
    body["capacity_hint"] = _capacity_hint(show)
    return body


@app.delete("/api/design/shows/{name}")
def delete_show(name: str) -> dict:
    p = _show_path(name)
    if not p.exists():
        raise HTTPException(404, f"no show named {name!r}")
    p.unlink()
    _approval_path(name).unlink(missing_ok=True)
    _undo.pop(p, None)
    _redo.pop(p, None)
    return {"ok": True}


@app.post("/api/design/shows/{name}/undo")
def undo_show(name: str) -> dict:
    return _step_history(name, _undo, _redo, "undo")


@app.post("/api/design/shows/{name}/redo")
def redo_show(name: str) -> dict:
    return _step_history(name, _redo, _undo, "redo")


@app.post("/api/design/shows/{name}/duplicate")
def duplicate_show(name: str, save_as: str | None = None) -> dict:
    show = _require_show(name)
    target = save_as or f"{name} copy"
    if _show_path(target).exists():
        raise HTTPException(409, f"a show named {target!r} already exists")
    show.name = target
    save_show(show, _show_path(target))
    return {"ok": True, "show": target}


class TemplateIn(ShowIn):
    # The words a template's text formation spells; blank uses its default.
    text: str = Field("", max_length=40)


@app.get("/api/design/templates")
def list_templates() -> dict:
    from . import templates

    return {"templates": templates.catalogue()}


@app.post("/api/design/templates/{key}")
def create_from_template(key: str, body: TemplateIn) -> dict:
    """Start a show from a built-in storyboard instead of a blank one.

    The result is an ordinary unprepared show: it still has to be prepared
    and pass the safety engine before it can be uploaded.
    """
    from . import templates

    if key not in templates.TEMPLATES:
        raise HTTPException(404, f"no template named {key!r}")
    path = _show_path(body.name)
    if path.exists():
        raise HTTPException(409, f"a show named {body.name!r} already exists")
    try:
        show = templates.build(
            key, body.name, body.drone_count, text=body.text,
            limits=ShowLimits(**body.limits.model_dump()),
            origin=ShowOrigin(**body.origin.model_dump()),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    show.fps = body.fps
    save_show(show, path)
    return {
        "ok": True,
        "show": show.name,
        "formations": len(show.formations),
        "warnings": [
            w for i in range(len(show.formations)) if (w := _fit_warning(show, i))
        ],
    }


LIMIT_FIELDS = ("min_separation", "min_nav_altitude", "max_altitude", "max_velocity_xy",
                "max_velocity_up", "max_velocity_down", "max_acceleration",
                "motion_speed_share", "battery_endurance_s", "battery_reserve_s", "keep_out")


@app.patch("/api/design/shows/{name}/limits")
def patch_limits(name: str, body: dict[str, Any]) -> dict:
    """Change some flight limits, leaving every other one as it is."""
    import dataclasses

    show = _require_show(name)
    unknown = sorted(set(body) - set(LIMIT_FIELDS))
    if unknown:
        raise HTTPException(400, f"cannot change {', '.join(unknown)} here")
    changes = dict(body)
    if "min_separation" in changes:
        changes["formation_separation"] = 0.0      # re-derive from the new minimum
    try:
        show.limits = dataclasses.replace(show.limits, **changes)
        show.limits.validate()
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from None
    save_show(show, _show_path(name))
    return {"ok": True, "limits": asdict(show.limits), "capacity_hint": _capacity_hint(show)}


@app.put("/api/design/shows/{name}/settings")
def update_settings(name: str, body: ShowIn) -> dict:
    show = _require_show(name)
    show.limits = ShowLimits(**body.limits.model_dump())
    show.origin = ShowOrigin(**body.origin.model_dump())
    show.fps = body.fps
    show.notes = body.notes
    show.limits.validate()
    save_show(show, _show_path(name))
    return {"ok": True, "capacity_hint": _capacity_hint(show)}


@app.post("/api/design/shows/{name}/formations")
def add_formation(name: str, body: FormationIn) -> dict:
    show = _require_show(name)
    use = body.drones or show.drone_count
    if not 1 <= use <= show.drone_count:
        raise HTTPException(400, f"use between 1 and {show.drone_count} drones")
    try:
        points = geometry.make(body.shape, use, **body.params)
    except TypeError as exc:
        raise HTTPException(400, f"bad parameters for shape {body.shape!r}: {exc}") from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None

    if body.yaw:
        points = geometry.rotate_z(points, body.yaw)
    points = geometry.lift(points, body.altitude)
    colors = np.asarray(body.color, dtype=np.int32)
    if use < show.drone_count:
        if use > 1 and geometry.min_pair_distance(points) < show.limits.formation_separation:
            points = geometry.fit_separation(
                points, planning.design_separation(show.limits, show.fps)) + points.mean(axis=0)
        points = planning.park_spares(
            points, show.drone_count, planning.design_separation(show.limits, show.fps))
        colors = np.vstack([np.tile(colors, (use, 1)),
                            np.zeros((show.drone_count - use, 3), dtype=np.int32)])
    try:
        show.add(
            Formation(
                name=body.name,
                points=points,
                colors=colors,
                transition=body.transition,
                hold=body.hold,
                easing=body.easing,
                motion=None if body.motion is None else Motion(**body.motion.model_dump()),
            )
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    save_show(show, _show_path(name))
    return {
        "ok": True,
        "index": len(show.formations) - 1,
        "raw_min_separation_m": round(geometry.min_pair_distance(points), 3),
        "extent_m": [round(float(v), 1) for v in np.ptp(points, axis=0)],
        "warning": _fit_warning(show),
    }


def _fit_warning(show: Show, index: int = -1) -> str | None:
    """Why a formation (the newest by default) will fail once prepared, or
    None if it fits.

    Runs the same fit `prepare` will, on a copy, so the designer hears about
    a ceiling breach when they add the shape instead of after preparing.
    """
    import copy

    probe = copy.deepcopy(show.formations[index])
    try:
        notes = planning.fit_formation(probe, show.limits, show.fps)
    except ValueError as exc:
        return str(exc)
    if planning.height_range(probe)[1] > show.limits.max_altitude:
        return notes[-1]
    return None


@app.post("/api/design/shows/{name}/import-design")
async def import_design(
    name: str,
    file: UploadFile = File(...),
    formation: str = Form(""),
    mode: str = Form("auto"),
    up: str = Form("y"),
    size: float = Form(0.0),
    altitude: float = Form(50.0),
    color: str = Form("255,255,255"),
    hold: float = Form(5.0),
    transition: float = Form(8.0),
) -> dict:
    """Add a formation from a design made elsewhere: a 3D model (OBJ, STL,
    PLY, GLB) or a drawing (SVG). Drones are spread evenly over it; after
    that it is an ordinary formation, editable like any other."""
    from . import models

    show = _require_show(name)
    data = await file.read()
    if size <= 0:
        size = round(max(geometry.billboard_for(
            show.drone_count, show.limits.formation_separation)), 1)
    try:
        rgb = [max(0, min(255, int(v))) for v in color.split(",")][:3]
        if len(rgb) != 3:
            raise ValueError("colour must be three numbers, like 255,200,0")
        points, info = models.place(data, file.filename or "", show.drone_count,
                                    mode=mode, up=up, size=size)
        show.add(Formation(
            name=(formation or Path(file.filename or "design").stem)[:40],
            points=geometry.lift(points, altitude),
            colors=np.asarray(rgb, dtype=np.int32),
            transition=transition, hold=hold,
        ))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    save_show(show, _show_path(name))
    return {"ok": True, "index": len(show.formations) - 1, "source": info,
            "raw_min_separation_m": round(geometry.min_pair_distance(points), 3),
            "warning": _fit_warning(show)}


class DrawIn(BaseModel):
    strokes: list[list[list[float]]] = Field(max_length=500)
    name: str = Field("drawing", max_length=40)
    size: float = 0.0
    altitude: float = 50.0
    color: list[int] = Field(default_factory=lambda: [255, 255, 255])
    hold: float = 5.0
    transition: float = 8.0


@app.post("/api/design/shows/{name}/draw")
def draw_formation(name: str, body: DrawIn) -> dict:
    """Add a formation drawn by hand: drones spread along the lines."""
    from . import models

    show = _require_show(name)
    if sum(len(s) for s in body.strokes) > 50_000:
        raise HTTPException(400, "the drawing has too many points; draw fewer lines")
    size = body.size if body.size > 0 else round(max(geometry.billboard_for(
        show.drone_count, show.limits.formation_separation)), 1)
    try:
        points = models.place_strokes(body.strokes, show.drone_count, size=size)
        show.add(Formation(name=body.name, points=geometry.lift(points, body.altitude),
                           colors=np.asarray(body.color, dtype=np.int32),
                           transition=body.transition, hold=body.hold))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    save_show(show, _show_path(name))
    return {"ok": True, "index": len(show.formations) - 1,
            "extent_m": [round(float(v), 1) for v in np.ptp(points, axis=0)],
            "warning": _fit_warning(show)}


class DronesIn(BaseModel):
    # Drones are named by where they are, not by number: the preview numbers
    # drones in flight order, which differs from the stored order until the
    # show is prepared.
    near: list[list[float]] = Field(min_length=1, max_length=2000)
    move: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


def _pair_with_previous(show: Show, index: int) -> Formation:
    """Store a formation in the order that best follows the one before it,
    so that locking its order locks the sensible pairing, not an accident
    of how it was drawn."""
    if index == 0:
        raise HTTPException(400, "the first formation has nothing to fly from; "
                                 "pick a later one")
    f = show.formations[index]
    if not f.keep_order:
        prev = show.formations[index - 1].end_points
        show.formations[index] = f = planning.reorder(f, planning.match(prev, f.points))
    return f


def _slots_near(f: Formation, near: list[list[float]]) -> list[int]:
    slots = []
    for p in near:
        if len(p) != 3:
            raise HTTPException(400, "each position needs east, north and up")
        d = np.linalg.norm(f.points - np.asarray(p, dtype=np.float64), axis=1)
        k = int(np.argmin(d))
        if d[k] > 2.0:
            raise HTTPException(400, f"no drone within 2 m of {p}; reload and pick again")
        slots.append(k)
    return slots


@app.post("/api/design/shows/{name}/formations/{index}/nudge")
def nudge_drones(name: str, index: int, body: DronesIn) -> dict:
    """Move picked drones of one formation by a fixed offset."""
    show = _require_show(name)
    if not 0 <= index < len(show.formations):
        raise HTTPException(404, "no such formation")
    if len(body.move) != 3:
        raise HTTPException(400, "move needs east, north and up metres")
    f = show.formations[index]
    slots = sorted(set(_slots_near(f, body.near)))
    f.points[slots] += np.asarray(body.move, dtype=np.float64)
    save_show(show, _show_path(name))
    return {"ok": True, "moved": len(slots),
            "closest_m": round(geometry.min_pair_distance(f.points), 2),
            "warning": _fit_warning(show, index)}


@app.post("/api/design/shows/{name}/formations/{index}/swap")
def swap_drones(name: str, index: int, body: DronesIn) -> dict:
    """Make two drones trade places in a formation, and keep that order.

    The picture stays the same; which drone flies where changes. Locks the
    formation's order so preparing does not undo it. Swapped paths can
    cross, so the safety engine has to pass the show again.
    """
    show = _require_show(name)
    if not 0 <= index < len(show.formations):
        raise HTTPException(404, "no such formation")
    if len(body.near) != 2:
        raise HTTPException(400, "pick exactly two drones to swap")
    f = _pair_with_previous(show, index)
    a, b = _slots_near(f, body.near)
    if a == b:
        raise HTTPException(400, "those are the same drone")
    order = np.arange(len(f.points))
    order[[a, b]] = order[[b, a]]
    f.points, f.colors = f.points[order], f.colors[order]
    f.color_keys = [(t, c[order]) for t, c in f.color_keys]
    if f.yaw is not None:
        f.yaw = f.yaw[order]
    f.keep_order = True
    save_show(show, _show_path(name))
    return {"ok": True, "swapped": [a, b], "keep_order": True}


class OrderIn(BaseModel):
    keep_order: bool


@app.put("/api/design/shows/{name}/formations/{index}/order")
def set_order_lock(name: str, index: int, body: OrderIn) -> dict:
    """Lock or unlock which drone flies to which slot in a formation."""
    show = _require_show(name)
    if not 0 <= index < len(show.formations):
        raise HTTPException(404, "no such formation")
    if body.keep_order:
        _pair_with_previous(show, index).keep_order = True
    else:
        show.formations[index].keep_order = False
    save_show(show, _show_path(name))
    return {"ok": True, "keep_order": body.keep_order}


class PaintIn(BaseModel):
    near: list[list[float]] = Field(min_length=1, max_length=2000)
    color: list[int] = Field(min_length=3, max_length=3)


@app.post("/api/design/shows/{name}/formations/{index}/paint")
def paint_drones(name: str, index: int, body: PaintIn) -> dict:
    """Give picked drones one colour, for the whole hold, over any effect."""
    show = _require_show(name)
    if not 0 <= index < len(show.formations):
        raise HTTPException(404, "no such formation")
    f = show.formations[index]
    slots = sorted(set(_slots_near(f, body.near)))
    rgb = np.clip(np.asarray(body.color, dtype=np.int32), 0, 255)
    f.colors = f.colors.copy()
    f.colors[slots] = rgb
    for _, c in f.color_keys:
        c[slots] = rgb
    save_show(show, _show_path(name))
    return {"ok": True, "painted": len(slots)}


class PictureIn(BaseModel):
    asset_id: str
    view: str = "front"      # front: as the crowd sees it; top: from above


@app.post("/api/design/shows/{name}/formations/{index}/picture")
def colour_from_picture(name: str, index: int, body: PictureIn) -> dict:
    """Colour a formation from an uploaded picture, stretched over it.

    front maps the picture onto east and up, as the crowd sees the show;
    top onto east and north. Replaces any light effect on the formation.
    """
    from PIL import Image

    from . import assets

    show = _require_show(name)
    if not 0 <= index < len(show.formations):
        raise HTTPException(404, "no such formation")
    if body.view not in ("front", "top"):
        raise HTTPException(400, "view must be front or top")
    try:
        path = assets.resolve(ASSET_DIR, body.asset_id)
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)
    except (assets.AssetError, OSError) as exc:
        raise HTTPException(404, f"no usable picture {body.asset_id!r} ({exc})") from None
    f = show.formations[index]
    u_axis, v_axis = (0, 2) if body.view == "front" else (0, 1)
    p = f.points
    lo, span = p.min(axis=0), np.maximum(np.ptp(p, axis=0), 1e-9)
    u = (p[:, u_axis] - lo[u_axis]) / span[u_axis]
    v = (p[:, v_axis] - lo[v_axis]) / span[v_axis]
    h, w = img.shape[:2]
    col = img[np.round((1.0 - v) * (h - 1)).astype(int), np.round(u * (w - 1)).astype(int)]
    f.colors, f.color_keys = col.astype(np.int32), []
    save_show(show, _show_path(name))
    return {"ok": True, "lit": int(np.count_nonzero(col.sum(axis=1) > 30))}


class GenerateIn(BaseModel):
    kind: str                                   # qr, countdown or ticker
    text: str = Field("", max_length=200)
    start: int = 10
    seconds: float = 1.0
    rows: int = 7
    speed: float = 4.0
    lit: str = "dark"
    flat: bool = False
    color: list[int] = Field(default_factory=lambda: [255, 255, 255])
    altitude: float = 50.0
    transition: float = 8.0


@app.post("/api/design/shows/{name}/generate")
def generate_formation(name: str, body: GenerateIn) -> dict:
    """Add a generated formation: a QR code, a countdown or scrolling text.

    Drones are laid on a grid at the show's formation spacing, so preparing
    does not rescale them. The countdown and the ticker change lights only.
    """
    from . import generators

    show = _require_show(name)
    spacing = planning.design_separation(show.limits, show.fps)
    rgb = np.clip(np.asarray(body.color, dtype=np.int32), 0, 255)
    try:
        if body.kind == "qr":
            f = generators.qr_formation(body.text, show.drone_count, spacing,
                                        color=rgb, lit=body.lit, flat=body.flat)
        elif body.kind == "countdown":
            f = generators.countdown_formation(
                body.start, show.drone_count, spacing, color=rgb, seconds=body.seconds,
                max_height=show.limits.max_altitude - body.altitude - 5.0)
        elif body.kind == "ticker":
            f = generators.ticker_formation(body.text, show.drone_count, spacing,
                                            rows=body.rows, color=rgb, speed=body.speed)
        else:
            raise ValueError("kind must be qr, countdown or ticker")
        f.points = geometry.lift(f.points, body.altitude)
        f.transition = body.transition
        show.add(f)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    save_show(show, _show_path(name))
    return {"ok": True, "index": len(show.formations) - 1, "name": f.name,
            "hold_s": round(f.hold, 1),
            "extent_m": [round(float(v), 1) for v in np.ptp(f.points, axis=0)],
            "warning": _fit_warning(show)}


# --------------------------------------------------------------------------
# Venue: a picture on the ground and a 3D model of the site
# --------------------------------------------------------------------------


@app.get("/api/design/assets/{asset_id}/file")
def asset_file(asset_id: str) -> Response:
    """An uploaded picture as it was uploaded, for the ground and previews."""
    from . import assets

    try:
        return FileResponse(assets.resolve(ASSET_DIR, asset_id))
    except assets.AssetError as exc:
        raise HTTPException(404, str(exc)) from None


class VenuePictureIn(BaseModel):
    asset_id: str | None = None          # None removes the picture
    width_m: float = Field(200.0, gt=1, lt=20000)
    rotation: float = 0.0                # degrees anticlockwise from north-up
    x: float = 0.0                       # where its centre is, east of the origin
    y: float = 0.0                       # and north of it


@app.put("/api/design/shows/{name}/venue/picture")
def set_venue_picture(name: str, body: VenuePictureIn) -> dict:
    """Lay an uploaded aerial photo or site plan on the ground under the show."""
    from . import assets

    show = _require_show(name)
    if body.asset_id is None:
        show.venue.pop("picture", None)
    else:
        try:
            assets.resolve(ASSET_DIR, body.asset_id)
        except assets.AssetError as exc:
            raise HTTPException(404, str(exc)) from None
        show.venue["picture"] = body.model_dump()
    save_show(show, _show_path(name))
    return {"ok": True, "venue": show.venue}


def _venue_model(show: Show):
    from . import models

    m = show.venue.get("model") or {}
    path = m.get("path", "")
    if not re.fullmatch(r"venue/[0-9a-f]{16}\.(obj|stl|ply|glb)", path):
        raise HTTPException(404, "this show has no venue model")
    target = ASSET_DIR / path
    if not target.exists():
        raise HTTPException(404, "the venue model file is missing; upload it again")
    try:
        return m, models.venue_mesh(target.read_bytes(), "." + path.rsplit(".", 1)[1],
                                    scale=m["scale"], up=m["up"], rotation=m["rotation"],
                                    offset=(m["x"], m["y"], m["z"]))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


@app.post("/api/design/shows/{name}/venue/model")
async def upload_venue_model(
    name: str,
    file: UploadFile = File(...),
    scale: float = Form(1.0),
    up: str = Form("y"),
    rotation: float = Form(0.0),
    x: float = Form(0.0),
    y: float = Form(0.0),
    z: float = Form(0.0),
) -> dict:
    """Store a 3D model of the site (OBJ, STL, PLY, GLB) to show with the show.

    scale is metres per model unit: 1 for files in metres, 0.01 for
    centimetres, 0.3048 for feet.
    """
    from . import models

    show = _require_show(name)
    data = await file.read()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in models.MODEL_SUFFIXES:
        raise HTTPException(400, "use an OBJ, STL, PLY or GLB model of the site")
    try:
        verts, tris = models.venue_mesh(data, suffix, scale=scale, up=up, rotation=rotation,
                                        offset=(x, y, z))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    folder = ASSET_DIR / "venue"
    folder.mkdir(parents=True, exist_ok=True)
    stored = f"venue/{hashlib.sha256(data).hexdigest()[:16]}{suffix}"
    (ASSET_DIR / stored).write_bytes(data)
    show.venue["model"] = {"path": stored, "name": Path(file.filename or "site").name[:80],
                           "scale": scale, "up": up, "rotation": rotation, "x": x, "y": y, "z": z}
    save_show(show, _show_path(name))
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    return {"ok": True, "triangles": int(len(tris)),
            "size_m": [round(float(v), 1) for v in hi - lo],
            "top_m": round(float(hi[2]), 1)}


@app.get("/api/design/shows/{name}/venue/mesh")
def venue_mesh(name: str) -> dict:
    """The site model in show metres, flattened for the 3D view."""
    _, (verts, tris) = _venue_model(_require_show(name))
    return {"vertices": np.round(verts, 2).ravel().tolist(),
            "triangles": tris.astype(int).ravel().tolist()}


@app.delete("/api/design/shows/{name}/venue/model")
def remove_venue_model(name: str) -> dict:
    show = _require_show(name)
    show.venue.pop("model", None)
    save_show(show, _show_path(name))
    return {"ok": True}


@app.post("/api/design/shows/{name}/venue/keep-out")
def keep_out_from_venue(name: str, margin: float = 3.0) -> dict:
    """Protect the site model with a keep-out zone around all of it.

    One cylinder covering the whole model plus a margin. Generous for an
    L-shaped site, but never too small; draw tighter zones by hand if the
    show must fly over part of it.
    """
    show = _require_show(name)
    m, (verts, _) = _venue_model(show)
    cx, cy = (verts[:, 0].min() + verts[:, 0].max()) / 2, (verts[:, 1].min() + verts[:, 1].max()) / 2
    radius = float(np.hypot(verts[:, 0] - cx, verts[:, 1] - cy).max()) + margin
    # Round outwards: a protective zone may grow by a centimetre, never shrink.
    out_ = lambda v: math.ceil(v * 100) / 100   # noqa: E731
    zone = {"name": f"site: {m['name']}", "x": round(float(cx), 2), "y": round(float(cy), 2),
            "radius": out_(radius + 0.01), "bottom": -out_(-float(min(verts[:, 2].min(), 0.0))),
            "top": out_(float(verts[:, 2].max()) + margin)}
    zones = [z for z in show.limits.keep_out if z.get("name") != zone["name"]] + [zone]
    import dataclasses

    show.limits = dataclasses.replace(show.limits, keep_out=zones)
    save_show(show, _show_path(name))
    return {"ok": True, "zone": zone}


class TransformIn(BaseModel):
    move: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    turn: float = 0.0     # degrees about the vertical axis, anticlockwise from above
    tilt: float = 0.0     # degrees about the east axis; positive leans the top south, toward the crowd
    scale: float = 1.0


@app.put("/api/design/shows/{name}/formations/{index}/transform")
def transform_formation(name: str, index: int, body: TransformIn) -> dict:
    """Move, turn, tilt or scale a formation about its centre, once.

    The edit-mode counterpart to motion: this changes where the formation
    is, motion animates it while it is held. Undo puts it back.
    """
    show = _require_show(name)
    if not 0 <= index < len(show.formations):
        raise HTTPException(404, "no such formation")
    if len(body.move) != 3 or not 0.05 <= body.scale <= 20:
        raise HTTPException(400, "move needs three numbers and scale must be 0.05 to 20")
    f = show.formations[index]
    c = f.points.mean(axis=0)
    p = (f.points - c) * body.scale
    p = geometry.rotate_x(p, body.tilt)
    p = geometry.rotate_z(p, body.turn)
    f.points = p + c + np.asarray(body.move, dtype=np.float64)
    save_show(show, _show_path(name))
    return {"ok": True, "extent_m": [round(float(v), 1) for v in np.ptp(f.points, axis=0)],
            "warning": _fit_warning(show, index)}


@app.put("/api/design/shows/{name}/formations/{index}/motion")
def set_motion(name: str, index: int, body: MotionIn) -> dict:
    """Make a formation spin, travel or grow while it is held.

    All zeros (and grow 1) makes it hold still again. Prepare stretches the
    hold if the motion would break a speed or acceleration limit.
    """
    show = _require_show(name)
    if not 0 <= index < len(show.formations):
        raise HTTPException(404, "no such formation")
    f = show.formations[index]
    try:
        motion = Motion(**body.model_dump())
        f.motion = None if motion.still else motion
        f.__post_init__()
        need = planning.motion_hold_need(f, show.limits)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    save_show(show, _show_path(name))
    return {"ok": True, "motion": None if f.motion is None else f.motion.to_dict(),
            "min_hold_s": round(need * 1.15, 1), "warning": _fit_warning(show, index)}


@app.delete("/api/design/shows/{name}/formations/{index}")
def remove_formation(name: str, index: int) -> dict:
    show = _require_show(name)
    if not 0 <= index < len(show.formations):
        raise HTTPException(404, "no such formation")
    show.formations.pop(index)
    save_show(show, _show_path(name))
    return {"ok": True}


@app.post("/api/design/shows/{name}/reorder")
def reorder_formations(name: str, order: list[int]) -> dict:
    show = _require_show(name)
    if sorted(order) != list(range(len(show.formations))):
        raise HTTPException(400, "order must be a permutation of the formation indices")
    show.formations = [show.formations[i] for i in order]
    save_show(show, _show_path(name))
    return {"ok": True}


# --------------------------------------------------------------------------
# Prepare, validate, preview
# --------------------------------------------------------------------------


@app.post("/api/design/shows/{name}/prepare")
def prepare_show(name: str, save_as: str | None = None) -> dict:
    """Scale, wrap with takeoff and landing, match and time the show."""
    show = _require_show(name)
    if not show.formations:
        raise HTTPException(400, "add at least one formation before preparing")
    try:
        prepared = planning.prepare(show)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    target = save_as or f"{name} prepared"
    prepared.show.name = target
    save_show(prepared.show, _show_path(target))
    return {
        "ok": True,
        "saved_as": target,
        "duration": round(prepared.show.duration, 2),
        "formations": [f.name for f in prepared.show.formations],
        "notes": prepared.notes,
        "launch_grid": np.round(prepared.grid, 2).tolist(),
    }


# --------------------------------------------------------------------------
# AI helpers: fix it for me, music to show, shape from words
# --------------------------------------------------------------------------


@app.get("/api/ai/status")
def ai_status() -> dict:
    from . import assistant

    return {"claude": assistant.available(), "model": assistant.MODEL,
            "key_source": assistant.key_source()}


class ClaudeKeyIn(BaseModel):
    key: str = Field(max_length=300)


@app.post("/api/ai/key")
def ai_key_save(body: ClaudeKeyIn) -> dict:
    """Keep a Claude API key on this computer. It is never sent back."""
    from . import assistant

    try:
        assistant.save_key(body.key)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return ai_status()


@app.delete("/api/ai/key")
def ai_key_remove() -> dict:
    from . import assistant

    assistant.KEY_FILE.unlink(missing_ok=True)
    return ai_status()


@app.post("/api/design/shows/{name}/ai/fix")
def ai_fix(name: str) -> dict:
    """Change the design until it passes the safety checks, and say how.
    One undo puts everything back."""
    from . import assistant

    show = _require_show(name)
    try:
        result = assistant.fix(show)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    if result["changes"]:
        save_show(show, _show_path(name))
    return {"ok": True, **result}


class MusicSyncIn(BaseModel):
    beats: int = Field(4, ge=1, le=32)


@app.post("/api/design/shows/{name}/ai/music")
def ai_music(name: str, body: MusicSyncIn) -> dict:
    """Retime every formation to land on the song's beat."""
    from . import assistant

    show = _require_show(name)
    try:
        notes = assistant.music_sync(show, beats=body.beats)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    save_show(show, _show_path(name))
    return {"ok": True, "notes": notes, "duration": round(show.duration, 2)}


class ShapeIn(BaseModel):
    description: str = Field(min_length=1, max_length=200)
    altitude: float = 50.0
    color: list[int] = Field(default_factory=lambda: [255, 255, 255])
    size: float = 0.0
    fill: bool = True
    hold: float = 5.0
    transition: float = 8.0


@app.post("/api/design/shows/{name}/ai/shape")
def ai_shape(name: str, body: ShapeIn) -> dict:
    """Add a formation described in words.

    A built-in shape name ("a red heart") is used directly. Anything else is
    drawn by Claude when a key is set; without one, the words themselves are
    written in the sky.
    """
    from . import assistant, models, voice

    show = _require_show(name)
    size = body.size if body.size > 0 else round(max(geometry.billboard_for(
        show.drone_count, show.limits.formation_separation)), 1)
    rgb = [max(0, min(255, int(v))) for v in body.color][:3]
    if len(rgb) != 3:
        raise HTTPException(400, "colour must be three numbers")
    found = voice.find_shape(voice.normalise(body.description))
    source, note = "", None
    try:
        if found and found[2] >= 1.0:   # an exact shape word, not a fuzzy guess
            shape, params, _ = found
            points = geometry.SHAPES[shape](show.drone_count, **params)
            ext = float(np.max(np.ptp(points, axis=0))) or 1.0
            points = (points - points.mean(axis=0)) * (size / ext)
            source = f"built-in {shape}"
        else:
            svg = None
            if assistant.available():
                try:
                    svg = assistant.shape_svg(body.description)
                    source = "drawn by Claude"
                except Exception as exc:  # fall back to writing the words
                    note = f"Claude could not draw it ({str(exc)[:120]}); wrote the words instead"
            if svg is not None:
                points, _ = models.place(svg.encode(), "shape.svg", show.drone_count,
                                         mode="fill" if body.fill else "outline", size=size)
            else:
                points = geometry.text(show.drone_count, body.description[:24], width=size)
                points = points - points.mean(axis=0)
                source = source or "written as text"
                if note is None and not assistant.available():
                    note = "no Claude key set, so the words were written in the sky"
        show.add(Formation(
            name=body.description[:40],
            points=geometry.lift(points, body.altitude),
            colors=np.asarray(rgb, dtype=np.int32),
            transition=body.transition, hold=body.hold,
        ))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    save_show(show, _show_path(name))
    return {"ok": True, "index": len(show.formations) - 1, "source": source,
            "note": note, "warning": _fit_warning(show)}


@app.post("/api/design/shows/{name}/validate")
def validate_show(name: str) -> dict:
    show = _require_show(name)
    if not show.formations:
        raise HTTPException(400, "nothing to validate")
    try:
        report = safety.validate(show)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return report.to_dict()


@app.get("/api/design/shows/{name}/plan")
def show_plan(name: str) -> dict:
    """Formations and timings for the browser to animate locally.

    The browser reproduces the motion with the same shared easing, so what
    the designer sees is what the validator checked and what the aircraft
    will fly - not a separately interpolated approximation.
    """
    show = _require_show(name)
    matched = planning.chain_match(show)
    cursor = 0.0
    out = []
    for f in matched.formations:
        out.append(
            {
                "name": f.name,
                "transition": f.transition,
                "hold": f.hold,
                "easing": f.easing,
                "t_start": round(cursor, 4),
                "points": np.round(f.points, 3).tolist(),
                "colors": f.colors.tolist(),
                "motion": None if f.motion is None else f.motion.to_dict(),
                "keep_order": f.keep_order,
            }
        )
        cursor += f.duration
    return {
        "name": show.name,
        "drone_count": show.drone_count,
        "duration": round(matched.duration, 3),
        "limits": asdict(show.limits),
        "formations": out,
    }


@app.get("/api/design/shows/{name}/replay")
def show_replay(name: str, fps: float = 4.0) -> dict:
    """The show sampled exactly as the validator sees it, lights included,
    packed small enough for a browser to play back.

    Positions are int16 decimetres and colours uint8 RGB, both base64, laid
    out frame by frame, drone by drone.
    """
    import base64

    from .trajectory import build_timeline

    show = _require_show(name)
    if not show.formations:
        raise HTTPException(400, "this show has no formations to play")
    fps = min(max(fps, 1.0), 10.0)
    timeline = build_timeline(show, fps=max(show.fps, fps))
    times = np.arange(0.0, float(timeline.t[-1]) + 1e-9, 1.0 / fps)
    pos = np.empty((len(times), show.drone_count, 3), np.int16)
    col = np.empty((len(times), show.drone_count, 3), np.uint8)
    for k, t in enumerate(times):
        p, c = timeline.sample(float(t))
        # ponytail: int16 decimetres cap positions at +-3.2 km, far past any
        # geofence a show is flown in.
        pos[k] = np.clip(np.round(p * 10), -32768, 32767)
        col[k] = c
    segments, cursor = [], 0.0
    for f in show.formations:
        segments.append({"name": f.name, "start": round(cursor, 2),
                         "end": round(cursor + f.duration, 2)})
        cursor += f.duration
    return {
        "name": show.name, "fps": fps, "frames": len(times),
        "drones": show.drone_count, "duration": round(float(timeline.t[-1]), 2),
        "segments": segments,
        "pos": base64.b64encode(pos.astype("<i2").tobytes()).decode("ascii"),
        "col": base64.b64encode(col.tobytes()).decode("ascii"),
    }


@app.get("/api/design/shows/{name}/charts")
def show_charts(name: str, points: int = 600) -> dict:
    """Height, speed, acceleration and spacing over the whole show.

    Taken from the same samples the safety engine checks. Each chart point
    is the worst value inside its slice of time, so a short spike is never
    averaged away. Spacing counts drones in the air only, as the safety
    engine does; on the ground the launch grid's spacing applies.
    """
    from scipy.spatial import cKDTree

    from .trajectory import build_timeline

    show = _require_show(name)
    if not show.formations:
        raise HTTPException(400, "this show has no formations to chart")
    lim = show.limits
    tl = build_timeline(show)
    pos, vel = tl.pos, tl.velocity()
    acc = tl.acceleration()
    z = pos[:, :, 2]
    series = {
        "altitude_max": z.max(axis=1), "altitude_min": z.min(axis=1),
        "speed_xy": np.linalg.norm(vel[:, :, :2], axis=2).max(axis=1),
        "climb": np.maximum(vel[:, :, 2], 0.0).max(axis=1),
        "descent": np.maximum(-vel[:, :, 2], 0.0).max(axis=1),
        "acceleration": np.linalg.norm(acc, axis=2).max(axis=1),
    }
    closest = np.full(len(tl.t), np.nan)
    for k in range(len(tl.t)):
        up = pos[k, z[k] > lim.min_nav_altitude]
        if len(up) >= 2:
            closest[k] = float(cKDTree(up).query(up, k=2)[0][:, 1].min())
    series["closest"] = closest

    edges = np.linspace(0, len(tl.t), min(max(points, 10), len(tl.t)) + 1).astype(int)
    keep_min = {"altitude_min", "closest"}
    # The preview's clock includes the first formation's lead-in; the sampler
    # starts at its hold. Zero for a prepared show, which starts on the ground.
    lead = show.formations[0].transition
    out = {"t": [round(float(tl.t[a:b].mean()) + lead, 3) for a, b in zip(edges[:-1], edges[1:])]}
    for key, arr in series.items():
        vals = []
        for a, b in zip(edges[:-1], edges[1:]):
            chunk = arr[a:b]
            chunk = chunk[~np.isnan(chunk)]
            vals.append(None if not len(chunk) else
                        round(float(chunk.min() if key in keep_min else chunk.max()), 3))
        out[key] = vals
    cursor, blocks = 0.0, []
    for f in show.formations:
        blocks.append({"name": f.name, "start": round(cursor, 2), "end": round(cursor + f.duration, 2)})
        cursor += f.duration
    out["formations"] = blocks
    out["limits"] = {k: getattr(lim, k) for k in (
        "min_separation", "max_altitude", "min_nav_altitude", "max_velocity_xy",
        "max_velocity_up", "max_velocity_down", "max_acceleration")}
    return out


class VoiceIn(BaseModel):
    text: str
    # What the UI already knows, so "make it red" resolves without repeating.
    context: dict[str, Any] = Field(default_factory=dict)
    # Set by the UI when the operator has clicked through a confirm card.
    confirmed: bool = False
    # Set on commands Claude wrote, so they are never sent to Claude again.
    translated: bool = False


@app.get("/api/voice/vocabulary")
def voice_vocabulary() -> dict:
    from . import voice

    return {**voice.vocabulary(), "help": voice.help_text()}


@app.post("/api/voice")
def voice_command(body: VoiceIn, request: Request) -> dict:
    """Parse a spoken phrase and, if it is safe to, carry it out.

    The risk level comes from the parser, and this is the only place that
    acts on it:

      refused  never executed, at any confidence
      confirm  parsed and returned for a click; nothing happens yet
      auto     executed now

    Arming, starting and disarming are refused here as well as in the
    parser. Two independent refusals for the three verbs that put aircraft
    in the air is deliberate - a grammar change should not be able to
    quietly make them reachable.
    """
    from . import voice

    intent = voice.parse(body.text, body.context)
    out: dict[str, Any] = {"intent": intent.to_dict(), "executed": False}

    if intent.name == "unknown" and not body.translated:
        from . import assistant

        if assistant.available():
            # Claude only rewrites the words. Each command it returns is
            # parsed and checked below exactly as if it had been typed, and
            # never arrives confirmed: anything risky still needs a click.
            try:
                commands = assistant.translate(body.text)
            except Exception as exc:  # network, refusal, bad key: say so, change nothing
                out["speech"] = f"{intent.speech} (Claude could not help: {str(exc)[:120]})"
                return out
            if commands:
                steps = []
                for c in commands:
                    r = voice_command(VoiceIn(text=c, context=body.context,
                                              translated=True), request)
                    r["command"] = c
                    steps.append(r)
                    if not r.get("executed") and not r.get("client_action"):
                        break
                last = dict(steps[-1])
                last["translated"] = commands
                last["steps"] = steps
                last["confirm_text"] = steps[-1]["command"]
                last["speech"] = " ".join(s.get("speech", "") for s in steps).strip()
                last["reload"] = any(s.get("reload") for s in steps) or last.get("reload")
                return last

    if intent.name in voice.REFUSED_INTENTS or intent.risk == voice.REFUSED:
        out["speech"] = intent.speech
        out["refused"] = True
        return out

    # The voice endpoint calls handlers directly, past the route checks, so
    # the intent's own role requirement is enforced here. Without this a
    # viewer could say "upload the plans" and it would happen.
    from . import auth

    needed = auth.INTENT_ROLES.get(intent.name)
    role = getattr(request.state, "role", None)
    if needed and not auth.allows(role, needed):
        out["speech"] = (
            f"{intent.name.replace('_', ' ').capitalize()} needs the {needed} "
            f"role, and you are signed in as {role or 'nobody'}."
        )
        out["forbidden"] = True
        return out

    if intent.risk == voice.CONFIRM and not body.confirmed:
        out["speech"] = intent.speech
        out["needs_confirmation"] = True
        out["confirm_prompt"] = intent.confirm_prompt
        return out

    try:
        result = _run_intent(intent, body.context)
    except HTTPException as exc:
        out["speech"] = f"That failed: {exc.detail}"
        out["error"] = exc.detail
        return out
    except (ValueError, RuntimeError, PermissionError, OSError) as exc:
        out["speech"] = f"That failed: {exc}"
        out["error"] = str(exc)
        return out

    out["executed"] = result.get("client_action") is None
    out.update(result)
    out.setdefault("speech", intent.speech)
    return out


def _run_intent(intent, ctx: dict) -> dict:
    """Carry out one parsed intent. Server actions run; UI actions are relayed."""
    from . import voice

    name = intent.name
    args = intent.args
    show_name = ctx.get("show")

    # Purely visual: the browser owns the camera and the playhead.
    if name in ("play", "pause", "seek", "camera", "next_formation",
                "previous_formation"):
        return {"client_action": {"name": name, **args}, "speech": intent.speech}

    if name == "help":
        return {"speech": voice.help_text(), "vocabulary": voice.vocabulary()}

    if name == "status":
        return {"speech": _spoken_status(show_name)}

    if name == "create_show":
        created = create_show(ShowIn(
            name=args["name"], drone_count=int(args["drone_count"]),
            limits=LimitsIn(), origin=OriginIn(),
        ))
        hint = created["capacity_hint"]
        speech = f"Created {created['show']} with {args['drone_count']} drones."
        if not hint["fits_under_ceiling"]:
            speech += (
                f" Careful: at {hint['formation_separation_m']} metre spacing they "
                f"need {hint['billboard_height_m']:.0f} metres of height, which is "
                "over the default ceiling."
            )
        return {"speech": speech, "result": created, "open_show": created["show"]}

    if not show_name:
        return {"speech": "Open a show first."}

    if name == "add_formation":
        r = add_formation(show_name, FormationIn(
            name=args["shape"].replace("_", " "), shape=args["shape"],
            params=args.get("params", {}),
            color=args.get("color", [255, 255, 255]),
            altitude=args.get("altitude", 50.0),
        ))
        return {"speech": f"{intent.speech} Raw spacing {r['raw_min_separation_m']} metres.",
                "result": r, "reload": True}

    if name == "apply_effect":
        r = apply_effect(show_name, int(args.get("formation", 0)), EffectIn(
            effect=args["effect"], params=args.get("params", {}),
            beats_per_cycle=args.get("beats_per_cycle"),
        ))
        return {"speech": f"{intent.speech} {r['keyframes']} keyframes.",
                "result": r, "reload": True}

    if name == "set_music":
        show = _require_show(show_name)
        r = set_music(show_name, MusicIn(
            path=show.music.path, bpm=args["bpm"],
            offset=show.music.offset, beat_phase=show.music.beat_phase,
            cues=show.music.cues,
        ))
        return {"speech": intent.speech, "result": r, "reload": True}

    if name == "prepare":
        r = prepare_show(show_name)
        speech = f"Prepared as {r['saved_as']}, {r['duration']:.0f} seconds."
        if r["notes"]:
            speech += f" {len(r['notes'])} notes, check the panel."
        return {"speech": speech, "result": r, "open_show": r["saved_as"]}

    if name == "validate":
        r = validate_show(show_name)
        m = r["metrics"]
        if r["passed"]:
            speech = (
                f"Passed. Closest approach {m['min_separation_m']:.2f} metres "
                f"against a {m['min_separation_limit_m']:.1f} metre limit."
            )
        else:
            first = r["violations"][0]["message"] if r["violations"] else ""
            speech = f"Failed with {r['error_count']} errors. {first}"
        return {"speech": speech, "result": r, "report": r}

    if name == "adapt":
        r = adapt_show(show_name, int(args["count"]))
        return {"speech": f"Adapted to {r['drone_count']} drones. "
                          f"{'Passes' if r['passed'] else 'Fails'} validation.",
                "result": r, "open_show": r["saved_as"]}

    if name == "auto_geofence":
        r = auto_fence(show_name, float(args.get("margin", 15.0)))
        return {"speech": f"Geofence drawn with {r['vertices']} vertices. "
                          "Check it against your permit.",
                "result": r, "reload": True}

    if name == "export":
        r = export_show(show_name, args["format"])
        return {"speech": f"Exported {args['format']}.", "result": r}

    if name == "report":
        return {"client_action": {"name": "open_report"}, "speech": intent.speech}

    if name == "ops_load":
        import asyncio

        r = asyncio.run(ops_load(LoadIn(show=show_name, driver=args["driver"])))
        return {"speech": f"Loaded onto the {args['driver']} fleet.", "result": r}

    if name == "preflight":
        r = ops_preflight()
        failing = [c["name"] for c in r["checks"] if c["blocking"] and not c["ok"]]
        speech = ("Preflight clear." if not failing
                  else f"Preflight failing on {', '.join(failing)}.")
        return {"speech": speech, "result": r}

    if name == "upload":
        r = ops_upload(OverrideIn())
        return {"speech": "Plans uploaded.", "result": r}

    if name == "suspend":
        r = ops_suspend()
        return {"speech": "Suspended. The fleet is holding.", "result": r}

    if name == "resume":
        r = ops_unsuspend()
        return {"speech": "Resumed.", "result": r}

    if name == "collective_rth":
        try:
            r = ops_collective_rth()
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            return {"speech": "Return refused, the fleet is holding instead. "
                              + str(detail.get("detail", ""))[:180],
                    "result": detail, "error": "refused"}
        return {"speech": f"Returning home. Closest approach "
                          f"{r['min_separation_m']} metres.", "result": r}

    if name == "land":
        r = ops_emergency(EmergencyIn(action="land", reason="voice, confirmed"))
        return {"speech": "Landing the fleet.", "result": r}

    if name == "save_log":
        return {"speech": "Flight log saved.", "result": ops_log_save()}

    if name == "compose":
        return {"client_action": {"name": "compose", **args}, "speech": intent.speech}

    return {"speech": "I understood that but there is nothing to do."}


def _spoken_status(show_name: str | None) -> str:
    bits = []
    if _fleet is not None:
        s = _fleet.snapshot()
        bits.append(
            f"Fleet {s['phase']}, {s['connected']} of {s['drone_count']} connected, "
            f"{s['armed']} armed, max deviation {s['max_deviation']:.1f} metres"
        )
        if s["alerts"]:
            bits.append(f"{len(s['alerts'])} active alerts")
    if show_name:
        try:
            show = _require_show(show_name)
            bits.append(f"Show {show.name}, {show.drone_count} drones, "
                        f"{show.duration:.0f} seconds, "
                        f"{len(show.formations)} formations")
        except HTTPException:
            pass
    return ". ".join(bits) + "." if bits else "Nothing loaded."


class ComposeIn(BaseModel):
    description: str
    drone_count: int = Field(default=100, ge=1, le=2000)
    name: str | None = None
    # Build the show for real, rather than only returning the draft.
    create: bool = False


@app.post("/api/voice/compose")
def voice_compose(body: ComposeIn) -> dict:
    """Draft a storyboard from a description, and optionally build it.

    The draft is never flyable on its own: building it runs the same
    generators and the same validation as a hand-built show, and the
    operator still has to prepare and validate before anything can be
    uploaded.
    """
    from . import voice

    spec = voice.compose(body.description, body.drone_count)
    if not body.create:
        return {"ok": True, "spec": spec, "created": None}

    name = body.name or f"{spec['mood']} show"
    create_show(ShowIn(name=name, drone_count=body.drone_count,
                       limits=LimitsIn(), origin=OriginIn()))
    if spec.get("bpm"):
        set_music(name, MusicIn(bpm=float(spec["bpm"])))

    built = []
    for i, f in enumerate(spec["formations"]):
        try:
            add_formation(name, FormationIn(
                name=f["name"], shape=f["shape"], params=f.get("params", {}),
                color=f["color"], altitude=f["altitude"],
                transition=f["transition"], hold=f["hold"],
            ))
            if f.get("effect") and f["effect"] != "solid":
                apply_effect(name, i, EffectIn(
                    effect=f["effect"],
                    params=f.get("effect_params") or {},
                ))
            built.append(f["name"])
        except HTTPException as exc:
            spec.setdefault("skipped", []).append(
                {"formation": f["name"], "why": str(exc.detail)[:160]}
            )
    return {
        "ok": True, "spec": spec, "created": name, "formations": built,
        "note": (
            "A draft. It has not been prepared and has not passed the safety "
            "engine, so it is not flyable yet."
        ),
    }


# --------------------------------------------------------------------------
# Fleet registry
# --------------------------------------------------------------------------

def _registry():
    from .registry import Registry

    return Registry.load(FLEET_FILE)


def _registry_call(fn):
    """Load, mutate, save - and turn registry errors into 400s."""
    from .registry import RegistryError

    reg = _registry()
    try:
        result = fn(reg)
    except RegistryError as exc:
        raise HTTPException(400, str(exc)) from None
    reg.save(FLEET_FILE)
    return reg, result


class DroneIn(BaseModel):
    id: str
    serial: str = ""
    model: str = ""
    controller: str = ""
    firmware: str = ""
    gnss: str = ""
    led: str = ""
    battery_id: str = ""
    sysid: int = 0
    notes: str = ""


class DronePatch(BaseModel):
    serial: str | None = None
    model: str | None = None
    controller: str | None = None
    firmware: str | None = None
    gnss: str | None = None
    led: str | None = None
    battery_id: str | None = None
    sysid: int | None = None
    notes: str | None = None
    status: str | None = None
    reason: str = ""


class BulkIn(BaseModel):
    count: int = Field(ge=1, le=2000)
    prefix: str = "DD"
    start: int = 1
    model: str = ""
    sysid_from: int = 1


class MaintenanceIn(BaseModel):
    event: str
    detail: str = ""


class AssignIn(BaseModel):
    drone_ids: list[str] | None = None
    auto: bool = False


@app.get("/api/fleet/drones")
def fleet_list() -> dict:
    reg = _registry()
    return {
        "drones": [d.to_dict() for d in sorted(reg.drones.values(), key=lambda d: d.id)],
        "summary": reg.summary(),
    }


@app.post("/api/fleet/drones")
def fleet_add(body: DroneIn) -> dict:
    from .registry import Drone

    def add(reg):
        return reg.add(Drone(**body.model_dump())).to_dict()

    _reg, drone = _registry_call(add)
    return {"ok": True, "drone": drone}


@app.post("/api/fleet/drones/bulk")
def fleet_bulk(body: BulkIn) -> dict:
    from .registry import bulk_register

    _reg, made = _registry_call(lambda reg: bulk_register(
        reg, body.count, prefix=body.prefix, start=body.start,
        model=body.model, sysid_from=body.sysid_from,
    ))
    return {"ok": True, "registered": [d.id for d in made]}


@app.patch("/api/fleet/drones/{drone_id}")
def fleet_patch(drone_id: str, body: DronePatch) -> dict:
    def patch(reg):
        drone = reg.get(drone_id)
        fields = body.model_dump(exclude_none=True)
        status = fields.pop("status", None)
        reason = fields.pop("reason", "")
        if "sysid" in fields and fields["sysid"] and not 0 < fields["sysid"] < 256:
            from .registry import RegistryError

            raise RegistryError("MAVLink system id must be 1-255")
        for key, value in fields.items():
            setattr(drone, key, value)
        if status and status != drone.status:
            reg.set_status(drone_id, status, reason)
        return drone.to_dict()

    _reg, drone = _registry_call(patch)
    return {"ok": True, "drone": drone}


@app.post("/api/fleet/drones/{drone_id}/maintenance")
def fleet_maintenance(drone_id: str, body: MaintenanceIn) -> dict:
    _reg, drone = _registry_call(
        lambda reg: reg.log_maintenance(drone_id, body.event, body.detail).to_dict()
    )
    return {"ok": True, "drone": drone}


@app.delete("/api/fleet/drones/{drone_id}")
def fleet_remove(drone_id: str) -> dict:
    _registry_call(lambda reg: reg.remove(drone_id))
    return {"ok": True}


class FleetImportIn(BaseModel):
    csv: str = Field(max_length=2_000_000)


@app.get("/api/fleet/export")
def fleet_export() -> Response:
    from .registry import export_csv

    return Response(
        export_csv(_registry()), media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="skyshine-fleet.csv"'},
    )


@app.post("/api/fleet/import")
def fleet_import(body: FleetImportIn) -> dict:
    """All or nothing: one bad row and the registry file is left untouched."""
    from .registry import import_csv

    _reg, result = _registry_call(lambda reg: import_csv(reg, body.csv))
    return {"ok": True, **result}


@app.get("/api/fleet/assign/{show}")
def fleet_assignment(show: str) -> dict:
    reg = _registry()
    detail = _require_show(show)
    problems = reg.verify(show, detail.drone_count)
    return {
        "show": show,
        "slots": detail.drone_count,
        "assignment": reg.assignment(show),
        "problems": problems,
        "ready": not [p for p in problems if p["kind"] != "advisory"],
    }


@app.post("/api/fleet/assign/{show}")
def fleet_assign(show: str, body: AssignIn) -> dict:
    detail = _require_show(show)

    def assign(reg):
        if body.auto:
            return reg.auto_assign(show, detail.drone_count)
        return reg.assign(show, body.drone_ids or [])

    reg, assigned = _registry_call(assign)
    problems = reg.verify(show, detail.drone_count)
    return {"ok": True, "assignment": assigned, "problems": problems}


@app.post("/api/design/assets")
async def upload_asset(file: UploadFile = File(...)) -> dict:
    """Accept a logo or image and describe how it will threshold.

    The response carries a suggested threshold from Otsu and the drone
    ceiling the mask can hold, so the UI can set sensible defaults and
    warn before a drone count that cannot be sampled.
    """
    from . import assets

    data = await file.read()
    try:
        asset = assets.save_upload(data, file.filename or "upload", ASSET_DIR)
        info = assets.describe(asset.path)
    except assets.AssetError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"ok": True, **asset.to_dict(), **info,
            "max_drones": assets.max_drones(
                asset.path, info["suggested_threshold"], info["suggested_invert"])}


@app.get("/api/design/assets")
def list_assets() -> dict:
    from . import assets

    return {"assets": assets.list_assets(ASSET_DIR)}


@app.get("/api/design/assets/{asset_id}/preview")
def preview_asset(
    asset_id: str,
    drones: int = 100,
    threshold: int = 128,
    invert: bool = False,
) -> Any:
    """The mask, with the drone positions that would actually be sampled.

    Looking at the mask alone is misleading: a logo that thresholds
    beautifully can still read as noise once it is only 120 points.
    """
    from fastapi.responses import Response

    from . import assets

    try:
        path = assets.resolve(ASSET_DIR, asset_id)
        png = assets.preview_png(
            path, drones=max(1, min(drones, 2000)),
            threshold=int(max(0, min(threshold, 255))), invert=invert,
        )
    except assets.AssetError as exc:
        raise HTTPException(404, str(exc)) from None
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/design/assets/{asset_id}/capacity")
def asset_capacity(asset_id: str, threshold: int = 128, invert: bool = False) -> dict:
    from . import assets

    try:
        path = assets.resolve(ASSET_DIR, asset_id)
    except assets.AssetError as exc:
        raise HTTPException(404, str(exc)) from None
    return {"max_drones": assets.max_drones(path, int(threshold), invert)}


@app.delete("/api/design/assets/{asset_id}")
def delete_asset(asset_id: str) -> dict:
    from . import assets

    try:
        assets.delete(ASSET_DIR, asset_id)
    except assets.AssetError as exc:
        raise HTTPException(404, str(exc)) from None
    return {"ok": True}


class ArtworkIn(BaseModel):
    """Add a formation from an uploaded image, or from text."""

    name: str
    # Exactly one of these.
    asset_id: str | None = None
    text: str | None = None
    threshold: int = 128
    invert: bool = False
    font: str | None = None
    width: float = 90.0
    altitude: float = 50.0
    transition: float = 9.0
    hold: float = 5.0
    color: list[int] = Field(default_factory=lambda: [255, 255, 255])
    yaw: float = 0.0


@app.post("/api/design/shows/{name}/artwork")
def add_artwork(name: str, body: ArtworkIn) -> dict:
    """Turn a logo or a word into a formation of this show's drone count."""
    from . import assets

    show = _require_show(name)
    if bool(body.asset_id) == bool(body.text):
        raise HTTPException(400, "give either an asset_id or text, not both")

    try:
        if body.asset_id:
            path = assets.resolve(ASSET_DIR, body.asset_id)
            ceiling = assets.max_drones(path, body.threshold, body.invert)
            if show.drone_count > ceiling:
                raise HTTPException(
                    400,
                    f"this artwork thresholds to {ceiling} usable pixels, fewer "
                    f"than the show's {show.drone_count} drones. Lower the "
                    "threshold, invert it, or use a higher-resolution image.",
                )
            points = geometry.image(
                show.drone_count, str(path), width=body.width,
                threshold=body.threshold, invert=body.invert,
            )
        else:
            points = geometry.text(
                show.drone_count, body.text or "", width=body.width,
                font_path=body.font,
            )
    except assets.AssetError as exc:
        raise HTTPException(404, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None

    if body.yaw:
        points = geometry.rotate_z(points, body.yaw)
    points = geometry.lift(points, body.altitude)
    try:
        show.add(Formation(
            name=body.name, points=points,
            colors=np.asarray(body.color, dtype=np.int32),
            transition=body.transition, hold=body.hold,
        ))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    save_show(show, _show_path(name))
    return {
        "ok": True,
        "index": len(show.formations) - 1,
        "raw_min_separation_m": round(geometry.min_pair_distance(points), 3),
        "extent_m": [round(float(v), 1) for v in np.ptp(points, axis=0)],
        "warning": _fit_warning(show),
        "note": (
            "Artwork is sampled, so the shape reads only as well as the drone "
            "count allows. Preview it before preparing."
        ),
    }


@app.get("/api/effects")
def list_effects() -> dict:
    """The light effect catalogue, with each effect's parameters."""
    from . import effects

    return {"effects": [effects.describe(n) for n in sorted(effects.EFFECTS)]}


class EffectIn(BaseModel):
    effect: str
    params: dict[str, Any] = Field(default_factory=dict)
    # Only these drones (named by position, as for nudge); None means all.
    near: list[list[float]] | None = None
    # Express a period in beats instead of seconds, so retiming the show to
    # a different tempo keeps the effect locked to the music.
    beats_per_cycle: float | None = None
    fps: float | None = None


@app.post("/api/design/shows/{name}/formations/{index}/effect")
def apply_effect(name: str, index: int, body: EffectIn) -> dict:
    from . import effects

    show = _require_show(name)
    if not 0 <= index < len(show.formations):
        raise HTTPException(404, "no such formation")
    params = dict(body.params)
    if body.beats_per_cycle:
        params["period"] = effects.beat_locked(show.music, body.beats_per_cycle)
    f = show.formations[index]
    if body.effect == "volume" and params.get("over_current", True):
        # Paint over what the formation already shows, not over black.
        params["background"] = f.colors
    before = f.copy()
    mask = None
    if body.near:
        mask = np.zeros(f.count, dtype=bool)
        mask[_slots_near(f, body.near)] = True
    try:
        fn = effects.make(body.effect, **params)
    except TypeError as exc:
        raise HTTPException(400, f"bad parameters for {body.effect!r}: {exc}") from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None

    effects.render(f, fn, fps=body.fps)
    if mask is not None:
        effects.keep_unmasked(before, f, mask)
    save_show(show, _show_path(name))
    return {
        "ok": True,
        "effect": body.effect,
        "keyframes": len(show.formations[index].color_keys),
        "render_fps": effects.recommended_fps(fn) if body.fps is None else body.fps,
        "period_s": round(params.get("period", 0.0), 4) if "period" in params else None,
    }


class MusicIn(BaseModel):
    path: str | None = None
    bpm: float | None = None
    offset: float = 0.0
    beat_phase: float = 0.0
    cues: list[tuple[float, str]] = Field(default_factory=list)


@app.put("/api/design/shows/{name}/music")
def set_music(name: str, body: MusicIn) -> dict:
    from .show import Music

    show = _require_show(name)
    show.music = Music(**body.model_dump())
    save_show(show, _show_path(name))
    return {
        "ok": True,
        "beat_seconds": show.music.beat_seconds,
        "beats": len(show.music.beat_times(show.duration)),
    }


AUDIO_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg",
               "flac": "audio/flac", "m4a": "audio/mp4"}
MAX_AUDIO_BYTES = 60 * 1024 * 1024


def _audio_kind(data: bytes) -> str | None:
    """Which audio format a file is, from its first bytes, or None."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return "mp3"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"fLaC":
        return "flac"
    if data[4:8] == b"ftyp":
        return "m4a"
    return None


@app.post("/api/design/shows/{name}/audio")
async def upload_audio(name: str, file: UploadFile = File(...)) -> dict:
    """Store the show's song. The browser decodes it for the waveform and
    for playback, so any format a browser can play works."""
    show = _require_show(name)
    data = await file.read()
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(400, f"the song is {len(data) / 1e6:.0f} MB; the limit is "
                                 f"{MAX_AUDIO_BYTES / 1e6:.0f} MB")
    kind = _audio_kind(data)
    if kind is None:
        raise HTTPException(400, "that is not a song file Sky Shine can store; use MP3, "
                                 "WAV, OGG, FLAC or M4A")
    folder = ASSET_DIR / "audio"
    folder.mkdir(parents=True, exist_ok=True)
    stored = f"audio/{hashlib.sha256(data).hexdigest()[:16]}.{kind}"
    (ASSET_DIR / stored).write_bytes(data)
    show.music.path = stored
    show.audio = Path(file.filename or "song").name[:80]
    save_show(show, _show_path(name))
    return {"ok": True, "path": stored, "title": show.audio, "bytes": len(data)}


@app.get("/api/design/shows/{name}/audio")
def get_audio(name: str) -> Response:
    show = _require_show(name)
    path = show.music.path or ""
    # Only songs stored by upload_audio are served, never an arbitrary path.
    if not re.fullmatch(r"audio/[0-9a-f]{16}\.(wav|mp3|ogg|flac|m4a)", path):
        raise HTTPException(404, "this show has no uploaded song")
    target = ASSET_DIR / path
    if not target.exists():
        raise HTTPException(404, "the song file is missing; upload it again")
    return FileResponse(target, media_type=AUDIO_TYPES[path.rsplit(".", 1)[1]])


class YawIn(BaseModel):
    # One angle for the whole formation, or one per drone. null clears it.
    yaw: float | list[float] | None


@app.put("/api/design/shows/{name}/formations/{index}/yaw")
def set_yaw(name: str, index: int, body: YawIn) -> dict:
    show = _require_show(name)
    if not 0 <= index < len(show.formations):
        raise HTTPException(404, "no such formation")
    f = show.formations[index]
    try:
        f.yaw = None if body.yaw is None else np.asarray(
            body.yaw, dtype=np.float64
        )
        show.formations[index] = Formation(**{
            **{k: getattr(f, k) for k in
               ("name", "points", "colors", "transition", "hold", "easing",
                "color_keys", "pyro", "motion", "keep_order")},
            "yaw": f.yaw,
        })
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    save_show(show, _show_path(name))
    return {"ok": True}


class PyroIn(BaseModel):
    at: float
    drones: list[int] = Field(default_factory=list)
    channel: int = 1
    description: str = ""
    heading_deg: float | None = None
    settle_s: float = 1.0


@app.post("/api/design/shows/{name}/formations/{index}/pyro")
def add_pyro(name: str, index: int, body: PyroIn) -> dict:
    from .show import PyroEvent

    show = _require_show(name)
    if not 0 <= index < len(show.formations):
        raise HTTPException(404, "no such formation")
    f = show.formations[index]
    event = PyroEvent(**body.model_dump())
    try:
        f.pyro.append(event)
        Formation(
            name=f.name, points=f.points, colors=f.colors, transition=f.transition,
            hold=f.hold, easing=f.easing, color_keys=f.color_keys, yaw=f.yaw,
            pyro=f.pyro,
        )
    except ValueError as exc:
        f.pyro.remove(event)
        raise HTTPException(400, str(exc)) from None
    save_show(show, _show_path(name))
    return {"ok": True, "events": len(f.pyro)}


@app.delete("/api/design/shows/{name}/formations/{index}/pyro/{event}")
def remove_pyro(name: str, index: int, event: int) -> dict:
    show = _require_show(name)
    if not 0 <= index < len(show.formations):
        raise HTTPException(404, "no such formation")
    f = show.formations[index]
    if not 0 <= event < len(f.pyro):
        raise HTTPException(404, "no such pyro event")
    f.pyro.pop(event)
    save_show(show, _show_path(name))
    return {"ok": True}


@app.post("/api/design/shows/{name}/geofence/auto")
def auto_fence(name: str, margin: float = 15.0) -> dict:
    show = _require_show(name)
    try:
        poly = planning.auto_geofence(show, margin=margin)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    show.limits.geofence = poly
    save_show(show, _show_path(name))
    return {
        "ok": True,
        "geofence": poly,
        "vertices": len(poly),
        "note": (
            "This hull wraps what the show actually does, plus margin. It is a "
            "starting point, not an authorisation - the real fence comes from "
            "the permit and is usually tighter."
        ),
    }


@app.post("/api/design/shows/{name}/adapt")
def adapt_show(name: str, count: int, save_as: str | None = None) -> dict:
    show = _require_show(name)
    try:
        adapted = planning.adapt(show, count)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    target = save_as or f"{name} x{count}"
    adapted.show.name = target
    save_show(adapted.show, _show_path(target))
    report = safety.validate(adapted.show)
    return {
        "ok": True,
        "saved_as": target,
        "drone_count": count,
        "duration": round(adapted.show.duration, 2),
        "notes": adapted.notes,
        "passed": report.passed,
        "min_separation_m": report.metrics.get("min_separation_m"),
    }


@app.post("/api/design/shows/{name}/export/{fmt}")
def export_show(name: str, fmt: str) -> dict:
    from . import interop

    show = _require_show(name)
    if fmt == "vviz":
        path = interop.export_vviz(show, SHOW_DIR / f"{name}.vviz")
        return {
            "ok": True, "path": str(path), "format": "vviz",
            "size_kb": round(path.stat().st_size / 1024, 1),
            "note": "Finale 3D and other vviz readers, for hybrid pyro shows.",
        }
    if fmt == "csv":
        files = interop.export_csv(show, SHOW_DIR / f"{name}-csv")
        return {
            "ok": True, "path": str(SHOW_DIR / f"{name}-csv"), "format": "csv",
            "files": len(files),
            "note": (
                "Sampled per-drone trajectories. Lossy by construction - for "
                "handing the show to another tool, not for round-tripping."
            ),
        }
    if fmt == "skyc":
        from . import skybrush

        try:
            path = skybrush.write_skyc(show, SHOW_DIR / f"{name}.skyc")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {
            "ok": True, "path": str(path), "format": "skyc",
            "size_kb": round(path.stat().st_size / 1024, 1),
            "note": ("Skybrush, EXPERIMENTAL: written from Skybrush's published format, "
                     "but the file's outer packaging is not documented. Load it into "
                     "Skybrush and check it before relying on it."),
        }
    raise HTTPException(400, f"unknown export format {fmt!r}; use vviz, csv or skyc")


class ImportIn(BaseModel):
    directory: str
    name: str | None = None


@app.post("/api/design/import/csv")
def import_csv_show(body: ImportIn) -> dict:
    from . import interop

    try:
        imported = interop.import_csv(body.directory, name=body.name)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from None
    return interop.validate_imported(imported)


@app.get("/api/design/shows/{name}/closest")
def closest_approaches(name: str, count: int = 10) -> dict:
    """The tightest encounters in the show, as a design aid."""
    show = _require_show(name)
    timeline = build_timeline(show)
    rows = safety.worst_pairs(timeline, count=count)
    return {
        "approaches": [
            {"time": round(t, 2), "a": a, "b": b, "distance_m": round(d, 3)}
            for t, a, b, d in rows
        ],
        "min_separation_limit_m": show.limits.min_separation,
    }


@app.get("/api/design/shows/{name}/report")
def validation_report(name: str) -> Any:
    """A printable flight-validation report, for the authority's file.

    HTML rather than PDF on purpose: every browser prints to PDF, and
    adding a PDF toolchain to a safety-critical package buys a dependency
    and a new way for the report to differ from the data it came from.
    """
    from html import escape as esc

    show = _require_show(name)
    report = safety.validate(show)
    body = report.to_dict()

    rows = "".join(
        f"<tr><td>{esc(k.replace('_', ' '))}</td><td class=v>{esc(str(v))}</td></tr>"
        for k, v in body["metrics"].items()
    )
    findings = "".join(
        f"<tr class={esc(v['severity'])}><td>{esc(v['severity'])}</td>"
        f"<td>{esc(v['kind'])}</td>"
        f"<td>{'' if v['time'] is None else str(v['time']) + ' s'}</td>"
        f"<td>{esc(v['message'])}</td></tr>"
        for v in body["violations"]
    ) or "<tr><td colspan=4>No findings.</td></tr>"

    storyboard = "".join(
        f"<tr><td>{i + 1}</td><td>{esc(f.name)}</td><td>{f.transition:.1f} s</td>"
        f"<td>{f.hold:.1f} s</td><td>{esc(f.easing)}</td>"
        f"<td>{'yes' if f.yaw is not None else 'no'}</td>"
        f"<td>{len(f.pyro)}</td></tr>"
        for i, f in enumerate(show.formations)
    )

    verdict = "PASS" if body["passed"] else "FAIL"
    html = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<title>Flight validation - {esc(show.name)}</title><style>
*{{box-sizing:border-box}}
body{{font:13px/1.5 system-ui,sans-serif;color:#111;background:#fff;max-width:900px;
margin:0 auto;padding:32px}}
h1{{font-size:21px;margin:0 0 2px}} h2{{font-size:13px;text-transform:uppercase;
letter-spacing:.09em;color:#666;margin:26px 0 8px;border-bottom:1px solid #ddd;
padding-bottom:4px}}
.verdict{{display:inline-block;padding:4px 14px;border-radius:4px;font-weight:700;
letter-spacing:.08em}}
.PASS{{background:#e3f6ea;color:#17683c;border:1px solid #9ad4b2}}
.FAIL{{background:#fdeaec;color:#8a1b28;border:1px solid #e3a3ac}}
table{{width:100%;border-collapse:collapse;margin-top:4px}}
th,td{{text-align:left;padding:5px 7px;border-bottom:1px solid #eee;
vertical-align:top}}
th{{color:#666;font-weight:600;font-size:11px;text-transform:uppercase}}
td.v{{font-family:ui-monospace,Consolas,monospace;text-align:right;width:30%}}
tr.error td{{background:#fdf3f4}} tr.warning td{{background:#fffaf0}}
footer{{margin-top:32px;padding-top:12px;border-top:1px solid #ddd;color:#666;
font-size:11px}}
@media print{{body{{padding:0}} h2{{break-after:avoid}} tr{{break-inside:avoid}}}}
</style></head><body>
<h1>Flight validation report</h1>
<p>{esc(show.name)} &middot; {show.drone_count} drones &middot;
{body['duration']} s &middot; generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}</p>
<p><span class="verdict {verdict}">{verdict}</span>
&nbsp;{body['error_count']} errors, {body['warning_count']} warnings</p>

<h2>Site</h2><table>
<tr><td>Origin</td><td class=v>{show.origin.latitude:.6f}, {show.origin.longitude:.6f}</td></tr>
<tr><td>AMSL reference</td><td class=v>{'none (AGL only)' if show.origin.amsl is None else f'{show.origin.amsl:.1f} m'}</td></tr>
<tr><td>Show heading</td><td class=v>{show.origin.orientation_deg:.1f}&deg;</td></tr>
<tr><td>Geofence vertices</td><td class=v>{len(show.limits.geofence) or 'none'}</td></tr>
</table>

<h2>Storyboard</h2><table>
<tr><th>#</th><th>Formation</th><th>Transition</th><th>Hold</th><th>Easing</th>
<th>Yaw</th><th>Pyro</th></tr>{storyboard}</table>

<h2>Measurements</h2><table>{rows}</table>

<h2>Findings</h2><table>
<tr><th>Severity</th><th>Check</th><th>Time</th><th>Detail</th></tr>{findings}</table>

<footer>
Generated by Sky Shine {__version__}. A passing report states that the
<em>plan</em> satisfies the configured limits. It is not a statement about
airworthiness, position accuracy, the airframe, the weather, or regulatory
authorisation, all of which remain the operator's responsibility. Separation
is measured centre to centre and continuously in time, including between
samples.
</footer></body></html>"""
    from fastapi.responses import HTMLResponse

    return HTMLResponse(html)


def _mission_path(name: str) -> Path:
    # Built from the sanitised show path, not the raw name: the name comes
    # from the URL, and a backslash in it would otherwise write the bundle
    # outside the shows folder.
    p = _show_path(name)
    return p.with_name(p.name.replace(".show.json", ".mission.json"))


def _mission_key(path: Path | None = None) -> bytes:
    try:
        return mission_key(path or KEY_FILE)
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from None


@app.get("/api/design/shows/{name}/mission/verify")
def verify_mission_file(name: str) -> dict:
    """Check the built bundle against this ground station's signing key."""
    path = _mission_path(name)
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        raise HTTPException(404, f"no mission has been built for {name!r}") from None
    except ValueError:
        body = None
    valid, detail = verify_mission(body, _mission_key())
    sig = dict(body.get("signature") or {}) if isinstance(body, dict) else {}
    sig.pop("value", None)
    return {"valid": valid, "detail": detail, "signature": sig}


@app.post("/api/design/shows/{name}/mission")
def export_mission(name: str, request: Request) -> dict:
    """Build the upload bundle. Refuses a show that has not passed safety."""
    show = _require_show(name)
    report = safety.validate(show)
    if not report.passed:
        raise HTTPException(
            409,
            {
                "error": "show failed safety validation, so no mission was built",
                "errors": [v.to_dict() for v in report.errors],
            },
        )
    mission = build_mission(show, safety=report.to_dict())
    path = _mission_path(name)
    signer =getattr(request.state, "user", None) or "local operator"
    mission.save(path, key=_mission_key(), signer=signer)
    return {
        "ok": True,
        "path": str(path),
        "signed_by": signer,
        "checksum": mission.to_dict()["checksum"],
        "duration": round(mission.duration, 2),
        "drone_count": mission.drone_count,
    }


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------


def _require_fleet() -> FleetManager:
    if _fleet is None:
        raise HTTPException(409, "no show is loaded; POST /api/ops/load first")
    return _fleet


@app.post("/api/ops/load")
async def ops_load(body: LoadIn) -> dict:
    global _fleet, _sitl
    show = _require_show(body.show)
    if body.driver not in DRIVERS and body.driver not in ("sitl", "sitl-guided"):
        raise HTTPException(400, f"unknown driver {body.driver!r}")
    if not show.formations:
        # Checked here, before any driver or virtual fleet is started:
        # building the timeline later raised a bare ValueError and the
        # operator got a 500 for what is simply an empty show.
        raise HTTPException(
            400, f"{show.name!r} has no formations, so there is nothing to fly. "
            "Add formations and prepare the show first.")

    async with _fleet_lock:
        if _fleet is not None and _fleet.phase in (
            Phase.ARMED,
            Phase.RUNNING,
            Phase.PAUSED,
        ):
            raise HTTPException(
                409,
                f"a show is {_fleet.phase.value}; land or disarm the fleet before "
                "loading another",
            )
        # Close the previous fleet's links before opening new ones. Without
        # this its reader thread kept running and its sockets stayed bound,
        # so reloading onto the same udpin ports could not bind them.
        if _fleet is not None:
            _fleet.driver.close()
            _fleet = None
        _stop_sitl()

        if body.signed and body.driver == "simulator":
            raise HTTPException(
                400, "the simulator has no MAVLink link to sign; pick a SITL "
                "or MAVLink driver")
        key = _mission_key(LINK_KEY_FILE)[:32] if body.signed else None

        if body.driver == "simulator":
            driver = SimDriver(
                faults=_sim_faults(body.faults),
                battery_endurance_s=show.limits.battery_endurance_s,
            )
        elif body.driver in ("sitl", "sitl-guided"):
            # Software in the loop: real MAVLink over real sockets, to
            # aircraft that exist only in this process. Everything in
            # drivers.py is exercised; nothing can leave the ground.
            from .sitl import VirtualFleet

            # Virtual aircraft come already holding the key, as aircraft
            # provisioned on the bench would.
            _sitl = VirtualFleet(show.drone_count, faults=_sitl_faults(body.faults),
                                 signing_key=key)
            if show.formations:
                _sitl.place(show.formations[0].points)
            _sitl.start()
            cls = (
                DRIVERS["mavlink-native"]
                if body.driver == "sitl"
                else DRIVERS["mavlink-guided"]
            )
            driver = cls(_sitl.endpoints, origin=show.origin, signing_key=key)
            try:
                driver.connect(show.drone_count, wait_heartbeat=20.0)
            except TimeoutError as exc:
                _stop_sitl()
                raise HTTPException(504, str(exc)) from None
        else:
            if not body.endpoints:
                raise HTTPException(
                    400,
                    f"driver {body.driver!r} needs one MAVLink endpoint per drone, "
                    "for example udpin:0.0.0.0:14551",
                )
            driver = DRIVERS[body.driver](
                body.endpoints, origin=show.origin, signing_key=key)
            try:
                driver.connect(show.drone_count, wait_heartbeat=20.0)
            except TimeoutError as exc:
                raise HTTPException(504, str(exc)) from None

        policy = PreflightPolicy(**body.policy)
        reg = _registry()
        _fleet = FleetManager(
            show, driver=driver, policy=policy,
            # Only bring the registry in when this show has airframes on its
            # slots; otherwise preflight would fail an identity check the
            # operator never opted into.
            registry=reg if reg.assignment(show.name) else None,
        )
        if driver is not None and getattr(driver, "name", "").startswith("mavlink"):
            _fleet.wait_for_fleet(10.0)
            _fleet.driver.set_origin(show.origin)
    return {"ok": True, "state": _fleet.snapshot()}


def _sim_faults(raw: dict) -> SimFaults:
    """Build simulator faults from the UI's form, refusing what it can't use.

    The fault form grew a field the simulator did not know about, and
    every load from the UI then failed with a 500 - which looked like the
    server being broken rather than a bad field. An unknown key is the
    caller's problem and is reported as a 400 naming it; nothing a client
    sends should be able to produce a 500.
    """
    import dataclasses

    known = {f.name for f in dataclasses.fields(SimFaults)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise HTTPException(
            400,
            f"unknown simulator fault(s): {', '.join(unknown)}. "
            f"Accepted: {', '.join(sorted(known))}.",
        )
    cleaned = dict(raw)
    # JSON object keys are strings; the simulator indexes drones by int.
    for key in ("link_loss", "weak_battery", "gps_degradation",
                "position_drift", "clock_offset"):
        if key in cleaned and isinstance(cleaned[key], dict):
            try:
                cleaned[key] = {int(k): float(v) for k, v in cleaned[key].items()}
            except (TypeError, ValueError):
                raise HTTPException(
                    400, f"{key} must map drone numbers to numbers"
                ) from None
    if "wind" in cleaned:
        wind = cleaned["wind"]
        if not (isinstance(wind, (list, tuple)) and len(wind) == 3):
            raise HTTPException(400, "wind must be three numbers: east, north, up")
        cleaned["wind"] = tuple(float(w) for w in wind)
    try:
        return SimFaults(**cleaned)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"bad simulator faults: {exc}") from None


def _sitl_faults(raw: dict) -> dict:
    """Map the UI's fault form onto per-aircraft SITL faults."""
    from .sitl import AircraftFaults

    out: dict[int, AircraftFaults] = {}
    for index, seconds in (raw.get("link_loss") or {}).items():
        out.setdefault(int(index), AircraftFaults()).go_silent_at = float(seconds)
    for index, pct in (raw.get("weak_battery") or {}).items():
        out.setdefault(int(index), AircraftFaults()).battery_percent = float(pct)
    for index, _ in (raw.get("gps_degradation") or {}).items():
        out.setdefault(int(index), AircraftFaults()).gps_fix = 3
    for index, offset in (raw.get("clock_offset") or {}).items():
        out.setdefault(int(index), AircraftFaults()).clock_offset = float(offset)
    return out


def _stop_sitl() -> None:
    global _sitl
    if _sitl is not None:
        _sitl.stop()
        _sitl = None


@app.get("/api/ops/state")
def ops_state() -> dict:
    fleet = _require_fleet()
    snap = fleet.snapshot()
    snap["link_signed"] = bool(getattr(fleet.driver, "signing_key", None))
    return snap


@app.post("/api/ops/signing/provision")
def provision_signing() -> dict:
    """Load link.key onto every connected aircraft and sign the link with it.

    The key crosses the link in the clear while this runs. Do it over USB or
    on a closed bench network, never on the show radio.
    """
    fleet = _require_fleet()
    if not hasattr(fleet.driver, "provision_signing"):
        raise HTTPException(400, "the simulator has no MAVLink link to sign")
    if fleet.phase in (Phase.ARMED, Phase.RUNNING, Phase.PAUSED):
        raise HTTPException(409, f"the fleet is {fleet.phase.value}; disarm first")
    key = _mission_key(LINK_KEY_FILE)[:32]
    try:
        result = fleet.driver.provision_signing(key)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    fleet.log("signing", f"signing key loaded on {len(result['confirmed'])} aircraft",
              **result)
    return {"ok": not result["unconfirmed"], **result}


@app.get("/api/ops/preflight")
def ops_preflight() -> dict:
    fleet = _require_fleet()
    checks = fleet.preflight()
    return {
        "checks": [c.to_dict() for c in checks],
        "ready": all(c.ok for c in checks if c.blocking),
    }


@app.post("/api/ops/validate")
def ops_validate() -> dict:
    return _require_fleet().validate().to_dict()


@app.post("/api/ops/upload")
def ops_upload(body: OverrideIn) -> dict:
    fleet = _require_fleet()
    approval = _approval(fleet.show)
    if _flies_hardware(fleet) and approval is None:
        # Unlike a failed safety check there is no override_reason for this:
        # an approval someone can skip by typing a sentence is not one.
        raise HTTPException(
            409,
            "this plan is not approved, so it cannot be uploaded to real "
            "aircraft. An operator must approve it first. Editing or "
            "re-preparing a show voids its approval, and approval cannot be "
            "overridden.",
        )
    try:
        mission = fleet.upload(override_reason=body.override_reason)
    except (PermissionError, RuntimeError, NotImplementedError) as exc:
        raise HTTPException(409, str(exc)) from None
    if approval:
        fleet.log("approval", f"plan approved by {approval['approved_by']}", **approval)
    return {"ok": True, "checksum": mission.to_dict()["checksum"]}


# --------------------------------------------------------------------------
# Mission approval
# --------------------------------------------------------------------------
#
# A named person signs off one exact plan. The record carries a fingerprint
# of the show, so any edit - a moved drone, a retimed transition, a new
# ceiling - voids it without anyone having to remember to revoke it.
# Rehearsal on the simulator or SITL needs no approval; real aircraft do.


class ApprovalIn(BaseModel):
    note: str = ""


def _approval_path(name: str) -> Path:
    p = _show_path(name)
    return p.with_name(p.name.replace(".show.json", ".approval.json"))


def _plan_hash(show: Show) -> str:
    from .show import show_to_dict

    body = json.dumps(show_to_dict(show), sort_keys=True).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _approval(show: Show) -> dict | None:
    """The approval on file, but only if it is for exactly this plan."""
    try:
        rec = json.loads(_approval_path(show.name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return rec if rec.get("plan_hash") == _plan_hash(show) else None


def _flies_hardware(fleet: FleetManager) -> bool:
    return _sitl is None and getattr(fleet.driver, "name", "").startswith("mavlink")


@app.get("/api/ops/approvals/{name}")
def get_approval(name: str) -> dict:
    rec = _approval(_require_show(name))
    return {
        "approved": rec is not None,
        "approval": rec,
        # A record exists but the show has changed since it was written.
        "stale": rec is None and _approval_path(name).exists(),
    }


@app.post("/api/ops/approvals/{name}")
def approve_show(name: str, body: ApprovalIn, request: Request) -> dict:
    """Sign off this exact plan. Refuses a show that fails safety."""
    show = _require_show(name)
    if not show.formations:
        raise HTTPException(400, f"{name!r} has no formations to approve")
    report = safety.validate(show)
    if not report.passed:
        raise HTTPException(
            409,
            f"{name!r} failed safety validation with {len(report.errors)} "
            f"error(s), so it cannot be approved. The first: "
            f"{report.errors[0].message}",
        )
    rec = {
        "show": name,
        "plan_hash": _plan_hash(show),
        # No accounts configured means local-only bootstrap mode: there is no
        # name to record, and saying so beats inventing one.
        "approved_by": getattr(request.state, "user", None) or "local operator",
        "role": getattr(request.state, "role", None),
        "ts": time.time(),
        "note": body.note,
    }
    _approval_path(name).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return {"ok": True, "approval": rec}


@app.delete("/api/ops/approvals/{name}")
def revoke_approval(name: str) -> dict:
    _approval_path(name).unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/ops/arm")
def ops_arm(body: OverrideIn) -> dict:
    fleet = _require_fleet()
    try:
        checks = fleet.arm(override_reason=body.override_reason)
    except PermissionError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"ok": True, "checks": [c.to_dict() for c in checks]}


@app.post("/api/ops/start")
async def ops_start(body: StartIn) -> dict:
    global _ops_task
    fleet = _require_fleet()
    try:
        when = fleet.start(at_unix_time=body.at_unix_time, lead_in=body.lead_in)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from None
    if _ops_task is None or _ops_task.done():
        _ops_task = asyncio.create_task(_run_loop())
    return {"ok": True, "start_unix": when, "starts_in_s": round(when - time.time(), 2)}


@app.post("/api/ops/pause")
def ops_pause() -> dict:
    try:
        _require_fleet().pause()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"ok": True}


@app.post("/api/ops/resume")
def ops_resume() -> dict:
    try:
        _require_fleet().resume()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"ok": True}


@app.post("/api/ops/suspend")
def ops_suspend() -> dict:
    fleet = _require_fleet()
    try:
        fleet.suspend(reason="operator, from UI")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"ok": True, "show_time": fleet.show_time}


@app.post("/api/ops/unsuspend")
def ops_unsuspend() -> dict:
    fleet = _require_fleet()
    try:
        fleet.unsuspend(reason="operator, from UI")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"ok": True, "show_time": fleet.show_time}


@app.post("/api/ops/collective-rth")
def ops_collective_rth() -> dict:
    """Bring the fleet home on one deconflicted plan, or hold if it cannot."""
    fleet = _require_fleet()
    report = fleet.collective_rth(reason="operator, from UI")
    if not report["executed"]:
        # Not an error: refusing to fly an unverified return is the correct
        # outcome. 409 so the UI shows it as a decision, not a success.
        raise HTTPException(409, report)
    return report


@app.post("/api/ops/emergency")
def ops_emergency(body: EmergencyIn) -> dict:
    fleet = _require_fleet()
    try:
        fleet.emergency(body.action, indices=body.drones, reason=body.reason)
    except (ValueError, PermissionError) as exc:
        raise HTTPException(409, str(exc)) from None
    return {"ok": True, "state": fleet.snapshot()}


@app.post("/api/ops/finish")
def ops_finish() -> dict:
    """Close out a show: credit airframes and write the flight log.

    Flight time comes from how far the show actually got, so an aborted
    show adds the seconds it flew rather than its designed length.
    """
    fleet = _require_fleet()
    touched = fleet.record_to_registry()
    if fleet.registry is not None:
        fleet.registry.save(FLEET_FILE)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    path = fleet.save_flight_log(LOG_DIR / f"{fleet.show.name}-{stamp}.flightlog.json")
    return {
        "ok": True,
        "flight_log": str(path),
        "airframes_credited": len(touched),
        "seconds": round(fleet.show_time, 1),
    }


@app.get("/api/ops/report")
def ops_report() -> Any:
    """The post-show report for the loaded fleet."""
    from fastapi.responses import HTMLResponse

    from . import reports

    fleet = _require_fleet()
    summary = fleet.registry.summary() if fleet.registry is not None else None
    return HTMLResponse(reports.post_show_html(fleet.flight_log(), registry_summary=summary))


@app.get("/api/ops/log")
def ops_log() -> dict:
    return _require_fleet().flight_log()


@app.get("/api/ops/sitl")
def ops_sitl() -> dict:
    """What the virtual aircraft themselves report, for comparison.

    The ground station only knows what telemetry tells it. This is ground
    truth, and the gap between the two is the whole point of running SITL.
    """
    if _sitl is None:
        return {"active": False}
    return {"active": True, **_sitl.summary()}




@app.post("/api/ops/log/save")
def ops_log_save() -> dict:
    fleet = _require_fleet()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    path = fleet.save_flight_log(LOG_DIR / f"{fleet.show.name}-{stamp}.flightlog.json")
    return {"ok": True, "path": str(path)}


async def _run_loop(rate: float = 10.0) -> None:
    """Drive the fleet's tick while a show is running.

    Runs in the server's event loop at a fixed rate. A guided-stream
    driver depends on this loop for its setpoints, so it must not be
    starved; nothing slow belongs here.
    """
    step = 1.0 / rate
    while True:
        fleet = _fleet
        if fleet is None:
            return
        if fleet.phase not in (Phase.RUNNING, Phase.PAUSED, Phase.ABORTING):
            return
        try:
            fleet.tick()
        except Exception as exc:  # a dead loop is worse than a logged error
            fleet.log("error", f"tick failed: {exc}")
        await asyncio.sleep(step)


@app.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    # The HTTP middleware does not see WebSocket upgrades, so the same rule
    # is applied here: live telemetry is for signed-in users, and with no
    # accounts yet, for this machine only.
    from . import auth

    host = socket.client.host if socket.client else None
    try:
        store = _auth()
    except auth.AuthError:
        await socket.close(code=4503)
        return
    if store.configured:
        token = socket.cookies.get(SESSION_COOKIE) or socket.query_params.get("token")
        if store.session(token) is None:
            await socket.close(code=4401)
            return
    elif not auth.is_local(host):
        await socket.close(code=4403)
        return

    await socket.accept()
    try:
        while True:
            fleet = _fleet
            if fleet is None:
                await socket.send_json({"phase": "unloaded"})
            else:
                if fleet.phase not in (Phase.RUNNING, Phase.PAUSED):
                    # Keep the simulated fleet's telemetry alive while idle
                    # so the operator sees batteries and GNSS on the pads.
                    fleet.tick()
                await socket.send_json(fleet.snapshot())
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        return
    except Exception:
        return


# --------------------------------------------------------------------------
# Static UI
# --------------------------------------------------------------------------


@app.get("/")
def index() -> Any:
    target = WEB_DIR / "index.html"
    if not target.exists():
        return JSONResponse(
            {"error": "web/index.html is missing", "api": "/docs"}, status_code=404
        )
    # no-cache = revalidate every load (ETag makes it a cheap 304). Without
    # it a browser heuristically keeps the old UI after an upgrade.
    return FileResponse(target, headers={"Cache-Control": "no-cache"})


@app.get("/simple")
def simple() -> Any:
    """Simple mode: pick a template, make it yours, check it, watch it."""
    target = WEB_DIR / "simple.html"
    if not target.exists():
        return JSONResponse({"error": "web/simple.html is missing"}, status_code=404)
    return FileResponse(target, headers={"Cache-Control": "no-cache"})


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def serve(host: str = "127.0.0.1", port: int = 8765, reload: bool = False) -> None:
    import uvicorn

    SHOW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run(
        "skyshine.server:app" if reload else app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
