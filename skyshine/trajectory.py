"""Sampling a show into a timeline, and compressing it for upload.

A Timeline is the dense form: every drone's position and colour at a fixed
rate. The validator and the 3D preview both read it.

A mission package is the compact form that actually goes to the aircraft:
per drone, the few control points and the shared easing that reproduce the
same motion exactly. One drone's package is a few kilobytes even for a
long show, which matters when 500 of them have to be uploaded over a radio
before the audience arrives.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import json
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .show import Show, Timeline, ease, iter_formation_windows


# Refuse to materialise a timeline larger than this many drone-samples.
# 60 M is ~1.4 GB of float64 position data: past that, something upstream
# has produced a nonsense duration and the honest answer is to say so
# rather than to exhaust memory.
MAX_DRONE_SAMPLES = 60_000_000
# Seconds between uploaded waypoints along a moving formation's path.
MOTION_STEP = 0.2


def build_timeline(show: Show, *, fps: float | None = None) -> Timeline:
    """Sample a show at a fixed rate, hitting every formation boundary exactly.

    Boundaries are forced into the sample set so that the validator sees
    the true held poses, not an interpolation that straddles them.
    """
    if not show.formations:
        raise ValueError("cannot sample a show with no formations")
    rate = float(fps if fps is not None else show.fps)
    if rate <= 0:
        raise ValueError("fps must be positive")
    step = 1.0 / rate

    projected = int((show.duration * rate + len(show.formations) + 1) * show.drone_count)
    if projected > MAX_DRONE_SAMPLES:
        raise ValueError(
            f"sampling this show would take {projected:,} drone-samples "
            f"({show.duration:.0f} s x {show.drone_count} drones at {rate:g} Hz). "
            "A show that long is almost always the symptom of a formation that "
            "got scaled far beyond its intended size, which then forces huge "
            "transition times. Check the show duration before sampling."
        )

    times: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    segments: list[np.ndarray] = []

    prev = show.formations[0].points
    clock = 0.0

    for index, f in enumerate(show.formations):
        if f.transition > 0 and index > 0:
            ts = _phase_times(clock, f.transition, step)
            tau = (ts - clock) / f.transition
            s = np.asarray(ease(f.easing, tau), dtype=np.float64)
            # Every drone shares `s`. This is the condition the separation
            # bound in planning.py rests on; do not make it per-drone.
            pos = prev[None, :, :] + (f.points - prev)[None, :, :] * s[:, None, None]
            col = _blend_colors(
                show.formations[index - 1].colors_at(
                    show.formations[index - 1].hold
                ),
                f.colors,
                s,
            )
            times.append(ts)
            positions.append(pos)
            colors.append(col)
            segments.append(np.full(len(ts), index, dtype=np.int32))
            clock += f.transition

        hold = max(f.hold, 0.0)
        ts = _phase_times(clock, hold, step, include_end=index == len(show.formations) - 1)
        if len(ts):
            if f.motion is None:
                pos = np.repeat(f.points[None, :, :], len(ts), axis=0)
            else:
                pos = np.stack([f.pose(float(t - clock)) for t in ts], axis=0)
            col = np.stack([f.colors_at(float(t - clock)) for t in ts], axis=0)
            times.append(ts)
            positions.append(pos)
            colors.append(col)
            segments.append(np.full(len(ts), index, dtype=np.int32))
        clock += hold
        prev = f.end_points

    t = np.concatenate(times)
    pos = np.concatenate(positions, axis=0)
    col = np.concatenate(colors, axis=0)
    seg = np.concatenate(segments)

    # Boundary samples can coincide; keep the first of each cluster so t
    # stays strictly increasing.
    keep = np.concatenate([[True], np.diff(t) > 1e-9])
    return Timeline(
        t=np.ascontiguousarray(t[keep]),
        pos=np.ascontiguousarray(pos[keep]),
        col=np.ascontiguousarray(col[keep]),
        segment=np.ascontiguousarray(seg[keep]),
        show=show,
    )


def _phase_times(
    start: float, length: float, step: float, *, include_end: bool = True
) -> np.ndarray:
    if length <= 0:
        return np.array([start]) if include_end else np.empty(0)
    count = max(2, int(np.ceil(length / step)) + 1)
    ts = np.linspace(start, start + length, count)
    return ts if include_end else ts[:-1]


def _blend_colors(a: np.ndarray, b: np.ndarray, s: np.ndarray) -> np.ndarray:
    af = a.astype(np.float64)[None, :, :]
    bf = b.astype(np.float64)[None, :, :]
    out = af + (bf - af) * s[:, None, None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Mission packages: what gets uploaded
# --------------------------------------------------------------------------


@dataclass
class DronePlan:
    """One aircraft's share of the show."""

    drone_index: int
    # Control points: (time_s, x, y, z). Motion between consecutive points
    # follows `easings[k]` over that interval.
    points: np.ndarray
    easings: list[str]
    # Light program: (time_s, r, g, b), linearly interpolated between keys.
    lights: np.ndarray

    def duration(self) -> float:
        return float(self.points[-1, 0]) if len(self.points) else 0.0

    def to_dict(self) -> dict:
        return {
            "drone": self.drone_index,
            "points": [
                [round(float(v), 4) for v in row] for row in self.points
            ],
            "easings": self.easings,
            "lights": [[round(float(r[0]), 3), int(r[1]), int(r[2]), int(r[3])] for r in self.lights],
        }


@dataclass
class MissionPackage:
    """The signed, versioned bundle an operator uploads and an auditor reads."""

    show_name: str
    drone_count: int
    duration: float
    origin: dict
    limits: dict
    plans: list[DronePlan] = field(default_factory=list)
    safety: dict | None = None
    checksum: str = ""

    def to_dict(self) -> dict:
        body = {
            "format": "skyshine.mission",
            "version": 1,
            "show": self.show_name,
            "drone_count": self.drone_count,
            "duration": round(self.duration, 3),
            "origin": self.origin,
            "limits": self.limits,
            "safety": self.safety,
            "plans": [p.to_dict() for p in self.plans],
        }
        body["checksum"] = _checksum(body)
        self.checksum = body["checksum"]
        return body

    def save(self, path, *, key: bytes | None = None, signer: str = "") -> str:
        """Write the bundle, signed when a key is given."""
        body = self.to_dict()
        if key is not None:
            body = sign_mission(body, key, signer)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(body, indent=1), encoding="utf-8")
        return str(p)


def _checksum(body: dict) -> str:
    clone = {k: v for k, v in body.items() if k != "checksum"}
    blob = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# -- signing ----------------------------------------------------------------
#
# The checksum says a package is internally consistent, which anyone who edits
# a plan can restore by recomputing it. The signature says this ground
# station built it: it needs a key the editor does not have.
#
# ponytail: HMAC is symmetric, so whoever holds the key to verify a package
# can also sign one. That is fine while one ground station signs and checks
# its own files. Move to Ed25519 (the `cryptography` package) when a third
# party has to verify without being trusted to sign.


def mission_key(path: str | Path) -> bytes:
    """The ground station's signing key, created on first use."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "x", encoding="ascii") as fh:
            fh.write(secrets.token_hex(32))
    except FileExistsError:
        pass
    try:
        key = bytes.fromhex(p.read_text(encoding="ascii").strip())
    except ValueError:
        key = b""
    if len(key) < 32:
        raise ValueError(
            f"{p} is not a usable signing key. It must hold at least 32 bytes "
            "as hex. Delete it to have a new one made; packages signed with "
            "the old key will then no longer verify."
        )
    return key


def _key_id(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()[:16]


def _mac(key: bytes, checksum: str, signer, signed_at) -> str:
    # JSON, not string joining: a signer name must not be able to slide
    # into the timestamp field and still produce the same bytes.
    msg = json.dumps([checksum, signer, signed_at]).encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def sign_mission(body: dict, key: bytes, signer: str) -> dict:
    signed_at = round(time.time(), 3)
    return {
        **body,
        "signature": {
            "alg": "HMAC-SHA256",
            "key_id": _key_id(key),
            "signer": signer,
            "signed_at": signed_at,
            "value": _mac(key, body["checksum"], signer, signed_at),
        },
    }


def verify_mission(body, key: bytes) -> tuple[bool, str]:
    """Is this exactly the package this ground station signed, and by whom?"""
    if not isinstance(body, dict) or body.get("format") != "skyshine.mission":
        return False, "not a Sky Shine mission package"
    sig = body.get("signature")
    if not isinstance(sig, dict):
        return False, "the package is not signed"
    content = {k: v for k, v in body.items() if k != "signature"}
    if _checksum(content) != body.get("checksum"):
        return False, "the contents were changed after the package was built"
    if sig.get("key_id") != _key_id(key):
        return False, "signed with a different key, not this ground station's"
    want = _mac(key, body["checksum"], sig.get("signer"), sig.get("signed_at"))
    if not hmac.compare_digest(want, str(sig.get("value", ""))):
        return False, "the signature does not match the package"
    return True, f"signed by {sig.get('signer') or 'unknown'}"


def build_mission(
    show: Show, *, safety: dict | None = None, light_step: float = 0.25
) -> MissionPackage:
    """Compress a show into one plan per drone.

    Positions reduce to the formation boundaries, which is lossless: the
    motion between them is fully described by the endpoints plus the shared
    easing. Lights are sampled, because colour keyframes inside a hold can
    be arbitrary.
    """
    windows = list(iter_formation_windows(show))
    plans: list[DronePlan] = []
    timeline = build_timeline(show, fps=max(1.0 / light_step, show.fps))
    # A moving hold has no single endpoint, so it goes up as short straight
    # steps. MOTION_STEP apart they follow the eased curve to well inside a
    # centimetre for any motion the autotimer allows.
    paths: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index, _t_start, t_hold, t_end in windows:
        f = show.formations[index]
        if f.motion is not None:
            k = max(2, int(math.ceil(f.hold / MOTION_STEP)))
            offsets = np.linspace(0.0, f.hold, k + 1)[1:]
            paths[index] = (t_hold + offsets, np.stack([f.pose(float(o)) for o in offsets]))

    for d in range(show.drone_count):
        # Start parked at the first formation's slot.
        times: list[float] = [0.0]
        xyz: list[np.ndarray] = [show.formations[0].points[d]]
        easings: list[str] = []
        for index, _t_start, t_hold, t_end in windows:
            f = show.formations[index]
            if index > 0 and f.transition > 0:
                times.append(t_hold)
                xyz.append(f.points[d])
                easings.append(f.easing)
            if index in paths:
                step_times, poses = paths[index]
                times.extend(step_times.tolist())
                xyz.extend(poses[:, d])
                easings.extend(["linear"] * len(step_times))
            elif f.hold > 0:
                times.append(t_end)
                xyz.append(f.points[d])
                easings.append("hold")
        pts = np.column_stack([np.asarray(times, dtype=np.float64), np.asarray(xyz)])
        lights = np.column_stack(
            [timeline.t, timeline.col[:, d, 0], timeline.col[:, d, 1], timeline.col[:, d, 2]]
        )
        plans.append(
            DronePlan(
                drone_index=d,
                points=pts,
                easings=easings,
                lights=_thin_lights(lights),
            )
        )

    return MissionPackage(
        show_name=show.name,
        drone_count=show.drone_count,
        duration=show.duration,
        origin=asdict(show.origin),
        limits=asdict(show.limits),
        plans=plans,
        safety=safety,
    )


def _thin_lights(lights: np.ndarray, tol: float = 0.5) -> np.ndarray:
    """Drop light keys that linear interpolation already reproduces.

    A show that holds one colour for 30 s should upload two keys, not 300.
    Every dropped key stays within `tol` of every colour channel.
    """
    return simplify(lights, tol, metric="max")


def simplify(rows: np.ndarray, tol: float, *, metric: str = "l2") -> np.ndarray:
    """Keep the fewest [t, values...] rows that straight lines in time
    reproduce to within `tol` (Douglas-Peucker in time).

    Every dropped row is checked against the segment that replaces it.
    Checking only neighbours, as this once did, let the error build up
    along a long gentle change: 11 cm on a flight path at a 1 cm setting.
    metric is "l2" for positions, "max" for per-channel colour error.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if len(rows) <= 2:
        return rows
    keep = np.zeros(len(rows), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(rows) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        t, v = rows[a:b + 1, 0], rows[a:b + 1, 1:]
        span = t[-1] - t[0]
        s = np.zeros(len(t)) if span <= 0 else (t - t[0]) / span
        diff = v - (v[0] + (v[-1] - v[0]) * s[:, None])
        err = np.linalg.norm(diff, axis=1) if metric == "l2" else np.abs(diff).max(axis=1)
        k = int(np.argmax(err))
        if err[k] > tol:
            keep[a + k] = True
            stack += [(a, a + k), (a + k, b)]
    return rows[keep]


def evaluate_plan(plan: DronePlan, time_s: float) -> np.ndarray:
    """Reconstruct a drone's commanded position from its compressed plan.

    Exists so that the uploaded artefact can be replayed and diffed against
    the designed timeline. If these two ever disagree, the compression is
    wrong and nothing should fly.
    """
    pts = plan.points
    if len(pts) == 0:
        raise ValueError("empty plan")
    t = float(np.clip(time_s, pts[0, 0], pts[-1, 0]))
    k = int(np.searchsorted(pts[:, 0], t, side="right")) - 1
    k = max(0, min(k, len(pts) - 2))
    span = pts[k + 1, 0] - pts[k, 0]
    tau = 0.0 if span <= 0 else (t - pts[k, 0]) / span
    kind = plan.easings[k] if k < len(plan.easings) else "hold"
    s = 0.0 if kind == "hold" else float(ease(kind, tau))
    return pts[k, 1:] + (pts[k + 1, 1:] - pts[k, 1:]) * s
