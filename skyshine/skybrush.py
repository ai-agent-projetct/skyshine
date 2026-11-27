"""Export to Skybrush's compiled show format (.skyc). EXPERIMENTAL.

Written from Skybrush's published specification (docs.skybrush.io, "Skybrush
file formats"): the show JSON schema, the version 1 trajectory, and the
light program bytecode. Every rule below is from that specification
except one: the outer container. The specification says only that a .skyc
is "a compressed JSON file". Here it is a ZIP holding show.json. That part
is unverified; load an export into Skybrush before relying on it, and
compare with a .skyc made by Skybrush itself.

No Skybrush code is used. Skybrush's own code is GPL-3.0; this file
implements the published format independently.
"""

from __future__ import annotations

import base64
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from .show import Show
from .trajectory import build_mission, build_timeline, simplify

FPS = 50                      # light program frames per second, per the spec
OP_SET_COLOR, OP_FADE_TO_COLOR = 0x04, 0x08


def varuint(n: int) -> bytes:
    """Unsigned little-endian base-128: 7 bits a byte, top bit = more follow."""
    n = max(0, min(int(n), 2**32 - 1))
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def light_program(keys: np.ndarray) -> bytes:
    """Bytecode for one drone's lights from [t, r, g, b] keys.

    The first colour is set at once; every later key is a linear fade from
    the previous one, which is how Sky Shine interpolates between keys.
    Durations are rounded on the absolute frame count, so rounding never
    accumulates over a long show.
    """
    keys = np.asarray(keys, dtype=np.float64)
    out = bytearray()
    first = np.clip(np.round(keys[0, 1:]), 0, 255).astype(int)
    out += bytes([OP_SET_COLOR, *first]) + varuint(round(keys[0, 0] * FPS))
    prev = round(keys[0, 0] * FPS)
    for t, *rgb in keys[1:]:
        frame = round(t * FPS)
        out += bytes([OP_FADE_TO_COLOR, *np.clip(np.round(rgb), 0, 255).astype(int)])
        out += varuint(max(frame - prev, 0))
        prev = frame
    return bytes(out)


def run_light_program(code: bytes, until_frame: int) -> list[tuple[int, tuple]]:
    """A minimal reader for the two opcodes written above, for checking.

    Returns (frame, colour) at each instruction boundary up to until_frame.
    """
    i, frame, colour, out = 0, 0, (0, 0, 0), []

    def read_varuint():
        nonlocal i
        value = shift = 0
        while True:
            b = code[i]
            i += 1
            value |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                return value

    while i < len(code) and frame <= until_frame:
        op = code[i]
        i += 1
        if op not in (OP_SET_COLOR, OP_FADE_TO_COLOR):
            raise ValueError(f"unexpected opcode {op:#x}")
        target = tuple(code[i:i + 3])
        i += 3
        frames = read_varuint()
        if op == OP_SET_COLOR:
            colour = target
            out.append((frame, colour))
            frame += frames
        else:
            frame += frames
            colour = target
            out.append((frame, colour))
    return out


def _thin_path(rows: np.ndarray, tol: float = 0.01) -> np.ndarray:
    """Drop trajectory samples that straight lines already reproduce."""
    return simplify(rows, tol, metric="l2")


def show_json(show: Show, *, sample_hz: float = 10.0) -> dict:
    """The show as the .skyc JSON document (specification version 1)."""
    if not show.formations:
        raise ValueError("this show has no formations to export")
    lim, origin = show.limits, show.origin
    timeline = build_timeline(show, fps=max(sample_hz, show.fps))
    step = max(1, int(round(timeline.t[1:].size / max(timeline.duration * sample_hz, 1))))
    idx = np.unique(np.concatenate([np.arange(0, len(timeline.t), step), [len(timeline.t) - 1]]))
    mission = build_mission(show)
    drones = []
    for d in range(show.drone_count):
        path = np.column_stack([timeline.t[idx], timeline.pos[idx, d]])
        path = _thin_path(path)
        points = [[round(float(t), 3), [round(float(v), 3) for v in p], []] for t, *p in path]
        drones.append({"type": "generic", "settings": {
            "name": f"Drone {d + 1}",
            "home": points[0][1], "landAt": points[-1][1],
            "trajectory": {"version": 1, "points": points, "takeoffTime": 0},
            "lights": {"version": 1, "data": base64.b64encode(
                light_program(mission.plans[d].lights)).decode("ascii")},
        }})
    # Skybrush x points at compass bearing `orientation`, y to its left. Sky
    # Shine's x is east turned clockwise by orientation_deg, so the frames
    # match exactly at bearing 90 + orientation_deg.
    location = {"orientation": (90.0 + origin.orientation_deg) % 360.0,
                "origin": [round(origin.latitude * 1e7), round(origin.longitude * 1e7)]
                + ([round(origin.amsl * 1000)] if origin.amsl is not None else [])}
    doc = {
        "version": 1,
        "environment": {"type": "outdoor", "location": location},
        "meta": {"title": show.name, "inputs": ["Sky Shine"]},
        "settings": {
            "validation": {
                "maxAltitude": lim.max_altitude, "minNavAltitude": lim.min_nav_altitude,
                "minDistance": lim.min_separation,
                "maxVelocityXY": lim.max_velocity_xy, "maxVelocityZUp": lim.max_velocity_up,
                "maxVelocityZ": lim.max_velocity_down,
                "maxAccelerationXY": lim.max_acceleration, "maxAccelerationZ": lim.max_acceleration,
            },
        },
        "swarm": {"drones": drones},
    }
    if show.music.cues:
        doc["settings"]["cues"] = {"version": 1, "items": [
            {"name": label, "time": round(float(t), 3)} for t, label in show.music.cues]}
    return doc


# --------------------------------------------------------------------------
# .skyb: the per-drone binary file Skybrush's ArduCopter firmware flies
# (it loads /COLLMOT/show.skyb). Layout, read from the open-source parser
# (libskybrush, src/formats/binary.c and src/trajectory/): no Skybrush code
# is used here.
#
#   "skyb", version (1)
#   blocks: type (u8), length (u16 LE), body
#     1 trajectory: scale (u8, mm per unit), start x y z (i16 LE, units),
#       2 legacy yaw bytes, then segments: header (u8, 2 bits per axis
#       x y z yaw: 0 constant, 1 linear, 2 cubic, 3 7th order Bezier),
#       duration (u16 LE, ms), control points 1..n-1 per axis (i16 LE).
#     2 light program: the same ledctrl bytecode as .skyc, ending in 0x00.
#   Coordinates are millimetres in the show frame: x at the bearing in
#   SHOW_ORIENTATION, y to its left, z up - the frame show_json uses.
# --------------------------------------------------------------------------

SKYB_TRAJECTORY, SKYB_LIGHTS, SKYB_COMMENT = 1, 2, 3
OP_END = 0x00
_SEG_MAX_MS = 65535


def read_skyb(data: bytes) -> dict[int, bytes]:
    """Blocks of a .skyb file by type (the first of each type)."""
    if data[:4] != b"skyb" or data[4] not in (1, 2):
        raise ValueError("not a Skybrush binary show file")
    i = 5
    if data[4] == 2:
        features = data[5]
        i = 6 + (4 if features & 1 else 0)
    blocks: dict[int, bytes] = {}
    while i < len(data):
        kind, length = data[i], data[i + 1] | (data[i + 2] << 8)
        body = data[i + 3:i + 3 + length]
        if len(body) != length:
            raise ValueError("truncated block")
        blocks.setdefault(kind, body)
        i += 3 + length
    return blocks


def _i16(b: bytes, i: int) -> int:
    return int.from_bytes(b[i:i + 2], "little", signed=True)


def trajectory_segments(block: bytes) -> list[tuple[int, int, list[list[float]]]]:
    """[(start_ms, duration_ms, [x_ctrl, y_ctrl, z_ctrl])] in millimetres."""
    scale = block[0] & 0x7F
    if not scale:
        return []
    pos = [_i16(block, 1 + 2 * k) * scale for k in range(3)]
    i, t, out = 9, 0, []
    while i < len(block):
        head, dur = block[i], block[i + 1] | (block[i + 2] << 8)
        i += 3
        axes = []
        for a in range(4):
            n = 1 << ((head >> (2 * a)) & 3)
            ctrl = [pos[a]] if a < 3 else [0]
            for _ in range(n - 1):
                ctrl.append(_i16(block, i) * scale)
                i += 2
            if a < 3:
                axes.append(ctrl)
                pos[a] = ctrl[-1]
        out.append((t, dur, axes))
        t += dur
    return out


def _bezier(ctrl: list[float], s: float) -> float:
    pts = list(ctrl)
    while len(pts) > 1:
        pts = [a + (b - a) * s for a, b in zip(pts, pts[1:])]
    return pts[0]


def skyb_position(segments, t_ms: float) -> tuple[float, float, float]:
    """Where a parsed trajectory says the drone is at t_ms (millimetres)."""
    for start, dur, axes in segments:
        if t_ms <= start + dur or (start, dur, axes) == segments[-1]:
            s = 0.0 if dur == 0 else min(max((t_ms - start) / dur, 0.0), 1.0)
            return tuple(_bezier(c, s) for c in axes)
    return (0.0, 0.0, 0.0)


def _trajectory_block(rows: np.ndarray) -> bytes:
    """Linear segments through (t_s, x_m, y_m, z_m) rows."""
    mm = rows[:, 1:4] * 1000.0
    scale = max(1, int(math.ceil(float(np.max(np.abs(mm))) / 32000.0)))
    if scale > 127:
        raise ValueError("show is too large for one Skybrush trajectory (over 4 km)")
    q = np.round(mm / scale).astype(int)
    out = bytearray([scale])
    for v in q[0]:
        out += int(v).to_bytes(2, "little", signed=True)
    out += b"\x00\x00"
    ms = np.round(rows[:, 0] * 1000.0).astype(np.int64)
    ms -= ms[0]
    for k in range(1, len(rows)):
        start, end = int(ms[k - 1]), int(ms[k])
        a, b = q[k - 1], q[k]
        steps = max(1, -(-(end - start) // _SEG_MAX_MS))      # split anything over 65.5 s
        for s in range(1, steps + 1):
            t0 = start + (end - start) * (s - 1) // steps
            t1 = start + (end - start) * s // steps
            p = np.round(a + (b - a) * (s / steps)).astype(int)
            prev = np.round(a + (b - a) * ((s - 1) / steps)).astype(int)
            head = 0
            coords = bytearray()
            for axis in range(3):
                if p[axis] != prev[axis]:
                    head |= 1 << (2 * axis)
                    coords += int(p[axis]).to_bytes(2, "little", signed=True)
            out += bytes([head]) + (t1 - t0).to_bytes(2, "little") + coords
    return bytes(out)


def _block(kind: int, body: bytes) -> bytes:
    if len(body) > 0xFFFF:
        raise ValueError(f"block {kind} is {len(body)} bytes; Skybrush blocks hold 65535")
    return bytes([kind]) + len(body).to_bytes(2, "little") + body


def plan_rows(plan, sample_hz: float = 10.0, tol: float = 0.01) -> np.ndarray:
    """A DronePlan's eased motion as (t, x, y, z) rows, thinned to `tol` metres."""
    from .trajectory import evaluate_plan

    pts = np.asarray(plan.points, dtype=np.float64)
    times = [pts[0, 0]]
    for k in range(1, len(pts)):     # every plan point, and samples in between
        t0, t1 = pts[k - 1, 0], pts[k, 0]
        n = max(1, int(math.ceil((t1 - t0) * sample_hz)))
        times.extend(t0 + (t1 - t0) * j / n for j in range(1, n + 1))
    rows = np.array([[t, *evaluate_plan(plan, t)] for t in times])
    return simplify(rows, tol, metric="l2")


def skyb_bytes(plan, title: str = "") -> bytes:
    """One drone's show as a .skyb file for Skybrush's ArduCopter firmware."""
    body = _block(SKYB_TRAJECTORY, _trajectory_block(plan_rows(plan)))
    lights = np.asarray(plan.lights, dtype=np.float64)
    if len(lights):
        body += _block(SKYB_LIGHTS, light_program(lights) + bytes([OP_END]))
    if title:
        body += _block(SKYB_COMMENT, title.encode("utf-8")[:200])
    return b"skyb\x01" + body


def write_skyc(show: Show, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("show.json", json.dumps(show_json(show), separators=(",", ":")))
    return path


if __name__ == "__main__":
    assert varuint(513) == bytes([0x81, 0x04])          # the specification's own example
    assert varuint(0) == b"\x00" and varuint(127) == b"\x7f"
    code = light_program(np.array([[0, 255, 0, 0], [2.0, 0, 0, 255], [2.5, 0, 0, 255]]))
    steps = run_light_program(code, 10**6)
    assert steps == [(0, (255, 0, 0)), (100, (0, 0, 255)), (125, (0, 0, 255))], steps
    print("skybrush.py self-check ok")
