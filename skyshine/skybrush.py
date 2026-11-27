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
