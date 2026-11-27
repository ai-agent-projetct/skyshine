"""The helpers behind the AI panel: fix it for me, music to show, and the
two places Claude is asked for help (a request the command grammar did not
understand, and a shape described in words).

Nothing here flies anything. Every result is an ordinary design edit that
goes through the same prepare and safety checks as a hand-made one, and
every Claude answer is narrowed to something the app already knows how to
do: a command the grammar parses, or a drawing that becomes a formation.
Without an API key the first two work as normal and the last two fall back
to rules, so the app never depends on the network.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from . import planning, safety
from .show import Show

MODEL = "claude-opus-5-5"
HEADROOM = 3.0            # metres kept clear under the ceiling by the fixer
SHAPE_SVG_LIMIT = 20_000  # characters of path data accepted from Claude
# A key pasted into the app is kept here, next to mission.key, and never
# sent back to the browser. An ANTHROPIC_API_KEY in the environment wins.
KEY_FILE = Path(__file__).resolve().parent.parent / "claude.key"


# --------------------------------------------------------------------------
# Fix it for me
# --------------------------------------------------------------------------


def _samples(f) -> list[np.ndarray]:
    """Where a formation's drones are across its hold (motion included)."""
    if f.motion is None:
        return [f.points]
    return [f.pose(f.hold * k / 8) for k in range(9)]


def _formation_at(show: Show, t: float) -> int:
    """Index of the formation flying at time t (transition counts as its own)."""
    cursor = 0.0
    for i, f in enumerate(show.formations):
        cursor += f.duration
        if t < cursor:
            return i
    return len(show.formations) - 1


def fix(show: Show) -> dict:
    """Change the design, in place, until it passes - or say what is left.

    Only moves an operator would make by hand: lower or raise a formation,
    slide it inside the geofence or out of a keep-out zone, shorten holds
    to fit the battery, let the matcher re-pair drones that collide. The
    shapes themselves are never squashed, since that is what keeps drones
    apart.
    """
    if not show.formations:
        raise ValueError("add a formation first")
    lim, changes = show.limits, []
    top_room = lim.max_altitude - HEADROOM
    low_room = lim.min_nav_altitude + 1.0

    for f in show.formations:
        fitted = f.copy()
        planning.fit_formation(fitted, lim, show.fps)
        low, top = planning.height_range(fitted)
        if top > top_room:
            if top - low <= top_room - low_room:
                d = top - top_room
                f.points[:, 2] -= d
                changes.append(f"{f.name}: lowered {d:.1f} m to fit under the "
                               f"{lim.max_altitude:.0f} m ceiling")
            else:
                changes.append(f"{f.name}: too tall for the {lim.max_altitude:.0f} m "
                               "ceiling at a safe spacing; use fewer drones on it or "
                               "a filled shape")
        fitted = f.copy()
        planning.fit_formation(fitted, lim, show.fps)

        if len(lim.geofence) >= 3:
            poly = np.asarray(lim.geofence, dtype=np.float64)
            xy = np.vstack([p[:, :2] for p in _samples(fitted)])
            if not safety.points_in_polygon(xy, poly).all():
                d = poly.mean(axis=0) - fitted.points[:, :2].mean(axis=0)
                f.points[:, :2] += d
                inside = safety.points_in_polygon(xy + d, poly).all()
                changes.append(f"{f.name}: moved {math.hypot(*d):.1f} m to the middle of "
                               "the geofence" + ("" if inside else
                                                 ", but it is still too wide for it"))
                fitted.points[:, :2] += d

        for zi, z in enumerate(lim.keep_out):
            pts = np.vstack(_samples(fitted))
            zone = np.array([float(z["x"]), float(z["y"])])
            near = (np.hypot(*(pts[:, :2] - zone).T)
                    < float(z["radius"]) + lim.min_separation)
            near &= (pts[:, 2] > float(z["bottom"]) - lim.min_separation)
            near &= (pts[:, 2] < float(z["top"]) + lim.min_separation)
            if not near.any():
                continue
            away = fitted.points[:, :2].mean(axis=0) - zone
            away = away / (np.linalg.norm(away) or 1.0) if np.linalg.norm(away) > 1e-6 \
                else np.array([1.0, 0.0])
            # Slide until the nearest drone clears the zone with a spacing to spare.
            reach = float(np.max(np.linalg.norm(pts[:, :2] - pts[:, :2].mean(axis=0), axis=1)))
            gap = float(np.linalg.norm(fitted.points[:, :2].mean(axis=0) - zone))
            d = away * (float(z["radius"]) + 2 * lim.min_separation + reach - gap)
            f.points[:, :2] += d
            fitted.points[:, :2] += d
            changes.append(f"{f.name}: moved {np.linalg.norm(d):.1f} m away from keep-out "
                           f"zone {z.get('name') or zi + 1!r}")

    report = None
    for _ in range(3):
        prepared = planning.prepare(show)
        report = safety.validate(prepared.show)
        if report.passed:
            break
        acted = False
        for v in report.errors:
            if v.kind == "endurance" and v.value and v.limit:
                excess = v.value - v.limit + 5.0
                holds = sum(f.hold for f in show.formations)
                need = [planning.motion_hold_need(f, lim) * 1.15 for f in show.formations]
                spare = sum(max(f.hold - max(n, 1.0), 0) for f, n in zip(show.formations, need))
                if spare <= 0:
                    continue
                k = min(excess / spare, 1.0)
                for f, n in zip(show.formations, need):
                    f.hold = round(f.hold - k * max(f.hold - max(n, 1.0), 0), 2)
                changes.append(f"shortened the holds by {k * spare:.0f} s in all "
                               f"(from {holds:.0f} s) to fit the battery")
                acted = True
            elif v.kind == "separation" and v.time is not None:
                i = _formation_at(prepared.show, v.time) - 2   # take-off is two formations
                if 0 <= i < len(show.formations) and show.formations[i].keep_order:
                    show.formations[i].keep_order = False
                    changes.append(f"{show.formations[i].name}: let the drones swap places "
                                   "again (lock order was making two of them collide)")
                    acted = True
        if not acted:
            break

    left = [] if report is None else [v.message for v in report.errors]
    return {"changes": changes, "passed": bool(report and report.passed),
            "left": left[:10],
            "summary": ("All checks pass." if report and report.passed else
                        f"{len(left)} problem(s) need a person: " + "; ".join(left[:3]))}


# --------------------------------------------------------------------------
# Music to show
# --------------------------------------------------------------------------


def music_sync(show: Show, *, beats: int = 4) -> list[str]:
    """Retime the design so every formation lands on the beat grid.

    Each formation arrives (finishes its transition) on a step of `beats`
    beats, nearest to where it already arrived, and the last hold ends on
    one too. No transition is made shorter than the flight limits allow -
    the minimums come from the same scaling and timing prepare uses, so
    prepare does not stretch them again and break the alignment. Extra
    time goes into the holds.
    """
    beat = show.music.beat_seconds
    if beat is None:
        raise ValueError("set the song's tempo first (Detect tempo, or type the BPM)")
    if not show.formations:
        raise ValueError("add a formation first")
    step = beat * max(1, int(beats))
    phase = show.music.beat_phase
    work = show.copy()
    for f in work.formations:
        planning.fit_formation(f, work.limits, work.fps)
    work = planning.autotime(work)
    min_t = [w.transition for w in work.formations]
    min_h = [max(w.hold, 1.0) for w in work.formations]

    def grid(t: float, floor: float) -> float:
        k = round((t - phase) / step)
        while phase + k * step < floor - 1e-9:
            k += 1
        return phase + k * step

    # Not rounded: per-step rounding adds up, and the beat drifts by the end.
    fs, before = show.formations, show.duration
    arrive = grid(max(fs[0].transition, min_t[0]), min_t[0])
    fs[0].transition = arrive
    for i in range(1, len(fs)):
        natural = arrive + fs[i - 1].hold + max(fs[i].transition, min_t[i])
        nxt = grid(natural, arrive + min_h[i - 1] + min_t[i])
        fs[i].transition = min(max(fs[i].transition, min_t[i]), nxt - arrive - min_h[i - 1])
        fs[i - 1].hold = nxt - arrive - fs[i].transition
        arrive = nxt
    fs[-1].hold = grid(arrive + fs[-1].hold, arrive + min_h[-1]) - arrive
    return [f"{len(fs)} formations now land every {beats} beats "
            f"({step:.2f} s at {show.music.bpm:g} BPM)",
            f"design runs {show.duration:.0f} s (was {before:.0f} s)"]


# --------------------------------------------------------------------------
# Claude
# --------------------------------------------------------------------------


def key_source() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "environment"
    return "saved" if KEY_FILE.exists() and KEY_FILE.read_text().strip() else None


def available() -> bool:
    return key_source() is not None


def save_key(key: str) -> None:
    key = (key or "").strip()
    if not key.startswith("sk-ant-") or not 20 <= len(key) <= 300 or any(c.isspace() for c in key):
        raise ValueError("that does not look like a Claude API key (it starts with sk-ant-)")
    KEY_FILE.write_text(key, encoding="utf-8")


def _client():
    import ssl

    import anthropic

    # Certificates are checked against the operating system's trust store,
    # not the SDK's bundled list. Antivirus HTTPS scanning and company
    # proxies re-sign traffic with a root that only the OS knows; with the
    # bundled list every call fails with CERTIFICATE_VERIFY_FAILED.
    # Verification stays on either way.
    http = anthropic.DefaultHttpxClient(verify=ssl.create_default_context())
    if key_source() == "saved":
        return anthropic.Anthropic(api_key=KEY_FILE.read_text().strip(), http_client=http)
    return anthropic.Anthropic(http_client=http)


def ask_json(system: str, prompt: str, schema: dict, *, max_tokens: int = 8000) -> dict:
    """One Claude call whose answer is JSON matching `schema`.

    Raises RuntimeError when Claude declines; any other failure is the SDK's
    own exception. Callers fall back to rules either way.
    """
    reply = _client().beta.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low",
                       "format": {"type": "json_schema", "schema": schema}},
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    if reply.stop_reason == "refusal":
        raise RuntimeError("Claude declined this request")
    if reply.stop_reason == "max_tokens":
        raise RuntimeError("Claude's answer was cut off")
    return json.loads(next(b.text for b in reply.content if b.type == "text"))


def translate(text: str) -> list[str]:
    """Rewrite a request the grammar missed as commands it understands.

    Claude only chooses words: each command is parsed and checked by the
    voice grammar exactly as if the operator had typed it, so it cannot
    reach anything typing could not (arming and starting stay refused).
    """
    from . import voice

    vocab = voice.vocabulary()
    system = (
        "You turn a drone-show designer's request into commands for Sky Shine's "
        "command grammar. Use only phrasings like these examples: "
        + "; ".join(vocab.get("examples", [])[:40])
        + ". " + voice.help_text()
        + " Return at most 4 commands, in order. If nothing fits, return none."
    )
    schema = {"type": "object", "additionalProperties": False, "required": ["commands"],
              "properties": {"commands": {"type": "array", "items": {"type": "string"}}}}
    out = ask_json(system, text[:500], schema, max_tokens=2000)
    return [str(c)[:200] for c in out["commands"][:4]]


def shape_svg(description: str) -> str:
    """An SVG of a described shape, drawn by Claude as filled outlines."""
    system = (
        "You draw simple, recognisable silhouettes for a drone light show. "
        "Answer with SVG path data in a 100 by 100 box, y down, using only "
        "M, L, C, Q, Z commands. One to five closed outlines, bold and "
        "readable from far away, no text, no fine detail."
    )
    schema = {"type": "object", "additionalProperties": False, "required": ["paths"],
              "properties": {"paths": {"type": "array", "items": {"type": "string"}}}}
    out = ask_json(system, f"Draw: {description[:200]}", schema)
    data = " ".join(str(p) for p in out["paths"][:5])[:SHAPE_SVG_LIMIT]
    if not data.strip():
        raise RuntimeError("Claude returned an empty drawing")
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            f'<path d="{data.replace(chr(34), "")}"/></svg>')


if __name__ == "__main__":
    from . import templates

    s = templates.build("showcase", "check", 30)
    s.music.bpm, s.music.beat_phase = 120.0, 0.25
    print(music_sync(s))
    p = planning.prepare(s).show
    step = 2.0
    cursor = 0.0
    for i, f in enumerate(p.formations):
        if 2 <= i < 2 + len(s.formations):
            k = (cursor + f.transition - p.music.beat_phase) / step
            assert abs(k - round(k)) < 1e-6, (f.name, k)
        cursor += f.duration
    s.limits.max_altitude = 60.0
    r = fix(s)
    print(r["changes"], r["summary"])
    print("assistant.py self-check ok")
