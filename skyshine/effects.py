"""Light effects: what the swarm does with colour while it holds a shape.

An effect is a pure function of (positions, time) -> colours. Rendering one
bakes it into the formation's `color_keys`, so effects leave nothing behind
at runtime: the result is ordinary keyframe data that serialises into the
show file, compresses into the mission package and plays back onboard. The
aircraft never has to know what a "chase" is.

That matters for more than tidiness. A light program the ground station
evaluates live would make the show depend on the radio; a light program
baked into keyframes is played from the drone's own clock, same as the
trajectory.

Spatial effects project each drone onto an axis of the formation and use
that as the parameter, so a gradient across a heart follows the heart
rather than the bounding box of the sky.
"""

from __future__ import annotations

import colorsys
import math
from typing import Callable

import numpy as np

from .show import Formation, Music

# An effect maps (points (n,3), time in seconds) to colours (n,3) uint8.
EffectFn = Callable[[np.ndarray, float], np.ndarray]

AXES = {"x": 0, "y": 1, "z": 2}


def _rgb(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size != 3:
        raise ValueError(f"expected an r,g,b triple, got {value!r}")
    return np.clip(arr, 0, 255)


def _axis_parameter(points: np.ndarray, axis: str) -> np.ndarray:
    """Each drone's position along `axis`, normalised to [0, 1].

    `radial` measures distance from the formation centroid instead, which
    is what you want for ripples and blooms.
    """
    p = np.asarray(points, dtype=np.float64)
    if axis == "radial":
        d = np.linalg.norm(p - p.mean(axis=0), axis=1)
    elif axis == "radial_xy":
        d = np.linalg.norm(p[:, :2] - p[:, :2].mean(axis=0), axis=1)
    elif axis in AXES:
        d = p[:, AXES[axis]]
    elif axis == "index":
        d = np.arange(len(p), dtype=np.float64)
    else:
        raise ValueError(
            f"unknown axis {axis!r}; expected one of x, y, z, radial, "
            "radial_xy, index"
        )
    span = float(np.ptp(d))
    if span < 1e-9:
        return np.zeros(len(p))
    return (d - d.min()) / span


def _mix(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Blend two colours per drone. t is (n,) in [0,1]."""
    t = np.clip(t, 0.0, 1.0)[:, None]
    return a[None, :] * (1.0 - t) + b[None, :] * t


def _out(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Effects
# --------------------------------------------------------------------------


def solid(color) -> EffectFn:
    c = _rgb(color)

    def fn(points, t):
        return _out(np.tile(c, (len(points), 1)))

    return fn


def fade(start, end, duration: float) -> EffectFn:
    """Cross-fade the whole swarm from one colour to another."""
    a, b = _rgb(start), _rgb(end)

    def fn(points, t):
        f = 0.0 if duration <= 0 else min(max(t / duration, 0.0), 1.0)
        return _out(np.tile(a + (b - a) * f, (len(points), 1)))

    return fn


def strobe(color, hz: float = 4.0, duty: float = 0.5, background=(0, 0, 0)) -> EffectFn:
    """Hard on/off flashing.

    Needs a render rate of at least 4x `hz` to land the edges anywhere near
    the right place; `recommended_fps` reports that, and `render` uses it.
    """
    c, bg = _rgb(color), _rgb(background)
    if hz <= 0:
        raise ValueError("strobe hz must be positive")

    def fn(points, t):
        phase = (t * hz) % 1.0
        value = c if phase < duty else bg
        return _out(np.tile(value, (len(points), 1)))

    fn.recommended_fps = max(20.0, hz * 8.0)  # type: ignore[attr-defined]
    return fn


def pulse(color, period: float = 2.0, floor: float = 0.15) -> EffectFn:
    """Breathing: a smooth sinusoidal brightness ramp."""
    c = _rgb(color)
    if period <= 0:
        raise ValueError("pulse period must be positive")

    def fn(points, t):
        k = floor + (1.0 - floor) * (0.5 + 0.5 * math.sin(2 * math.pi * t / period - math.pi / 2))
        return _out(np.tile(c * k, (len(points), 1)))

    return fn


def gradient(start, end, axis: str = "z") -> EffectFn:
    """A static colour ramp across the formation."""
    a, b = _rgb(start), _rgb(end)

    def fn(points, t):
        return _out(_mix(a, b, _axis_parameter(points, axis)))

    return fn


def rainbow(period: float = 6.0, axis: str = "x", saturation: float = 1.0,
            value: float = 1.0) -> EffectFn:
    """Hue sweeping across the formation and around over time."""
    if period <= 0:
        raise ValueError("rainbow period must be positive")

    def fn(points, t):
        u = _axis_parameter(points, axis)
        hues = (u + t / period) % 1.0
        rgb = np.array(
            [colorsys.hsv_to_rgb(float(h), saturation, value) for h in hues]
        )
        return _out(rgb * 255.0)

    return fn


def wipe(start, end, duration: float, axis: str = "x", softness: float = 0.08) -> EffectFn:
    """A colour front sweeping across the formation once."""
    a, b = _rgb(start), _rgb(end)
    if duration <= 0:
        raise ValueError("wipe duration must be positive")
    soft = max(softness, 1e-6)

    def fn(points, t):
        u = _axis_parameter(points, axis)
        front = min(max(t / duration, 0.0), 1.0) * (1.0 + 2 * soft) - soft
        f = np.clip((front - u) / soft + 0.5, 0.0, 1.0)
        return _out(_mix(a, b, f))

    return fn


def chase(color, period: float = 2.0, axis: str = "x", width: float = 0.18,
          background=(0, 0, 0)) -> EffectFn:
    """A bright band travelling across the formation, looping."""
    c, bg = _rgb(color), _rgb(background)
    if period <= 0:
        raise ValueError("chase period must be positive")
    w = max(width, 1e-6)

    def fn(points, t):
        u = _axis_parameter(points, axis)
        head = (t / period) % 1.0
        # Distance behind the head, wrapped, so the band re-enters cleanly.
        d = (head - u) % 1.0
        f = np.clip(1.0 - d / w, 0.0, 1.0)
        return _out(_mix(bg, c, f))

    return fn


def ripple(color, period: float = 2.5, background=(0, 0, 0), width: float = 0.2,
           axis: str = "radial") -> EffectFn:
    """Concentric waves running outward from the formation centre."""
    return chase(color, period=period, axis=axis, width=width, background=background)


def twinkle(color, background=(0, 0, 0), density: float = 0.12, rate: float = 6.0,
            seed: int = 0) -> EffectFn:
    """Random drones sparkle on and decay. Deterministic for a given seed.

    Determinism is the point: an effect that looked right in the preview
    has to look the same in the air, and a show file that renders
    differently each time cannot be reviewed.
    """
    c, bg = _rgb(color), _rgb(background)
    if rate <= 0:
        raise ValueError("twinkle rate must be positive")

    def fn(points, t):
        n = len(points)
        step = int(t * rate)
        frac = (t * rate) % 1.0
        out = np.zeros((n, 3))
        for offset, weight in ((0, 1.0 - frac), (1, frac)):
            rng = np.random.default_rng((seed * 1_000_003) ^ (step + offset))
            lit = rng.random(n) < density
            out += np.where(lit[:, None], c[None, :], bg[None, :]) * weight
        return _out(out)

    fn.recommended_fps = max(20.0, rate * 4.0)  # type: ignore[attr-defined]
    return fn


def meteor(color, period: float = 3.0, axis: str = "z", tail: float = 0.35,
           background=(0, 0, 0)) -> EffectFn:
    """A bright head with a fading tail falling along an axis."""
    c, bg = _rgb(color), _rgb(background)
    tail = max(tail, 1e-6)

    def fn(points, t):
        u = _axis_parameter(points, axis)
        head = 1.0 - (t / period) % 1.0
        d = np.clip(u - head, 0.0, None)
        f = np.exp(-d / tail)
        return _out(_mix(bg, c, f))

    return fn


def alternate(color_a, color_b, period: float = 1.0, axis: str = "index") -> EffectFn:
    """Split the swarm in two interleaved halves and swap their colours."""
    a, b = _rgb(color_a), _rgb(color_b)

    def fn(points, t):
        if axis == "index":
            group = (np.arange(len(points)) % 2).astype(np.float64)
        else:
            group = (_axis_parameter(points, axis) > 0.5).astype(np.float64)
        if int(t / period) % 2:
            group = 1.0 - group
        return _out(_mix(a, b, group))

    return fn


def volume(color, axis: str = "x", radius: float = 0.3, duration: float = 4.0,
           keep: bool = True, over_current: bool = True,
           background=(0, 0, 0)) -> EffectFn:
    """A ball of colour sweeping through the formation; keep leaves a trail.

    The ball crosses the formation along `axis` in `duration` seconds.
    radius is a share of the formation's size (0.05 to 1). With keep, every
    drone it has touched stays coloured, so it paints the shape in; without,
    only drones inside it light up. background may be one colour or one per
    drone. over_current asks the caller to pass the formation's present
    colours as background (the server does); on its own it changes nothing.
    """
    c = _rgb(color)
    bg = np.asarray(background, dtype=np.float64)
    if axis not in AXES:
        raise ValueError("volume sweeps along x, y or z")
    if not 0.05 <= radius <= 1.0:
        raise ValueError("radius must be between 0.05 and 1")
    if duration <= 0:
        raise ValueError("duration must be positive")
    k = AXES[axis]

    def fn(points, t):
        p = np.asarray(points, dtype=np.float64)
        lo, hi = p.min(axis=0), p.max(axis=0)
        half = max(float(np.max(hi - lo)) / 2.0, 1e-9)
        q = (p - (lo + hi) / 2.0) / half              # -1..1 on the longest side
        s = min(max(t / duration, 0.0), 1.0)
        s = s * s * (3.0 - 2.0 * s)                   # smoothstep: eases in and out
        centre = -1.0 - radius + s * 2.0 * (1.0 + radius)
        perp = np.linalg.norm(np.delete(q, k, axis=1), axis=1)
        reach = np.sqrt(np.maximum(radius * radius - perp * perp, 0.0))
        along = q[:, k]
        inside = (perp <= radius) & (np.abs(along - centre) <= reach)
        if keep:
            inside |= (perp <= radius) & (along <= centre + reach)
        base = np.broadcast_to(bg if bg.ndim == 2 else _rgb(bg)[None, :], p.shape)
        return _out(np.where(inside[:, None], c[None, :], base))

    return fn


EFFECTS: dict[str, Callable[..., EffectFn]] = {
    "solid": solid,
    "fade": fade,
    "strobe": strobe,
    "pulse": pulse,
    "gradient": gradient,
    "rainbow": rainbow,
    "wipe": wipe,
    "chase": chase,
    "ripple": ripple,
    "twinkle": twinkle,
    "meteor": meteor,
    "alternate": alternate,
    "volume": volume,
}


def make(name: str, **kwargs) -> EffectFn:
    try:
        factory = EFFECTS[name]
    except KeyError:
        raise ValueError(
            f"unknown effect {name!r}; available: {', '.join(sorted(EFFECTS))}"
        ) from None
    return factory(**kwargs)


# --------------------------------------------------------------------------
# Rendering an effect into a formation
# --------------------------------------------------------------------------


def recommended_fps(effect: EffectFn, default: float = 10.0) -> float:
    return float(getattr(effect, "recommended_fps", default))


def render(
    formation: Formation,
    effect: EffectFn,
    *,
    fps: float | None = None,
    duration: float | None = None,
) -> Formation:
    """Bake an effect into a formation's colour keyframes, in place.

    Samples over the formation's hold. Fast effects raise the sample rate
    themselves: rendering a 10 Hz strobe at 10 Hz would alias into a solid
    colour, or into nothing, depending on phase.
    """
    span = formation.hold if duration is None else duration
    rate = recommended_fps(effect) if fps is None else float(fps)
    if rate <= 0:
        raise ValueError("fps must be positive")

    formation.colors = np.asarray(effect(formation.points, 0.0), dtype=np.int32)
    if span <= 0:
        formation.color_keys = []
        return formation

    count = max(1, int(math.ceil(span * rate)))
    keys: list[tuple[float, np.ndarray]] = []
    for i in range(1, count + 1):
        t = min(span, i / rate)
        keys.append((t, np.asarray(effect(formation.points, t), dtype=np.int32)))
        if t >= span:
            break
    formation.color_keys = keys
    return formation


def keep_unmasked(old: Formation, new: Formation, mask: np.ndarray) -> Formation:
    """Restrict a freshly rendered effect to the masked drones, in place.

    Every other drone keeps the colours it had, keyframes included. Keys
    are taken at every time either version had one, so neither loses detail.
    """
    mask = np.asarray(mask, dtype=bool)[:, None]
    times = sorted({t for t, _ in old.color_keys} | {t for t, _ in new.color_keys})
    keys = [(t, np.where(mask, new.colors_at(t), old.colors_at(t)).astype(np.int32))
            for t in times]
    new.colors = np.where(mask, new.colors, old.colors).astype(np.int32)
    new.color_keys = keys
    return new


def render_named(formation: Formation, name: str, **kwargs) -> Formation:
    """Render by effect name, splitting out the render-rate argument."""
    fps = kwargs.pop("fps", None)
    return render(formation, make(name, **kwargs), fps=fps)


def beat_locked(music: Music, beats_per_cycle: float = 1.0, fallback: float = 2.0) -> float:
    """A period in seconds equal to `beats_per_cycle` beats of the track.

    Lets a chase or strobe be written in musical terms and come out in
    seconds, so retiming the show to a different tempo does not silently
    desynchronise every effect from the music.
    """
    step = music.beat_seconds
    if step is None:
        return fallback
    return step * max(beats_per_cycle, 1e-6)


def describe(effect_name: str) -> dict:
    """Parameter names and defaults for one effect, for the UI to render."""
    import inspect

    factory = EFFECTS[effect_name]
    sig = inspect.signature(factory)
    params = []
    for p in sig.parameters.values():
        default = None if p.default is inspect.Parameter.empty else p.default
        kind = "color" if "color" in p.name or p.name == "background" else (
            "axis" if p.name == "axis" else "number"
        )
        params.append({"name": p.name, "default": default, "kind": kind})
    return {
        "name": effect_name,
        "params": params,
        "doc": (factory.__doc__ or "").strip().splitlines()[0]
        if factory.__doc__
        else "",
    }
