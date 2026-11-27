"""Show data model and on-disk format.

Coordinate frame: local ENU in metres, origin = the show origin on the
ground. x = east, y = north, z = up (AGL). The origin is tied to a
lat/lon/AMSL triple only at upload time, so a design is site-independent.

Times are show-relative seconds. t=0 is the moment the show clock starts,
which is a scheduled absolute UTC instant at flight time.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

SHOW_FORMAT_VERSION = 1

# Easing functions. Every drone in a transition MUST share one easing: the
# collision bound in safety.py only holds when the time-warp is common.
# See docs/SAFETY.md.
EASINGS = ("linear", "smoothstep", "minjerk")

# Peak |s'| and |s''| of each easing on a unit displacement over unit time.
# Used to size transition durations against velocity/acceleration limits.
EASING_PEAKS = {
    "linear": (1.0, math.inf),          # instant accel at the ends
    "smoothstep": (1.5, 6.0),           # 3t^2-2t^3
    "minjerk": (1.875, 10.0 / math.sqrt(3.0)),  # 10t^3-15t^4+6t^5
}


def ease(name: str, tau: np.ndarray | float) -> np.ndarray | float:
    """Map normalised time tau in [0,1] to normalised progress s in [0,1]."""
    t = np.clip(tau, 0.0, 1.0)
    if name == "linear":
        return t
    if name == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    if name == "minjerk":
        return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
    raise ValueError(f"unknown easing {name!r}; expected one of {EASINGS}")


@dataclass
class ShowLimits:
    """Flight envelope and safety thresholds.

    min_separation is the hard floor: the distance below which two drones
    are considered to have collided, measured centre to centre. Set it
    from airframe size, not from taste - a 450 mm arm length drone with
    props needs well over 1 m of that budget. 3 m is the common choice
    for sub-metre-class show drones.
    """

    min_separation: float = 3.0
    # Formations are scaled so neighbours sit at least this far apart.
    # sqrt(2) * min_separation is the value that makes transitions safe by
    # construction (see safety.transition_lower_bound).
    formation_separation: float = 0.0  # 0 -> derive as sqrt(2)*min_separation
    takeoff_grid_spacing: float = 3.0
    takeoff_layer_gap: float = 3.0
    takeoff_layers: int = 4
    min_nav_altitude: float = 2.5
    max_altitude: float = 120.0
    max_velocity_xy: float = 8.0
    max_velocity_up: float = 4.0
    max_velocity_down: float = 3.0
    max_acceleration: float = 4.0
    # Share of the speed and acceleration limits a moving formation may use
    # during its hold. A transition touches top speed for a moment; a spin
    # holds it for the whole hold, and a position controller trails its
    # plan by about speed x lag the whole time. At the full limit virtual
    # aircraft trailed by 3.5 m and crossed the 3 m abort line. Tune it from
    # measured tracking on the real airframe.
    motion_speed_share: float = 0.5
    # Degrees per second of heading change a drone can track while holding
    # position accurately. Only matters when a formation sets yaw.
    max_yaw_rate_deg_s: float = 90.0
    # Pyro constraints. A gerb throws sparks well beyond the airframe, so
    # firing drones need more room from each other than flying ones, and
    # enough height that the spray burns out above the crowd.
    pyro_min_altitude: float = 40.0
    pyro_separation: float = 10.0
    pyro_heading_tolerance_deg: float = 10.0
    # Horizontal polygon the whole show must stay inside, list of (x, y).
    # Empty -> no lateral geofence (altitude limits still apply).
    geofence: list[list[float]] = field(default_factory=list)
    # No-fly cylinders around things in the venue: buildings, trees, cranes,
    # a stage. Each is {"name", "x", "y", "radius", "bottom", "top"} in show
    # metres (x east, y north, heights above ground).
    keep_out: list[dict] = field(default_factory=list)
    # Usable flight time per battery, seconds. Show duration plus reserve
    # must fit inside it.
    battery_endurance_s: float = 900.0
    battery_reserve_s: float = 180.0

    def __post_init__(self) -> None:
        if self.formation_separation <= 0.0:
            self.formation_separation = self.min_separation * math.sqrt(2.0)

    def validate(self) -> None:
        if self.min_separation <= 0:
            raise ValueError("min_separation must be positive")
        if self.formation_separation < self.min_separation:
            raise ValueError(
                "formation_separation must be >= min_separation; "
                "sqrt(2)*min_separation is the safe choice"
            )
        if self.max_altitude <= self.min_nav_altitude:
            raise ValueError("max_altitude must exceed min_nav_altitude")
        if self.takeoff_layers < 1:
            raise ValueError("takeoff_layers must be >= 1")
        if not 0.1 <= self.motion_speed_share <= 1.0:
            raise ValueError("motion_speed_share must be between 0.1 and 1")
        for i, z in enumerate(self.keep_out):
            try:
                x, y, r, lo, hi = (float(z[k]) for k in ("x", "y", "radius", "bottom", "top"))
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"keep-out zone {i + 1} needs x, y, radius, bottom and top") from None
            if r <= 0 or hi <= lo:
                raise ValueError(f"keep-out zone {z.get('name') or i + 1}: radius must be "
                                 "positive and top above bottom")
        if len(self.geofence) not in (0,) and len(self.geofence) < 3:
            raise ValueError("geofence needs at least 3 vertices, or none")


@dataclass
class Motion:
    """How a formation moves while it is held.

    All three happen together, eased so the formation starts and ends at
    rest: spin turns it about the vertical axis through its centre (degrees,
    anticlockwise seen from above), move carries it east, north and up
    (metres), and grow scales it about its centre (end size / start size).
    """

    spin: float = 0.0
    move: tuple = (0.0, 0.0, 0.0)
    grow: float = 1.0

    def __post_init__(self) -> None:
        self.spin = float(self.spin)
        self.move = tuple(float(v) for v in self.move)
        self.grow = float(self.grow)
        if len(self.move) != 3:
            raise ValueError("move needs three numbers: east, north and up metres")
        if not 0.2 <= self.grow <= 5.0:
            raise ValueError("grow must be between 0.2 and 5 times the start size")
        if abs(self.spin) > 3600:
            raise ValueError("spin is limited to 10 full turns per hold")

    @property
    def still(self) -> bool:
        return self.spin == 0 and not any(self.move) and self.grow == 1.0

    def to_dict(self) -> dict:
        return {"spin": self.spin, "move": list(self.move), "grow": self.grow}


def moved(points: np.ndarray, motion: Motion, s: float) -> np.ndarray:
    """Points carried `s` of the way (0..1, already eased) through a motion."""
    p = np.asarray(points, dtype=np.float64)
    c = p.mean(axis=0)
    th = math.radians(motion.spin * s)
    cs, sn = math.cos(th), math.sin(th)
    d = (p - c) * (1.0 + (motion.grow - 1.0) * s)
    out = np.column_stack([d[:, 0] * cs - d[:, 1] * sn, d[:, 0] * sn + d[:, 1] * cs, d[:, 2]])
    return out + c + np.asarray(motion.move, dtype=np.float64) * s


@dataclass
class PyroEvent:
    """A pyrotechnic trigger carried by drones.

    `at` is seconds from the start of this formation's hold. `drones` lists
    which aircraft fire; empty means all of them.

    Pyro is the one payload that cannot be taken back once it leaves, so it
    carries its own constraints: a minimum altitude, and a required heading
    with a settle time, because a gerb fired while the airframe is still
    yawing sprays in the wrong direction.
    """

    at: float
    drones: list[int] = field(default_factory=list)
    channel: int = 1
    description: str = ""
    # Heading the drone must hold when firing, degrees clockwise from show
    # north. None means heading does not matter for this effect.
    heading_deg: float | None = None
    # How long the heading must already have been held before ignition.
    settle_s: float = 1.0

    def to_dict(self) -> dict:
        return {
            "at": round(float(self.at), 3),
            "drones": list(self.drones),
            "channel": int(self.channel),
            "description": self.description,
            "heading_deg": self.heading_deg,
            "settle_s": round(float(self.settle_s), 3),
        }

    @staticmethod
    def from_dict(data: dict) -> "PyroEvent":
        return PyroEvent(
            at=float(data["at"]),
            drones=[int(d) for d in data.get("drones", [])],
            channel=int(data.get("channel", 1)),
            description=data.get("description", ""),
            heading_deg=data.get("heading_deg"),
            settle_s=float(data.get("settle_s", 1.0)),
        )


@dataclass
class Formation:
    """One held pose of the whole swarm, plus the transition that reaches it.

    points is (n, 3) in show ENU metres. colors is (n, 3) uint8 sRGB; a
    single (3,) row is broadcast to every drone.

    transition is the time in seconds spent flying from the previous
    formation to this one; hold is the time spent sitting in it.

    yaw is optional per-drone heading in degrees clockwise from show north.
    None leaves heading to the autopilot, which is right for a light-only
    show: a drone with an omnidirectional LED has no reason to burn energy
    pointing anywhere in particular. Set it for pyro, directional payloads
    or camera work.
    """

    name: str
    points: np.ndarray
    colors: np.ndarray
    transition: float = 8.0
    hold: float = 4.0
    easing: str = "minjerk"
    # Optional per-drone colour animation during the hold: a list of
    # (offset_seconds, colors) keyframes, offsets relative to hold start.
    color_keys: list[tuple[float, np.ndarray]] = field(default_factory=list)
    yaw: np.ndarray | None = None
    pyro: list[PyroEvent] = field(default_factory=list)
    # Movement during the hold; None for a formation that holds still.
    motion: Motion | None = None
    # Keep drone i on slot i: matching leaves this formation's order alone,
    # so a hand swap survives preparing. The safety engine still checks it.
    keep_order: bool = False

    def __post_init__(self) -> None:
        self.points = np.ascontiguousarray(self.points, dtype=np.float64)
        if isinstance(self.motion, dict):
            self.motion = Motion(**self.motion)
        if self.motion is not None and self.motion.still:
            self.motion = None
        if self.motion is not None:
            if self.hold <= 0:
                raise ValueError(f"{self.name}: a moving formation needs a hold to move in")
            if self.pyro:
                raise ValueError(
                    f"{self.name}: pyro needs a formation that holds still; "
                    "remove the motion or the pyro"
                )
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"{self.name}: points must be (n, 3)")
        self.colors = _broadcast_colors(self.colors, len(self.points), self.name)
        if self.easing not in EASINGS:
            raise ValueError(f"{self.name}: easing must be one of {EASINGS}")
        if self.transition < 0 or self.hold < 0:
            raise ValueError(f"{self.name}: transition and hold must be >= 0")
        if self.yaw is not None:
            self.yaw = np.asarray(self.yaw, dtype=np.float64).ravel()
            if self.yaw.size == 1:
                self.yaw = np.full(len(self.points), float(self.yaw[0]))
            if self.yaw.shape != (len(self.points),):
                raise ValueError(
                    f"{self.name}: yaw must be a scalar or one angle per drone"
                )
            self.yaw = np.mod(self.yaw, 360.0)
        for event in self.pyro:
            if event.at < 0 or event.at > self.hold + 1e-9:
                raise ValueError(
                    f"{self.name}: pyro event at {event.at}s falls outside the "
                    f"{self.hold}s hold; pyro must fire while the formation is "
                    "stationary, not mid-transition"
                )
            for d in event.drones:
                if not 0 <= d < len(self.points):
                    raise ValueError(
                        f"{self.name}: pyro event targets drone {d}, which does "
                        f"not exist in a {len(self.points)} drone formation"
                    )

    @property
    def count(self) -> int:
        return len(self.points)

    @property
    def duration(self) -> float:
        return self.transition + self.hold

    def pose(self, t_in_hold: float) -> np.ndarray:
        """Where every drone is t_in_hold seconds into the hold."""
        if self.motion is None:
            return self.points
        tau = min(max(t_in_hold / self.hold, 0.0), 1.0)
        return moved(self.points, self.motion, float(ease("minjerk", tau)))

    @property
    def end_points(self) -> np.ndarray:
        """Where the hold ends, which is where the next transition starts."""
        return self.pose(self.hold)

    def colors_at(self, t_in_hold: float) -> np.ndarray:
        """Colour of every drone t_in_hold seconds into this formation's hold."""
        if not self.color_keys:
            return self.colors
        keys = sorted(self.color_keys, key=lambda k: k[0])
        prev_t, prev_c = 0.0, self.colors
        for kt, kc in keys:
            if t_in_hold < kt:
                span = kt - prev_t
                a = 0.0 if span <= 0 else (t_in_hold - prev_t) / span
                return _lerp_colors(prev_c, kc, a)
            prev_t, prev_c = kt, kc
        return prev_c

    def copy(self) -> "Formation":
        return Formation(
            name=self.name,
            points=self.points.copy(),
            colors=self.colors.copy(),
            transition=self.transition,
            hold=self.hold,
            easing=self.easing,
            color_keys=[(t, c.copy()) for t, c in self.color_keys],
            yaw=None if self.yaw is None else self.yaw.copy(),
            pyro=[PyroEvent(**asdict(e)) for e in self.pyro],
            motion=None if self.motion is None else Motion(**self.motion.to_dict()),
            keep_order=self.keep_order,
        )

    def firing_drones(self, event: PyroEvent) -> list[int]:
        return list(event.drones) if event.drones else list(range(self.count))


def _broadcast_colors(colors: Any, n: int, where: str) -> np.ndarray:
    arr = np.asarray(colors, dtype=np.int32)
    if arr.ndim == 1 and arr.size == 3:
        arr = np.tile(arr, (n, 1))
    if arr.shape != (n, 3):
        raise ValueError(f"{where}: colors must be (3,) or ({n}, 3), got {arr.shape}")
    if arr.min() < 0 or arr.max() > 255:
        raise ValueError(f"{where}: colors must be 0-255")
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _lerp_colors(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    return np.clip(af + (bf - af) * alpha, 0, 255).astype(np.uint8)


@dataclass
class ShowOrigin:
    """Where the local ENU frame sits on the planet, set at flight time."""

    latitude: float = 0.0
    longitude: float = 0.0
    # AMSL altitude of the takeoff surface, metres. None -> AGL-only
    # operation, which is only acceptable without RTK.
    amsl: float | None = None
    # Rotation of show +y (north) clockwise to true north, degrees. A show
    # designed facing the audience is aimed by setting this.
    orientation_deg: float = 0.0


@dataclass
class Music:
    """The soundtrack, and the grid the choreography is cut against.

    The audio itself is never the show clock. Playback is started from the
    same absolute instant the fleet is, and if the two drift it is the
    audio that is wrong. A drone show that chases a media player is a drone
    show whose timing depends on a media player.

    `offset` is where the show's t=0 sits inside the audio file, so a track
    with a silent intro does not force silence into the flight plan.
    """

    path: str | None = None
    bpm: float | None = None
    # Seconds into the audio file at which show time zero occurs.
    offset: float = 0.0
    # Where the downbeat sits relative to show time zero, seconds.
    beat_phase: float = 0.0
    # Named moments the designer wants to hit, as (show_seconds, label).
    cues: list[tuple[float, str]] = field(default_factory=list)

    @property
    def beat_seconds(self) -> float | None:
        if not self.bpm or self.bpm <= 0:
            return None
        return 60.0 / self.bpm

    def beat_times(self, duration: float) -> list[float]:
        """Every beat instant from show time zero to `duration`."""
        step = self.beat_seconds
        if step is None:
            return []
        out = []
        t = self.beat_phase % step
        while t <= duration + 1e-9:
            out.append(round(t, 6))
            t += step
        return out

    def snap(self, time_s: float, *, division: int = 1) -> float:
        """Round a time onto the nearest beat, or sub-beat.

        division=1 snaps to beats, 2 to half beats, 4 to sixteenths of a
        4/4 bar. Returns the input unchanged when no tempo is set, so
        snapping is always safe to call.
        """
        step = self.beat_seconds
        if step is None or division < 1:
            return time_s
        step /= division
        return round(
            self.beat_phase + round((time_s - self.beat_phase) / step) * step, 6
        )

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "bpm": self.bpm,
            "offset": round(self.offset, 4),
            "beat_phase": round(self.beat_phase, 4),
            "cues": [[round(float(t), 3), str(label)] for t, label in self.cues],
        }

    @staticmethod
    def from_dict(data: dict) -> "Music":
        return Music(
            path=data.get("path"),
            bpm=data.get("bpm"),
            offset=float(data.get("offset", 0.0)),
            beat_phase=float(data.get("beat_phase", 0.0)),
            cues=[(float(t), str(label)) for t, label in data.get("cues", [])],
        )


@dataclass
class Show:
    name: str
    drone_count: int
    formations: list[Formation] = field(default_factory=list)
    limits: ShowLimits = field(default_factory=ShowLimits)
    origin: ShowOrigin = field(default_factory=ShowOrigin)
    # Sampling rate of the generated timeline, Hz. 10 Hz is plenty for a
    # show that is then re-checked continuously between samples.
    fps: float = 10.0
    audio: str | None = None
    music: Music = field(default_factory=Music)
    notes: str = ""
    # The site, for the designer's eyes only: {"picture": {...}, "model": {...}}.
    # Nothing here is flown; a keep-out zone is what protects the site.
    venue: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.limits.validate()
        if self.drone_count < 1:
            raise ValueError("drone_count must be >= 1")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        for f in self.formations:
            if f.count != self.drone_count:
                raise ValueError(
                    f"formation {f.name!r} has {f.count} points but the show "
                    f"has {self.drone_count} drones"
                )

    @property
    def duration(self) -> float:
        return float(sum(f.duration for f in self.formations))

    def add(self, formation: Formation) -> "Show":
        if formation.count != self.drone_count:
            raise ValueError(
                f"formation {formation.name!r} has {formation.count} points, "
                f"show has {self.drone_count} drones"
            )
        self.formations.append(formation)
        return self

    def pyro_events(self) -> list[tuple[float, int, "PyroEvent"]]:
        """Every pyro event as (absolute_show_time, formation_index, event)."""
        out = []
        for index, _t_start, t_hold, _t_end in iter_formation_windows(self):
            for event in self.formations[index].pyro:
                out.append((t_hold + event.at, index, event))
        return sorted(out, key=lambda row: row[0])

    def copy(self) -> "Show":
        return Show(
            name=self.name,
            drone_count=self.drone_count,
            formations=[f.copy() for f in self.formations],
            limits=ShowLimits(**asdict(self.limits)),
            origin=ShowOrigin(**asdict(self.origin)),
            fps=self.fps,
            audio=self.audio,
            music=Music.from_dict(self.music.to_dict()),
            notes=self.notes,
            venue=json.loads(json.dumps(self.venue)),
        )


@dataclass
class Timeline:
    """A fully sampled show: what every drone does at every sample time.

    t     (T,)        show seconds, strictly increasing
    pos   (T, n, 3)   ENU metres
    col   (T, n, 3)   uint8 sRGB
    """

    t: np.ndarray
    pos: np.ndarray
    col: np.ndarray
    # Index of the formation active at each sample, for UI scrubbing.
    segment: np.ndarray
    show: Show

    def __post_init__(self) -> None:
        T, n = self.pos.shape[0], self.pos.shape[1]
        if self.t.shape != (T,):
            raise ValueError("t and pos disagree on sample count")
        if self.col.shape != (T, n, 3):
            raise ValueError("col and pos disagree on shape")
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("t must be strictly increasing")

    @property
    def drone_count(self) -> int:
        return self.pos.shape[1]

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def sample(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Interpolated positions and colours at an arbitrary show time."""
        time_s = float(np.clip(time_s, self.t[0], self.t[-1]))
        i = int(np.searchsorted(self.t, time_s, side="right")) - 1
        i = max(0, min(i, len(self.t) - 2))
        span = self.t[i + 1] - self.t[i]
        a = 0.0 if span <= 0 else (time_s - self.t[i]) / span
        pos = self.pos[i] + (self.pos[i + 1] - self.pos[i]) * a
        col = _lerp_colors(self.col[i], self.col[i + 1], a)
        return pos, col

    def velocity(self) -> np.ndarray:
        """(T, n, 3) finite-difference velocity, m/s."""
        return _gradient_time(self.pos, self.t)

    def acceleration(self) -> np.ndarray:
        """(T, n, 3) finite-difference acceleration, m/s^2."""
        return _gradient_time(self.velocity(), self.t)

    def trajectory_of(self, drone: int) -> np.ndarray:
        return self.pos[:, drone, :]


def _gradient_time(arr: np.ndarray, t: np.ndarray) -> np.ndarray:
    if arr.shape[0] < 2:
        return np.zeros_like(arr)
    return np.gradient(arr, t, axis=0)


# --------------------------------------------------------------------------
# Serialisation. One JSON object per show; arrays stored as nested lists so
# a show file stays diffable and human-auditable, which matters when the
# file is the artefact a safety review signs off.
# --------------------------------------------------------------------------


def show_to_dict(show: Show) -> dict:
    return {
        "format": "skyshine.show",
        "version": SHOW_FORMAT_VERSION,
        "name": show.name,
        "drone_count": show.drone_count,
        "fps": show.fps,
        "audio": show.audio,
        "music": show.music.to_dict(),
        "notes": show.notes,
        **({"venue": show.venue} if show.venue else {}),
        "limits": asdict(show.limits),
        "origin": asdict(show.origin),
        "formations": [
            {
                "name": f.name,
                "transition": f.transition,
                "hold": f.hold,
                "easing": f.easing,
                "points": np.round(f.points, 4).tolist(),
                "colors": f.colors.tolist(),
                "color_keys": [
                    [float(t), c.tolist()] for t, c in f.color_keys
                ],
                "yaw": None if f.yaw is None else np.round(f.yaw, 2).tolist(),
                "pyro": [e.to_dict() for e in f.pyro],
                **({"motion": f.motion.to_dict()} if f.motion else {}),
                **({"keep_order": True} if f.keep_order else {}),
            }
            for f in show.formations
        ],
    }


def show_from_dict(data: dict) -> Show:
    if data.get("format") != "skyshine.show":
        raise ValueError("not a Sky Shine show file")
    version = int(data.get("version", 0))
    if version > SHOW_FORMAT_VERSION:
        raise ValueError(
            f"show file version {version} is newer than this build "
            f"supports ({SHOW_FORMAT_VERSION})"
        )
    limits = ShowLimits(**data.get("limits", {}))
    origin = ShowOrigin(**data.get("origin", {}))
    formations = [
        Formation(
            name=f["name"],
            points=np.asarray(f["points"], dtype=np.float64),
            colors=np.asarray(f["colors"], dtype=np.int32),
            transition=float(f.get("transition", 8.0)),
            hold=float(f.get("hold", 4.0)),
            easing=f.get("easing", "minjerk"),
            color_keys=[
                (float(t), np.asarray(c, dtype=np.int32))
                for t, c in f.get("color_keys", [])
            ],
            yaw=None if f.get("yaw") is None else np.asarray(f["yaw"], dtype=np.float64),
            pyro=[PyroEvent.from_dict(e) for e in f.get("pyro", [])],
            motion=f.get("motion"),
            keep_order=bool(f.get("keep_order", False)),
        )
        for f in data.get("formations", [])
    ]
    return Show(
        name=data.get("name", "untitled"),
        drone_count=int(data["drone_count"]),
        formations=formations,
        limits=limits,
        origin=origin,
        fps=float(data.get("fps", 10.0)),
        audio=data.get("audio"),
        music=Music.from_dict(data.get("music", {})),
        notes=data.get("notes", ""),
        venue=data.get("venue") or {},
    )


def save_show(show: Show, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(show_to_dict(show), indent=1), encoding="utf-8")
    return p


def load_show(path: str | Path) -> Show:
    return show_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def enu_to_wgs84(
    xy: Sequence[float] | np.ndarray, origin: ShowOrigin
) -> tuple[float, float]:
    """Show-frame (x east, y north) to lat/lon, after applying orientation.

    Equirectangular approximation about the origin. Good to well under a
    centimetre over a show box of a few hundred metres, which is below
    RTK noise.
    """
    x, y = float(xy[0]), float(xy[1])
    th = math.radians(origin.orientation_deg)
    # Rotate show frame clockwise by orientation to land on true north.
    east = x * math.cos(th) + y * math.sin(th)
    north = -x * math.sin(th) + y * math.cos(th)
    r = 6378137.0
    dlat = north / r
    dlon = east / (r * math.cos(math.radians(origin.latitude)))
    return (
        origin.latitude + math.degrees(dlat),
        origin.longitude + math.degrees(dlon),
    )


def iter_formation_windows(show: Show) -> Iterable[tuple[int, float, float, float]]:
    """Yield (index, t_transition_start, t_hold_start, t_end) per formation."""
    t = 0.0
    for i, f in enumerate(show.formations):
        yield i, t, t + f.transition, t + f.duration
        t += f.duration
