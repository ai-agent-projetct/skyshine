"""Safety validation. The only gate that may authorise a show for flight.

Deliberately independent of planning.py. The planner builds shows that
should be safe by construction; this module assumes none of that and
measures the finished trajectories instead. If the two ever disagree,
believe this one and fix the planner.

Separation checking is continuous in time, not sampled. Sampling a 500
drone show at 10 Hz and checking only the samples would miss a pair that
dips below the minimum between two frames. Instead, each sample interval
is treated as straight-line motion and the exact minimum of the pair
distance over that interval is solved in closed form, with a measured
curvature margin added to cover the difference between the real eased path
and the chord. Candidate pairs come from a KD-tree query whose radius is
wide enough that no breaching pair can escape it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from . import geometry
from .show import Show, Timeline

ERROR = "error"
WARNING = "warning"

# How many individual breaches of one kind to record before summarising.
# A show with a broken transition can generate millions; the operator
# needs the worst few and a count, not a dump.
MAX_DETAIL = 25


@dataclass
class Violation:
    kind: str
    severity: str
    message: str
    time: float | None = None
    drones: tuple[int, ...] = ()
    value: float | None = None
    limit: float | None = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "time": None if self.time is None else round(self.time, 3),
            "drones": list(self.drones),
            "value": None if self.value is None else round(self.value, 4),
            "limit": None if self.limit is None else round(self.limit, 4),
        }


@dataclass
class SafetyReport:
    show_name: str
    drone_count: int
    duration: float
    violations: list[Violation] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == ERROR]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == WARNING]

    @property
    def passed(self) -> bool:
        """True only when nothing is an error. Warnings do not block."""
        return not self.errors

    def add(self, v: Violation) -> None:
        self.violations.append(v)

    def to_dict(self) -> dict:
        return {
            "show": self.show_name,
            "drone_count": self.drone_count,
            "duration": round(self.duration, 3),
            "passed": self.passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.metrics.items()
            },
            "violations": [v.to_dict() for v in self.violations],
        }

    def summary(self) -> str:
        lines = [
            f"Sky Shine safety report: {self.show_name}",
            f"  drones      {self.drone_count}",
            f"  duration    {self.duration:.1f} s",
            f"  verdict     {'PASS' if self.passed else 'FAIL'}"
            f"  ({len(self.errors)} errors, {len(self.warnings)} warnings)",
            "",
            "  measured:",
        ]
        for k, v in self.metrics.items():
            lines.append(f"    {k:<28} {v if not isinstance(v, float) else f'{v:.3f}'}")
        if self.violations:
            lines.append("")
            lines.append("  findings:")
            for v in self.violations:
                where = "" if v.time is None else f" t={v.time:7.2f}s"
                who = "" if not v.drones else f" drones={list(v.drones)}"
                lines.append(f"    [{v.severity.upper():<7}] {v.kind}{where}{who}")
                lines.append(f"              {v.message}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Continuous separation
# --------------------------------------------------------------------------


def curvature_margin(pos: np.ndarray) -> float:
    """Upper estimate of how far the real path strays from the chord.

    For three consecutive samples the sagitta of the curve through them is
    |p[k-1] - 2 p[k] + p[k+1]| / 8. Taking the maximum over every drone and
    interval bounds the modelling error of the straight-line assumption
    used by the sweep below. The caller doubles it, since two drones can
    bow towards each other.
    """
    if pos.shape[0] < 3:
        return 0.0
    second = pos[:-2] - 2.0 * pos[1:-1] + pos[2:]
    return float(np.max(np.linalg.norm(second, axis=2)) / 8.0)


def check_separation(
    timeline: Timeline,
    min_separation: float,
    *,
    ignore_below_altitude: float | None = None,
) -> tuple[list[Violation], float]:
    """Exact-in-time pairwise separation sweep.

    ignore_below_altitude suppresses findings where both drones are under
    that height, which is where the launch grid lives. Grid spacing is
    checked separately and statically; without this the report would be
    nothing but takeoff-pad noise. Set it to None to check everything.

    Returns the violations and the worst separation seen anywhere.
    """
    pos = np.ascontiguousarray(timeline.pos, dtype=np.float64)
    t = timeline.t
    n_steps = pos.shape[0]
    margin = 2.0 * curvature_margin(pos)
    threshold = min_separation + margin

    worst = math.inf
    found: list[tuple[float, int, int, float]] = []

    for k in range(n_steps - 1):
        p0 = pos[k]
        delta = pos[k + 1] - p0
        max_disp = float(np.max(np.linalg.norm(delta, axis=1))) if len(delta) else 0.0
        # Any pair that breaches during the interval is within this radius
        # at the start of it, so the query cannot miss one.
        radius = threshold + 2.0 * max_disp
        pairs = cKDTree(p0).query_pairs(radius, output_type="ndarray")
        if len(pairs) == 0:
            continue

        i, j = pairs[:, 0], pairs[:, 1]
        r0 = p0[i] - p0[j]
        dr = delta[i] - delta[j]
        dd = np.einsum("ij,ij->i", dr, dr)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(dd > 1e-18, -np.einsum("ij,ij->i", r0, dr) / dd, 0.0)
        s = np.clip(s, 0.0, 1.0)
        closest = r0 + dr * s[:, None]
        dist = np.linalg.norm(closest, axis=1)

        if ignore_below_altitude is not None:
            zi = p0[i, 2] + delta[i, 2] * s
            zj = p0[j, 2] + delta[j, 2] * s
            relevant = (zi > ignore_below_altitude) | (zj > ignore_below_altitude)
        else:
            relevant = np.ones(len(dist), dtype=bool)

        if np.any(relevant):
            # Track the tightest encounter anywhere, so a passing report can
            # still say how much room was left. Which two drones they were is
            # the designer's question, answered by worst_pairs().
            worst = min(worst, float(np.min(dist[relevant])))

        bad = relevant & (dist < threshold)
        if np.any(bad):
            dt = t[k + 1] - t[k]
            for idx in np.nonzero(bad)[0]:
                found.append(
                    (float(t[k] + dt * s[idx]), int(i[idx]), int(j[idx]), float(dist[idx]))
                )

    violations: list[Violation] = []
    found.sort(key=lambda r: r[3])
    for time_s, a, b, d in found[:MAX_DETAIL]:
        violations.append(
            Violation(
                kind="separation",
                severity=ERROR,
                message=(
                    f"drones {a} and {b} close to {d:.2f} m, "
                    f"minimum is {min_separation:.2f} m "
                    f"(threshold {threshold:.2f} m includes a {margin:.3f} m "
                    "path-curvature margin)"
                ),
                time=time_s,
                drones=(a, b),
                value=d,
                limit=min_separation,
            )
        )
    if len(found) > MAX_DETAIL:
        violations.append(
            Violation(
                kind="separation",
                severity=ERROR,
                message=(
                    f"{len(found) - MAX_DETAIL} further separation breaches not "
                    "listed; the transition design is wrong, not marginal"
                ),
            )
        )
    return violations, (0.0 if worst is math.inf else worst)


# --------------------------------------------------------------------------
# Envelope checks
# --------------------------------------------------------------------------


def check_altitude(
    timeline: Timeline, min_nav: float, max_alt: float
) -> tuple[list[Violation], float, float]:
    z = timeline.pos[:, :, 2]
    lo, hi = float(z.min()), float(z.max())
    out: list[Violation] = []
    if hi > max_alt:
        k, d = np.unravel_index(int(np.argmax(z)), z.shape)
        out.append(
            Violation(
                kind="altitude_max",
                severity=ERROR,
                message=f"drone {d} reaches {hi:.2f} m, ceiling is {max_alt:.2f} m",
                time=float(timeline.t[k]),
                drones=(int(d),),
                value=hi,
                limit=max_alt,
            )
        )
    if lo < -0.01:
        k, d = np.unravel_index(int(np.argmin(z)), z.shape)
        out.append(
            Violation(
                kind="altitude_min",
                severity=ERROR,
                message=f"drone {d} is commanded to {lo:.2f} m, below the ground",
                time=float(timeline.t[k]),
                drones=(int(d),),
                value=lo,
                limit=0.0,
            )
        )
    return out, lo, hi


def check_geofence(
    timeline: Timeline, polygon: Sequence[Sequence[float]]
) -> list[Violation]:
    if len(polygon) < 3:
        return []
    poly = np.asarray(polygon, dtype=np.float64)
    xy = timeline.pos[:, :, :2].reshape(-1, 2)
    inside = points_in_polygon(xy, poly).reshape(timeline.pos.shape[:2])
    if bool(np.all(inside)):
        return []
    bad = np.argwhere(~inside)
    k, d = int(bad[0][0]), int(bad[0][1])
    return [
        Violation(
            kind="geofence",
            severity=ERROR,
            message=(
                f"{len(bad)} sample(s) leave the geofence; first is drone {d} at "
                f"({timeline.pos[k, d, 0]:.1f}, {timeline.pos[k, d, 1]:.1f})"
            ),
            time=float(timeline.t[k]),
            drones=(d,),
            value=float(len(bad)),
        )
    ]


def check_keep_out(timeline: Timeline, lim) -> list[Violation]:
    """No drone may come near a keep-out zone at any time.

    The zone is grown by half the minimum separation, the drone's own size
    budget, plus half the largest move between two samples, so a drone
    cannot slip through the corner of a zone between samples.
    """
    if not lim.keep_out:
        return []
    pos = timeline.pos
    step = float(np.max(np.linalg.norm(np.diff(pos, axis=0), axis=2))) if len(pos) > 1 else 0.0
    margin = lim.min_separation / 2.0 + step / 2.0
    out = []
    for i, z in enumerate(lim.keep_out):
        name = z.get("name") or f"zone {i + 1}"
        d = np.hypot(pos[:, :, 0] - float(z["x"]), pos[:, :, 1] - float(z["y"]))
        inside = ((d < float(z["radius"]) + margin)
                  & (pos[:, :, 2] > float(z["bottom"]) - margin)
                  & (pos[:, :, 2] < float(z["top"]) + margin))
        if not inside.any():
            continue
        bad = np.argwhere(inside)
        k, first = int(bad[0][0]), int(bad[0][1])
        drones = sorted(set(bad[:, 1].tolist()))
        out.append(Violation(
            kind="keep_out", severity=ERROR,
            message=(f"{len(drones)} drone(s) enter keep-out zone {name!r}, first drone "
                     f"{first} at {float(timeline.t[k]):.1f}s; move the formations or the "
                     "flight path away from it"),
            time=float(timeline.t[k]), drones=tuple(drones[:12]), value=float(len(drones)),
        ))
    return out


def points_in_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray casting. Boundary counts as inside-ish.

    Points exactly on an edge are not treated specially; a geofence should
    be drawn with margin, not hugged.
    """
    p = np.asarray(points, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    inside = np.zeros(len(p), dtype=bool)
    n = len(poly)
    for a in range(n):
        b = (a + 1) % n
        x1, y1 = poly[a]
        x2, y2 = poly[b]
        straddles = (y1 > y) != (y2 > y)
        if not np.any(straddles):
            continue
        # Horizontal edges never straddle, so the division is safe wherever
        # `straddles` is true; nan elsewhere compares false and drops out.
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = np.nan if y2 == y1 else (y2 - y1)
            x_cross = x1 + (y - y1) * (x2 - x1) / denom
        inside ^= straddles & (x < x_cross)
    return inside


def check_kinematics(timeline: Timeline, limits) -> tuple[list[Violation], dict]:
    # Compute the derivative chain once; timeline.acceleration() would
    # re-differentiate positions and double the peak memory for a large show.
    vel = timeline.velocity()
    acc = np.gradient(vel, timeline.t, axis=0) if vel.shape[0] > 1 else np.zeros_like(vel)
    v_xy = np.linalg.norm(vel[:, :, :2], axis=2)
    v_up = np.maximum(vel[:, :, 2], 0.0)
    v_dn = np.maximum(-vel[:, :, 2], 0.0)
    a_mag = np.linalg.norm(acc, axis=2)

    out: list[Violation] = []
    for arr, limit, kind, label in (
        (v_xy, limits.max_velocity_xy, "velocity_xy", "horizontal speed"),
        (v_up, limits.max_velocity_up, "velocity_up", "climb rate"),
        (v_dn, limits.max_velocity_down, "velocity_down", "descent rate"),
        (a_mag, limits.max_acceleration, "acceleration", "acceleration"),
    ):
        peak = float(arr.max()) if arr.size else 0.0
        # 1% slack: finite differences on a sampled curve overshoot the
        # analytic peak slightly, and failing a show for that is noise.
        if peak > limit * 1.01:
            k, d = np.unravel_index(int(np.argmax(arr)), arr.shape)
            out.append(
                Violation(
                    kind=kind,
                    severity=ERROR,
                    message=(
                        f"drone {d} peaks at {peak:.2f} {label} units, "
                        f"limit is {limit:.2f}"
                    ),
                    time=float(timeline.t[k]),
                    drones=(int(d),),
                    value=peak,
                    limit=limit,
                )
            )

    metrics = {
        "peak_horizontal_speed_mps": float(v_xy.max()) if v_xy.size else 0.0,
        "peak_climb_rate_mps": float(v_up.max()) if v_up.size else 0.0,
        "peak_descent_rate_mps": float(v_dn.max()) if v_dn.size else 0.0,
        "peak_acceleration_mps2": float(a_mag.max()) if a_mag.size else 0.0,
        "total_path_length_m": float(
            np.sum(np.linalg.norm(np.diff(timeline.pos, axis=0), axis=2))
        ),
    }
    return out, metrics


def check_yaw(show: Show) -> list[Violation]:
    """Heading changes must be trackable, and consistently specified.

    Yaw only matters when something on the airframe points somewhere -
    pyro, a directional light, a camera. When it does, two things go wrong
    cheaply: asking for a turn faster than the aircraft can hold position
    through, and setting yaw on some formations but not others, so the
    heading between them is whatever the autopilot felt like.
    """
    out: list[Violation] = []
    lim = show.limits
    with_yaw = [f for f in show.formations if f.yaw is not None]
    if not with_yaw:
        return out

    if len(with_yaw) != len(show.formations):
        missing = [f.name for f in show.formations if f.yaw is None][:6]
        out.append(
            Violation(
                kind="yaw",
                severity=WARNING,
                message=(
                    f"{len(with_yaw)} of {len(show.formations)} formations set a "
                    f"heading; {missing} do not, so heading through those is "
                    "undefined. Set yaw everywhere or nowhere."
                ),
            )
        )

    cursor = 0.0
    for i, f in enumerate(show.formations):
        if i > 0 and f.transition > 0 and f.yaw is not None:
            prev = show.formations[i - 1].yaw
            if prev is not None:
                # Shortest way round: a 350 degree command is a 10 degree turn.
                delta = np.abs((f.yaw - prev + 180.0) % 360.0 - 180.0)
                rate = float(np.max(delta)) / f.transition
                if rate > lim.max_yaw_rate_deg_s * 1.01:
                    drone = int(np.argmax(delta))
                    out.append(
                        Violation(
                            kind="yaw_rate",
                            severity=ERROR,
                            message=(
                                f"entering {f.name!r}, drone {drone} must turn "
                                f"{delta[drone]:.0f} degrees in {f.transition:.1f}s "
                                f"({rate:.0f} deg/s), over the "
                                f"{lim.max_yaw_rate_deg_s:.0f} deg/s limit"
                            ),
                            time=cursor,
                            drones=(drone,),
                            value=rate,
                            limit=lim.max_yaw_rate_deg_s,
                        )
                    )
        cursor += f.duration
    return out


def check_pyro(show: Show) -> list[Violation]:
    """Pyro cannot be recalled once lit, so its constraints are hard.

    Checked here: altitude, extra separation between firing drones,
    a defined and correct heading, and enough settle time after the
    aircraft stopped moving. Not checked here, and not checkable by
    software: whether the article is legal where you are flying, whether
    the fallout zone is clear, and whether anybody is standing in it.
    """
    out: list[Violation] = []
    lim = show.limits
    events = show.pyro_events()
    if not events:
        return out

    for when, index, event in events:
        f = show.formations[index]
        firing = f.firing_drones(event)
        pts = f.points[firing]

        low = float(pts[:, 2].min())
        if low < lim.pyro_min_altitude:
            out.append(
                Violation(
                    kind="pyro_altitude",
                    severity=ERROR,
                    message=(
                        f"pyro in {f.name!r} fires at {low:.1f} m, under the "
                        f"{lim.pyro_min_altitude:.0f} m minimum: the spray needs "
                        "height to burn out before it reaches the ground"
                    ),
                    time=when,
                    value=low,
                    limit=lim.pyro_min_altitude,
                )
            )

        if len(firing) > 1:
            gap = geometry.min_pair_distance(pts)
            if gap < lim.pyro_separation:
                out.append(
                    Violation(
                        kind="pyro_separation",
                        severity=ERROR,
                        message=(
                            f"two firing drones in {f.name!r} are {gap:.1f} m "
                            f"apart, under the {lim.pyro_separation:.0f} m pyro "
                            "minimum: a gerb throws sparks well past the airframe"
                        ),
                        time=when,
                        value=gap,
                        limit=lim.pyro_separation,
                    )
                )

        if event.heading_deg is not None:
            if f.yaw is None:
                out.append(
                    Violation(
                        kind="pyro_heading",
                        severity=ERROR,
                        message=(
                            f"pyro in {f.name!r} requires a heading of "
                            f"{event.heading_deg:.0f} degrees, but the formation "
                            "sets no yaw, so the aircraft will fire whichever way "
                            "it happens to be facing"
                        ),
                        time=when,
                    )
                )
            else:
                err = np.abs(
                    (f.yaw[firing] - event.heading_deg + 180.0) % 360.0 - 180.0
                )
                worst = float(np.max(err))
                if worst > lim.pyro_heading_tolerance_deg:
                    drone = firing[int(np.argmax(err))]
                    out.append(
                        Violation(
                            kind="pyro_heading",
                            severity=ERROR,
                            message=(
                                f"drone {drone} fires in {f.name!r} while facing "
                                f"{f.yaw[drone]:.0f} degrees, {worst:.0f} off the "
                                f"required {event.heading_deg:.0f}"
                            ),
                            time=when,
                            drones=(drone,),
                            value=worst,
                            limit=lim.pyro_heading_tolerance_deg,
                        )
                    )

        if event.at < event.settle_s:
            out.append(
                Violation(
                    kind="pyro_settle",
                    severity=ERROR,
                    message=(
                        f"pyro in {f.name!r} fires {event.at:.1f}s into the hold "
                        f"but needs {event.settle_s:.1f}s of settled heading "
                        "first; the airframe is still rotating into place"
                    ),
                    time=when,
                    value=event.at,
                    limit=event.settle_s,
                )
            )
    return out


def check_endurance(show: Show, duration: float) -> list[Violation]:
    lim = show.limits
    usable = lim.battery_endurance_s - lim.battery_reserve_s
    if duration > usable:
        return [
            Violation(
                kind="endurance",
                severity=ERROR,
                message=(
                    f"show runs {duration:.0f} s but only {usable:.0f} s is usable "
                    f"({lim.battery_endurance_s:.0f} s endurance minus a "
                    f"{lim.battery_reserve_s:.0f} s reserve)"
                ),
                value=duration,
                limit=usable,
            )
        ]
    headroom = usable - duration
    if headroom < 0.2 * usable:
        return [
            Violation(
                kind="endurance",
                severity=WARNING,
                message=(
                    f"only {headroom:.0f} s of usable battery left after the show; "
                    "a delayed start or a headwind will eat that"
                ),
                value=duration,
                limit=usable,
            )
        ]
    return []


# --------------------------------------------------------------------------
# Design-time checks on the show itself, before any sampling
# --------------------------------------------------------------------------


def check_design(show: Show) -> list[Violation]:
    out: list[Violation] = []
    lim = show.limits

    if not show.formations:
        out.append(
            Violation(
                kind="empty", severity=ERROR, message="show contains no formations"
            )
        )
        return out

    for i, f in enumerate(show.formations):
        if f.count != show.drone_count:
            out.append(
                Violation(
                    kind="drone_count",
                    severity=ERROR,
                    message=(
                        f"formation {f.name!r} has {f.count} positions, show has "
                        f"{show.drone_count} drones"
                    ),
                )
            )
        d = geometry.min_pair_distance(f.points)
        on_ground = float(f.points[:, 2].max()) <= 0.01
        floor = lim.min_separation if on_ground else lim.formation_separation
        if d < floor - 1e-6:
            out.append(
                Violation(
                    kind="formation_separation",
                    severity=ERROR if d < lim.min_separation else WARNING,
                    message=(
                        f"formation {f.name!r} packs drones {d:.2f} m apart; "
                        f"{floor:.2f} m is needed "
                        + (
                            "(sqrt(2) x min_separation, so that transitions in and "
                            "out of it stay legal)"
                            if not on_ground
                            else "(launch grid minimum)"
                        )
                    ),
                    value=d,
                    limit=floor,
                )
            )
        if f.easing == "linear" and f.transition > 0 and i > 0:
            out.append(
                Violation(
                    kind="easing",
                    severity=WARNING,
                    message=(
                        f"formation {f.name!r} uses linear easing, which demands "
                        "an instantaneous velocity step at both ends; prefer "
                        "minjerk unless the airframe is known to cope"
                    ),
                )
            )
        if f.transition <= 0 and i > 0:
            moved = float(
                np.max(np.linalg.norm(f.points - show.formations[i - 1].end_points, axis=1))
            )
            if moved > 1e-6:
                out.append(
                    Violation(
                        kind="timing",
                        severity=ERROR,
                        message=(
                            f"formation {f.name!r} asks for a {moved:.1f} m move in "
                            "zero time"
                        ),
                        value=moved,
                    )
                )

    easings = {f.easing for f in show.formations if f.transition > 0}
    if len(easings) > 1:
        out.append(
            Violation(
                kind="easing",
                severity=WARNING,
                message=(
                    "formations use mixed easings "
                    f"({', '.join(sorted(easings))}); that is fine between "
                    "transitions but every drone inside one transition must "
                    "share an easing for the separation bound to hold"
                ),
            )
        )

    if show.origin.amsl is None:
        out.append(
            Violation(
                kind="origin",
                severity=WARNING,
                message=(
                    "no AMSL reference set, so the show will fly on AGL only; "
                    "acceptable without RTK, not for a dense show"
                ),
            )
        )
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def validate(
    show: Show,
    timeline: Timeline | None = None,
    *,
    check_takeoff_proximity: bool = False,
) -> SafetyReport:
    """Run every check. A show is flyable only if the result passes.

    check_takeoff_proximity=True also flags drones that sit close together
    on the launch pads, which is normal and expected; leave it off for an
    operational report.
    """
    from .trajectory import build_timeline  # local import: avoids a cycle

    if timeline is None:
        timeline = build_timeline(show)

    report = SafetyReport(
        show_name=show.name,
        drone_count=show.drone_count,
        duration=timeline.duration,
    )

    for v in check_design(show):
        report.add(v)

    lim = show.limits
    ignore_below = None if check_takeoff_proximity else lim.min_nav_altitude
    sep_violations, worst_sep = check_separation(
        timeline, lim.min_separation, ignore_below_altitude=ignore_below
    )
    for v in sep_violations:
        report.add(v)

    alt_violations, lo, hi = check_altitude(timeline, lim.min_nav_altitude, lim.max_altitude)
    for v in alt_violations:
        report.add(v)

    for v in check_geofence(timeline, lim.geofence):
        report.add(v)

    for v in check_keep_out(timeline, lim):
        report.add(v)

    kin_violations, metrics = check_kinematics(timeline, lim)
    for v in kin_violations:
        report.add(v)

    for v in check_yaw(show):
        report.add(v)

    for v in check_pyro(show):
        report.add(v)

    for v in check_endurance(show, timeline.duration):
        report.add(v)

    grid_sep = (
        geometry.min_pair_distance(show.formations[0].points)
        if show.formations
        else math.inf
    )

    report.metrics = {
        "min_separation_m": worst_sep,
        "min_separation_limit_m": lim.min_separation,
        "separation_headroom_m": worst_sep - lim.min_separation,
        "launch_grid_spacing_m": grid_sep,
        "min_altitude_m": lo,
        "max_altitude_m": hi,
        "altitude_ceiling_m": lim.max_altitude,
        "samples_checked": int(timeline.pos.shape[0]),
        "pair_seconds_checked": int(
            timeline.pos.shape[0] * show.drone_count * (show.drone_count - 1) / 2
        ),
        "curvature_margin_m": 2.0 * curvature_margin(timeline.pos),
        "pyro_events": len(show.pyro_events()),
        **metrics,
    }
    return report


def worst_pairs(timeline: Timeline, count: int = 10) -> list[tuple[float, int, int, float]]:
    """The `count` closest encounters in the show, for the designer's UI.

    Sampled at frame resolution only - this is a design aid, not the
    verdict. validate() is the verdict.
    """
    out: list[tuple[float, int, int, float]] = []
    for k in range(timeline.pos.shape[0]):
        tree = cKDTree(timeline.pos[k])
        d, idx = tree.query(timeline.pos[k], k=2)
        lo = int(np.argmin(d[:, 1]))
        out.append((float(timeline.t[k]), lo, int(idx[lo, 1]), float(d[lo, 1])))
    out.sort(key=lambda r: r[3])
    return out[:count]
