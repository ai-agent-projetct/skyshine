"""Turning artwork into a flyable plan: matching, takeoff, landing, timing.

This module is where collision freedom is *designed in*. safety.py only
verifies it afterwards. The two must stay independent - a validator that
shares assumptions with the planner validates nothing.

The guarantee
-------------
Let formation A have points a_i and formation B have points b_i, and let
sigma be a matching that minimises sum_i |a_i - b_sigma(i)|^2. For any pair
i, j, optimality means swapping them cannot help:

    |a_i-b_i|^2 + |a_j-b_j|^2  <=  |a_i-b_j|^2 + |a_j-b_i|^2

which rearranges to

    (a_i - a_j) . (b_i - b_j)  >=  0                                  (*)

Now fly every drone on a straight line under one shared easing s(t) that
goes 0 -> 1, so p_i(t) = (1-s) a_i + s b_i. Writing u = a_i - a_j and
v = b_i - b_j, the pair separation is

    |p_i - p_j|^2 = (1-s)^2 |u|^2 + 2 s(1-s) (u.v) + s^2 |v|^2

Every term is non-negative by (*) and s in [0,1], so

    |p_i - p_j|^2 >= (1-s)^2 |u|^2 + s^2 |v|^2
                  >= min(|u|^2, |v|^2) * ((1-s)^2 + s^2)
                  >= min(|u|^2, |v|^2) / 2

giving a separation floor of min(d_A, d_B) / sqrt(2) throughout the move.
Hence: scale formations to min_separation * sqrt(2) and transitions can
never breach min_separation. No search, no re-planning, no luck.

Two conditions are load-bearing and are enforced, not assumed:
  1. the matching is optimal (linear_sum_assignment, not a heuristic)
  2. all drones share one easing - per-drone timing breaks the proof
A drone sitting out a transition (identical start and end) is fine; a drone
on a different clock is not.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import geometry
from .show import EASING_PEAKS, Formation, Show, ShowLimits, ease, moved

SAFE_TRANSITION_FACTOR = math.sqrt(2.0)


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimal assignment of points in `a` to points in `b`.

    Returns perm such that drone i flies a[i] -> b[perm[i]], minimising the
    total squared distance flown. Squared (not plain) distance is required:
    it is what yields inequality (*) in the module docstring, and it is
    also what keeps peak speed down, since the slowest drone sets the
    transition duration.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"cannot match {a.shape} against {b.shape}")
    # (n, n) squared distances without forming the (n, n, 3) difference.
    cost = (
        np.sum(a * a, axis=1)[:, None]
        - 2.0 * (a @ b.T)
        + np.sum(b * b, axis=1)[None, :]
    )
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(len(a), dtype=np.int64)
    perm[rows] = cols
    # Symmetric shapes (a cube, a grid) have many equally short answers, and
    # the solver may pick a different one each time it runs. Keeping the
    # current order when it is already optimal makes matching a no-op on a
    # matched show; without this the preview re-matched a prepared show and
    # drew paths up to 14 m from the ones validated and flown.
    keep = float(np.trace(cost))
    best = float(cost[rows, cols].sum())
    if keep <= best + 1e-9 * max(1.0, abs(best)):
        return np.arange(len(a), dtype=np.int64)
    return perm


def reorder(formation: Formation, perm: np.ndarray) -> Formation:
    """Relabel a formation so drone i owns the slot it was matched to."""
    out = formation.copy()
    # perm maps drone -> slot, so slot perm[i] must move to index i.
    out.points = formation.points[perm]
    out.colors = formation.colors[perm]
    out.color_keys = [(t, c[perm]) for t, c in formation.color_keys]
    return out


def transition_lower_bound(a: np.ndarray, b: np.ndarray) -> float:
    """Guaranteed minimum separation during an optimally matched transition.

    Only valid for the already-matched pair of formations - call it after
    `reorder`. Independent of the easing, as long as it is shared.
    """
    da = geometry.min_pair_distance(a)
    db = geometry.min_pair_distance(b)
    return min(da, db) / SAFE_TRANSITION_FACTOR


def chain_match(show: Show) -> Show:
    """Relabel every formation so consecutive moves are optimally matched.

    The first formation fixes drone identity; each later one is permuted to
    fit. Done in place on a copy; returns the new show.

    A formation with keep_order keeps its pairing with the one before it:
    the drone on stored slot j of the previous formation flies to stored
    slot j of this one. It is renumbered exactly as its predecessor was, so
    the pairing survives whatever renumbering happens upstream (take-off is
    added in front of everything by prepare).
    """
    out = show.copy()
    applied = np.arange(out.drone_count)    # renumbering of the previous formation
    for i in range(1, len(out.formations)):
        f = out.formations[i]
        if not f.keep_order:
            applied = match(out.formations[i - 1].end_points, f.points)
        out.formations[i] = reorder(f, applied)
    return out


# --------------------------------------------------------------------------
# Takeoff and landing
# --------------------------------------------------------------------------


def takeoff_grid(
    n: int, spacing: float = 3.0, *, stagger: bool = True, cols: int | None = None
) -> np.ndarray:
    """Ground positions for n drones, centred on the show origin, z = 0.

    Odd rows are offset by half the spacing when `stagger` is set, which
    buys 15% more clearance between rows for free - the usual trick when
    the launch area is tight.
    """
    if n < 1:
        raise ValueError("need at least one drone")
    if cols is None:
        cols = max(1, math.ceil(math.sqrt(n)))
    idx = np.arange(n)
    row = idx // cols
    col = idx % cols
    x = col.astype(np.float64) * spacing
    y = row.astype(np.float64) * spacing
    if stagger:
        x = x + (row % 2) * (spacing / 2.0)
    p = np.column_stack([x, y, np.zeros(n)])
    return p - np.array([p[:, 0].mean(), p[:, 1].mean(), 0.0])


def grid_layers(n: int, layers: int, cols: int | None = None) -> np.ndarray:
    """Layer index per drone, as an interleaved checkerboard over the grid.

    With 4 layers the pattern is a 2x2 tile, so same-layer neighbours are
    two grid cells apart (2*spacing) and cross-layer neighbours are at
    least one cell plus one layer gap apart. That is what makes the climb
    end in a well-separated 3D cloud instead of a flat grid.
    """
    if layers < 1:
        raise ValueError("layers must be >= 1")
    if cols is None:
        cols = max(1, math.ceil(math.sqrt(n)))
    idx = np.arange(n)
    row = idx // cols
    col = idx % cols
    if layers == 1:
        return np.zeros(n, dtype=np.int64)
    if layers == 2:
        return ((row + col) % 2).astype(np.int64)
    if layers == 4:
        return ((row % 2) * 2 + (col % 2)).astype(np.int64)
    # General case: diagonal stripes, still spreading neighbours apart.
    return ((row * 2 + col) % layers).astype(np.int64)


def launch_cloud(
    grid: np.ndarray,
    *,
    base_altitude: float,
    layers: int,
    layer_gap: float,
    cols: int | None = None,
) -> np.ndarray:
    """Lift a ground grid straight up into altitude-separated layers.

    A pure vertical climb from a grid whose spacing already meets the
    minimum is collision-free by construction: horizontal separation never
    changes and vertical motion is monotone and shared. The resulting
    cloud is what the first artistic formation is matched against.
    """
    n = len(grid)
    layer = grid_layers(n, layers, cols=cols)
    out = np.array(grid, dtype=np.float64, copy=True)
    out[:, 2] = base_altitude + layer * layer_gap
    return out


def build_takeoff(
    show: Show,
    *,
    climb_duration: float | None = None,
    base_altitude: float | None = None,
    grid: np.ndarray | None = None,
    color: tuple[int, int, int] = (0, 0, 0),
    hold: float = 2.0,
) -> tuple[Show, np.ndarray]:
    """Prepend a ground pose and a layered climb to a show.

    Returns the extended show and the ground grid, which the operator
    needs in order to physically place the drones. Lights are off on the
    ground by default - a lit drone on the ground is a tripping hazard and
    confuses the crew about which units are armed.
    """
    lim = show.limits
    n = show.drone_count
    if grid is None:
        grid = takeoff_grid(n, lim.takeoff_grid_spacing)
    if len(grid) != n:
        raise ValueError(f"grid has {len(grid)} slots for {n} drones")

    spacing_ok = geometry.min_pair_distance(grid)
    # Tolerance, because a staggered grid laid out at exactly the minimum
    # can measure 2.99999999999999911 for 3.0 depending on the drone count,
    # and a safety guard that trips on 0.9 femtometres is a bug, not a guard.
    if spacing_ok < lim.min_separation - 1e-6:
        raise ValueError(
            f"takeoff grid spacing is {spacing_ok:.2f} m but min_separation is "
            f"{lim.min_separation:.2f} m - widen the grid or use fewer drones"
        )

    if base_altitude is None:
        base_altitude = max(lim.min_nav_altitude, 10.0)
    cloud = launch_cloud(
        grid,
        base_altitude=base_altitude,
        layers=lim.takeoff_layers,
        layer_gap=lim.takeoff_layer_gap,
    )

    climb = float(np.max(cloud[:, 2]) - 0.0)
    if climb_duration is None:
        climb_duration = _duration_for(climb, lim.max_velocity_up, lim.max_acceleration, "minjerk")

    out = show.copy()
    ground = Formation(
        name="ground",
        points=grid,
        colors=np.array(color, dtype=np.int32),
        transition=0.0,
        hold=0.0,
        easing="linear",
    )
    lifted = Formation(
        name="climb",
        points=cloud,
        colors=np.array(color, dtype=np.int32),
        transition=climb_duration,
        hold=hold,
        easing="minjerk",
    )
    out.formations = [ground, lifted] + out.formations
    return chain_match(out), grid


def build_landing(
    show: Show,
    *,
    grid: np.ndarray | None = None,
    rth_altitude: float | None = None,
    rth_duration: float | None = None,
    descent_duration: float | None = None,
    fade_duration: float = 3.0,
) -> Show:
    """Append return-to-home and a vertical descent to the ground grid.

    RTH deliberately does not send each drone back to its own launch pad.
    The matching is free to pick any pad, which is both shorter and safer;
    the crew collects by pad, not by serial number. If you need a drone
    back on its own pad, pass a `grid` ordered to match drone index and
    accept the longer, more crowded transit.
    """
    lim = show.limits
    n = show.drone_count
    if not show.formations:
        raise ValueError("cannot land a show with no formations")
    if grid is None:
        grid = takeoff_grid(n, lim.takeoff_grid_spacing)

    if rth_altitude is None:
        rth_altitude = max(lim.min_nav_altitude, 15.0)
    cloud = launch_cloud(
        grid,
        base_altitude=rth_altitude,
        layers=lim.takeoff_layers,
        layer_gap=lim.takeoff_layer_gap,
    )

    # Match now, not later, so the duration is sized from the real
    # per-drone displacement rather than a loose bound.
    last = show.formations[-1].end_points
    cloud = cloud[match(last, cloud)]
    span = float(np.max(np.linalg.norm(cloud - last, axis=1)))
    if rth_duration is None:
        rth_duration = _duration_for(
            span, lim.max_velocity_xy, lim.max_acceleration, "minjerk"
        )
    descent = float(np.max(cloud[:, 2]))
    if descent_duration is None:
        descent_duration = _duration_for(
            descent, lim.max_velocity_down, lim.max_acceleration, "minjerk"
        )

    out = show.copy()
    out.formations.append(
        Formation(
            name="rth",
            points=cloud,
            colors=show.formations[-1].colors.copy(),
            transition=rth_duration,
            hold=fade_duration,
            easing="minjerk",
            color_keys=[(fade_duration, np.zeros((n, 3), dtype=np.int32))],
        )
    )
    out.formations.append(
        Formation(
            name="land",
            # Straight down from wherever RTH parked each drone: the descent
            # must not move laterally, or drones in different layers would
            # cross while sinking through each other's altitude.
            points=np.column_stack([cloud[:, 0], cloud[:, 1], np.zeros(n)]),
            colors=np.zeros((n, 3), dtype=np.int32),
            transition=descent_duration,
            hold=1.0,
            easing="minjerk",
        )
    )
    return chain_match(out)


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def _duration_for(
    distance: float, v_max: float, a_max: float, easing: str
) -> float:
    """Shortest duration for a shared-easing move of `distance` within limits.

    For easing s with peak |s'| = kv and peak |s''| = ka on unit time and
    unit displacement, a move of distance D over duration T peaks at
    kv*D/T and ka*D/T^2, so T must satisfy both bounds.
    """
    if distance <= 0:
        return 0.0
    kv, ka = EASING_PEAKS[easing]
    t_v = kv * distance / max(v_max, 1e-9)
    t_a = math.sqrt(ka * distance / max(a_max, 1e-9)) if math.isfinite(ka) else 0.0
    return max(t_v, t_a, 0.1)


def autotime(show: Show, *, margin: float = 1.15, quantise: float = 0.1) -> Show:
    """Stretch any transition that would breach the velocity or accel limit.

    Never shortens a transition: a designer's slow move is a choice, a
    designer's too-fast move is a bug. `margin` keeps a cushion above the
    bare minimum so that finite-difference checks in safety.py do not sit
    exactly on the threshold.
    """
    out = chain_match(show)
    lim = out.limits
    for i in range(1, len(out.formations)):
        f = out.formations[i]
        delta = f.points - out.formations[i - 1].end_points
        if not len(delta):
            continue
        dist = float(np.max(np.linalg.norm(delta, axis=1)))
        horiz = float(np.max(np.linalg.norm(delta[:, :2], axis=1)))
        up = float(np.max(np.maximum(delta[:, 2], 0.0)))
        down = float(np.max(np.maximum(-delta[:, 2], 0.0)))
        need = max(
            _duration_for(horiz, lim.max_velocity_xy, lim.max_acceleration, f.easing),
            _duration_for(up, lim.max_velocity_up, lim.max_acceleration, f.easing),
            _duration_for(down, lim.max_velocity_down, lim.max_acceleration, f.easing),
            _duration_for(dist, math.inf, lim.max_acceleration, f.easing),
        )
        need *= margin
        if quantise > 0:
            need = math.ceil(need / quantise) * quantise
        if need > f.transition:
            f.transition = round(need, 4)
    for f in out.formations:
        need = motion_hold_need(f, lim) * margin
        if quantise > 0:
            need = math.ceil(need / quantise) * quantise
        if need > f.hold:
            f.hold = round(need, 4)
    return out


def motion_hold_need(f: Formation, lim: ShowLimits) -> float:
    """Shortest hold in which a formation's motion stays inside its share
    of the speed and acceleration limits. Zero for one that holds still."""
    if f.motion is None:
        return 0.0
    tau = np.linspace(0.0, 1.0, 241)
    poses = np.stack([moved(f.points, f.motion, float(ease("minjerk", x))) for x in tau])
    dt = tau[1] - tau[0]
    vel = np.diff(poses, axis=0) / dt           # metres per unit of hold
    acc = np.diff(poses, n=2, axis=0) / dt**2
    k = lim.motion_speed_share
    return max(
        float(np.max(np.linalg.norm(vel[..., :2], axis=-1))) / (lim.max_velocity_xy * k),
        float(np.max(np.maximum(vel[..., 2], 0.0))) / (lim.max_velocity_up * k),
        float(np.max(np.maximum(-vel[..., 2], 0.0))) / (lim.max_velocity_down * k),
        math.sqrt(float(np.max(np.linalg.norm(acc, axis=-1))) / (lim.max_acceleration * k)),
    )


def park_spares(points: np.ndarray, count: int, spacing: float) -> np.ndarray:
    """Add positions for the drones a formation does not use.

    They wait, unlit, on a square grid two spacings behind (north of) the
    formation, centred on it and no lower than its lowest drone.
    """
    from .generators import _park

    p = np.asarray(points, dtype=np.float64)
    extra = count - len(p)
    if extra <= 0:
        return p
    park = _park(extra, spacing)
    park[:, 0] += p[:, 0].mean()
    park[:, 1] = p[:, 1].max() + 2.0 * spacing
    park[:, 2] += p[:, 2].mean()
    park[:, 2] += max(0.0, p[:, 2].min() - park[:, 2].min())
    return np.vstack([p, park])


def height_range(f: Formation) -> tuple[float, float]:
    """Lowest and highest point a formation reaches, motion included."""
    if f.motion is None:
        return float(f.points[:, 2].min()), float(f.points[:, 2].max())
    z = np.stack([f.pose(f.hold * x)[:, 2] for x in np.linspace(0.0, 1.0, 21)])
    return float(z.min()), float(z.max())


# --------------------------------------------------------------------------
# One-call preparation
# --------------------------------------------------------------------------


def build_escape(
    positions: np.ndarray,
    limits: ShowLimits,
    *,
    grid: np.ndarray | None = None,
    rth_altitude: float | None = None,
    name: str = "collective rth",
) -> Show:
    """A collision-free route home from wherever the fleet currently is.

    This is what a fleet-wide abort should fly. Sending every aircraft its
    own RTL instead gives each autopilot a private straight line to its own
    launch pad, computed with no knowledge of the other 499 - which is a
    plan for a mid-air, not a failsafe.

    Here the return is a short show: current positions, matched optimally
    onto a layered cloud above the launch grid, then a vertical descent.
    The matching and the shared easing are the same machinery as the main
    show, so the same separation bound applies.

    One caveat that cannot be engineered away and is therefore verified
    instead: the bound guarantees min(d_start, d_end)/sqrt(2), and d_start
    here is whatever separation the fleet happens to have right now, which
    may be as low as min_separation. The caller must run the sweep on the
    result - FleetManager.collective_rth does - and fall back to holding if
    it does not clear.
    """
    p = np.asarray(positions, dtype=np.float64)
    n = len(p)
    if n == 0:
        raise ValueError("no positions to return")

    if grid is None:
        grid = takeoff_grid(n, limits.takeoff_grid_spacing)
    if rth_altitude is None:
        rth_altitude = max(limits.min_nav_altitude, 15.0)

    cloud = launch_cloud(
        grid,
        base_altitude=rth_altitude,
        layers=limits.takeoff_layers,
        layer_gap=limits.takeoff_layer_gap,
    )
    cloud = cloud[match(p, cloud)]

    show = Show(name=name, drone_count=n, limits=ShowLimits(**asdict(limits)))
    show.formations = [
        Formation(
            name="now",
            points=p.copy(),
            colors=np.array([120, 120, 120], dtype=np.int32),
            transition=0.0,
            hold=0.0,
            easing="minjerk",
        ),
        Formation(
            name="regroup",
            points=cloud,
            colors=np.array([255, 160, 0], dtype=np.int32),
            transition=1.0,  # autotime widens this to fit the envelope
            hold=2.0,
            easing="minjerk",
        ),
        Formation(
            name="land",
            points=np.column_stack([cloud[:, 0], cloud[:, 1], np.zeros(n)]),
            colors=np.zeros((n, 3), dtype=np.int32),
            transition=1.0,
            hold=1.0,
            easing="minjerk",
        ),
    ]
    return autotime(show)


def auto_geofence(show: Show, margin: float = 15.0, *, simplify: int = 16) -> list[list[float]]:
    """A geofence hull around everything the show actually does, plus margin.

    Drawn from the trajectories rather than the formations, so the bulge a
    transition makes is inside the fence. The result is a starting point for
    the operator, not an authorisation: the real fence comes from the
    permit, and is usually tighter.
    """
    from scipy.spatial import ConvexHull

    from .trajectory import build_timeline

    if not show.formations:
        raise ValueError("cannot fence a show with no formations")
    xy = build_timeline(show).pos[:, :, :2].reshape(-1, 2)
    hull = ConvexHull(xy)
    poly = xy[hull.vertices]

    centre = poly.mean(axis=0)
    # Push each vertex out along its own outward direction, which over-covers
    # at corners. Over-covering a geofence is the safe direction to err.
    offsets = poly - centre
    norms = np.linalg.norm(offsets, axis=1, keepdims=True)
    poly = poly + np.divide(offsets, np.maximum(norms, 1e-9)) * margin

    if simplify and len(poly) > simplify:
        # Keep evenly spaced vertices: a 400-vertex fence is unreadable on a
        # map and most flight controllers cap the vertex count anyway.
        idx = np.round(np.linspace(0, len(poly) - 1, simplify)).astype(int)
        poly = poly[np.unique(idx)]
    return [[round(float(x), 2), round(float(y), 2)] for x, y in poly]


def adapt(show: Show, drone_count: int, *, keep_shape: bool = True) -> PreparedShow:
    """Re-cut a show for a different number of drones.

    The on-site reality this exists for: two aircraft fail preflight an hour
    before the show and the choreography was built for the full fleet. Rather
    than flying with holes in every formation, each formation is re-sampled
    down to the available count and the whole plan is rebuilt - rescaled,
    rematched and retimed - so the result is a show that passes validation in
    its own right, not a damaged version of one that did.

    Re-sampling uses farthest-point selection, which keeps the silhouette: a
    heart drawn with 300 drones still reads as a heart with 220. Taking the
    first N points instead would delete one side of it.
    """
    if drone_count < 1:
        raise ValueError("need at least one drone")
    if drone_count > show.drone_count:
        raise ValueError(
            f"cannot adapt a {show.drone_count} drone show up to {drone_count}; "
            "formations would have to invent positions. Re-generate the show "
            "at the larger count instead."
        )

    out = show.copy()
    out.drone_count = drone_count
    if drone_count == show.drone_count:
        return prepare(out)

    for f in out.formations:
        keep = (
            _farthest_point_subset(f.points, drone_count)
            if keep_shape
            else np.arange(drone_count)
        )
        f.points = f.points[keep]
        f.colors = f.colors[keep]
        f.color_keys = [(t, c[keep]) for t, c in f.color_keys]
        if f.yaw is not None:
            f.yaw = f.yaw[keep]
        # Pyro targets drone indices, which have just been renumbered. An
        # event with no explicit drones means "all of them" and survives as
        # is; a targeted one survives only if any of its drones did, and is
        # dropped rather than silently re-aimed at whoever inherited the
        # index.
        remap = {old: new for new, old in enumerate(keep.tolist())}
        surviving = []
        for event in f.pyro:
            if not event.drones:
                surviving.append(event)
                continue
            event.drones = sorted(remap[d] for d in event.drones if d in remap)
            if event.drones:
                surviving.append(event)
        f.pyro = surviving

    out.notes = (
        f"{out.notes}\nadapted from {show.drone_count} to {drone_count} drones"
    ).strip()
    return prepare(out)


def _farthest_point_subset(points: np.ndarray, count: int) -> np.ndarray:
    """Pick `count` well-spread points, preserving the shape's extent.

    Greedy farthest-point sampling: start at the point furthest from the
    centroid, then repeatedly take whichever remaining point is furthest
    from everything already chosen. O(count x n), which is nothing next to
    the Hungarian matching that follows.
    """
    p = np.asarray(points, dtype=np.float64)
    n = len(p)
    if count >= n:
        return np.arange(n)

    first = int(np.argmax(np.linalg.norm(p - p.mean(axis=0), axis=1)))
    chosen = [first]
    best = np.linalg.norm(p - p[first], axis=1)
    for _ in range(count - 1):
        nxt = int(np.argmax(best))
        chosen.append(nxt)
        best = np.minimum(best, np.linalg.norm(p - p[nxt], axis=1))
        best[nxt] = -1.0
    return np.sort(np.asarray(chosen, dtype=np.int64))


@dataclass
class PreparedShow:
    show: Show
    grid: np.ndarray
    notes: list[str]


def design_separation(lim: ShowLimits, fps: float = 10.0) -> float:
    """The spacing formations are packed to: formation_separation plus slack.

    The proof at the top of this module gives min_separation exactly, and
    the validator asks for more: it adds a path-curvature margin to its
    threshold, because it checks chords between samples. A formation packed
    to exactly formation_separation, flown to a launch cloud that is also
    exactly that tight, passes at the bound and so fails the validator by
    millimetres. Any lattice does it (a wave, a grid).

    The margin is at most max_acceleration / (4 fps^2): twice the sagitta
    of a path at the acceleration limit. Against a cloud at exactly
    sqrt(2) * min_separation, slack d buys d / (2 sqrt(2)) at the bound, so
    d = 2 sqrt(2) * margin. Deliberately the worst case: at 10 Hz it costs
    under 1 % of show size, and a low sample rate costs more because the
    validator can then see less.
    """
    margin = lim.max_acceleration / (4.0 * fps * fps)
    return lim.formation_separation + 2.0 * math.sqrt(2.0) * margin


def fit_formation(f, lim, fps: float = 10.0) -> list[str]:
    """Scale and raise one formation to legal size, in place.

    Returns the notes. If the result is over the ceiling, the last note says
    so - `prepare` reports it, and the server uses it to warn at design time.
    """
    notes: list[str] = []
    target = design_separation(lim, fps)
    if f.motion is not None and f.motion.grow < 1.0:
        # Shrinking pulls drones together; start wide enough to end legal.
        target /= f.motion.grow
    before = geometry.min_pair_distance(f.points)
    if before < target:
        centre = f.points.mean(axis=0)
        scaled = geometry.fit_separation(f.points, target, center=True)
        f.points = scaled + centre
        notes.append(
            f"{f.name}: scaled x{target / max(before, 1e-9):.2f} "
            f"to reach {target:.2f} m separation"
        )
    low = height_range(f)[0]
    if low < lim.min_nav_altitude:
        f.points[:, 2] += lim.min_nav_altitude - low
        notes.append(
            f"{f.name}: raised {lim.min_nav_altitude - low:.2f} m to clear "
            "the minimum navigation altitude"
        )
    top = height_range(f)[1]
    if top > lim.max_altitude:
        # Not clamped: shrinking it back would breach separation, and
        # silently moving the artwork is worse than refusing it. The
        # validator will fail the show; this note says why and what to
        # change, which the validator cannot know.
        notes.append(
            f"{f.name}: now {top:.0f} m tall, over the {lim.max_altitude:.0f} m "
            f"ceiling. {f.count} drones at {lim.formation_separation:.2f} m "
            "spacing do not fit this shape at legal size. Use a filled "
            "shape (solid=True, or disc/ball) so the drone count spreads "
            "over an area instead of a line, or put fewer drones on it."
        )
    return notes


def prepare(
    show: Show,
    *,
    add_takeoff: bool = True,
    add_landing: bool = True,
    enforce_separation: bool = True,
    grid: np.ndarray | None = None,
) -> PreparedShow:
    """Take designed artwork to a flyable plan.

    Order matters and is not negotiable:
      1. scale each artistic formation up to design_separation
      2. wrap with takeoff and landing
      3. optimally match every consecutive pair
      4. stretch transitions to fit the flight envelope

    Scaling first means the transition bound applies to the final geometry.
    Matching after wrapping means the climb and RTH are matched too.
    """
    notes: list[str] = []
    work = show.copy()
    lim = work.limits

    if enforce_separation:
        for f in work.formations:
            notes.extend(fit_formation(f, lim, work.fps))

    used_grid = grid if grid is not None else takeoff_grid(
        work.drone_count, lim.takeoff_grid_spacing
    )
    if add_takeoff:
        work, used_grid = build_takeoff(work, grid=used_grid)
    if add_landing:
        work = build_landing(work, grid=used_grid)

    work = autotime(work)
    if add_takeoff and show.formations:
        # Take-off now comes first, so the music must start later by the same
        # amount. Line it up on the moment the first designed formation
        # appears; in the design that is the end of its transition.
        design = show.formations[0].transition
        flown = sum(f.duration for f in work.formations[:2]) + work.formations[2].transition
        _shift_music(work.music, flown - design)
    return PreparedShow(show=work, grid=used_grid, notes=notes)


def _shift_music(music, seconds: float) -> None:
    """Move the song, its beat grid and its cues later in show time."""
    music.offset -= seconds
    music.beat_phase += seconds
    music.cues = [(t + seconds, label) for t, label in music.cues]
