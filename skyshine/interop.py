"""Talking to the rest of the industry.

Three directions:

  vviz out   the interchange format Finale 3D imports, for hybrid shows
             where pyro is scripted against the drones
  csv out    per-drone sampled trajectories, the lowest common denominator
             every tool reads
  csv in     somebody else's show, sampled, so Sky Shine's validator can be
             pointed at it

The import path is the one that earns its keep. A sampled trajectory from
another tool carries no formations, no matching and no easing - so none of
the design-time guarantees in planning.py apply to it. What it can still
get is the measurement: safety.py does not care where a trajectory came
from, so an imported show gets the same continuous-time separation sweep
as a native one.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .show import Show, ShowLimits, ShowOrigin, Timeline
from .trajectory import build_timeline

# --------------------------------------------------------------------------
# vviz
# --------------------------------------------------------------------------

# vviz uses the OpenGL convention: +x right, +y up, +z toward the viewer.
# With the audience to the south of the show, +z is south, so north is -z.
# Getting this wrong mirrors the whole show front to back, which looks
# plausible on a symmetric formation and wrong on a logo.
VVIZ_POSITION_RATE = 2.0
VVIZ_COLOR_RATE = 10.0


def enu_to_ogl(p: np.ndarray) -> np.ndarray:
    """(east, north, up) -> (x, y, z) in vviz's ogl frame."""
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def ogl_to_enu(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def _rle_colors(samples: np.ndarray) -> list[dict]:
    """Run-length encode a colour track the way vviz payloadActions do.

    Each action occupies `frames` frames, defaulting to 1 when absent, and
    the frame counts must sum to the track length or the importer will
    drift against the motion.
    """
    actions: list[dict] = []
    if len(samples) == 0:
        return actions
    current = samples[0]
    run = 1
    for value in samples[1:]:
        if np.array_equal(value, current):
            run += 1
            continue
        actions.append(_action(current, run))
        current = value
        run = 1
    actions.append(_action(current, run))
    return actions


def _action(color: np.ndarray, frames: int) -> dict:
    out = {"r": int(color[0]), "g": int(color[1]), "b": int(color[2])}
    if frames > 1:
        out["frames"] = int(frames)
    return out


def export_vviz(
    show: Show,
    path: str | Path,
    *,
    timeline: Timeline | None = None,
    airframe: str = "SkyShine",
    lumens: int = 900,
    position_rate: float = VVIZ_POSITION_RATE,
    color_rate: float = VVIZ_COLOR_RATE,
) -> Path:
    """Write a .vviz file for Finale 3D and other vviz readers.

    Motion is stored as per-sample deltas from the previous position, which
    is what makes the format compress: a drone holding station writes a run
    of zeroes.
    """
    if timeline is None:
        timeline = build_timeline(show, fps=max(position_rate, color_rate))

    duration = timeline.duration
    pos_times = np.arange(0.0, duration + 1e-9, 1.0 / position_rate)
    col_times = np.arange(0.0, duration + 1e-9, 1.0 / color_rate)

    positions = np.stack([timeline.sample(float(t))[0] for t in pos_times])
    colors = np.stack([timeline.sample(float(t))[1] for t in col_times])

    pyro_by_drone: dict[int, list[tuple[float, str]]] = {}
    for when, index, event in show.pyro_events():
        for drone in show.formations[index].firing_drones(event):
            pyro_by_drone.setdefault(drone, []).append((when, event.description))

    performances = []
    for d in range(show.drone_count):
        track = enu_to_ogl(positions[:, d, :])
        home = track[0]
        deltas = np.diff(track, axis=0, prepend=track[:1])
        traversal = [
            {"dx": round(float(v[0]), 4), "dy": round(float(v[1]), 4),
             "dz": round(float(v[2]), 4)}
            for v in deltas
        ]

        payloads: list[dict] = [
            {
                "id": 0,
                "type": "Light",
                "lumens": lumens,
                "colorType": "RGB",
                "sourceType": "Dome",
                "payloadActions": _rle_colors(colors[:, d, :]),
            }
        ]
        for i, (when, label) in enumerate(sorted(pyro_by_drone.get(d, []))):
            payloads.append(
                {
                    "id": i + 1,
                    "type": "Pyro",
                    "eventTime": round(float(when), 3),
                    "vdl": label or "drone pyro",
                    "partNumber": "",
                }
            )

        performances.append(
            {
                "id": d,
                "agentDescription": {
                    "homeX": round(float(home[0]), 4),
                    "homeY": round(float(home[1]), 4),
                    "homeZ": round(float(home[2]), 4),
                    "airframe": airframe,
                    "agentTraversal": traversal,
                },
                "payloadDescription": payloads,
            }
        )

    body = {
        "version": "1.0",
        "performanceName": show.name,
        "coordinateFrame": "ogl",
        "globalReferenceFrame": {
            "lat": show.origin.latitude,
            "lon": show.origin.longitude,
            "alt": show.origin.amsl or 0.0,
        },
        "defaultPositionRate": position_rate,
        "defaultColorRate": color_rate,
        "performances": performances,
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=1), encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------

CSV_HEADER = ["Time [msec]", "x [m]", "y [m]", "z [m]", "Red", "Green", "Blue"]


def export_csv(
    show: Show,
    directory: str | Path,
    *,
    timeline: Timeline | None = None,
    fps: float = 5.0,
) -> list[Path]:
    """One CSV per drone: time in milliseconds, position in metres, RGB.

    The de facto interchange shape. Lossy by construction - it is a
    sampling of the motion, not the motion - so it is for handing a show to
    another tool, never for round-tripping your own.
    """
    if timeline is None:
        timeline = build_timeline(show, fps=fps)
    times = np.arange(0.0, timeline.duration + 1e-9, 1.0 / fps)
    samples = [timeline.sample(float(t)) for t in times]

    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for d in range(show.drone_count):
        target = out_dir / f"drone_{d + 1:03d}.csv"
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_HEADER)
            for t, (pos, col) in zip(times, samples):
                writer.writerow(
                    [
                        int(round(t * 1000)),
                        f"{pos[d, 0]:.3f}",
                        f"{pos[d, 1]:.3f}",
                        f"{pos[d, 2]:.3f}",
                        int(col[d, 0]),
                        int(col[d, 1]),
                        int(col[d, 2]),
                    ]
                )
        written.append(target)
    return written


@dataclass
class ImportedShow:
    """A trajectory set from another tool, ready for validation only."""

    timeline: Timeline
    show: Show
    source: str
    drone_count: int
    duration: float
    notes: list[str]


def import_csv(
    directory: str | Path,
    *,
    name: str | None = None,
    limits: ShowLimits | None = None,
    origin: ShowOrigin | None = None,
) -> ImportedShow:
    """Read a directory of per-drone CSVs into something validate() accepts.

    Files are sorted by name, and that order becomes drone identity. All
    files must share a time column: a fleet sampled on different clocks
    cannot be checked for separation, because there is no single instant to
    compare them at. That is reported as an error rather than interpolated
    over, since quietly resampling someone else's show would hide exactly
    the timing defect worth finding.
    """
    folder = Path(directory)
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise ValueError(f"no CSV files in {folder}")

    notes: list[str] = []
    times_ref: np.ndarray | None = None
    positions: list[np.ndarray] = []
    colors: list[np.ndarray] = []

    for f in files:
        t, p, c = _read_one_csv(f)
        if times_ref is None:
            times_ref = t
        elif len(t) != len(times_ref) or not np.allclose(t, times_ref, atol=1e-3):
            raise ValueError(
                f"{f.name} is sampled on a different time base than "
                f"{files[0].name}. Separation cannot be checked across drones "
                "that do not share an instant; resample them onto one clock "
                "before importing."
            )
        positions.append(p)
        colors.append(c)

    assert times_ref is not None
    pos = np.stack(positions, axis=1)
    col = np.stack(colors, axis=1)
    n = pos.shape[1]

    show = Show(
        name=name or folder.name,
        drone_count=n,
        limits=limits or ShowLimits(),
        origin=origin or ShowOrigin(),
        notes=f"imported from {folder}",
    )
    # A synthetic single formation keeps the Show valid for reporting. It is
    # NOT the design: there are no formations in a sampled trajectory, and
    # check_design would otherwise invent findings about artwork that does
    # not exist here.
    from .show import Formation

    show.formations = [
        Formation(
            name="imported",
            points=pos[0].copy(),
            colors=col[0].astype(np.int32),
            transition=0.0,
            hold=float(times_ref[-1]),
        )
    ]

    timeline = Timeline(
        t=np.asarray(times_ref, dtype=np.float64),
        pos=np.ascontiguousarray(pos),
        col=np.ascontiguousarray(col),
        segment=np.zeros(len(times_ref), dtype=np.int32),
        show=show,
    )

    step = float(np.median(np.diff(times_ref))) if len(times_ref) > 1 else 0.0
    if step > 0.25:
        notes.append(
            f"samples are {step:.2f}s apart. The sweep interpolates linearly "
            "between them, so a manoeuvre shorter than that is invisible to "
            "this check - ask the source tool for a denser export."
        )
    notes.append(f"{n} drones, {len(times_ref)} samples, {times_ref[-1]:.1f}s")

    return ImportedShow(
        timeline=timeline,
        show=show,
        source=str(folder),
        drone_count=n,
        duration=float(times_ref[-1]),
        notes=notes,
    )


def _read_one_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: list[float] = []
    pos: list[list[float]] = []
    col: list[list[int]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{path.name} is empty")

    start = 1 if _looks_like_header(rows[0]) else 0
    for lineno, row in enumerate(rows[start:], start=start + 1):
        cells = [c.strip() for c in row if c.strip() != ""]
        if not cells:
            continue
        if len(cells) < 4:
            raise ValueError(
                f"{path.name} line {lineno}: expected at least time,x,y,z, "
                f"got {len(cells)} values"
            )
        try:
            t = float(cells[0])
            xyz = [float(cells[1]), float(cells[2]), float(cells[3])]
            rgb = [int(float(cells[i])) for i in (4, 5, 6)] if len(cells) >= 7 else [255, 255, 255]
        except ValueError as exc:
            raise ValueError(f"{path.name} line {lineno}: {exc}") from None
        # Milliseconds if the header says so, or if the values are large
        # enough that seconds would mean a show lasting days.
        times.append(t / 1000.0 if _time_in_ms(rows[0], start) else t)
        pos.append(xyz)
        col.append([max(0, min(255, v)) for v in rgb])

    if len(times) < 2:
        raise ValueError(f"{path.name} has fewer than two samples")
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.asarray(col, dtype=np.uint8),
    )


def _looks_like_header(row: list[str]) -> bool:
    try:
        float(row[0])
        return False
    except (ValueError, IndexError):
        return True


def _time_in_ms(header: list[str], start: int) -> bool:
    if start == 0:
        return False
    first = (header[0] if header else "").lower()
    return "msec" in first or "millis" in first or "[ms]" in first


def validate_imported(imported: ImportedShow) -> dict:
    """Run the separation and envelope checks against an imported show."""
    from . import safety

    report = safety.SafetyReport(
        show_name=imported.show.name,
        drone_count=imported.drone_count,
        duration=imported.duration,
    )
    limits = imported.show.limits
    violations, worst = safety.check_separation(
        imported.timeline, limits.min_separation,
        ignore_below_altitude=limits.min_nav_altitude,
    )
    for v in violations:
        report.add(v)
    alt, lo, hi = safety.check_altitude(
        imported.timeline, limits.min_nav_altitude, limits.max_altitude
    )
    for v in alt:
        report.add(v)
    for v in safety.check_geofence(imported.timeline, limits.geofence):
        report.add(v)
    kin, metrics = safety.check_kinematics(imported.timeline, limits)
    for v in kin:
        report.add(v)

    report.metrics = {
        "min_separation_m": worst,
        "min_separation_limit_m": limits.min_separation,
        "separation_headroom_m": worst - limits.min_separation,
        "min_altitude_m": lo,
        "max_altitude_m": hi,
        "samples_checked": int(imported.timeline.pos.shape[0]),
        **metrics,
    }
    out = report.to_dict()
    out["import_notes"] = imported.notes
    out["source"] = imported.source
    # Say plainly what this verdict does and does not cover.
    out["coverage"] = (
        "Measured checks only. An imported trajectory has no formations, no "
        "matching and no shared easing, so none of the design-time collision "
        "guarantees apply to it - only what was measured here."
    )
    return out
