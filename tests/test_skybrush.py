"""Skybrush .skyc export: the parts the published specification defines."""

from __future__ import annotations

import base64
import json
import zipfile

import numpy as np
import pytest

from skyshine import planning, skybrush, templates
from skyshine.trajectory import build_mission, build_timeline


@pytest.fixture(scope="module")
def prepared():
    show = templates.build("showcase", "sky", 16)
    show.origin.latitude, show.origin.longitude, show.origin.amsl = 47.5, 19.05, 110.0
    show.origin.orientation_deg = 30.0
    return planning.prepare(show).show


def test_varuint_matches_the_specification():
    assert skybrush.varuint(513) == bytes([0x81, 0x04])
    assert skybrush.varuint(2**40) == bytes([0xFF, 0xFF, 0xFF, 0xFF, 0x0F])   # clamped


def test_document_shape(prepared):
    doc = skybrush.show_json(prepared)
    assert doc["version"] == 1 and doc["environment"]["type"] == "outdoor"
    loc = doc["environment"]["location"]
    assert loc["orientation"] == pytest.approx(120.0)             # 90 + 30
    assert loc["origin"] == [475000000, 190500000, 110000]
    drones = doc["swarm"]["drones"]
    assert len(drones) == prepared.drone_count
    traj = drones[0]["settings"]["trajectory"]
    assert traj["version"] == 1 and traj["takeoffTime"] == 0
    times = [p[0] for p in traj["points"]]
    assert times[0] == 0 and times == sorted(times)
    assert all(len(p) == 3 and len(p[1]) == 3 and p[2] == [] for p in traj["points"])
    assert doc["settings"]["validation"]["minDistance"] == prepared.limits.min_separation


def test_paths_follow_the_validated_timeline(prepared):
    doc = skybrush.show_json(prepared)
    tl = build_timeline(prepared)
    for d in (0, 7, 15):
        pts = doc["swarm"]["drones"][d]["settings"]["trajectory"]["points"]
        t = np.array([p[0] for p in pts])
        xyz = np.array([p[1] for p in pts])
        est = np.column_stack([np.interp(tl.t, t, xyz[:, k]) for k in range(3)])
        assert np.max(np.linalg.norm(est - tl.pos[:, d], axis=1)) < 0.02   # 2 cm


def test_light_programs_replay_the_show_lights(prepared):
    doc = skybrush.show_json(prepared)
    mission = build_mission(prepared)
    for d in (0, 9):
        code = base64.b64decode(doc["swarm"]["drones"][d]["settings"]["lights"]["data"])
        steps = dict(skybrush.run_light_program(code, 10**9))
        for t, *rgb in mission.plans[d].lights:
            frame = round(t * skybrush.FPS)
            assert steps[frame] == tuple(int(round(v)) for v in rgb)


def test_written_file_holds_the_json(prepared, tmp_path):
    path = skybrush.write_skyc(prepared, tmp_path / "sky.skyc")
    with zipfile.ZipFile(path) as z:
        doc = json.loads(z.read("show.json"))
    assert doc["meta"]["title"] == prepared.name


@pytest.mark.parametrize("metric,tol", [("max", 0.5), ("l2", 0.01)])
def test_thinning_keeps_every_dropped_row_within_tolerance(metric, tol):
    """A slow, gently curving change is where neighbour-only thinning drifted."""
    from skyshine.trajectory import simplify

    t = np.linspace(0, 30, 3001)
    vals = np.column_stack([100 + 80 * np.sin(t / 9), 50 + 0.4 * t ** 1.5, 200 - 3 * t])
    rows = np.column_stack([t, vals])
    kept = simplify(rows, tol, metric=metric)
    assert len(kept) < len(rows) / 5
    rebuilt = np.column_stack([np.interp(t, kept[:, 0], kept[:, k]) for k in (1, 2, 3)])
    diff = rebuilt - vals
    err = np.abs(diff).max(axis=1) if metric == "max" else np.linalg.norm(diff, axis=1)
    assert err.max() <= tol + 1e-9
