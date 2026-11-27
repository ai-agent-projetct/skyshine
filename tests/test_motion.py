"""Moving formations: a formation that spins, travels or grows while held.

The motion lives in the sampler, so the safety engine, the uploaded mission
and the preview all see the same path. These tests hold those three to it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from skyshine import geometry, planning, safety
from skyshine.show import (
    Formation, Motion, PyroEvent, Show, moved, show_from_dict, show_to_dict,
)
from skyshine.trajectory import build_mission, build_timeline


def moving_show(n=40, **motion) -> Show:
    show = Show(name="moving", drone_count=n)
    show.add(Formation(name="ring", points=geometry.lift(geometry.circle(n, 20.0), 40.0),
                       colors=np.array([255, 0, 0]), hold=3.0))
    show.add(Formation(name="disc", points=geometry.lift(geometry.disc(n, 18.0), 40.0),
                       colors=np.array([0, 200, 255]), hold=3.0,
                       motion=Motion(**(motion or {"spin": 180, "move": (8, 0, 4)}))))
    show.add(Formation(name="ball", points=geometry.lift(geometry.ball(n, 15.0), 40.0),
                       colors=np.array([255, 255, 255]), hold=3.0))
    return show


class TestMotionMaths:
    def test_spin_move_and_grow(self):
        p = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])  # centre at the origin
        q = moved(p, Motion(spin=90), 1.0)
        assert np.allclose(q[0], [0.0, 1.0, 0.0], atol=1e-12)  # anticlockwise from above
        assert np.allclose(moved(p, Motion(grow=2.0), 1.0)[0], [2.0, 0.0, 0.0])
        assert np.allclose(moved(p, Motion(move=(3, 4, 5)), 0.5), p + [1.5, 2.0, 2.5])
        assert np.allclose(moved(p, Motion(spin=90, grow=3), 0.0), p)

    @pytest.mark.parametrize("bad", [
        {"grow": 0.1}, {"grow": 6}, {"spin": 4000}, {"move": (1, 2)},
    ])
    def test_out_of_range_motion_is_refused(self, bad):
        with pytest.raises(ValueError):
            Motion(**bad)

    def test_still_motion_is_dropped_and_conflicts_are_refused(self):
        pts = geometry.lift(geometry.circle(6, 10.0), 30.0)
        assert Formation("f", pts, np.array([1, 2, 3]), motion=Motion()).motion is None
        with pytest.raises(ValueError, match="needs a hold"):
            Formation("f", pts, np.array([1, 2, 3]), hold=0.0, motion=Motion(spin=90))
        with pytest.raises(ValueError, match="pyro needs a formation that holds still"):
            Formation("f", pts, np.array([1, 2, 3]), hold=5.0, motion=Motion(spin=90),
                      pyro=[PyroEvent(at=1.0, heading_deg=0.0)])


class TestMovingShow:
    def test_prepared_show_passes_and_the_hold_is_stretched_to_the_limits(self):
        prep = planning.prepare(moving_show()).show
        disc = next(f for f in prep.formations if f.name == "disc")
        assert disc.hold > 3.0, "a fast spin must be slowed down, not flown"
        report = safety.validate(prep)
        assert report.passed, report.summary()
        assert report.metrics["peak_horizontal_speed_mps"] <= prep.limits.max_velocity_xy

    def test_the_formation_really_moves_and_the_next_move_starts_where_it_ended(self):
        prep = planning.prepare(moving_show()).show
        tl = build_timeline(prep)
        i = next(k for k, f in enumerate(prep.formations) if f.name == "disc")
        f = prep.formations[i]
        span = np.nonzero(tl.segment == i)[0]
        start_hold = span[np.searchsorted(tl.t[span], tl.t[span][0] + f.transition)]
        assert not np.allclose(tl.pos[start_hold], tl.pos[span[-1]], atol=1.0)
        # No jump at the hand-over to the next formation.
        assert np.allclose(tl.pos[span[-1] + 1], tl.pos[span[-1]], atol=0.5)

    def test_shrinking_starts_wide_enough_to_end_legal(self):
        prep = planning.prepare(moving_show(grow=0.5)).show
        report = safety.validate(prep)
        assert report.passed, report.summary()

    def test_motion_survives_the_file_format(self):
        show = moving_show(spin=-90, move=(0, 5, 0), grow=1.2)
        again = show_from_dict(show_to_dict(show))
        assert again.formations[1].motion.to_dict() == show.formations[1].motion.to_dict()
        assert again.formations[0].motion is None

    def test_uploaded_path_follows_the_validated_path(self):
        from skyshine.sitl import VirtualAircraft

        prep = planning.prepare(moving_show()).show
        tl = build_timeline(prep)
        plan = build_mission(prep).to_dict()["plans"][3]
        air = VirtualAircraft(0, 14894, noise=0.0)
        try:
            air.plan = plan
            gap = max(float(np.linalg.norm(air.evaluate_plan(float(t)) - tl.pos[k, 3]))
                      for k, t in enumerate(tl.t))
        finally:
            air.close()
        assert gap < 0.02, f"aircraft path strays {gap:.3f} m from what was validated"


class TestMotionApi:
    @pytest.fixture()
    def made(self, client):
        client.post("/api/design/shows", json={"name": "m", "drone_count": 20})
        client.post("/api/design/shows/m/formations", json={"name": "d", "shape": "disc"})
        return client

    def test_set_clear_and_plan(self, made):
        r = made.put("/api/design/shows/m/formations/0/motion",
                     json={"spin": 360, "move": [0, 0, 0], "grow": 1})
        assert r.status_code == 200, r.text
        assert r.json()["motion"]["spin"] == 360 and r.json()["min_hold_s"] > 0
        plan = made.get("/api/design/shows/m/plan").json()
        assert plan["formations"][0]["motion"]["spin"] == 360

        r = made.put("/api/design/shows/m/formations/0/motion", json={})
        assert r.json()["motion"] is None
        assert made.get("/api/design/shows/m/plan").json()["formations"][0]["motion"] is None

    def test_bad_motion_and_missing_formation(self, made):
        assert made.put("/api/design/shows/m/formations/0/motion",
                        json={"grow": 9}).status_code == 400
        assert made.put("/api/design/shows/m/formations/5/motion",
                        json={"spin": 90}).status_code == 404


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from skyshine import server

    monkeypatch.setattr(server, "SHOW_DIR", tmp_path / "shows")
    monkeypatch.setattr(server, "LOG_DIR", tmp_path / "logs")
    (tmp_path / "shows").mkdir()
    return TestClient(server.app)


def test_shortest_hold_sits_on_a_limit_without_crossing_it():
    """At exactly the hold the autotimer asks for, the motion touches its
    binding limit (speed or acceleration) and breaks neither."""
    f = moving_show().formations[1]
    lim = moving_show().limits
    f.hold = planning.motion_hold_need(f, lim)
    tau = np.linspace(0, f.hold, 4001)
    dt = tau[1] - tau[0]
    path = np.stack([f.pose(t) for t in tau])
    speed = float(np.max(np.linalg.norm(np.diff(path[..., :2], axis=0), axis=-1))) / dt
    accel = float(np.max(np.linalg.norm(np.diff(path, n=2, axis=0), axis=-1))) / dt**2
    k = lim.motion_speed_share
    assert speed <= lim.max_velocity_xy * k * 1.01
    assert accel <= lim.max_acceleration * k * 1.01
    assert math.isclose(speed, lim.max_velocity_xy * k, rel_tol=0.02) or \
        math.isclose(accel, lim.max_acceleration * k, rel_tol=0.02)


def test_matching_a_matched_show_changes_nothing():
    """Re-matching must be a no-op, or the preview draws other paths than
    the ones validated. A cube has many equally short assignments."""
    from skyshine.show import Formation as F

    n = 64
    show = Show(name="cube", drone_count=n)
    show.add(F(name="grid", points=geometry.lift(geometry.grid(n, 4.0), 30.0),
               colors=np.array([255, 255, 255])))
    show.add(F(name="cube", points=geometry.lift(geometry.cube_shell(n, 24.0), 30.0),
               colors=np.array([255, 0, 0])))
    once = planning.chain_match(show)
    twice = planning.chain_match(once)
    for a, b in zip(once.formations, twice.formations):
        assert np.array_equal(a.points, b.points), a.name
