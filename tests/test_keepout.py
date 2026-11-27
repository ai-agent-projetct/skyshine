"""Keep-out zones, editable flight limits, and formations using fewer drones."""

from __future__ import annotations

import numpy as np
import pytest

from skyshine import geometry, planning, safety
from skyshine.show import Formation, Show, ShowLimits


def two_walls(limits: ShowLimits) -> Show:
    """A wall west of the origin, then the same wall east of it."""
    n = 16
    wall = geometry.grid(n, 5.0, upright=True)
    show = Show(name="ko", drone_count=n, limits=limits)
    show.add(Formation(name="west", points=geometry.lift(wall, 30.0) - [60, 0, 0],
                       colors=np.array([255, 255, 255])))
    show.add(Formation(name="east", points=geometry.lift(wall, 30.0) + [60, 0, 0],
                       colors=np.array([255, 255, 255])))
    return show


def errors(show, kind="keep_out"):
    report = safety.validate(planning.prepare(show).show)
    return [v for v in report.errors if v.kind == kind]


class TestKeepOut:
    def test_a_zone_the_show_flies_through_is_refused(self):
        zone = {"name": "crane", "x": 0, "y": 0, "radius": 6, "bottom": 0, "top": 80}
        found = errors(two_walls(ShowLimits(keep_out=[zone])))
        assert found and "crane" in found[0].message and found[0].drones

    def test_a_zone_off_to_the_side_is_fine(self):
        zone = {"name": "tree", "x": 0, "y": 200, "radius": 6, "bottom": 0, "top": 20}
        assert errors(two_walls(ShowLimits(keep_out=[zone]))) == []

    def test_a_zone_below_the_flight_is_fine_but_its_top_counts(self):
        # Right under the west wall's path east, well clear of the launch pad.
        low = {"name": "stage", "x": -30, "y": 0, "radius": 5, "bottom": 0, "top": 8}
        tall = {**low, "top": 60}
        assert errors(two_walls(ShowLimits(keep_out=[low]))) == []
        assert errors(two_walls(ShowLimits(keep_out=[tall])))

    @pytest.mark.parametrize("zone", [
        {"x": 0, "y": 0, "radius": 0, "bottom": 0, "top": 10},
        {"x": 0, "y": 0, "radius": 5, "bottom": 10, "top": 5},
        {"x": 0, "radius": 5, "bottom": 0, "top": 5},
    ])
    def test_malformed_zones_are_refused(self, zone):
        with pytest.raises(ValueError, match="keep-out zone"):
            ShowLimits(keep_out=[zone]).validate()


def test_spare_drones_park_unlit_behind_the_shape():
    shape = geometry.lift(geometry.circle(10, 15.0), 30.0)
    pts = planning.park_spares(shape, 30, 4.3)
    assert pts.shape == (30, 3)
    parked = pts[10:]
    assert parked[:, 1].min() >= shape[:, 1].max() + 2 * 4.3 - 1e-9     # behind
    assert parked[:, 2].min() >= shape[:, 2].min() - 1e-9               # not lower
    assert geometry.min_pair_distance(parked) >= 4.3 - 1e-9


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from skyshine import server

    monkeypatch.setattr(server, "SHOW_DIR", tmp_path / "shows")
    (tmp_path / "shows").mkdir()
    c = TestClient(server.app)
    c.post("/api/design/shows", json={"name": "k", "drone_count": 30})
    return c


class TestApi:
    def test_patch_changes_only_what_is_sent(self, client):
        before = client.get("/api/design/shows/k").json()["limits"]
        r = client.patch("/api/design/shows/k/limits",
                         json={"max_velocity_xy": 5.5, "min_separation": 2.0})
        assert r.status_code == 200, r.text
        after = r.json()["limits"]
        assert after["max_velocity_xy"] == 5.5 and after["min_separation"] == 2.0
        assert after["formation_separation"] == pytest.approx(2.0 * 2 ** 0.5)  # re-derived
        assert after["pyro_min_altitude"] == before["pyro_min_altitude"]       # untouched
        assert client.patch("/api/design/shows/k/limits",
                            json={"geofence": []}).status_code == 400
        assert client.patch("/api/design/shows/k/limits",
                            json={"keep_out": [{"x": 0}]}).status_code == 400

    def test_zone_round_trip_and_validation(self, client):
        client.post("/api/design/shows/k/formations", json={"name": "ring", "shape": "circle",
                                                            "params": {"radius": 15}})
        zone = {"name": "mast", "x": 15, "y": 0, "radius": 3, "bottom": 0, "top": 200}
        client.patch("/api/design/shows/k/limits", json={"keep_out": [zone]})
        assert client.get("/api/design/shows/k/plan").json()["limits"]["keep_out"] == [zone]
        prep = client.post("/api/design/shows/k/prepare").json()
        report = client.post(f"/api/design/shows/{prep['saved_as']}/validate").json()
        assert any(v["kind"] == "keep_out" for v in report["violations"])

    def test_formation_using_fewer_drones(self, client):
        r = client.post("/api/design/shows/k/formations", json={
            "name": "small heart", "shape": "heart", "params": {"solid": True},
            "drones": 12, "color": [255, 0, 0]})
        assert r.status_code == 200, r.text
        f = client.get("/api/design/shows/k").json()["formations"][0]
        lit = [c for c in f["colors"] if c != [0, 0, 0]]
        assert len(f["points"]) == 30 and len(lit) == 12
        prep = client.post("/api/design/shows/k/prepare").json()
        report = client.post(f"/api/design/shows/{prep['saved_as']}/validate").json()
        assert report["passed"], report["violations"][:3]
        assert client.post("/api/design/shows/k/formations", json={
            "name": "x", "shape": "disc", "drones": 31}).status_code == 400
