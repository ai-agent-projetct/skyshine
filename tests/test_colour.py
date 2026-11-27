"""Colour tools: painting picked drones, masked effects, the colour volume,
and colouring a formation from a picture."""

from __future__ import annotations

import io

import numpy as np
import pytest

from skyshine import effects, geometry
from skyshine.show import Formation

RED = [255, 0, 0]


def wall(n=36):
    return Formation(name="wall", points=geometry.lift(geometry.grid(n, 4.0, upright=True), 30.0),
                     colors=np.array([0, 0, 255]), hold=6.0)


class TestVolume:
    def test_keep_paints_the_whole_shape_by_the_end(self):
        f = wall()
        fn = effects.volume(RED, axis="x", radius=1.0, duration=4.0, keep=True,
                            background=f.colors)
        start, end = fn(f.points, 0.0), fn(f.points, 4.0)
        assert (start == [0, 0, 255]).all()          # the ball starts outside
        assert (end == RED).all()                    # and has touched everyone

    def test_without_keep_only_drones_inside_light(self):
        f = wall()
        fn = effects.volume(RED, axis="x", radius=0.3, duration=4.0, keep=False)
        mid = fn(f.points, 2.0)
        lit = (mid == RED).all(axis=1)
        assert 0 < lit.sum() < len(lit)
        assert (mid[~lit] == 0).all()               # black background

    def test_the_sweep_moves_along_its_axis(self):
        f = wall()
        fn = effects.volume(RED, axis="x", radius=0.3, duration=4.0, keep=False)
        x = f.points[:, 0]
        early = x[(fn(f.points, 1.2) == RED).all(axis=1)].mean()
        late = x[(fn(f.points, 2.8) == RED).all(axis=1)].mean()
        assert early < late

    @pytest.mark.parametrize("bad", [{"axis": "radial"}, {"radius": 2.0}, {"duration": 0}])
    def test_bad_settings_are_refused(self, bad):
        with pytest.raises(ValueError):
            effects.volume(RED, **bad)


def test_masked_effect_leaves_other_drones_alone():
    old = wall()
    effects.render(old, effects.make("pulse", color=[0, 255, 0], period=2.0))
    new = old.copy()
    effects.render(new, effects.make("solid", color=RED))
    mask = np.zeros(old.count, dtype=bool)
    mask[:5] = True
    effects.keep_unmasked(old, new, mask)
    for t in (0.0, 0.7, 1.9, 5.5):
        got, was = new.colors_at(t), old.colors_at(t)
        assert (got[:5] == RED).all()
        assert np.array_equal(got[5:], was[5:])


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from skyshine import server

    monkeypatch.setattr(server, "SHOW_DIR", tmp_path / "shows")
    monkeypatch.setattr(server, "ASSET_DIR", tmp_path / "assets")
    (tmp_path / "shows").mkdir()
    c = TestClient(server.app)
    c.post("/api/design/shows", json={"name": "c", "drone_count": 36})
    c.post("/api/design/shows/c/formations", json={
        "name": "wall", "shape": "grid", "params": {"spacing": 4, "upright": True},
        "color": [0, 0, 255], "hold": 6})
    return c


def formation(client):
    return client.get("/api/design/shows/c").json()["formations"][0]


class TestApi:
    def test_paint_picked(self, client):
        pts = formation(client)["points"]
        r = client.post("/api/design/shows/c/formations/0/paint",
                        json={"near": pts[:3], "color": RED})
        assert r.status_code == 200 and r.json()["painted"] == 3
        cols = formation(client)["colors"]
        assert cols[:3] == [RED] * 3 and cols[3] == [0, 0, 255]

    def test_effect_on_picked_drones_only(self, client):
        pts = formation(client)["points"]
        r = client.post("/api/design/shows/c/formations/0/effect",
                        json={"effect": "solid", "params": {"color": RED}, "near": pts[:4]})
        assert r.status_code == 200, r.text
        cols = formation(client)["colors"]
        assert cols[:4] == [RED] * 4 and all(c == [0, 0, 255] for c in cols[4:])

    def test_volume_paints_over_the_current_colours(self, client):
        r = client.post("/api/design/shows/c/formations/0/effect", json={
            "effect": "volume", "params": {"color": RED, "radius": 0.2, "duration": 6}})
        assert r.status_code == 200, r.text
        f = formation(client)
        # Before the ball arrives the wall still shows its own blue.
        assert all(c == [0, 0, 255] for c in f["colors"])
        # A small ball on a straight line paints a stripe through the middle
        # and leaves the rest of the wall blue.
        end = np.array(f["color_keys"][-1][1])
        z = np.array(f["points"])[:, 2]
        stripe = np.abs(z - (z.min() + z.max()) / 2) <= 0.2 * np.ptp(z) / 2 + 1e-6
        assert (end[stripe] == RED).all() and (end[~stripe] == [0, 0, 255]).all()

    def test_colour_from_picture(self, client):
        from PIL import Image

        img = Image.new("RGB", (40, 40), (0, 0, 0))
        img.paste((255, 255, 0), (0, 0, 40, 20))        # top half yellow
        buf = io.BytesIO()
        img.save(buf, "PNG")
        up = client.post("/api/design/assets", files={"file": ("half.png", buf.getvalue(), "image/png")})
        assert up.status_code == 200, up.text
        r = client.post("/api/design/shows/c/formations/0/picture",
                        json={"asset_id": up.json()["id"], "view": "front"})
        assert r.status_code == 200, r.text
        f = formation(client)
        z = np.array(f["points"])[:, 2]
        cols = np.array(f["colors"])
        assert (cols[z > np.median(z) + 1] == [255, 255, 0]).all()   # top is yellow
        assert (cols[z < np.median(z) - 1] == [0, 0, 0]).all()       # bottom is dark
        assert client.post("/api/design/shows/c/formations/0/picture",
                           json={"asset_id": "nope"}).status_code == 404
