"""The venue: a picture on the ground, a site model, and a zone around it."""

from __future__ import annotations

import io

import numpy as np
import pytest

from skyshine import models, planning
from skyshine.show import Show, show_from_dict, show_to_dict

# A 2 x 2 x 1 box in centimetres, z up, corner at its origin.
BOX = "\n".join(
    [f"v {x} {y} {z}" for x in (0, 200) for y in (0, 200) for z in (0, 100)]
    + ["f 1 2 4 3", "f 5 6 8 7", "f 1 2 6 5", "f 3 4 8 7", "f 1 3 7 5", "f 2 4 8 6"]
).encode()


def test_site_model_lands_in_show_metres():
    v, t = models.venue_mesh(BOX, ".obj", scale=0.01, up="z", rotation=90, offset=(10, 20, 0))
    assert len(t) == 12
    # 2 m square turned a quarter anticlockwise about the origin, then moved.
    assert np.allclose(v.min(axis=0), [8, 20, 0]) and np.allclose(v.max(axis=0), [10, 22, 1])


def test_big_models_are_thinned_for_display():
    rng = np.random.default_rng(0)
    pts = rng.random((3000, 3))
    obj = "\n".join(f"v {a} {b} {c}" for a, b, c in pts) + "\n" + "\n".join(
        f"f {i + 1} {i + 2} {i + 3}" for i in range(0, 2997, 3))
    _, t = models.venue_mesh(obj.encode(), ".obj", max_triangles=300)
    assert len(t) <= 300


def test_venue_survives_files_and_preparing():
    show = Show(name="v", drone_count=4, venue={"picture": {"asset_id": "abc", "width_m": 100}})
    assert show_from_dict(show_to_dict(show)).venue == show.venue
    assert "venue" not in show_to_dict(Show(name="w", drone_count=4))
    assert show.copy().venue == show.venue and show.copy().venue is not show.venue


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from skyshine import server

    monkeypatch.setattr(server, "SHOW_DIR", tmp_path / "shows")
    monkeypatch.setattr(server, "ASSET_DIR", tmp_path / "assets")
    (tmp_path / "shows").mkdir()
    c = TestClient(server.app)
    c.post("/api/design/shows", json={"name": "site", "drone_count": 20})
    return c


class TestApi:
    def test_ground_picture(self, client):
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (40, 20), (0, 120, 0)).save(buf, "PNG")
        asset = client.post("/api/design/assets",
                            files={"file": ("field.png", buf.getvalue(), "image/png")}).json()
        r = client.put("/api/design/shows/site/venue/picture",
                       json={"asset_id": asset["id"], "width_m": 250, "rotation": 15})
        assert r.status_code == 200 and r.json()["venue"]["picture"]["width_m"] == 250
        assert client.get(f"/api/design/assets/{asset['id']}/file").content == buf.getvalue()
        assert client.put("/api/design/shows/site/venue/picture",
                          json={"asset_id": "missing"}).status_code == 404
        client.put("/api/design/shows/site/venue/picture", json={"asset_id": None})
        assert "picture" not in client.get("/api/design/shows/site").json().get("venue", {})

    def test_site_model_and_its_keep_out_zone(self, client):
        r = client.post("/api/design/shows/site/venue/model",
                        files={"file": ("hall.obj", BOX, "text/plain")},
                        data={"scale": "0.05", "up": "z", "x": "40", "y": "30"})
        assert r.status_code == 200, r.text
        assert r.json()["size_m"] == [10.0, 10.0, 5.0]
        mesh = client.get("/api/design/shows/site/venue/mesh").json()
        v = np.array(mesh["vertices"]).reshape(-1, 3)
        assert np.allclose(v.min(axis=0), [40, 30, 0])

        zone = client.post("/api/design/shows/site/venue/keep-out").json()["zone"]
        d = np.hypot(v[:, 0] - zone["x"], v[:, 1] - zone["y"])
        assert (d <= zone["radius"] - 3 + 1e-6).all() and zone["top"] == pytest.approx(8.0)
        # Asking again replaces the zone rather than stacking copies.
        client.post("/api/design/shows/site/venue/keep-out")
        assert len(client.get("/api/design/shows/site").json()["limits"]["keep_out"]) == 1

        client.delete("/api/design/shows/site/venue/model")
        assert client.get("/api/design/shows/site/venue/mesh").status_code == 404

    def test_bad_models_are_refused(self, client):
        bad = client.post("/api/design/shows/site/venue/model",
                          files={"file": ("hall.fbx", b"Kaydara", "application/octet-stream")})
        assert bad.status_code == 400
        assert client.post("/api/design/shows/site/venue/keep-out").status_code == 404
