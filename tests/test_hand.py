"""Hand placement: drawn formations, nudged drones, swaps and order locks."""

from __future__ import annotations

import numpy as np
import pytest

from skyshine import geometry, models, planning
from skyshine.show import Formation, Show


def test_strokes_become_an_upright_picture():
    strokes = [[[0, 0], [100, 0]], [[0, 50], [100, 50]]]   # two lines, y down
    pts = models.place_strokes(strokes, 20, size=40.0)
    assert pts.shape == (20, 3) and np.allclose(pts[:, 1], 0.0)
    assert np.ptp(pts[:, 0]) == pytest.approx(40.0)
    # Screen y grows downwards, so the first line drawn is the top one.
    top = pts[:, 2] > 0
    assert 8 <= top.sum() <= 12
    with pytest.raises(ValueError, match="draw at least one line"):
        models.place_strokes([[[1, 1]]], 5)


def test_a_locked_formation_keeps_its_pairing_through_renumbering():
    """Stored slot j of the previous formation always flies to stored slot j
    of a keep_order formation, however preparing renumbers things."""
    rng = np.random.default_rng(3)
    n = 30
    a = geometry.lift(geometry.grid(n, 5.0, upright=True), 30.0)
    b = geometry.lift(geometry.disc(n, 20.0), 30.0)[rng.permutation(n)]
    c = geometry.lift(geometry.ball(n, 15.0), 30.0)[rng.permutation(n)]
    show = Show(name="lock", drone_count=n)
    for name, p in (("a", a), ("b", b), ("c", c)):
        show.add(Formation(name=name, points=p, colors=np.array([255, 255, 255])))
    show.formations[2].keep_order = True

    out = planning.chain_match(show)
    for d in range(n):
        s1 = int(np.argmin(np.linalg.norm(b - out.formations[1].points[d], axis=1)))
        assert np.allclose(out.formations[2].points[d], c[s1])


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from skyshine import server

    monkeypatch.setattr(server, "SHOW_DIR", tmp_path / "shows")
    (tmp_path / "shows").mkdir()
    c = TestClient(server.app)
    c.post("/api/design/shows", json={"name": "h", "drone_count": 24})
    c.post("/api/design/shows/h/formations", json={"name": "ring", "shape": "circle",
                                                    "params": {"radius": 20}})
    c.post("/api/design/shows/h/formations", json={"name": "grid", "shape": "grid",
                                                    "params": {"spacing": 6, "upright": True}})
    return c


def points(client, i):
    return np.array(client.get("/api/design/shows/h").json()["formations"][i]["points"])


class TestApi:
    def test_draw(self, client):
        r = client.post("/api/design/shows/h/draw", json={
            "strokes": [[[0, 0], [50, 80], [100, 0]]], "color": [255, 0, 0], "size": 30})
        assert r.status_code == 200, r.text
        f = client.get("/api/design/shows/h").json()["formations"][2]
        assert f["name"] == "drawing" and len(f["points"]) == 24
        assert client.post("/api/design/shows/h/draw", json={"strokes": []}).status_code == 400

    def test_nudge_moves_only_the_picked_drones(self, client):
        before = points(client, 1)
        r = client.post("/api/design/shows/h/formations/1/nudge",
                        json={"near": before[:2].tolist(), "move": [0, 0, 3]})
        assert r.status_code == 200 and r.json()["moved"] == 2
        after = points(client, 1)
        assert np.allclose(after[:2, 2], before[:2, 2] + 3)
        assert np.allclose(after[2:], before[2:], atol=1e-3)
        far = client.post("/api/design/shows/h/formations/1/nudge",
                          json={"near": [[999, 999, 999]], "move": [1, 0, 0]})
        assert far.status_code == 400 and "no drone within 2 m" in far.text

    def test_swap_keeps_the_picture_and_locks_the_order(self, client):
        grid = points(client, 1)
        pick = [grid[0].tolist(), grid[5].tolist()]
        r = client.post("/api/design/shows/h/formations/1/swap", json={"near": pick})
        assert r.status_code == 200 and r.json()["keep_order"]
        f = client.get("/api/design/shows/h").json()["formations"][1]
        assert f["keep_order"] is True
        # Same set of positions: the picture is unchanged.
        after = np.array(f["points"])
        assert np.allclose(np.sort(after, axis=0), np.sort(grid, axis=0), atol=1e-3)
        prep = client.post("/api/design/shows/h/prepare")
        assert prep.status_code == 200, prep.text

    def test_lock_and_its_refusals(self, client):
        r = client.put("/api/design/shows/h/formations/1/order", json={"keep_order": True})
        assert r.status_code == 200
        assert client.get("/api/design/shows/h/plan").json()["formations"][1]["keep_order"]
        assert client.put("/api/design/shows/h/formations/0/order",
                          json={"keep_order": True}).status_code == 400
        one = points(client, 1)[0].tolist()
        assert client.post("/api/design/shows/h/formations/1/swap",
                           json={"near": [one]}).status_code == 400
        assert client.post("/api/design/shows/h/formations/1/swap",
                           json={"near": [one, one]}).status_code == 400
