"""Flight charts must show what the safety engine measured."""

from __future__ import annotations

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from skyshine import server

    monkeypatch.setattr(server, "SHOW_DIR", tmp_path / "shows")
    (tmp_path / "shows").mkdir()
    c = TestClient(server.app)
    c.post("/api/design/templates/festival", json={"name": "f", "drone_count": 30})
    prep = c.post("/api/design/shows/f/prepare").json()
    c.prepared = prep["saved_as"]
    return c


def test_chart_peaks_match_the_safety_report(client):
    name = client.prepared
    charts = client.get(f"/api/design/shows/{name}/charts?points=400").json()
    report = client.post(f"/api/design/shows/{name}/validate").json()
    m = report["metrics"]
    n = len(charts["t"])
    assert 10 <= n <= 400
    for key in ("altitude_max", "altitude_min", "speed_xy", "climb", "descent",
                "acceleration", "closest"):
        assert len(charts[key]) == n, key
    peak = lambda k: max(v for v in charts[k] if v is not None)  # noqa: E731
    assert peak("altitude_max") == pytest.approx(m["max_altitude_m"], abs=0.05)
    assert peak("speed_xy") == pytest.approx(m["peak_horizontal_speed_mps"], rel=0.02)
    assert peak("climb") == pytest.approx(m["peak_climb_rate_mps"], rel=0.02)
    assert peak("acceleration") == pytest.approx(m["peak_acceleration_mps2"], rel=0.05)
    # Sampled spacing can only be at or above the continuous sweep's minimum.
    closest = min(v for v in charts["closest"] if v is not None)
    assert m["min_separation_m"] - 1e-6 <= closest <= m["min_separation_m"] + 0.2
    assert charts["limits"]["min_separation"] == 3.0
    assert charts["formations"][0]["name"] == "ground"
    # Times run forward from zero to the end of the show.
    assert charts["t"][0] >= 0 and charts["t"] == sorted(charts["t"])


def test_empty_show_has_no_charts(client):
    client.post("/api/design/shows", json={"name": "e", "drone_count": 5})
    assert client.get("/api/design/shows/e/charts").status_code == 400
