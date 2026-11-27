"""Generated formations: QR code, countdown, scrolling text."""

from __future__ import annotations

import numpy as np
import pytest

from skyshine import generators, geometry

SPACING = 4.3


def scan(dark: np.ndarray) -> str:
    """Read a QR matrix the way a camera would, with OpenCV's decoder."""
    cv2 = pytest.importorskip("cv2")
    img = np.pad(np.where(dark, 0, 255).astype(np.uint8), 4, constant_values=255)
    img = cv2.resize(img, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
    return cv2.QRCodeDetector().detectAndDecode(img)[0]


def sky_image(f) -> np.ndarray:
    """What the crowd sees: True where a drone is lit, on the module grid."""
    lit = f.colors.sum(axis=1) > 0
    p = f.points[lit & (np.abs(f.points[:, 1]) < 1e-6)]
    cols = np.round((p[:, 0] - p[:, 0].min()) / SPACING).astype(int)
    rows = np.round((p[:, 2].max() - p[:, 2]) / SPACING).astype(int)
    img = np.zeros((rows.max() + 1, cols.max() + 1), dtype=bool)
    img[rows, cols] = True
    return img


class TestQr:
    @pytest.mark.parametrize("text", ["https://example.com", "SKY SHINE 2027",
                                      "https://skyshine.example/a/very/long/link?with=query", "x" * 120])
    @pytest.mark.parametrize("level", ["L", "M"])
    def test_codes_scan(self, text, level):
        assert scan(generators.qr_matrix(text, level)) == text

    def test_dark_lit_code_in_the_sky_scans_when_inverted(self):
        f = generators.qr_formation("https://example.com", 400, SPACING)
        assert f.count == 400 and geometry.min_pair_distance(f.points) >= SPACING - 1e-9
        assert scan(sky_image(f)) == "https://example.com"     # lit drones are the dark modules

    def test_flat_code_lies_level_and_reads_from_below(self):
        f = generators.qr_formation("https://example.com", 400, SPACING, flat=True)
        lit = f.colors.sum(axis=1) > 0
        assert np.ptp(f.points[lit, 2]) < 1e-9          # one altitude
        p = f.points[lit]
        cols = np.round((p[:, 0] - p[:, 0].min()) / SPACING).astype(int)
        rows = np.round((p[:, 1].max() - p[:, 1]) / SPACING).astype(int)   # north is up
        img = np.zeros((rows.max() + 1, cols.max() + 1), dtype=bool)
        img[rows, cols] = True
        assert scan(img) == "https://example.com"

    def test_light_lit_code_scans_as_is(self):
        need = int((~np.pad(~generators.qr_matrix("SKY"), 2, constant_values=True)).sum())
        f = generators.qr_formation("SKY", 700, SPACING, lit="light")
        assert scan(~sky_image(f)) == "SKY"
        assert need > 0

    def test_too_few_drones_and_too_much_text(self):
        with pytest.raises(ValueError, match="needs .* drones"):
            generators.qr_formation("https://example.com", 50, SPACING)
        with pytest.raises(ValueError, match="too long"):
            generators.qr_matrix("x" * 400)


class TestCountdown:
    def test_digits_change_every_second_and_drones_never_move(self):
        f = generators.countdown_formation(8, 70, SPACING)
        assert f.count == 70 and f.hold == 8.0
        assert geometry.min_pair_distance(f.points) >= SPACING - 1e-9
        digit = int((np.abs(f.points[:, 1]) < 1e-9).sum())   # the rest wait behind
        lit = lambda t: int((f.colors_at(t).sum(axis=1) > 0).sum())  # noqa: E731
        assert lit(0.0) == digit                 # "8" lights every segment
        assert lit(1.0) == digit * 3 // 7        # "7": three segments
        assert lit(7.0) == digit * 2 // 7        # "1": two segments

    def test_two_digit_countdown(self):
        f = generators.countdown_formation(12, 200, SPACING)
        assert f.count == 200 and len(f.color_keys) == 11
        assert geometry.min_pair_distance(f.points) >= SPACING - 1e-9

    def test_spare_drones_wait_behind_instead_of_making_digits_taller(self):
        f = generators.countdown_formation(10, 600, SPACING, max_height=60.0)
        assert f.count == 600 and geometry.min_pair_distance(f.points) >= SPACING - 1e-9
        assert np.ptp(f.points[:, 2]) <= 60.0 + 1e-6 or (f.points[:, 1] > 0).any()
        digits = f.points[np.abs(f.points[:, 1]) < 1e-9]
        assert np.ptp(digits[:, 2]) <= 60.0 + 1e-6
        assert (f.colors[f.points[:, 1] > 0] == 0).all()     # parked drones stay dark
        with pytest.raises(ValueError, match="at least 28 drones"):
            generators.countdown_formation(12, 20, SPACING)


class TestTicker:
    def test_text_enters_crosses_and_leaves(self):
        f = generators.ticker_formation("HI", 150, SPACING, rows=7, speed=5.0)
        assert f.count == 150
        assert geometry.min_pair_distance(f.points) >= SPACING - 1e-9
        lit = lambda t: int((f.colors_at(t).sum(axis=1) > 0).sum())  # noqa: E731
        assert lit(0.0) == 0 and lit(f.hold) == 0
        assert max(lit(t) for t in np.linspace(0, f.hold, 60)) > 10
        with pytest.raises(ValueError):
            generators.ticker_formation("HI", 20, SPACING)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from skyshine import server

    monkeypatch.setattr(server, "SHOW_DIR", tmp_path / "shows")
    (tmp_path / "shows").mkdir()
    c = TestClient(server.app)
    c.post("/api/design/shows", json={"name": "g", "drone_count": 400})
    return c


@pytest.mark.parametrize("body", [
    {"kind": "qr", "text": "https://example.com", "flat": True},
    {"kind": "countdown", "start": 10},
    {"kind": "ticker", "text": "HAPPY 2027"},
])
def test_generated_shows_prepare_and_pass(client, body):
    r = client.post("/api/design/shows/g/generate", json={**body, "altitude": 30})
    assert r.status_code == 200, r.text
    assert r.json()["warning"] is None
    prep = client.post("/api/design/shows/g/prepare").json()
    report = client.post(f"/api/design/shows/{prep['saved_as']}/validate").json()
    assert report["passed"], report["violations"][:3]


def test_unknown_generator_is_refused(client):
    assert client.post("/api/design/shows/g/generate", json={"kind": "spiral"}).status_code == 400
