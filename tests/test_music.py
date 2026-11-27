"""The show's song: upload, serving, and staying in step when preparing."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from skyshine import geometry, planning
from skyshine.show import Formation, Music, Show


def click_track(bpm: float = 128.0, seconds: float = 8.0, rate: int = 8000) -> bytes:
    """A WAV with a short click on every beat."""
    t = np.arange(int(seconds * rate)) / rate
    sig = np.zeros_like(t)
    for beat in np.arange(0, seconds, 60.0 / bpm):
        i = int(beat * rate)
        sig[i:i + 80] = np.sin(2 * np.pi * 1000 * t[i:i + 80])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((sig * 30000).astype("<i2").tobytes())
    return buf.getvalue()


def test_preparing_keeps_the_song_in_step():
    n = 12
    show = Show(name="m", drone_count=n,
                music=Music(bpm=120, offset=3.0, beat_phase=0.25, cues=[(10.0, "drop")]))
    show.add(Formation(name="ring", points=geometry.lift(geometry.circle(n, 15.0), 40.0),
                       colors=np.array([255, 0, 0]), transition=6.0, hold=4.0))
    show.add(Formation(name="disc", points=geometry.lift(geometry.disc(n, 15.0), 40.0),
                       colors=np.array([0, 0, 255]), hold=4.0))
    prep = planning.prepare(show).show
    first = next(i for i, f in enumerate(prep.formations) if f.name == "ring")
    appears = sum(f.duration for f in prep.formations[:first]) + prep.formations[first].transition
    shift = appears - show.formations[0].transition
    assert shift > 0
    # The song plays the same moment when the first formation appears.
    assert prep.music.offset + appears == pytest.approx(show.music.offset + show.formations[0].transition)
    assert prep.music.beat_phase == pytest.approx(0.25 + shift)
    assert prep.music.cues == [(pytest.approx(10.0 + shift), "drop")]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from skyshine import server

    monkeypatch.setattr(server, "SHOW_DIR", tmp_path / "shows")
    monkeypatch.setattr(server, "ASSET_DIR", tmp_path / "assets")
    (tmp_path / "shows").mkdir()
    c = TestClient(server.app)
    c.post("/api/design/shows", json={"name": "s", "drone_count": 12})
    return c


class TestApi:
    def test_upload_and_play_back(self, client):
        data = click_track()
        r = client.post("/api/design/shows/s/audio", files={"file": ("beat.wav", data, "audio/wav")})
        assert r.status_code == 200, r.text
        assert r.json()["path"].startswith("audio/") and r.json()["path"].endswith(".wav")
        show = client.get("/api/design/shows/s").json()
        assert show["audio"] == "beat.wav" and show["music"]["path"] == r.json()["path"]
        got = client.get("/api/design/shows/s/audio")
        assert got.status_code == 200 and got.content == data
        assert got.headers["content-type"] == "audio/wav"

    @pytest.mark.parametrize("head,kind", [
        (b"ID3\x04" + b"\0" * 20, "mp3"), (b"\xff\xfb\x90" + b"\0" * 20, "mp3"),
        (b"OggS" + b"\0" * 20, "ogg"), (b"fLaC" + b"\0" * 20, "flac"),
        (b"\0\0\0\x20ftypM4A " + b"\0" * 20, "m4a"),
    ])
    def test_formats_are_recognised(self, head, kind):
        from skyshine import server

        assert server._audio_kind(head) == kind

    def test_non_audio_is_refused(self, client):
        r = client.post("/api/design/shows/s/audio", files={"file": ("x.mp3", b"hello", "audio/mpeg")})
        assert r.status_code == 400

    def test_only_uploaded_songs_are_served(self, client):
        assert client.get("/api/design/shows/s/audio").status_code == 404
        for path in ("../../users.json", "audio/../../users.json", "C:/Windows/win.ini"):
            client.put("/api/design/shows/s/music", json={"path": path})
            assert client.get("/api/design/shows/s/audio").status_code == 404, path
