"""AI panel: fix it for me, music to show, shape from words, free-form
commands. Claude itself is never called here; its two entry points are
replaced, and the tests check that whatever it says still goes through
the grammar and the safety checks."""

from __future__ import annotations

import pytest

from skyshine import assistant, planning, safety, templates


def _arrivals(prepared, design_count):
    out, cursor = [], 0.0
    for i, f in enumerate(prepared.formations):
        if 2 <= i < 2 + design_count:            # after the two take-off formations
            out.append(cursor + f.transition)
        cursor += f.duration
    return out


@pytest.mark.parametrize("beats", [2, 4, 8])
def test_music_sync_lands_every_formation_on_the_grid_after_prepare(beats):
    show = templates.build("showcase", "m", 40)
    show.music.bpm, show.music.beat_phase = 128.0, 0.3
    assistant.music_sync(show, beats=beats)
    prepared = planning.prepare(show).show
    step = 60.0 / 128.0 * beats
    for t in _arrivals(prepared, len(show.formations)):
        k = (t - prepared.music.beat_phase) / step
        assert abs(k - round(k)) < 1e-6
    assert safety.validate(prepared).passed


def test_music_sync_needs_a_tempo():
    show = templates.build("showcase", "m", 10)
    with pytest.raises(ValueError, match="tempo"):
        assistant.music_sync(show)


def test_fix_lowers_formations_under_a_lower_ceiling():
    show = templates.build("showcase", "f", 30)
    show.limits.max_altitude = 60.0
    assert not safety.validate(planning.prepare(show).show).passed
    r = assistant.fix(show)
    assert r["passed"] and any("lowered" in c for c in r["changes"])


def test_fix_shortens_holds_to_fit_the_battery():
    show = templates.build("showcase", "f", 30)
    usable = show.limits.battery_endurance_s - show.limits.battery_reserve_s
    for f in show.formations:
        f.hold = usable / len(show.formations)
    assert not safety.validate(planning.prepare(show).show).passed
    r = assistant.fix(show)
    assert r["passed"], r["left"]
    assert any("battery" in c for c in r["changes"])


def test_fix_moves_a_formation_out_of_a_keep_out_zone():
    show = templates.build("showcase", "f", 30)
    c = show.formations[2].points.mean(axis=0)
    show.limits.keep_out = [{"name": "tower", "x": float(c[0]), "y": float(c[1]),
                             "radius": 4.0, "bottom": 0.0, "top": 200.0}]
    r = assistant.fix(show)
    assert any("keep-out" in c for c in r["changes"])
    # Only formations move; a transition may still cross the tower, and
    # then the result says so instead of passing.
    fitted = show.formations[2].copy()
    planning.fit_formation(fitted, show.limits, show.fps)
    gap = ((fitted.points[:, :2] - c[:2]) ** 2).sum(axis=1) ** 0.5
    assert gap.min() > 4.0 + show.limits.min_separation
    assert r["passed"] or r["left"]


def test_fix_says_when_a_shape_cannot_fit():
    show = templates.build("showcase", "f", 30)
    show.limits.max_altitude = 25.0
    r = assistant.fix(show)
    assert not r["passed"] and any("too tall" in c for c in r["changes"])


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from skyshine import server

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(server, "SHOW_DIR", tmp_path / "shows")
    (tmp_path / "shows").mkdir()
    c = TestClient(server.app)
    assert c.post("/api/design/templates/showcase",
                  json={"name": "a", "drone_count": 30}).status_code == 200
    return c


def test_fix_and_music_endpoints_and_undo(client):
    assert client.get("/api/ai/status").json()["claude"] is False
    r = client.post("/api/design/shows/a/ai/music", json={"beats": 4})
    assert r.status_code == 400 and "tempo" in r.json()["detail"]
    client.put("/api/design/shows/a/music", json={"bpm": 120})
    r = client.post("/api/design/shows/a/ai/music", json={"beats": 4})
    assert r.status_code == 200, r.text
    client.patch("/api/design/shows/a/limits", json={"max_altitude": 60})
    r = client.post("/api/design/shows/a/ai/fix").json()
    assert r["passed"] and r["changes"]
    assert client.post("/api/design/shows/a/undo").status_code == 200


def test_shape_from_words_uses_built_in_shapes_then_text(client):
    r = client.post("/api/design/shows/a/ai/shape",
                    json={"description": "a big heart", "color": [255, 0, 0]}).json()
    assert r["source"] == "built-in heart"
    r = client.post("/api/design/shows/a/ai/shape",
                    json={"description": "a dolphin"}).json()
    assert r["source"] == "written as text" and "no Claude key" in r["note"]


def test_shape_from_words_uses_claudes_drawing(client, monkeypatch):
    monkeypatch.setattr(assistant, "available", lambda: True)
    monkeypatch.setattr(assistant, "shape_svg", lambda d: (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<path d="M10 10 L90 10 L50 90 Z"/></svg>'))
    r = client.post("/api/design/shows/a/ai/shape", json={"description": "a dolphin"})
    assert r.status_code == 200 and r.json()["source"] == "drawn by Claude"


def test_free_form_commands_go_through_the_grammar(client, monkeypatch):
    monkeypatch.setattr(assistant, "available", lambda: True)
    monkeypatch.setattr(assistant, "translate",
                        lambda text: ["add a red heart at 60 metres"])
    before = len(client.get("/api/design/shows/a").json()["formations"])
    r = client.post("/api/voice", json={"text": "something lovely please",
                                        "context": {"show": "a"}}).json()
    assert r["translated"] == ["add a red heart at 60 metres"] and r["executed"]
    assert len(client.get("/api/design/shows/a").json()["formations"]) == before + 1


def test_claude_cannot_reach_refused_commands(client, monkeypatch):
    monkeypatch.setattr(assistant, "available", lambda: True)
    monkeypatch.setattr(assistant, "translate", lambda text: ["arm the drones"])
    r = client.post("/api/voice", json={"text": "get them ready to go",
                                        "context": {"show": "a"}}).json()
    assert r.get("refused") and not r["executed"]


def test_claude_failure_changes_nothing(client, monkeypatch):
    monkeypatch.setattr(assistant, "available", lambda: True)

    def boom(text):
        raise RuntimeError("offline")
    monkeypatch.setattr(assistant, "translate", boom)
    r = client.post("/api/voice", json={"text": "blorp", "context": {"show": "a"}}).json()
    assert not r["executed"] and "offline" in r["speech"]


def test_claude_key_is_saved_never_returned_and_removable(client):
    from skyshine import assistant

    assert client.post("/api/ai/key", json={"key": "hello"}).status_code == 400
    key = "sk-ant-test-" + "x" * 30
    r = client.post("/api/ai/key", json={"key": key})
    assert r.status_code == 200 and key not in r.text
    assert r.json() == {**r.json(), "claude": True, "key_source": "saved"}
    assert assistant.KEY_FILE.read_text() == key
    assert key not in client.get("/api/ai/status").text
    assert client.delete("/api/ai/key").json()["claude"] is False
    assert not assistant.KEY_FILE.exists()
