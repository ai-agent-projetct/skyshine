"""Onboard shows on Skybrush's ArduCopter firmware: the .skyb file and the
commands that load and start it."""

from __future__ import annotations

import numpy as np
import pytest

from skyshine import drivers, planning, skybrush, templates
from skyshine.show import ShowOrigin
from skyshine.trajectory import build_mission, build_timeline


@pytest.fixture(scope="module")
def prepared():
    return planning.prepare(templates.build("showcase", "sb", 20)).show


def test_skyb_reproduces_the_validated_flight_and_lights(prepared):
    mission, tl = build_mission(prepared), build_timeline(prepared)
    for d in (0, 11, 19):
        plan = mission.plans[d]
        data = skybrush.skyb_bytes(plan, title="sb")
        assert data[:5] == b"skyb\x01"
        blocks = skybrush.read_skyb(data)
        segs = skybrush.trajectory_segments(blocks[skybrush.SKYB_TRAJECTORY])
        est = np.array([skybrush.skyb_position(segs, t * 1000) for t in tl.t]) / 1000.0
        assert np.max(np.linalg.norm(est - tl.pos[:, d], axis=1)) < 0.015     # 15 mm
        code = blocks[skybrush.SKYB_LIGHTS]
        assert code[-1] == skybrush.OP_END
        steps = dict(skybrush.run_light_program(code[:-1], 10**9))
        for t, *rgb in plan.lights:
            assert steps[round(t * skybrush.FPS)] == tuple(int(round(v)) for v in rgb)


def test_long_holds_are_split_into_segments_the_format_can_hold():
    class Plan:
        points = np.array([[0, 0, 0, 0], [5, 0, 0, 10], [200, 0, 0, 10], [205, 0, 0, 0]], float)
        easings = ["linear", "hold", "linear"]
        lights = np.zeros((0, 4))
    segs = skybrush.trajectory_segments(skybrush.read_skyb(skybrush.skyb_bytes(Plan()))[1])
    assert all(dur <= 65535 for _, dur, _ in segs)
    assert segs[-1][0] + segs[-1][1] == 205000
    assert skybrush.skyb_position(segs, 100000)[2] == pytest.approx(10000)


def test_gps_time_of_week():
    # 2023-11-14 22:13:20 UTC is Tuesday 22:13:38 in GPS time.
    assert drivers.gps_time_of_week(1700000000.25) == (2 * 86400 + 22 * 3600 + 13 * 60 + 38, 250)


def test_driver_uploads_reloads_and_starts(prepared, monkeypatch):
    d = drivers.MavlinkSkybrushDriver(["udpin:0.0.0.0:1"] * 2,
                                      origin=ShowOrigin(latitude=47.5, longitude=19.05,
                                                        orientation_deg=30, amsl=110.0))
    d.links = [object(), object()]
    d.states = [drivers.DroneState(index=i) for i in range(2)]
    calls = []
    monkeypatch.setattr(d, "ftp_write", lambda i, blob, path: calls.append(("ftp", i, path, blob[:4])))
    monkeypatch.setattr(d, "_command", lambda i, cmd, params, **kw: calls.append(("cmd", i, cmd, params, kw)))
    monkeypatch.setattr(d, "_set_param", lambda i, name, v: calls.append(("param", i, name, v)))
    mission = build_mission(prepared)
    mission.plans = mission.plans[:2]
    d.upload(mission)
    from pymavlink import mavutil
    assert calls[0] == ("ftp", 0, "/COLLMOT/show.skyb", b"skyb")
    assert calls[1][:4] == ("cmd", 0, mavutil.mavlink.MAV_CMD_USER_1, [0.0])
    assert all(s.mission_uploaded for s in d.states)
    calls.clear()
    d.start(1700000000.25)
    origin = calls[0]
    assert origin[2] == mavutil.mavlink.MAV_CMD_USER_2 and origin[3][3] == 120.0
    assert origin[4] == {"use_int": True, "x": 475000000, "y": 190500000, "z": 110000.0}
    params = {c[2]: c[3] for c in calls if c[0] == "param" and c[1] == 0}
    assert params == {"SHOW_START_TIME": 252818, "SHOW_START_MSEC": 250, "SHOW_START_AUTH": 1}


def test_a_failed_reload_fails_the_upload(prepared, monkeypatch):
    d = drivers.MavlinkSkybrushDriver(["udpin:0.0.0.0:1"])
    d.links, d.states = [object()], [drivers.DroneState(index=0)]
    monkeypatch.setattr(d, "ftp_write", lambda *a: None)

    def reject(*a, **k):
        raise IOError("drone 0 rejected command 31010 (result 4)")
    monkeypatch.setattr(d, "_command", reject)
    mission = build_mission(prepared)
    mission.plans = mission.plans[:1]
    with pytest.raises(IOError, match="mixed state"):
        d.upload(mission)
    assert not d.states[0].mission_uploaded
