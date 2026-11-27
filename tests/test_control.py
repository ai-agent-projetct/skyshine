"""Tests for the control half: drivers, operational gates, API.

The point of these tests is not that the happy path works. It is that the
gates refuse: that a show which failed validation cannot be uploaded, that
a fleet with a flat battery cannot be armed, and that an airborne drone
cannot be disarmed. Those refusals are the product.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from skyshine import geometry, planning
from skyshine.drivers import (
    GUIDED_STREAM_MAX_DRONES,
    SHOW_CUSTOM_MODE,
    MavlinkGuidedDriver,
    MavlinkNativeDriver,
    SimDriver,
    SimFaults,
)
from skyshine.fleet import FleetManager, Phase, PreflightPolicy, rehearse
from skyshine.show import Formation, Show, ShowLimits
from skyshine.trajectory import build_mission


def flyable_show(n=20, name="ops test") -> Show:
    show = Show(
        name=name,
        drone_count=n,
        limits=ShowLimits(max_altitude=260.0, battery_endurance_s=1800.0),
    )
    show.origin.latitude = 12.9716
    show.origin.longitude = 77.5946
    show.origin.amsl = 920.0
    for i, pts in enumerate(
        [geometry.ball(n, 18), geometry.disc(n, 20), geometry.cube_shell(n, 24)]
    ):
        show.add(
            Formation(
                name=f"f{i}",
                points=geometry.lift(pts, 40.0),
                colors=np.array([255, 120, 30]),
                transition=8.0,
                hold=4.0,
            )
        )
    return planning.prepare(show).show


def broken_show(n=12) -> Show:
    """A show that genuinely fails validation: it busts the ceiling."""
    show = Show(
        name="broken",
        drone_count=n,
        limits=ShowLimits(max_altitude=25.0, battery_endurance_s=1800.0),
    )
    show.origin.amsl = 900.0
    for i in range(2):
        show.add(
            Formation(
                name=f"high{i}",
                points=geometry.lift(geometry.ball(n, 15), 120.0 + 10 * i),
                colors=np.array([10, 200, 255]),
                transition=10.0,
                hold=4.0,
            )
        )
    return show


class ManualClock:
    def __init__(self, start=1_700_000_000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# ==========================================================================
# Simulator
# ==========================================================================


class TestSimDriver:
    def test_connect_populates_a_healthy_fleet(self):
        d = SimDriver()
        d.connect(25)
        states = d.poll()
        assert len(states) == 25
        assert all(s.connected for s in states)
        assert all(s.gps_fix >= 5 for s in states)
        assert all(s.battery_percent == 100.0 for s in states)
        assert all(not s.armed for s in states)

    def test_arm_and_disarm(self):
        d = SimDriver()
        d.connect(5)
        d.arm([0, 1])
        assert [s.armed for s in d.poll()] == [True, True, False, False, False]
        d.disarm()
        assert not any(s.armed for s in d.poll())

    def test_noise_is_bounded_and_centred(self):
        d = SimDriver(faults=SimFaults(position_noise=0.05, wind=(0, 0, 0)), seed=3)
        d.connect(200)
        planned = np.zeros((200, 3))
        planned[:, 2] = 50.0
        d.advance(10.0, planned)
        devs = np.array([s.deviation for s in d.poll()])
        # Three axes at sigma=0.05 gives a typical radius near 0.087 m.
        assert devs.mean() < 0.2
        assert devs.max() < 0.6

    def test_injected_faults_fire_at_the_right_time(self):
        faults = SimFaults(
            link_loss={2: 5.0},
            gps_degradation={3: 7.0},
            weak_battery={4: 40.0},
            position_drift={5: 0.5},
        )
        d = SimDriver(faults=faults, battery_endurance_s=600.0)
        d.connect(8)
        planned = np.tile(np.array([0.0, 0.0, 50.0]), (8, 1))

        d.advance(1.0, planned)
        assert d.poll()[2].connected, "link loss fired before its time"
        assert d.poll()[3].gps_fix >= 5
        assert d.poll()[4].battery_percent < 100.0

        d.advance(6.0, planned)
        assert not d.poll()[2].connected
        assert d.poll()[3].gps_fix >= 5, "gps degradation fired early"

        d.advance(8.0, planned)
        assert d.poll()[3].gps_fix == 3
        assert d.poll()[5].deviation > 3.0

    def test_position_maps_to_sensible_coordinates(self):
        from skyshine.show import ShowOrigin

        d = SimDriver(faults=SimFaults(position_noise=0.0))
        d.connect(1)
        d.set_origin(ShowOrigin(latitude=12.9716, longitude=77.5946, amsl=920.0))
        d.advance(0.0, np.array([[0.0, 100.0, 50.0]]))
        s = d.poll()[0]
        assert s.latitude > 12.9716
        assert s.latitude == pytest.approx(12.9716 + 0.000899, abs=5e-6)
        assert s.altitude_agl == pytest.approx(50.0, abs=0.01)


class TestMavlinkDrivers:
    def test_guided_driver_refuses_a_show_sized_fleet(self):
        endpoints = [f"udpout:10.0.0.{i}:14550" for i in range(GUIDED_STREAM_MAX_DRONES + 1)]
        with pytest.raises(ValueError, match="ceiling for ground-streamed setpoints"):
            MavlinkGuidedDriver(endpoints)

    def test_guided_driver_accepts_a_small_fleet(self):
        driver = MavlinkGuidedDriver(["udpout:10.0.0.1:14550"])
        assert driver.streams_setpoints is True

    def test_native_driver_refuses_to_pretend_it_uploaded(self):
        """An upload to aircraft that are not there must fail, loudly.

        The transport is real MAVLink FTP (tests/test_sitl.py exercises it
        against aircraft that answer). With nothing listening, every
        transfer times out, and the one thing that must not happen is a
        report of success that an operator would fly on.
        """
        driver = MavlinkNativeDriver(["udpin:127.0.0.1:14894"])
        mission = build_mission(flyable_show(n=1))
        try:
            driver.connect(1)
            with pytest.raises(IOError, match="plan uploads failed"):
                driver.upload(mission)
            assert not any(s.mission_uploaded for s in driver.poll())
        finally:
            driver.close()

    def test_native_driver_uses_the_hook_and_records_the_checksum(self):
        seen = []
        driver = MavlinkNativeDriver(
            [f"udpout:10.0.0.{i}:14550" for i in range(6)],
            upload_hook=lambda link, index, plan: seen.append(index),
        )
        show = flyable_show(n=6)
        mission = build_mission(show)
        # Stand in for connect(), which would open real sockets.
        from skyshine.drivers import DroneState

        driver.links = [object()] * 6
        driver.states = [DroneState(index=i) for i in range(6)]
        driver.upload(mission)
        assert sorted(seen) == list(range(6))
        want = mission.to_dict()["checksum"]
        assert all(s.mission_checksum == want for s in driver.states)
        assert all(s.mission_uploaded for s in driver.states)

    def test_native_driver_takes_no_setpoints(self):
        driver = MavlinkNativeDriver(["udpout:10.0.0.1:14550"])
        with pytest.raises(NotImplementedError, match="onboard"):
            driver.setpoint(0, np.zeros(3), np.zeros(3))

    def test_emergency_rejects_an_unknown_action(self):
        driver = MavlinkNativeDriver(["udpout:10.0.0.1:14550"])
        with pytest.raises(ValueError, match="unknown emergency action"):
            driver.emergency("kill")

    def test_show_mode_number_is_the_expected_convention(self):
        # Shared with the firmware; a silent change here desynchronises the
        # ground station from every aircraft in the fleet.
        assert SHOW_CUSTOM_MODE == 127

    def test_enu_conversion_is_the_inverse_of_the_outgoing_one(self):
        from skyshine.show import ShowOrigin, enu_to_wgs84

        origin = ShowOrigin(latitude=12.9716, longitude=77.5946, orientation_deg=35.0)
        driver = MavlinkNativeDriver(["udpout:10.0.0.1:14550"], origin=origin)
        for xy in ([0.0, 0.0], [120.0, -40.0], [-75.0, 210.0]):
            lat, lon = enu_to_wgs84(xy, origin)
            back = driver._to_enu(lat, lon, 50.0)
            assert back[0] == pytest.approx(xy[0], abs=0.02)
            assert back[1] == pytest.approx(xy[1], abs=0.02)


# ==========================================================================
# Operational gates
# ==========================================================================


class TestGates:
    def test_upload_requires_validation_first(self):
        fleet = FleetManager(flyable_show(), clock=ManualClock())
        with pytest.raises(RuntimeError, match="validate"):
            fleet.upload()

    def test_upload_is_refused_when_safety_failed(self):
        fleet = FleetManager(broken_show(), clock=ManualClock())
        report = fleet.validate()
        assert not report.passed
        with pytest.raises(PermissionError, match="cannot be uploaded"):
            fleet.upload()
        assert fleet.phase == Phase.IDLE

    def test_override_is_possible_but_recorded(self):
        fleet = FleetManager(broken_show(), clock=ManualClock())
        fleet.validate()
        fleet.upload(override_reason="ground test on a tethered rig")
        assert fleet.phase == Phase.UPLOADED
        log = fleet.flight_log()
        assert log["overrides"] == ["ground test on a tethered rig"]
        assert any(e["kind"] == "override" for e in log["events"]), (
            "an override that is not in the flight log is not an override, "
            "it is a loophole"
        )

    def test_arm_is_refused_on_a_failing_preflight(self):
        show = flyable_show()
        driver = SimDriver(faults=SimFaults(weak_battery={3: 20.0}))
        fleet = FleetManager(show, driver=driver, clock=ManualClock())
        fleet.validate()
        fleet.upload()
        with pytest.raises(PermissionError, match="preflight failed"):
            fleet.arm()
        assert fleet.phase == Phase.UPLOADED

    def test_arm_override_is_recorded(self):
        show = flyable_show()
        driver = SimDriver(faults=SimFaults(weak_battery={3: 20.0}))
        fleet = FleetManager(show, driver=driver, clock=ManualClock())
        fleet.validate()
        fleet.upload()
        fleet.arm(override_reason="drone 3 is a spare, staying on the ground")
        assert fleet.phase == Phase.ARMED
        assert any(e.kind == "override" for e in fleet.events)

    def test_start_requires_an_armed_fleet(self):
        fleet = FleetManager(flyable_show(), clock=ManualClock())
        with pytest.raises(RuntimeError, match="arm the fleet first"):
            fleet.start()

    def test_start_rejects_a_time_in_the_past(self):
        clock = ManualClock()
        fleet = FleetManager(flyable_show(), clock=clock)
        fleet.validate()
        fleet.upload()
        fleet.arm()
        with pytest.raises(ValueError, match="in the past"):
            fleet.start(at_unix_time=clock.t - 10.0)

    def test_start_schedules_against_absolute_time(self):
        clock = ManualClock()
        fleet = FleetManager(flyable_show(), clock=clock)
        fleet.validate()
        fleet.upload()
        fleet.arm()
        when = fleet.start(lead_in=6.0)
        assert when == pytest.approx(clock.t + 6.0)
        # Before the instant arrives, the show clock stays at zero.
        clock.advance(2.0)
        fleet.tick()
        assert fleet.show_time == pytest.approx(0.0), (
            "the show must not creep forward before its scheduled start"
        )
        clock.advance(8.0)
        fleet.tick()
        assert fleet.show_time == pytest.approx(4.0)

    def test_preflight_catches_a_checksum_mismatch(self):
        show = flyable_show()
        fleet = FleetManager(show, clock=ManualClock())
        fleet.validate()
        fleet.upload()
        assert all(c.ok for c in fleet.preflight() if c.blocking)

        # Re-plan without re-uploading: the fleet now holds a stale plan.
        show.formations[3].points[:, 2] += 4.0
        fleet.timeline = __import__(
            "skyshine.trajectory", fromlist=["build_timeline"]
        ).build_timeline(show)
        fleet.validate()
        fleet.mission = build_mission(show)
        checks = {c.name: c for c in fleet.preflight()}
        assert not checks["plan checksums match"].ok, (
            "a drone holding yesterday's plan is the classic way a show goes "
            "wrong, and it must be caught on the ground"
        )

    def test_preflight_catches_a_drone_on_the_wrong_pad(self):
        show = flyable_show(n=16)
        driver = SimDriver(faults=SimFaults(position_noise=0.0))
        fleet = FleetManager(show, driver=driver, clock=ManualClock())
        assert {c.name: c.ok for c in fleet.preflight()}["launch positions"]

        # Swap two airframes on the ground.
        swapped = show.formations[0].points.copy()
        swapped[[0, 5]] = swapped[[5, 0]]
        driver.advance(0.0, swapped)
        checks = {c.name: c for c in fleet.preflight()}
        assert not checks["launch positions"].ok
        assert "swapped" in checks["launch positions"].detail

    def test_offline_drone_blocks_preflight(self):
        show = flyable_show()
        driver = SimDriver(faults=SimFaults(link_loss={4: 0.0}))
        fleet = FleetManager(show, driver=driver, clock=ManualClock())
        driver.advance(1.0, show.formations[0].points)
        checks = {c.name: c for c in fleet.preflight()}
        assert not checks["all drones online"].ok


class TestEmergency:
    def _running_fleet(self, **fault_kwargs):
        clock = ManualClock()
        show = flyable_show()
        driver = SimDriver(faults=SimFaults(**fault_kwargs))
        fleet = FleetManager(show, driver=driver, clock=clock)
        fleet.validate()
        fleet.upload()
        fleet.arm()
        fleet.start(lead_in=0.0)
        clock.advance(40.0)
        fleet.tick()
        return fleet, clock

    def test_there_is_no_kill_switch(self):
        fleet, _ = self._running_fleet()
        with pytest.raises(ValueError, match="unknown action"):
            fleet.emergency("kill")
        with pytest.raises(ValueError):
            fleet.emergency("motor_cut")

    def test_disarm_is_refused_for_airborne_drones(self):
        fleet, _ = self._running_fleet()
        airborne = [d for d in fleet.drones if d.altitude_agl > 0.5]
        assert airborne, "the fleet should be in the air 40 s into the show"
        with pytest.raises(PermissionError, match="refusing to disarm airborne"):
            fleet.emergency("disarm")

    def test_disarm_is_allowed_on_the_ground(self):
        show = flyable_show()
        fleet = FleetManager(show, clock=ManualClock())
        fleet.validate()
        fleet.upload()
        fleet.arm()
        fleet.emergency("disarm")
        assert not any(d.armed for d in fleet.drones)
        # Standing down before take-off is not an abort: the plan is still
        # on board, so the fleet can be armed again.
        assert fleet.phase == Phase.UPLOADED
        assert [e.kind for e in fleet.events][-1] == "disarm"
        assert not any(e.kind == "emergency" for e in fleet.events)

    def test_disarm_with_nothing_armed_is_not_an_abort(self):
        fleet = FleetManager(flyable_show(), clock=ManualClock())
        fleet.validate()
        fleet.upload()
        fleet.emergency("disarm")
        assert fleet.phase == Phase.UPLOADED

    def test_disarm_after_the_show_keeps_it_completed(self):
        from skyshine.reports import summarise

        fleet, clock = self._running_fleet()
        while fleet.phase == Phase.RUNNING:
            clock.advance(5.0)
            fleet.tick()
        assert fleet.phase == Phase.FINISHED
        fleet.emergency("disarm")
        assert fleet.phase == Phase.FINISHED
        assert summarise(fleet.flight_log())["outcome"] == "completed"

    @pytest.mark.parametrize("action", ["hold", "rth", "land"])
    def test_each_emergency_action_is_logged(self, action):
        fleet, _ = self._running_fleet()
        fleet.emergency(action, reason="test")
        events = [e for e in fleet.events if e.kind == "emergency"]
        assert events and events[-1].data["action"] == action
        assert fleet.phase == Phase.ABORTING

    def test_emergency_can_target_individual_drones(self):
        fleet, _ = self._running_fleet()
        before = fleet.phase
        fleet.emergency("land", indices=[2, 7], reason="low battery")
        # A per-drone action is not a fleet abort.
        assert fleet.phase == before
        assert fleet.drones[2].mode == "LAND"
        assert fleet.drones[0].mode != "LAND"


class TestWatchdog:
    def test_drift_is_graded_and_logged_once(self):
        clock = ManualClock()
        show = flyable_show()
        driver = SimDriver(
            faults=SimFaults(position_drift={5: 0.12}, position_noise=0.0)
        )
        fleet = FleetManager(show, driver=driver, clock=clock)
        fleet.validate()
        fleet.upload()
        fleet.arm()
        fleet.start(lead_in=0.0)

        for _ in range(900):
            fleet.tick()
            clock.advance(0.1)
            if fleet.phase != Phase.RUNNING:
                break

        kinds = {(e.data.get("drone"), e.data.get("condition")) for e in fleet.events if e.kind == "alert"}
        assert (5, "deviation_warn") in kinds
        assert (5, "deviation_abort") in kinds

        # Edge triggered: one event per condition, not one per tick.
        warn_events = [
            e
            for e in fleet.events
            if e.kind == "alert" and e.data.get("condition") == "deviation_warn"
        ]
        assert len(warn_events) <= 3, (
            f"{len(warn_events)} log lines for one condition; an operator cannot "
            "read a log that repeats itself ten times a second"
        )

    def test_auto_action_fires_once_per_drone(self):
        clock = ManualClock()
        show = flyable_show()
        driver = SimDriver(faults=SimFaults(position_drift={5: 0.3}, position_noise=0.0))
        policy = PreflightPolicy(auto_action="land", abort_deviation_m=2.0)
        fleet = FleetManager(show, driver=driver, policy=policy, clock=clock)
        fleet.validate()
        fleet.upload()
        fleet.arm()
        fleet.start(lead_in=0.0)
        for _ in range(400):
            fleet.tick()
            clock.advance(0.1)

        auto = [
            e
            for e in fleet.events
            if e.kind == "emergency" and e.data.get("reason") == "watchdog"
        ]
        assert len(auto) == 1, f"watchdog acted {len(auto)} times on one drone"

    def test_watchdog_is_quiet_before_the_show_runs(self):
        fleet = FleetManager(flyable_show(), clock=ManualClock())
        fleet.validate()
        assert fleet.tick()["alerts"] == []


class TestRehearsal:
    def test_clean_rehearsal_completes(self):
        fleet, summary = rehearse(flyable_show(n=24))
        assert summary["safety_passed"]
        assert summary["completed"]
        assert summary["issues"] == []
        assert summary["worst_deviation_m"] < 0.5
        assert fleet.phase == Phase.FINISHED

    def test_rehearsal_reports_every_injected_fault_once(self):
        show = flyable_show(n=24)
        driver = SimDriver(
            faults=SimFaults(
                link_loss={3: 30.0},
                gps_degradation={11: 50.0},
                position_drift={5: 0.1},
                wind=(4.0, 2.0, 0.0),
            ),
            battery_endurance_s=show.limits.battery_endurance_s,
        )
        _, summary = rehearse(show, driver=driver)
        kinds = {(i["drone"], i["kind"]) for i in summary["issues"]}
        assert (3, "link_lost") in kinds
        assert (11, "gnss_degraded") in kinds
        assert any(d == 5 and "deviation" in k for d, k in kinds)
        # Reported once each, in show-time order.
        times = [i["first_seen_s"] for i in summary["issues"]]
        assert times == sorted(times)

    def test_flight_log_is_complete_enough_to_audit(self):
        fleet, _ = rehearse(flyable_show(n=12))
        log = fleet.flight_log()
        for key in (
            "show", "drone_count", "driver", "start_unix", "mission_checksum",
            "safety", "overrides", "final_state", "events", "completed",
        ):
            assert key in log, f"flight log is missing {key}"
        assert log["completed"] is True
        assert log["safety"]["passed"] is True
        assert len(log["final_state"]) == 12
        kinds = {e["kind"] for e in log["events"]}
        assert {"fleet", "validate", "upload", "arm", "start", "finish"} <= kinds

    def test_flight_log_writes_and_reloads(self, tmp_path):
        import json

        fleet, _ = rehearse(flyable_show(n=8))
        path = fleet.save_flight_log(tmp_path / "flight.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["format"] == "skyshine.flightlog"
        assert data["drone_count"] == 8

    def test_pause_and_resume_hold_the_show_clock(self):
        clock = ManualClock()
        show = flyable_show(n=10)
        fleet = FleetManager(show, clock=clock)
        fleet.validate()
        fleet.upload()
        fleet.arm()
        fleet.start(lead_in=0.0)
        clock.advance(12.0)
        fleet.tick()
        at_pause = fleet.show_time
        fleet.pause()

        clock.advance(30.0)
        fleet.tick()
        assert fleet.show_time == pytest.approx(at_pause), "paused clock moved"

        fleet.resume()
        clock.advance(5.0)
        fleet.tick()
        assert fleet.show_time == pytest.approx(at_pause + 5.0, abs=1e-6)

    def test_pause_is_refused_when_the_aircraft_owns_the_clock(self):
        show = flyable_show(n=4)
        driver = MavlinkNativeDriver([f"udpout:10.0.0.{i}:14550" for i in range(4)])
        # Avoid opening sockets: stand in for connect().
        from skyshine.drivers import DroneState

        driver.connect = lambda count: (
            setattr(driver, "links", [object()] * count),
            setattr(driver, "states", [DroneState(index=i) for i in range(count)]),
        )
        fleet = FleetManager(show, driver=driver, clock=ManualClock())
        fleet.phase = Phase.RUNNING
        with pytest.raises(RuntimeError, match="cannot pause"):
            fleet.pause()


# ==========================================================================
# HTTP API
# ==========================================================================


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from skyshine import server

    monkeypatch.setattr(server, "SHOW_DIR", tmp_path / "shows")
    monkeypatch.setattr(server, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(server, "_fleet", None)
    (tmp_path / "shows").mkdir()
    return TestClient(server.app)


class TestApi:
    def test_shape_catalogue(self, client):
        body = client.get("/api/shapes").json()
        names = {s["name"] for s in body["shapes"]}
        assert {"ball", "heart", "star", "disc", "text"} <= names

    def test_design_flow_end_to_end(self, client):
        created = client.post(
            "/api/design/shows",
            json={
                "name": "api show",
                "drone_count": 30,
                "limits": {"min_separation": 3.0, "max_altitude": 200.0,
                           "battery_endurance_s": 1800.0},
                "origin": {"latitude": 12.97, "longitude": 77.59, "amsl": 920.0},
            },
        )
        assert created.status_code == 200, created.text
        hint = created.json()["capacity_hint"]
        assert hint["billboard_width_m"] > 0 and hint["fits_under_ceiling"]

        for shape, label in (("ball", "rise"), ("disc", "spin"), ("cube", "box")):
            r = client.post(
                "/api/design/shows/api show/formations",
                json={"name": label, "shape": shape, "params": {},
                      "color": [255, 80, 20], "altitude": 50.0,
                      "transition": 9.0, "hold": 5.0},
            )
            assert r.status_code == 200, r.text
            assert r.json()["raw_min_separation_m"] > 0

        assert len(client.get("/api/design/shows/api show").json()["formations"]) == 3

        prepared = client.post("/api/design/shows/api show/prepare").json()
        name = prepared["saved_as"]
        assert prepared["duration"] > 0
        assert len(prepared["launch_grid"]) == 30

        report = client.post(f"/api/design/shows/{name}/validate").json()
        assert report["passed"], report["violations"]
        assert report["metrics"]["min_separation_m"] >= 3.0

        plan = client.get(f"/api/design/shows/{name}/plan").json()
        assert plan["drone_count"] == 30
        assert len(plan["formations"][0]["points"]) == 30

        mission = client.post(f"/api/design/shows/{name}/mission").json()
        assert len(mission["checksum"]) == 64

    def test_mission_export_is_refused_for_an_unsafe_show(self, client):
        from skyshine.show import save_show

        from skyshine import server

        save_show(broken_show(), server.SHOW_DIR / "broken.show.json")
        r = client.post("/api/design/shows/broken/mission")
        assert r.status_code == 409
        assert "failed safety validation" in r.text

    def test_bad_shape_parameters_are_reported_not_swallowed(self, client):
        client.post(
            "/api/design/shows",
            json={"name": "p", "drone_count": 10,
                  "limits": {"max_altitude": 200.0}},
        )
        r = client.post(
            "/api/design/shows/p/formations",
            json={"name": "x", "shape": "ball", "params": {"nonsense": 1}},
        )
        assert r.status_code == 400
        assert "bad parameters" in r.text

        r = client.post(
            "/api/design/shows/p/formations",
            json={"name": "x", "shape": "not_a_shape", "params": {}},
        )
        assert r.status_code == 400
        assert "unknown shape" in r.text

    def test_missing_show_is_a_404(self, client):
        assert client.get("/api/design/shows/nope").status_code == 404
        assert client.post("/api/design/shows/nope/validate").status_code == 404

    def test_ops_requires_a_loaded_show(self, client):
        assert client.get("/api/ops/state").status_code == 409
        assert client.post("/api/ops/arm", json={}).status_code == 409

    def test_ops_launch_sequence(self, client):
        from skyshine.show import save_show

        from skyshine import server

        save_show(flyable_show(n=14, name="ops"), server.SHOW_DIR / "ops.show.json")

        assert client.post(
            "/api/ops/load", json={"show": "ops", "driver": "simulator"}
        ).status_code == 200

        pre = client.get("/api/ops/preflight").json()
        assert any(c["name"] == "safety validation" for c in pre["checks"])

        assert client.post("/api/ops/validate").json()["passed"]
        assert client.post("/api/ops/upload", json={}).status_code == 200
        assert client.post("/api/ops/arm", json={}).status_code == 200

        started = client.post("/api/ops/start", json={"lead_in": 3.0}).json()
        assert started["starts_in_s"] > 0

        state = client.get("/api/ops/state").json()
        assert state["phase"] == "running"
        assert state["drone_count"] == 14
        assert len(state["planned"]) == 14

        log = client.get("/api/ops/log").json()
        assert log["format"] == "skyshine.flightlog"

    def test_ops_upload_refused_for_unsafe_show_via_api(self, client):
        from skyshine.show import save_show

        from skyshine import server

        save_show(broken_show(), server.SHOW_DIR / "broken.show.json")
        client.post("/api/ops/load", json={"show": "broken", "driver": "simulator"})
        client.post("/api/ops/validate")
        r = client.post("/api/ops/upload", json={})
        assert r.status_code == 409
        assert "cannot be uploaded" in r.text

        ok = client.post(
            "/api/ops/upload", json={"override_reason": "bench test only"}
        )
        assert ok.status_code == 200

    def test_mavlink_driver_needs_endpoints(self, client):
        from skyshine.show import save_show

        from skyshine import server

        save_show(flyable_show(n=4, name="m"), server.SHOW_DIR / "m.show.json")
        r = client.post(
            "/api/ops/load", json={"show": "m", "driver": "mavlink-guided"}
        )
        assert r.status_code == 400
        assert "endpoint per drone" in r.text

    def test_unknown_driver_is_rejected(self, client):
        from skyshine.show import save_show

        from skyshine import server

        save_show(flyable_show(n=4, name="m"), server.SHOW_DIR / "m.show.json")
        r = client.post("/api/ops/load", json={"show": "m", "driver": "telepathy"})
        assert r.status_code == 400

    def test_emergency_endpoint_rejects_a_kill_request(self, client):
        from skyshine.show import save_show

        from skyshine import server

        save_show(flyable_show(n=6, name="e"), server.SHOW_DIR / "e.show.json")
        client.post("/api/ops/load", json={"show": "e", "driver": "simulator"})
        r = client.post("/api/ops/emergency", json={"action": "kill"})
        assert r.status_code == 409
        assert "unknown action" in r.text

    def test_loading_over_a_running_show_is_refused(self, client):
        from skyshine.show import save_show

        from skyshine import server

        save_show(flyable_show(n=6, name="a"), server.SHOW_DIR / "a.show.json")
        client.post("/api/ops/load", json={"show": "a", "driver": "simulator"})
        client.post("/api/ops/validate")
        client.post("/api/ops/upload", json={})
        client.post("/api/ops/arm", json={})
        r = client.post("/api/ops/load", json={"show": "a", "driver": "simulator"})
        assert r.status_code == 409
        assert "land or disarm" in r.text


class TestUiFaultPayload:
    """Regression: the UI's fault form broke every simulator load.

    The form gained a clock-error field the simulator did not know about,
    and POST /api/ops/load answered 500 for every show. The API tests never
    sent that key, which is why this is tested with the payload the browser
    actually builds.
    """

    UI_PAYLOAD = {
        "wind": [0, 0, 0],
        "position_noise": 0.03,
        "link_loss": {},
        "weak_battery": {},
        "gps_degradation": {},
        "position_drift": {},
        "clock_offset": {},
    }

    def _save(self, name="ui"):
        from skyshine import server
        from skyshine.show import save_show

        save_show(flyable_show(n=6, name=name), server.SHOW_DIR / f"{name}.show.json")

    def test_the_exact_ui_payload_loads(self, client):
        self._save()
        r = client.post("/api/ops/load", json={
            "show": "ui", "driver": "simulator", "faults": self.UI_PAYLOAD,
        })
        assert r.status_code == 200, r.text

    def test_string_keys_from_json_are_accepted(self, client):
        """JSON object keys are always strings; drones are indexed by int."""
        self._save()
        payload = dict(self.UI_PAYLOAD, link_loss={"3": 40}, weak_battery={"1": 55})
        r = client.post("/api/ops/load", json={
            "show": "ui", "driver": "simulator", "faults": payload,
        })
        assert r.status_code == 200, r.text

    def test_unknown_fault_is_a_400_naming_it(self, client):
        self._save()
        r = client.post("/api/ops/load", json={
            "show": "ui", "driver": "simulator", "faults": {"gremlins": 3},
        })
        assert r.status_code == 400
        assert "gremlins" in r.text

    def test_malformed_wind_is_a_400(self, client):
        self._save()
        r = client.post("/api/ops/load", json={
            "show": "ui", "driver": "simulator", "faults": {"wind": [1, 2]},
        })
        assert r.status_code == 400

    def test_clock_offset_fails_the_clock_check_on_the_simulator(self, client):
        """Rehearsable without SITL: right path, wrong time, caught on the ground."""
        self._save()
        client.post("/api/ops/load", json={
            "show": "ui", "driver": "simulator",
            "faults": dict(self.UI_PAYLOAD, clock_offset={"2": 0.4}),
        })
        checks = {c["name"]: c for c in client.get("/api/ops/preflight").json()["checks"]}
        clock = checks["clock synchronisation"]
        assert not clock["ok"] and clock["blocking"]
        assert "(2," in clock["detail"].replace(" ", "")


def test_loading_an_empty_show_is_a_400_not_a_500(client):
    """Regression: an empty show crashed /api/ops/load with an unhandled error."""
    client.post("/api/design/shows", json={"name": "empty", "drone_count": 5})
    r = client.post("/api/ops/load", json={"show": "empty", "driver": "simulator"})
    assert r.status_code == 400
    assert "no formations" in r.json()["detail"]


class TestMissionApproval:
    """One named sign-off for one exact plan, required before real aircraft."""

    def _save(self, show):
        from skyshine import server
        from skyshine.show import save_show

        save_show(show, server.SHOW_DIR / f"{show.name}.show.json")

    def test_approval_is_recorded_and_any_edit_voids_it(self, client):
        show = flyable_show(n=14, name="ops")
        self._save(show)
        assert not client.get("/api/ops/approvals/ops").json()["approved"]

        rec = client.post("/api/ops/approvals/ops", json={"note": "site A"}).json()
        assert rec["approval"]["approved_by"] == "local operator"
        assert client.get("/api/ops/approvals/ops").json()["approved"]

        show.formations[-1].hold += 1.0
        self._save(show)
        after = client.get("/api/ops/approvals/ops").json()
        assert not after["approved"] and after["stale"]

    def test_a_show_that_fails_safety_cannot_be_approved(self, client):
        self._save(broken_show())
        r = client.post("/api/ops/approvals/broken", json={})
        assert r.status_code == 409 and "cannot be approved" in r.text

    def test_real_aircraft_need_approval_and_no_override_gets_round_it(
        self, client, monkeypatch
    ):
        from skyshine import server

        self._save(flyable_show(n=14, name="ops"))
        client.post("/api/ops/load", json={"show": "ops", "driver": "simulator"})
        client.post("/api/ops/validate")
        # Stand in for hardware: the gate keys on the driver, not the wire.
        monkeypatch.setattr(server._fleet.driver, "name", "mavlink-native",
                            raising=False)

        for body in ({}, {"override_reason": "trust me"}):
            r = client.post("/api/ops/upload", json=body)
            assert r.status_code == 409 and "not approved" in r.text

        client.post("/api/ops/approvals/ops", json={})
        assert client.post("/api/ops/upload", json={}).status_code == 200
        # The sign-off travels into the flight log with the upload.
        assert any(e.kind == "approval" for e in server._fleet.events)

    def test_rehearsal_on_the_simulator_needs_no_approval(self, client):
        self._save(flyable_show(n=14, name="ops"))
        client.post("/api/ops/load", json={"show": "ops", "driver": "simulator"})
        client.post("/api/ops/validate")
        assert client.post("/api/ops/upload", json={}).status_code == 200

    def test_deleting_a_show_deletes_its_approval(self, client):
        from skyshine import server

        self._save(flyable_show(n=14, name="ops"))
        client.post("/api/ops/approvals/ops", json={})
        assert server._approval_path("ops").exists()
        client.delete("/api/design/shows/ops")
        assert not server._approval_path("ops").exists()


class TestUndoAndDuplicate:
    def _count(self, client, name="u"):
        return len(client.get(f"/api/design/shows/{name}").json()["formations"])

    def test_undo_and_redo_walk_the_edit_history(self, client):
        client.post("/api/design/shows", json={"name": "u", "drone_count": 20})
        assert client.post("/api/design/shows/u/undo").status_code == 409

        for shape in ("ring_stack", "sphere"):
            assert client.post("/api/design/shows/u/formations", json={
                "name": shape, "shape": shape}).status_code == 200
        assert self._count(client) == 2

        client.post("/api/design/shows/u/undo")
        assert self._count(client) == 1
        client.post("/api/design/shows/u/redo")
        assert self._count(client) == 2

        # A fresh edit after an undo drops the redo branch, as in any editor.
        client.post("/api/design/shows/u/undo")
        client.delete("/api/design/shows/u/formations/0")
        assert client.post("/api/design/shows/u/redo").status_code == 409
        assert self._count(client) == 0

    def test_duplicate_copies_the_show_and_refuses_to_overwrite(self, client):
        client.post("/api/design/shows", json={"name": "u", "drone_count": 20})
        client.post("/api/design/shows/u/formations", json={
            "name": "s", "shape": "sphere"})
        r = client.post("/api/design/shows/u/duplicate?save_as=u%20two").json()
        assert r["show"] == "u two" and self._count(client, "u two") == 1
        assert client.post(
            "/api/design/shows/u/duplicate?save_as=u%20two").status_code == 409


class TestSignedMission:
    """The checksum proves consistency; the signature proves who built it."""

    def _built(self, client):
        from skyshine import server
        from skyshine.show import save_show

        save_show(flyable_show(n=14, name="ops"), server.SHOW_DIR / "ops.show.json")
        r = client.post("/api/design/shows/ops/mission").json()
        assert r["signed_by"] == "local operator"
        return server.SHOW_DIR / "ops.mission.json"

    def _verify(self, client):
        return client.get("/api/design/shows/ops/mission/verify").json()

    def test_a_built_mission_verifies(self, client):
        self._built(client)
        r = self._verify(client)
        assert r["valid"] and r["detail"] == "signed by local operator"
        assert "value" not in r["signature"]

    def test_an_edited_plan_fails_even_with_its_checksum_repaired(self, client):
        from skyshine.trajectory import _checksum

        path = self._built(client)
        body = json.loads(path.read_text(encoding="utf-8"))
        body["plans"][0]["points"][1][3] += 5.0     # lift one drone 5 m
        path.write_text(json.dumps(body), encoding="utf-8")
        assert "changed after" in self._verify(client)["detail"]

        # The attacker who also recomputes the checksum still lacks the key.
        body["checksum"] = _checksum(
            {k: v for k, v in body.items() if k != "signature"})
        path.write_text(json.dumps(body), encoding="utf-8")
        r = self._verify(client)
        assert not r["valid"] and "signature does not match" in r["detail"]

    def test_the_signer_name_cannot_be_rewritten(self, client):
        path = self._built(client)
        body = json.loads(path.read_text(encoding="utf-8"))
        body["signature"]["signer"] = "someone else"
        path.write_text(json.dumps(body), encoding="utf-8")
        assert not self._verify(client)["valid"]

    def test_unsigned_other_key_and_missing_are_told_apart(self, client, monkeypatch):
        from skyshine import server

        assert client.get("/api/design/shows/ops/mission/verify").status_code == 404
        path = self._built(client)

        monkeypatch.setattr(server, "KEY_FILE", path.parent / "other.key")
        assert "different key" in self._verify(client)["detail"]

        body = json.loads(path.read_text(encoding="utf-8"))
        del body["signature"]
        path.write_text(json.dumps(body), encoding="utf-8")
        assert "not signed" in self._verify(client)["detail"]

    def test_a_corrupt_key_file_is_a_clear_503_not_a_crash(self, client):
        from skyshine import server

        self._built(client)
        server.KEY_FILE.write_text("not hex", encoding="ascii")
        r = client.get("/api/design/shows/ops/mission/verify")
        assert r.status_code == 503 and "not a usable signing key" in r.text


class TestTemplates:
    def test_catalogue_lists_every_template(self, client):
        from skyshine import templates

        rows = client.get("/api/design/templates").json()["templates"]
        assert [r["id"] for r in rows] == list(templates.TEMPLATES)
        assert all(r["title"] and r["summary"] and r["formations"] for r in rows)

    def test_creates_a_show_and_refuses_to_overwrite(self, client):
        body = {"name": "gala", "drone_count": 24, "text": "R & S"}
        r = client.post("/api/design/templates/wedding", json=body)
        assert r.status_code == 200, r.text
        assert r.json()["warnings"] == []

        show = client.get("/api/design/shows/gala").json()
        names = [f["name"] for f in show["formations"]]
        assert len(names) == r.json()["formations"] and "initials" in names
        assert show["drone_count"] == 24

        again = client.post("/api/design/templates/wedding", json=body)
        assert again.status_code == 409

    def test_unknown_template_and_bad_input_are_refused(self, client):
        body = {"name": "x", "drone_count": 10}
        assert client.post("/api/design/templates/nope", json=body).status_code == 404
        r = client.post("/api/design/templates/wedding",
                        json={**body, "limits": {"min_separation": -1}})
        assert r.status_code == 400 and "min_separation" in r.text
        assert client.get("/api/design/shows/x").status_code == 404
        assert client.post("/api/design/templates/wedding",
                           json={**body, "text": "x" * 41}).status_code == 422

    def test_a_fleet_too_big_for_the_ceiling_is_warned_about(self, client):
        r = client.post("/api/design/templates/showcase",
                        json={"name": "big", "drone_count": 600}).json()
        assert r["ok"] and any("ceiling" in w for w in r["warnings"])


class TestLinkSigning:
    def _ops(self):
        from skyshine import server
        from skyshine.show import save_show

        save_show(flyable_show(n=3, name="ops"), server.SHOW_DIR / "ops.show.json")

    def test_signed_sitl_load_and_provisioning(self, client):
        from skyshine import server

        self._ops()
        try:
            r = client.post("/api/ops/load",
                            json={"show": "ops", "driver": "sitl", "signed": True})
            assert r.status_code == 200, r.text
            assert client.get("/api/ops/state").json()["link_signed"] is True
            assert server._sitl.summary()["signed"] == 3
            assert server.LINK_KEY_FILE.exists()

            # Reload unsigned, then provision over the open link.
            client.post("/api/ops/load", json={"show": "ops", "driver": "sitl"})
            assert client.get("/api/ops/state").json()["link_signed"] is False
            r = client.post("/api/ops/signing/provision").json()
            assert r["ok"] and r["confirmed"] == [0, 1, 2]
            assert client.get("/api/ops/state").json()["link_signed"] is True
            assert any(e.kind == "signing" for e in server._fleet.events)
        finally:
            server._stop_sitl()

    def test_the_simulator_cannot_be_signed(self, client):
        self._ops()
        r = client.post("/api/ops/load",
                        json={"show": "ops", "driver": "simulator", "signed": True})
        assert r.status_code == 400
        client.post("/api/ops/load", json={"show": "ops", "driver": "simulator"})
        assert client.post("/api/ops/signing/provision").status_code == 400


class TestSimpleMode:
    def test_page_is_served_without_signing_in(self, client):
        from skyshine import auth

        r = client.get("/simple")
        assert r.status_code == 200 and "Make my show" in r.text
        assert auth.required_role("GET", "/simple") is None

    def test_replay_matches_the_prepared_show(self, client):
        import base64

        client.post("/api/design/templates/festival",
                     json={"name": "kid", "drone_count": 12})
        client.post("/api/design/shows/kid/prepare")
        r = client.get("/api/design/shows/kid prepared/replay?fps=2").json()
        pos = np.frombuffer(base64.b64decode(r["pos"]), "<i2").reshape(
            r["frames"], r["drones"], 3)
        col = np.frombuffer(base64.b64decode(r["col"]), "u1")
        assert r["drones"] == 12 and len(col) == r["frames"] * 12 * 3
        assert r["frames"] == int(r["duration"] * 2) + 1
        # Starts and ends on the ground, and climbs in between.
        assert pos[0, :, 2].max() == 0 and pos[-1, :, 2].max() == 0
        assert pos[:, :, 2].max() > 300  # decimetres
        assert [s["name"] for s in r["segments"]][:2] == ["ground", "climb"]

    def test_replay_of_an_empty_show_is_a_400(self, client):
        client.post("/api/design/shows", json={"name": "empty", "drone_count": 5})
        assert client.get("/api/design/shows/empty/replay").status_code == 400
