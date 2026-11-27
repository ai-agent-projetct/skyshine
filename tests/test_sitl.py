"""Tests for the MAVLink wire path, against aircraft on real sockets.

These are the tests the pure-Python SimDriver cannot provide: nothing in
test_control.py ever serialises a MAVLink frame, so a mis-encoded command,
a wrong frame convention or a broken FTP transfer would pass everything
there and fail on the flight line.

They are slower than the rest of the suite because they wait for real
timers and real packets. That is the point - a protocol test that does not
use the protocol is a mock test.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from skyshine import geometry, planning
from skyshine.drivers import (
    COPTER_MODE_NAMES,
    SHOW_CUSTOM_MODE,
    SHOW_PLAN_PATH,
    MavlinkGuidedDriver,
    MavlinkNativeDriver,
)
from skyshine.fleet import FleetManager, Phase, PreflightPolicy
from skyshine.show import Formation, Show, ShowLimits
from skyshine.sitl import (
    AircraftFaults,
    ArduPilotSITL,
    VirtualAircraft,
    VirtualFleet,
    decode_ftp,
    encode_ftp,
    sitl_available,
)
from skyshine.trajectory import build_mission

CONNECT_TIMEOUT = 25.0


def wire_show(n: int = 4, name: str = "wire") -> Show:
    show = Show(
        name=name,
        drone_count=n,
        limits=ShowLimits(max_altitude=220.0, battery_endurance_s=1800.0),
    )
    show.origin.latitude = 12.9716
    show.origin.longitude = 77.5946
    show.origin.amsl = 920.0
    for label, pts in (("ball", geometry.ball(n, 12)), ("disc", geometry.disc(n, 14))):
        show.add(
            Formation(
                name=label,
                points=geometry.lift(pts, 30.0),
                colors=np.array([255, 60, 20]),
                transition=6.0,
                hold=3.0,
            )
        )
    return planning.prepare(show).show


@pytest.fixture()
def wired():
    """A virtual fleet plus a connected native driver, torn down cleanly."""
    created: list = []

    def build(show: Show, driver_cls=MavlinkNativeDriver, faults=None, place=True):
        fleet = VirtualFleet(show.drone_count, faults=faults or {})
        if place and show.formations:
            fleet.place(show.formations[0].points)
        fleet.start()
        driver = driver_cls(fleet.endpoints, origin=show.origin)
        driver.connect(show.drone_count, wait_heartbeat=CONNECT_TIMEOUT)
        created.append((fleet, driver))
        return fleet, driver

    yield build
    for fleet, driver in created:
        driver.close()
        fleet.stop()


# ==========================================================================
# FTP wire format
# ==========================================================================


class TestFtpFraming:
    def test_header_round_trips(self):
        frame = encode_ftp(
            seq=7, session=3, opcode=7, req_opcode=0, offset=1234, data=b"hello"
        )
        assert len(frame) == 251, "FILE_TRANSFER_PROTOCOL payload is a fixed 251 bytes"
        op = decode_ftp(frame)
        assert op["seq"] == 7
        assert op["session"] == 3
        assert op["opcode"] == 7
        assert op["offset"] == 1234
        assert op["data"] == b"hello"

    def test_oversized_chunk_is_refused(self):
        with pytest.raises(ValueError, match="exceeds 239"):
            encode_ftp(1, 1, 7, data=b"x" * 240)

    def test_payload_is_padded_not_truncated(self):
        frame = encode_ftp(1, 1, 7, data=b"ab")
        assert frame[12:14] == b"ab"
        assert set(frame[14:]) == {0}


# ==========================================================================
# Link bring-up
# ==========================================================================


class TestLink:
    def test_heartbeat_handshake(self, wired):
        show = wire_show(3)
        fleet, driver = wired(show)
        states = driver.poll()
        assert all(s.connected for s in states)
        assert [s.sysid for s in states] == [1, 2, 3], (
            "system ids must come from the aircraft, not be assumed"
        )

    def test_connect_times_out_on_a_dead_endpoint(self):
        # Nothing is listening, so the wait must end in a clear timeout
        # rather than reporting a fleet that is not there.
        driver = MavlinkNativeDriver(["udpin:127.0.0.1:14899"])
        try:
            with pytest.raises(TimeoutError, match="no heartbeat"):
                driver.connect(1, wait_heartbeat=2.0)
        finally:
            driver.close()

    def test_connect_refuses_more_drones_than_endpoints(self):
        driver = MavlinkNativeDriver(["udpin:127.0.0.1:14898"])
        try:
            with pytest.raises(ValueError, match="only 1"):
                driver.connect(4)
        finally:
            driver.close()

    def test_telemetry_decodes_into_show_coordinates(self, wired):
        show = wire_show(3)
        fleet, driver = wired(show)
        driver.set_origin(show.origin)
        time.sleep(1.5)
        truth = fleet.positions()
        for state in driver.poll():
            assert state.position is not None, "no position decoded from telemetry"
            # Round trip: aircraft ENU -> lat/lon on the wire -> driver ENU.
            error = float(np.linalg.norm(state.position[:2] - truth[state.index][:2]))
            assert error < 0.5, (
                f"drone {state.index} decoded {error:.2f} m from where it is; "
                "the ENU/WGS84 conversion disagrees between the two ends"
            )

    def test_gnss_and_battery_reach_the_ground_station(self, wired):
        show = wire_show(2)
        _, driver = wired(show)
        time.sleep(1.5)
        for state in driver.poll():
            assert state.gps_fix == 6
            assert state.satellites >= 12
            assert state.hdop < 1.2
            assert state.battery_percent > 90.0
            assert state.battery_voltage > 10.0

    def test_mode_numbers_decode_to_names(self):
        assert COPTER_MODE_NAMES[SHOW_CUSTOM_MODE] == "SHOW"
        assert COPTER_MODE_NAMES[4] == "GUIDED"
        assert COPTER_MODE_NAMES[9] == "LAND"


# ==========================================================================
# Commands
# ==========================================================================


class TestCommands:
    def test_origin_is_accepted_as_command_int(self, wired):
        show = wire_show(3)
        fleet, driver = wired(show)
        driver.set_origin(show.origin)
        time.sleep(1.0)
        assert fleet.summary()["origin_set"] == 3
        for aircraft in fleet.aircraft:
            assert aircraft.origin.latitude == pytest.approx(12.9716, abs=1e-7)
            assert aircraft.origin.longitude == pytest.approx(77.5946, abs=1e-7)
            assert aircraft.origin.amsl == pytest.approx(920.0, abs=1e-3)

    def test_origin_as_command_long_is_denied(self, wired):
        """The precision guard, exercised over the wire.

        Latitude in a float32 param loses about a centimetre, which is the
        same order as the RTK accuracy the whole show rests on. Firmware
        refuses it, so the ground station must never send it that way.
        """
        from pymavlink import mavutil

        show = wire_show(1)
        fleet, driver = wired(show)
        link = driver.links[0]
        link.mav.command_long_send(
            link.target_system, link.target_component,
            mavutil.mavlink.MAV_CMD_USER_2, 0,
            0, 0, 0, 35.0, 12.97, 77.59, 920.0,
        )
        time.sleep(1.0)
        nacks = fleet.summary()["nacks"]
        assert 0 in nacks
        assert any(
            result == mavutil.mavlink.MAV_RESULT_DENIED for _cmd, result in nacks[0]
        )
        assert not fleet.aircraft[0].origin_set
        assert any("rejected" in f for f in driver.poll()[0].faults), (
            "a refused command must surface to the operator, not vanish"
        )

    def test_arm_and_disarm_over_the_wire(self, wired):
        show = wire_show(3)
        fleet, driver = wired(show)
        driver.arm()
        time.sleep(1.5)
        assert all(a.armed for a in fleet.aircraft)
        assert all(s.armed for s in driver.poll()), "armed state must come back"
        driver.disarm()
        time.sleep(1.5)
        assert not any(a.armed for a in fleet.aircraft)

    def test_a_refused_arm_is_reported_not_assumed(self, wired):
        show = wire_show(3)
        fleet, driver = wired(show, faults={0: AircraftFaults(refuse_arm=True)})
        driver.arm()
        time.sleep(1.5)
        assert [a.armed for a in fleet.aircraft] == [False, True, True]
        assert [s.armed for s in driver.poll()] == [False, True, True], (
            "the ground station must not believe a drone armed because it "
            "sent the command"
        )
        assert driver.poll()[0].faults

    def test_show_start_survives_the_float32_trap(self, wired):
        """A unix timestamp cannot travel in a MAVLink float param.

        Around 1.79e9 the gap between representable float32 values is 128
        seconds. The start instant therefore goes as an int32 in a
        COMMAND_INT, and this checks the aircraft receives it to the
        millisecond.
        """
        show = wire_show(2)
        fleet, driver = wired(show)
        target = time.time() + 123.456
        driver.set_show_start(target)
        # Wait for the command to arrive rather than a fixed second: on a
        # busy machine the virtual fleet's loop can take longer than that.
        deadline = time.time() + 5.0
        while time.time() < deadline and any(a.show_start_unix is None for a in fleet.aircraft):
            time.sleep(0.05)
        for aircraft in fleet.aircraft:
            assert aircraft.show_start_unix is not None
            assert aircraft.show_start_unix == pytest.approx(target, abs=0.002)

        # And prove the trap is real, so this test cannot be "simplified"
        # back into a float param by someone who thinks it is tidier.
        float32_error = abs(float(np.float32(target)) - target)
        assert float32_error > 1.0, (
            f"a float32 only loses {float32_error:.3f}s on this timestamp; "
            "re-check the epoch before trusting a float param"
        )

    def test_start_time_beyond_int32_is_refused(self, wired):
        show = wire_show(1)
        _, driver = wired(show)
        with pytest.raises(ValueError, match="int32"):
            driver.set_show_start(3e9)

    def test_emergency_actions_change_aircraft_mode(self, wired):
        show = wire_show(3)
        fleet, driver = wired(show)
        driver.arm()
        time.sleep(1.2)
        for action, expected in (("rth", "RTL"), ("land", "LAND"), ("hold", "BRAKE")):
            driver.emergency(action)
            time.sleep(1.2)
            modes = fleet.summary()["modes"]
            assert modes == [expected], f"{action} produced {modes}"

    def test_show_control_clears_a_loaded_plan(self, wired):
        show = wire_show(2)
        fleet, driver = wired(show)
        driver.upload(build_mission(show))
        assert fleet.summary()["plans_loaded"] == 2
        driver.show_control("clear")
        time.sleep(1.0)
        assert fleet.summary()["plans_loaded"] == 0

    def test_unknown_show_action_is_rejected_locally(self, wired):
        show = wire_show(1)
        _, driver = wired(show)
        with pytest.raises(ValueError, match="unknown show action"):
            driver.show_control("explode")


# ==========================================================================
# Plan upload
# ==========================================================================


class TestUpload:
    def test_plans_arrive_intact_and_crc_verified(self, wired):
        show = wire_show(4)
        fleet, driver = wired(show)
        mission = build_mission(show)
        driver.upload(mission)

        assert fleet.summary()["plans_loaded"] == 4
        checksum = mission.to_dict()["checksum"]
        assert all(s.mission_checksum == checksum for s in driver.poll())

        # The aircraft must hold exactly the bytes we sent.
        for aircraft, plan in zip(fleet.aircraft, mission.plans):
            stored = aircraft.plan
            assert stored is not None
            assert stored["drone"] == plan.drone_index
            assert len(stored["points"]) == len(plan.points)
            assert np.allclose(
                np.asarray(stored["points"], dtype=float), plan.points, atol=1e-4
            )
            assert stored["easings"] == plan.easings

    def test_uploaded_plan_reproduces_the_designed_path(self, wired):
        """The point of the whole transfer: same motion at both ends."""
        from skyshine.trajectory import evaluate_plan

        show = wire_show(3)
        fleet, driver = wired(show)
        mission = build_mission(show)
        driver.upload(mission)

        # The mission file stores coordinates rounded to 0.1 mm so it stays
        # readable and diffable for a safety review, so that quantisation -
        # not a protocol error - is the floor on agreement here.
        for aircraft, plan in zip(fleet.aircraft, mission.plans):
            for t in np.linspace(0.0, show.duration, 41):
                onboard = aircraft.evaluate_plan(float(t))
                ground = evaluate_plan(plan, float(t))
                assert np.allclose(onboard, ground, atol=2e-4), (
                    f"drone {plan.drone_index} would fly a different path at "
                    f"t={t:.1f}s than the ground station designed: "
                    f"{np.max(np.abs(onboard - ground)) * 1000:.4f} mm apart"
                )

    def test_a_single_failed_upload_fails_the_whole_fleet(self, wired):
        show = wire_show(3)
        fleet, driver = wired(show, faults={1: AircraftFaults(fail_upload=True)})
        with pytest.raises(IOError, match="mixed state"):
            driver.upload(build_mission(show))
        # And the operator can see exactly who is loaded.
        assert [s.mission_uploaded for s in driver.poll()] == [True, False, True]

    def test_upload_hook_overrides_the_default_transport(self, wired):
        seen: list[int] = []
        show = wire_show(3)
        fleet = VirtualFleet(3)
        fleet.start()
        driver = MavlinkNativeDriver(
            fleet.endpoints,
            origin=show.origin,
            upload_hook=lambda link, index, plan: seen.append(index),
        )
        try:
            driver.connect(3, wait_heartbeat=CONNECT_TIMEOUT)
            driver.upload(build_mission(show))
            assert sorted(seen) == [0, 1, 2]
            # Nothing reached the aircraft: the hook replaced the transport.
            assert fleet.summary()["plans_loaded"] == 0
        finally:
            driver.close()
            fleet.stop()

    def test_ftp_write_and_read_back(self, wired):
        show = wire_show(1)
        fleet, driver = wired(show)
        blob = bytes(range(256)) * 7  # 1792 bytes, several FTP chunks
        written = driver.ftp_write(0, blob, "@TEST/blob.bin")
        assert written == len(blob)
        assert bytes(fleet.aircraft[0]._ftp_files["@TEST/blob.bin"]) == blob

    def test_plan_path_is_agreed_between_both_ends(self, wired):
        from skyshine import sitl

        assert SHOW_PLAN_PATH == sitl.SHOW_PLAN_PATH, (
            "the driver writes the plan where the aircraft does not read it"
        )


# ==========================================================================
# Flying a show
# ==========================================================================


class TestShowExecution:
    def test_native_show_runs_from_the_aircraft_clock(self, wired):
        show = wire_show(4)
        fleet, driver = wired(show)
        driver.set_origin(show.origin)
        driver.upload(build_mission(show))
        driver.arm()
        time.sleep(1.2)

        start = time.time() + 1.0
        driver.start(start)
        time.sleep(2.0)
        assert fleet.summary()["modes"] == ["SHOW"]
        assert fleet.summary()["start_scheduled"] == 4

        time.sleep(8.0)
        # They should be climbing, each from its own clock, not waiting for
        # a packet from the ground.
        heights = [a.position[2] for a in fleet.aircraft]
        assert min(heights) > 1.0, f"nobody left the ground: {heights}"
        assert driver.poll()[0].mode == "SHOW"

    def test_a_skewed_clock_desynchronises_that_aircraft(self, wired):
        """The fault that collides a fleet, made visible.

        Every drone flies the correct path. One of them flies it four
        seconds early, so it is in the right place at the wrong time, and
        nothing in trajectory validation can see it. Only a clock check on
        the ground can.
        """
        show = wire_show(4)
        fleet, driver = wired(show, faults={2: AircraftFaults(clock_offset=4.0)})
        driver.set_origin(show.origin)
        driver.upload(build_mission(show))
        driver.arm()
        time.sleep(1.2)
        driver.start(time.time() + 0.5)
        time.sleep(10.0)

        skewed = fleet.aircraft[2]
        others = [a for i, a in enumerate(fleet.aircraft) if i != 2]
        assert skewed.show_time is not None
        lead = skewed.show_time - float(
            np.mean([a.show_time for a in others])
        )
        assert lead == pytest.approx(4.0, abs=0.5), (
            f"the skewed aircraft leads by {lead:.2f}s, expected about 4"
        )

        # Comparing raw altitudes between drones would prove nothing: the
        # layered launch cloud puts each drone at a different height by
        # design. What matters is that this one is at the right place for
        # the wrong time - closer to where its own plan puts it four
        # seconds from now than to where it should be now.
        fleet_time = float(np.mean([a.show_time for a in others]))
        here_now = skewed.evaluate_plan(fleet_time)
        here_later = skewed.evaluate_plan(fleet_time + 4.0)
        to_now = float(np.linalg.norm(skewed.position - here_now))
        to_later = float(np.linalg.norm(skewed.position - here_later))
        assert to_later < to_now, (
            f"the skewed drone is {to_now:.2f} m from its position for now and "
            f"{to_later:.2f} m from its position four seconds ahead; it should "
            "be flying ahead of the fleet"
        )

    def test_guided_stream_moves_the_fleet(self, wired):
        show = wire_show(3)
        fleet, driver = wired(show, driver_cls=MavlinkGuidedDriver)
        driver.arm()
        time.sleep(1.2)
        driver.start(time.time())
        time.sleep(0.5)
        assert fleet.summary()["modes"] == ["GUIDED"]

        # Stream a climb by hand, the way the fleet manager's tick does.
        target = np.array([0.0, 0.0, 40.0])
        for _ in range(60):
            for i in range(3):
                driver.setpoint(i, target, np.array([255, 128, 0]))
            time.sleep(0.1)
        time.sleep(1.0)
        for aircraft in fleet.aircraft:
            assert aircraft.position[2] > 20.0, (
                f"drone {aircraft.index} did not follow setpoints: "
                f"{aircraft.position}"
            )

    def test_guided_stream_drives_the_led_payload(self, wired):
        show = wire_show(2)
        fleet, driver = wired(show, driver_cls=MavlinkGuidedDriver)
        driver.setpoint(0, np.array([0.0, 0.0, 20.0]), np.array([255, 60, 20]))
        time.sleep(1.0)
        leds = fleet.aircraft[0].leds
        # PWM is 1000-2000us mapped onto 0-255, so expect close, not exact.
        assert abs(int(leds[0]) - 255) <= 3
        assert abs(int(leds[1]) - 60) <= 3
        assert abs(int(leds[2]) - 20) <= 3

    def test_setpoint_frame_is_ned_not_enu(self, wired):
        """A swapped axis here puts the whole show sideways.

        The show frame is x-east, y-north, z-up. MAVLink LOCAL_NED is
        north-east-down. Send a purely eastward setpoint and confirm the
        aircraft ends up east, not north.
        """
        show = wire_show(1)
        fleet, driver = wired(show, driver_cls=MavlinkGuidedDriver)
        driver.arm()
        time.sleep(1.0)
        for _ in range(40):
            driver.setpoint(0, np.array([30.0, 0.0, 25.0]), np.zeros(3))
            time.sleep(0.05)
        time.sleep(1.5)
        p = fleet.aircraft[0].position
        assert p[0] > 20.0, f"east component wrong: {p}"
        assert abs(p[1]) < 5.0, f"leaked into north: {p}"
        assert p[2] > 15.0, f"altitude sign wrong: {p}"

    def test_native_driver_still_refuses_setpoints(self, wired):
        show = wire_show(1)
        _, driver = wired(show)
        with pytest.raises(NotImplementedError, match="onboard"):
            driver.setpoint(0, np.zeros(3), np.zeros(3))


# ==========================================================================
# Through the ground station
# ==========================================================================


class TestFleetManagerOverMavlink:
    def test_manager_adopts_a_connected_driver(self, wired):
        """Regression: the manager used to reconnect and kill the link.

        Rebuilding the sockets of a driver that was already connected -
        with its heartbeat timeout already satisfied - left every drone
        looking offline and every preflight check failing.
        """
        show = wire_show(3)
        fleet, driver = wired(show)
        manager = FleetManager(show, driver=driver)
        assert manager.wait_for_fleet(10.0) == 3
        assert all(d.connected for d in manager.drones)
        assert any("adopted" in e.message for e in manager.events)

    def test_full_sequence_over_the_wire(self, wired):
        show = wire_show(4)
        fleet, driver = wired(show)
        manager = FleetManager(show, driver=driver)
        manager.wait_for_fleet(10.0)
        driver.set_origin(show.origin)

        assert manager.validate().passed
        manager.upload()
        failing = [c.name for c in manager.preflight() if c.blocking and not c.ok]
        assert not failing, failing

        manager.arm()
        time.sleep(1.5)
        assert sum(a.armed for a in fleet.aircraft) == 4

        manager.start(lead_in=0.5)
        deadline = time.time() + 14.0
        while time.time() < deadline and manager.phase == Phase.RUNNING:
            manager.tick()
            time.sleep(0.1)
        assert manager.phase == Phase.RUNNING
        assert fleet.summary()["modes"] == ["SHOW"]
        assert max(a.position[2] for a in fleet.aircraft) > 2.0

        manager.emergency("land")
        time.sleep(1.5)
        assert fleet.summary()["modes"] == ["LAND"]

    def test_launch_position_check_sees_real_telemetry(self, wired):
        show = wire_show(4)
        # Not placed on the pads: they spawn at the show origin instead.
        fleet, driver = wired(show, place=False)
        manager = FleetManager(show, driver=driver)
        manager.wait_for_fleet(10.0)
        driver.set_origin(show.origin)
        time.sleep(1.0)
        checks = {c.name: c for c in manager.preflight()}
        assert not checks["launch positions"].ok, (
            "aircraft sitting at the origin are not on their pads"
        )

        fleet.place(show.formations[0].points)
        time.sleep(1.5)
        checks = {c.name: c for c in manager.preflight()}
        assert checks["launch positions"].ok

    def test_link_loss_mid_show_is_detected(self, wired):
        show = wire_show(3)
        fleet, driver = wired(show, faults={1: AircraftFaults(go_silent_at=3.0)})
        manager = FleetManager(show, driver=driver)
        manager.wait_for_fleet(10.0)
        driver.set_origin(show.origin)
        manager.validate()
        manager.upload()
        manager.arm()
        time.sleep(1.2)
        manager.start(lead_in=0.5)

        deadline = time.time() + 16.0
        saw_loss = False
        while time.time() < deadline:
            snapshot = manager.tick()
            if any(
                a["drone"] == 1 and a["kind"] == "link_lost" for a in snapshot["alerts"]
            ):
                saw_loss = True
                break
            time.sleep(0.1)
        assert saw_loss, "a drone that stopped transmitting was not noticed"
        assert 1 in fleet.summary()["silent"]

    def test_deviation_threshold_check_flags_an_unrealistic_policy(self, wired):
        show = wire_show(3)
        _, driver = wired(show)
        tight = FleetManager(
            show, driver=driver, policy=PreflightPolicy(abort_deviation_m=0.2)
        )
        check = {c.name: c for c in tight.preflight()}["deviation thresholds"]
        assert not check.ok
        assert "trail its setpoint" in check.detail
        assert not check.blocking, "threshold tuning must not block a launch"


# ==========================================================================
# Real ArduPilot SITL
# ==========================================================================


class TestArduPilotSITL:
    def test_endpoints_follow_the_instance_port_block(self):
        sitl = ArduPilotSITL(count=3)
        assert sitl.endpoints == [
            "tcp:127.0.0.1:5760",
            "tcp:127.0.0.1:5770",
            "tcp:127.0.0.1:5780",
        ], "ArduPilot offsets each instance's port block by 10"

    def test_absence_is_explained_not_faked(self):
        sitl = ArduPilotSITL(sim_vehicle="/definitely/not/here/sim_vehicle.py")
        with pytest.raises(FileNotFoundError) as excinfo:
            sitl.locate()
        message = str(excinfo.value)
        assert "waf" in message and "ardupilot" in message.lower(), (
            "the error must tell the operator how to install it"
        )
        assert "VirtualFleet" in message

    def test_availability_probe_does_not_raise(self):
        assert isinstance(sitl_available("/nope/sim_vehicle.py"), bool)

    @pytest.mark.skipif(not sitl_available(), reason="ArduPilot SITL is not installed")
    def test_real_sitl_boots_and_heartbeats(self):
        show = wire_show(1)
        sitl = ArduPilotSITL(
            count=1, latitude=show.origin.latitude, longitude=show.origin.longitude,
            speedup=5.0,
        ).start()
        driver = MavlinkGuidedDriver(sitl.endpoints, origin=show.origin)
        try:
            driver.connect(1, wait_heartbeat=90.0)
            time.sleep(5.0)
            state = driver.poll()[0]
            assert state.connected
            assert state.sysid == 1
        finally:
            driver.close()
            sitl.stop()


# ==========================================================================
# Harness sanity
# ==========================================================================


class TestHarness:
    def test_each_aircraft_gets_its_own_port(self):
        fleet = VirtualFleet(6)
        try:
            ports = [a.port for a in fleet.aircraft]
            assert len(set(ports)) == 6
            assert all(e.startswith("udpin:") for e in fleet.endpoints)
        finally:
            fleet.stop()

    def test_fleet_records_errors_instead_of_hiding_them(self):
        # A swallowed exception in the service loop presents as "no
        # heartbeat", which cost real debugging time once already.
        fleet = VirtualFleet(1)
        try:
            fleet.aircraft[0].step = lambda dt: (_ for _ in ()).throw(RuntimeError("boom"))
            fleet.start()
            time.sleep(0.5)
            assert fleet.errors, "the service loop must not hide aircraft errors"
            assert "boom" in fleet.errors[0][1]
        finally:
            fleet.stop()

    def test_place_rejects_a_mismatched_count(self):
        fleet = VirtualFleet(3)
        try:
            with pytest.raises(ValueError, match="for 3 aircraft"):
                fleet.place(np.zeros((5, 3)))
        finally:
            fleet.stop()

    def test_aircraft_disarms_itself_after_landing_but_not_on_the_pad(self):
        aircraft = VirtualAircraft(0, 14896, noise=0.0)
        aircraft.disarm_delay = 1.0
        try:
            aircraft.armed = True  # waiting on the pad for the show start
            for _ in range(40):
                aircraft.fly(0.1)
            assert aircraft.armed

            aircraft.target = np.array([0.0, 0.0, 20.0])
            for _ in range(60):
                aircraft.fly(0.1)
            aircraft.custom_mode = 9  # LAND
            for _ in range(40):
                aircraft.fly(0.1)
                if not aircraft.armed:
                    break
            assert not aircraft.armed and aircraft.position[2] <= 0.05
        finally:
            aircraft.close()

    def test_aircraft_tracks_its_target_with_lag_not_teleportation(self):
        aircraft = VirtualAircraft(0, 14897, noise=0.0)
        try:
            aircraft.armed = True
            aircraft.target = np.array([0.0, 0.0, 50.0])
            aircraft.fly(0.1)
            first = aircraft.position[2]
            assert 0.0 < first < 50.0, (
                "an aircraft that reaches its setpoint instantly tests nothing"
            )
            for _ in range(200):
                aircraft.fly(0.05)
            assert aircraft.position[2] == pytest.approx(50.0, abs=0.2)
        finally:
            aircraft.close()

    def test_disarmed_aircraft_stays_on_the_ground(self):
        aircraft = VirtualAircraft(0, 14896, noise=0.0)
        try:
            aircraft.armed = False
            aircraft.target = np.array([10.0, 10.0, 50.0])
            for _ in range(50):
                aircraft.fly(0.1)
            assert aircraft.position[2] == 0.0
        finally:
            aircraft.close()


# ==========================================================================
# MAVLink 2 signing
# ==========================================================================


class TestSigning:
    """Each case is one key on the ground station and one on the aircraft."""

    KEY = bytes(range(32))
    OTHER = bytes(range(1, 33))

    def _fleet(self, gcs_key, air_key, count=2):
        fleet = VirtualFleet(count, signing_key=air_key).start()
        driver = MavlinkNativeDriver(fleet.endpoints, signing_key=gcs_key)
        return fleet, driver

    def test_matching_keys_connect_and_commands_land(self):
        fleet, driver = self._fleet(self.KEY, self.KEY)
        try:
            driver.connect(2, wait_heartbeat=CONNECT_TIMEOUT)
            driver.arm()
            time.sleep(0.6)
            # Regression: mavutil swapped its parser on the first MAVLink 2
            # packet and dropped the key, so the link ran unsigned and
            # signed aircraft silently ignored every command.
            assert fleet.summary()["armed"] == 2
            assert all(a.mav.signing.goodsig_count for a in fleet.aircraft)
        finally:
            driver.close()
            fleet.stop()

    @pytest.mark.parametrize("air", [None, OTHER])
    def test_unsigned_or_wrong_key_aircraft_are_refused_by_name(self, air):
        fleet, driver = self._fleet(self.KEY, air)
        try:
            with pytest.raises(TimeoutError, match="signature check"):
                driver.connect(2, wait_heartbeat=3.0)
        finally:
            driver.close()
            fleet.stop()

    def test_signed_aircraft_ignore_an_unsigned_ground_station(self):
        fleet, driver = self._fleet(None, self.KEY)
        try:
            driver.connect(2, wait_heartbeat=CONNECT_TIMEOUT)
            driver.arm()
            time.sleep(0.6)
            assert fleet.summary()["armed"] == 0
        finally:
            driver.close()
            fleet.stop()

    def test_provisioning_loads_the_key_and_refuses_when_armed(self):
        fleet, driver = self._fleet(None, None, count=3)
        try:
            driver.connect(3, wait_heartbeat=CONNECT_TIMEOUT)
            assert driver.provision_signing(self.KEY) == {
                "confirmed": [0, 1, 2], "unconfirmed": []}
            assert fleet.summary()["signed"] == 3
            driver.arm()
            time.sleep(0.6)
            assert fleet.summary()["armed"] == 3
            assert all(s.connected for s in driver.poll())
            # The driver learns it is armed from the next 1 Hz heartbeat.
            deadline = time.time() + 3.0
            while not all(s.armed for s in driver.poll()) and time.time() < deadline:
                time.sleep(0.05)
            with pytest.raises(RuntimeError, match="armed"):
                driver.provision_signing(self.OTHER)
        finally:
            driver.close()
            fleet.stop()
