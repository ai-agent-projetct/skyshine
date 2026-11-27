"""Links to the aircraft. Three of them, with very different trust levels.

SimDriver            no hardware. Flies the plan with injected noise,
                     wind, clock error and faults. Everything in tests
                     runs here, and every show should be rehearsed here.

MavlinkNativeDriver  for firmware with a show mode: the whole trajectory
                     is uploaded, the aircraft executes it from its own
                     clock, and the ground station only supervises. This
                     is the only approach that scales to hundreds of
                     drones, because nothing time-critical crosses the
                     radio during the show.

MavlinkGuidedDriver  for unmodified ArduPilot: GUIDED mode with position
                     setpoints streamed from the ground at 10 Hz. Works
                     today with stock firmware, but every setpoint is a
                     packet that can be late or lost, so it is capped at a
                     small fleet and refuses to pretend otherwise.

HARDWARE STATUS: the two MAVLink drivers are written against the MAVLink 2
message definitions and the documented Skybrush show-mode command set.
They have NOT been flown. Treat them as a starting point for bench and
SITL integration, not as flight-proven code. The simulator is what this
repository's tests exercise.
"""

from __future__ import annotations

import json
import math
import queue
import struct
import threading
import time
import zlib
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

from .show import ShowOrigin, enu_to_wgs84
from .trajectory import MissionPackage

# Custom flight mode number used for show execution. 127 is the value the
# Skybrush ArduCopter build uses, and is the de facto convention.
SHOW_CUSTOM_MODE = 127

# Unix epoch seconds cannot travel in a MAVLink float32 param: around
# 1.79e9 the spacing between representable float32 values is 128 seconds,
# so a start instant sent that way could land over two minutes out. The
# show start command is therefore COMMAND_INT, with whole seconds in the
# int32 x field and the millisecond remainder in y.
SHOW_START_COMMAND = "MAV_CMD_USER_3"

# ArduCopter custom mode numbers, so telemetry reads as words.
COPTER_MODE_NAMES = {
    0: "STABILIZE", 1: "ACRO", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED",
    5: "LOITER", 6: "RTL", 7: "CIRCLE", 9: "LAND", 11: "DRIFT",
    13: "SPORT", 14: "FLIP", 15: "AUTOTUNE", 16: "POSHOLD", 17: "BRAKE",
    18: "THROW", 20: "GUIDED_NOGPS", 21: "SMART_RTL", 22: "FLOWHOLD",
    23: "FOLLOW", 24: "ZIGZAG", SHOW_CUSTOM_MODE: "SHOW",
}

# MAVLink FTP, the transport show-mode firmware uses for plan upload.
FTP_HEADER = "<HBBBBBBI"
FTP_HEADER_LEN = 12
FTP_MAX_DATA = 239
FTP_PAYLOAD_LEN = 251
FTP_OP_TERMINATE = 1
FTP_OP_CREATE = 6
FTP_OP_WRITE = 7
FTP_OP_CRC32 = 14
FTP_OP_ACK = 128
FTP_OP_NAK = 129

# Path the show plan is written to on the aircraft. Both ends must agree;
# only the agreement matters, not the spelling.
SHOW_PLAN_PATH = "@SHOW/plan.json"

# Beyond this fleet size, streaming setpoints from the ground is not a
# defensible architecture: see MavlinkGuidedDriver.
GUIDED_STREAM_MAX_DRONES = 50


@dataclass
class DroneState:
    """Everything the operator needs to know about one aircraft."""

    index: int
    sysid: int = 0
    connected: bool = False
    armed: bool = False
    mode: str = "UNKNOWN"
    # Measured position in show ENU metres. None until first fix.
    position: np.ndarray | None = None
    velocity: np.ndarray | None = None
    latitude: float = 0.0
    longitude: float = 0.0
    altitude_agl: float = 0.0
    battery_voltage: float = 0.0
    battery_percent: float = 0.0
    gps_fix: int = 0
    satellites: int = 0
    hdop: float = 99.9
    # Seconds between this drone's clock and the ground station's.
    clock_offset: float = 0.0
    last_heartbeat: float = 0.0
    # Distance from where the plan says it should be, metres.
    deviation: float = 0.0
    mission_uploaded: bool = False
    mission_checksum: str = ""
    faults: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "sysid": self.sysid,
            "connected": self.connected,
            "armed": self.armed,
            "mode": self.mode,
            "position": None if self.position is None else [round(float(v), 3) for v in self.position],
            "latitude": round(self.latitude, 7),
            "longitude": round(self.longitude, 7),
            "altitude_agl": round(self.altitude_agl, 2),
            "battery_voltage": round(self.battery_voltage, 2),
            "battery_percent": round(self.battery_percent, 1),
            "gps_fix": self.gps_fix,
            "satellites": self.satellites,
            "hdop": round(self.hdop, 2),
            "clock_offset": round(self.clock_offset, 4),
            "deviation": round(self.deviation, 3),
            "mission_uploaded": self.mission_uploaded,
            "faults": list(self.faults),
        }


class ShowDriver(Protocol):
    """What the fleet manager needs from any link to the aircraft."""

    name: str
    streams_setpoints: bool

    def connect(self, count: int) -> None: ...
    def close(self) -> None: ...
    def poll(self) -> list[DroneState]: ...
    def set_origin(self, origin: ShowOrigin) -> None: ...
    def upload(self, mission: MissionPackage) -> None: ...
    def arm(self, indices: Sequence[int] | None = None) -> None: ...
    def disarm(self, indices: Sequence[int] | None = None) -> None: ...
    def start(self, at_unix_time: float) -> None: ...
    def setpoint(self, index: int, position: np.ndarray, color: np.ndarray) -> None: ...
    def emergency(self, action: str, indices: Sequence[int] | None = None) -> None: ...


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------


@dataclass
class SimFaults:
    """Faults to inject, so the operator rehearses the bad night too."""

    # Drones that lose their link, as {index: show_time_seconds}.
    link_loss: dict[int, float] = field(default_factory=dict)
    # Drones that start with a weak battery, as {index: percent}.
    weak_battery: dict[int, float] = field(default_factory=dict)
    # Drones whose GPS degrades, as {index: show_time_seconds}.
    gps_degradation: dict[int, float] = field(default_factory=dict)
    # Drones that drift off plan, as {index: metres_per_second}.
    position_drift: dict[int, float] = field(default_factory=dict)
    # Steady wind in show ENU, m/s. Shifts every drone slightly downwind.
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Standard deviation of per-axis position noise, metres. RTK is about
    # 0.02 m horizontal; plain GPS is well over a metre, which is why a
    # dense show needs RTK.
    position_noise: float = 0.03
    # Per-drone clock error standard deviation, seconds.
    clock_jitter: float = 0.002
    # Drones whose clock is simply wrong, as {index: seconds}. The fault
    # that collides a fleet - right path, wrong time - and the one the
    # preflight clock check exists to catch, so it has to be rehearsable
    # here and not only under SITL.
    clock_offset: dict[int, float] = field(default_factory=dict)


class SimDriver:
    """A fleet that exists only in memory. The default, and the safe one."""

    name = "simulator"
    streams_setpoints = False

    def __init__(
        self,
        *,
        faults: SimFaults | None = None,
        seed: int = 0,
        battery_endurance_s: float = 900.0,
    ) -> None:
        self.faults = faults or SimFaults()
        self.rng = np.random.default_rng(seed)
        self.battery_endurance_s = battery_endurance_s
        self.states: list[DroneState] = []
        self.mission: MissionPackage | None = None
        self.origin = ShowOrigin()
        self.started_at: float | None = None
        self._commanded: dict[int, np.ndarray] = {}
        self._emergency: dict[int, str] = {}

    def connect(self, count: int) -> None:
        self.states = []
        for i in range(count):
            s = DroneState(index=i, sysid=i + 1, connected=True, mode="LOITER")
            s.battery_percent = self.faults.weak_battery.get(i, 100.0)
            s.battery_voltage = 16.8 * s.battery_percent / 100.0
            s.gps_fix = 6  # RTK fixed
            s.satellites = 22
            s.hdop = 0.6
            s.clock_offset = float(self.rng.normal(0.0, self.faults.clock_jitter))
            s.clock_offset += float(self.faults.clock_offset.get(i, 0.0))
            s.last_heartbeat = time.time()
            self.states.append(s)

    def close(self) -> None:
        self.states = []

    def set_origin(self, origin: ShowOrigin) -> None:
        self.origin = origin

    def upload(self, mission: MissionPackage) -> None:
        self.mission = mission
        checksum = mission.to_dict()["checksum"]
        for s in self.states:
            s.mission_uploaded = True
            s.mission_checksum = checksum

    def arm(self, indices: Sequence[int] | None = None) -> None:
        for s in self._select(indices):
            s.armed = True
            s.mode = "SHOW"

    def disarm(self, indices: Sequence[int] | None = None) -> None:
        for s in self._select(indices):
            s.armed = False
            s.mode = "LOITER"

    def start(self, at_unix_time: float) -> None:
        self.started_at = at_unix_time

    def setpoint(self, index: int, position: np.ndarray, color: np.ndarray) -> None:
        self._commanded[index] = np.asarray(position, dtype=np.float64)

    def emergency(self, action: str, indices: Sequence[int] | None = None) -> None:
        for s in self._select(indices):
            self._emergency[s.index] = action
            s.mode = {"hold": "BRAKE", "rth": "RTL", "land": "LAND", "disarm": "DISARMED"}.get(
                action, "LOITER"
            )
            if action == "disarm":
                s.armed = False

    def release_hold(self, indices: Sequence[int] | None = None) -> None:
        """Clear a latched emergency so the fleet accepts guidance again.

        A hold is a mode, not a momentary action: until it is released the
        aircraft ignores the plan. Leaving it latched made resume and
        collective RTH appear to run while the fleet sat motionless, which
        is the worst possible failure for an abort path - it looks like it
        worked.
        """
        for s in self._select(indices):
            self._emergency.pop(s.index, None)
            if s.armed:
                s.mode = "SHOW"

    def advance(self, show_time: float, planned: np.ndarray) -> list[DroneState]:
        """Step the simulation to `show_time` given the planned positions.

        `planned` is (n, 3) in show ENU. The fleet manager calls this; a
        real driver gets the same information from telemetry instead.
        """
        f = self.faults
        wind = np.asarray(f.wind, dtype=np.float64)
        now = time.time()
        for s in self.states:
            i = s.index
            if i in f.link_loss and show_time >= f.link_loss[i]:
                s.connected = False
                s.faults = ["link lost"]
                continue
            s.connected = True
            s.last_heartbeat = now

            if i in f.gps_degradation and show_time >= f.gps_degradation[i]:
                s.gps_fix = 3
                s.satellites = 7
                s.hdop = 3.4
                s.faults = ["GPS degraded"]

            target = planned[i] if i < len(planned) else np.zeros(3)
            if self._emergency.get(i) == "hold":
                target = s.position if s.position is not None else target
            elif self._emergency.get(i) == "land":
                target = np.array([target[0], target[1], max(0.0, target[2] - 2.0 * 1.0)])

            noise_scale = f.position_noise * (6.0 if s.gps_fix < 5 else 1.0)
            noise = self.rng.normal(0.0, noise_scale, size=3)
            drift = f.position_drift.get(i, 0.0) * show_time
            offset = wind * 0.12 + noise
            if drift:
                offset = offset + np.array([drift, 0.0, 0.0])

            new_pos = np.asarray(target, dtype=np.float64) + offset
            if s.position is not None and show_time > 0:
                s.velocity = (new_pos - s.position) * 1.0
            s.position = new_pos
            s.deviation = float(np.linalg.norm(new_pos - target))
            s.altitude_agl = float(new_pos[2])
            s.latitude, s.longitude = enu_to_wgs84(new_pos[:2], self.origin)

            start_pct = f.weak_battery.get(i, 100.0)
            used = (show_time / max(self.battery_endurance_s, 1.0)) * 100.0
            s.battery_percent = max(0.0, start_pct - used)
            s.battery_voltage = 13.2 + 3.6 * s.battery_percent / 100.0
            if s.battery_percent < 20.0 and "low battery" not in s.faults:
                s.faults.append("low battery")
        return self.states

    def poll(self) -> list[DroneState]:
        return self.states

    def _select(self, indices: Sequence[int] | None) -> list[DroneState]:
        if indices is None:
            return self.states
        return [self.states[i] for i in indices if 0 <= i < len(self.states)]


# --------------------------------------------------------------------------
# MAVLink, shared plumbing
# --------------------------------------------------------------------------


class _MavlinkBase:
    """Connection handling and telemetry decoding shared by both drivers.

    One MAVLink connection per aircraft. A single shared UDP port with 500
    system IDs multiplexed onto it is tempting and wrong: one noisy drone
    then delays telemetry for the whole fleet, and there is no way to
    rate-limit a single aircraft.
    """

    name = "mavlink"
    streams_setpoints = False

    def __init__(
        self,
        endpoints: Sequence[str],
        *,
        origin: ShowOrigin | None = None,
        heartbeat_timeout: float = 3.0,
        signing_key: bytes | None = None,
    ) -> None:
        self.endpoints = list(endpoints)
        self.origin = origin or ShowOrigin()
        self.heartbeat_timeout = heartbeat_timeout
        # MAVLink 2 signing. With a key, every message out is signed and
        # every message in must carry a valid signature, or it is dropped.
        self.signing_key = signing_key
        self._rejected: list[int] = []
        self.links: list = []
        self.states: list[DroneState] = []
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()
        # The reader thread owns the socket, so replies to a request/response
        # exchange have to be handed back to the requesting call rather than
        # read from the socket a second time.
        self._ftp_replies: list[queue.Queue] = []
        self._ack_replies: list[queue.Queue] = []
        self._ftp_seq = 0

    def connect(self, count: int, *, wait_heartbeat: float = 0.0) -> None:
        from pymavlink import mavutil

        if count > len(self.endpoints):
            raise ValueError(
                f"{count} drones requested but only {len(self.endpoints)} "
                "endpoints configured"
            )
        self.links = []
        self.states = []
        self._ftp_replies = []
        self._ack_replies = []
        self._rejected = []
        for i in range(count):
            link = mavutil.mavlink_connection(self.endpoints[i], source_system=255)
            if self.signing_key:
                self._sign(link, self.signing_key)
            self.links.append(link)
            self._rejected.append(0)
            self.states.append(DroneState(index=i, sysid=0))
            self._ftp_replies.append(queue.Queue())
            self._ack_replies.append(queue.Queue())
        self._stop.clear()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        if wait_heartbeat > 0:
            # A UDP "connection" is not one. Until an aircraft has been
            # heard from, pymavlink does not know its system id and every
            # command goes out addressed to 0, so wait for contact rather
            # than reporting a fleet that may not be there.
            deadline = time.time() + wait_heartbeat
            while time.time() < deadline:
                if all(s.connected for s in self.poll()):
                    break
                time.sleep(0.05)
            missing = [s.index for s in self.poll() if not s.connected]
            if missing:
                refused = [i for i in missing if self._refused(i)]
                raise TimeoutError(
                    f"no heartbeat from drones {missing[:12]} within "
                    f"{wait_heartbeat:.1f}s on endpoints "
                    f"{[self.endpoints[i] for i in missing[:3]]}"
                    + (
                        f". Drones {refused[:12]} were transmitting, but every "
                        "message failed the MAVLink signature check: those "
                        "aircraft have no signing key, or a different one"
                        if refused else ""
                    )
                )

    # -- MAVLink 2 signing --------------------------------------------------

    @staticmethod
    def _sign(link, key: bytes) -> None:
        # The callback refuses every unsigned message, but counts it, so a
        # silent aircraft can be told apart from one with the wrong key.
        # ponytail: stricter than ArduPilot, which lets RADIO_STATUS through
        # unsigned. Nothing here reads RADIO_STATUS yet; allow msgid 109 if
        # it ever does.
        #
        # mavutil opens a link on MAVLink 1 and, on the first MAVLink 2
        # packet, swaps in a fresh parser - silently dropping any signing
        # set up before then, so the link ran unsigned and checked nothing.
        # Make that swap happen now, with the same call mavutil makes on
        # seeing a 0xFD start byte, and refuse to carry on if it did not.
        link.auto_mavlink_version(b"\xfd")
        if link.WIRE_PROTOCOL_VERSION != "2.0":
            raise RuntimeError("MAVLink signing needs a MAVLink 2 link")
        link.setup_signing(
            bytes(key), sign_outgoing=True,
            allow_unsigned_callback=lambda mav, msgid: False,
        )

    def _refused(self, i: int) -> int:
        """Messages from aircraft i dropped for a missing or bad signature."""
        sig = self.links[i].mav.signing
        return sig.badsig_count + sig.reject_count

    def provision_signing(
        self, key: bytes, indices: Sequence[int] | None = None, *, confirm: float = 5.0
    ) -> dict:
        """Load a signing key onto the aircraft, then sign this link with it.

        Sends SETUP_SIGNING, which ArduPilot stores and switches on at once,
        on every link. The message itself carries the key in the clear: do
        this over USB or on a closed bench network, never over the show
        radio. ArduPilot sends no reply, so success is confirmed by signed
        traffic coming back; an aircraft that stays unconfirmed did not take
        the key and will now be dropped as unsigned.
        """
        key = bytes(key)
        if len(key) != 32:
            raise ValueError("a MAVLink signing key is exactly 32 bytes")
        targets = self._indices(indices)
        with self._lock:
            armed = [i for i in targets if self.states[i].armed]
            sysids = {i: self.states[i].sysid for i in targets}
        if armed:
            raise RuntimeError(
                f"drones {armed[:12]} are armed; ArduPilot will not take a "
                "signing key while armed"
            )
        # MAVLink signing time: 10 microsecond units since 2015-01-01.
        stamp = int((time.time() - 1420070400) * 100_000)
        baseline = {}
        for i in targets:
            link = self.links[i]
            link.mav.setup_signing_send(sysids[i], 0, list(key), stamp)
            self._sign(link, key)
            baseline[i] = link.mav.signing.goodsig_count
        if indices is None:
            self.signing_key = key

        deadline = time.time() + confirm
        pending = set(targets)
        while pending and time.time() < deadline:
            pending = {i for i in pending
                       if self.links[i].mav.signing.goodsig_count <= baseline[i]}
            time.sleep(0.05)
        return {
            "confirmed": sorted(set(targets) - pending),
            "unconfirmed": sorted(pending),
        }

    def close(self) -> None:
        self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=2.0)
        for link in self.links:
            try:
                link.close()
            except Exception:
                pass
        self.links = []

    def poll(self) -> list[DroneState]:
        now = time.time()
        with self._lock:
            for i, s in enumerate(self.states):
                if self.signing_key and i < len(self._rejected):
                    refused = self._refused(i)
                    note = "MAVLink messages failing the signature check"
                    if refused > self._rejected[i] and note not in s.faults:
                        s.faults.append(note)
                    self._rejected[i] = refused
            for s in self.states:
                if s.last_heartbeat and now - s.last_heartbeat > self.heartbeat_timeout:
                    if s.connected:
                        s.faults.append("heartbeat timeout")
                    s.connected = False
            return [s for s in self.states]

    def _read_loop(self) -> None:
        next_heartbeat = 0.0
        while not self._stop.is_set():
            if time.time() >= next_heartbeat:
                next_heartbeat = time.time() + 1.0
                self._send_gcs_heartbeat()
            idle = True
            for i, link in enumerate(self.links):
                while (msg := self._recv(link)) is not None:
                    idle = False
                    self._handle(i, msg)
            if idle:
                time.sleep(0.002)

    @staticmethod
    def _recv(link):
        try:
            return link.recv_match(blocking=False)
        except OSError:
            # Windows reports an ICMP "port unreachable" from an earlier
            # send as WinError 10054 on the next receive. That raised out of
            # the reader thread and stopped telemetry for the whole fleet
            # because one aircraft was unreachable. Treat it as no message:
            # the heartbeat timeout still flags that aircraft.
            return None

    def _send_gcs_heartbeat(self) -> None:
        """Announce the ground station once a second on every link.

        Not decoration. A drone reached over `udpin` has no idea where to
        send telemetry until it has received a packet, so without this the
        ground station waits for a heartbeat that the aircraft cannot send
        for want of a return address. Firmware also uses the GCS heartbeat
        to detect a lost link, so stopping it is how a GCS-failsafe is
        meant to trigger.
        """
        from pymavlink import mavutil

        for link in self.links:
            try:
                link.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0,
                    0,
                    mavutil.mavlink.MAV_STATE_ACTIVE,
                )
            except Exception:
                continue

    def _handle(self, i: int, msg) -> None:
        kind = msg.get_type()
        with self._lock:
            s = self.states[i]
            if kind == "HEARTBEAT":
                from pymavlink import mavutil

                s.sysid = msg.get_srcSystem()
                s.connected = True
                s.last_heartbeat = time.time()
                s.armed = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                s.mode = COPTER_MODE_NAMES.get(msg.custom_mode, str(msg.custom_mode))
            elif kind == "GLOBAL_POSITION_INT":
                s.latitude = msg.lat / 1e7
                s.longitude = msg.lon / 1e7
                s.altitude_agl = msg.relative_alt / 1000.0
                s.position = self._to_enu(s.latitude, s.longitude, s.altitude_agl)
                s.velocity = np.array(
                    [msg.vx / 100.0, msg.vy / 100.0, -msg.vz / 100.0], dtype=np.float64
                )
            elif kind == "GPS_RAW_INT":
                s.gps_fix = msg.fix_type
                s.satellites = msg.satellites_visible
                s.hdop = msg.eph / 100.0 if msg.eph not in (0, 65535) else 99.9
            elif kind == "SYS_STATUS":
                s.battery_voltage = msg.voltage_battery / 1000.0
                s.battery_percent = float(max(msg.battery_remaining, 0))
            elif kind == "SYSTEM_TIME":
                if msg.time_unix_usec:
                    s.clock_offset = time.time() - msg.time_unix_usec / 1e6
            elif kind == "STATUSTEXT":
                text = (
                    msg.text.decode() if isinstance(msg.text, bytes) else str(msg.text)
                )
                if msg.severity <= 3 and text not in s.faults:
                    s.faults.append(text)
            elif kind == "COMMAND_ACK":
                self._ack_replies[i].put(msg)
                if msg.result != 0:  # MAV_RESULT_ACCEPTED
                    note = f"command {msg.command} rejected (result {msg.result})"
                    if note not in s.faults:
                        s.faults.append(note)
            elif kind == "FILE_TRANSFER_PROTOCOL":
                self._ftp_replies[i].put(bytes(msg.payload))

    def _to_enu(self, lat: float, lon: float, alt: float) -> np.ndarray:
        r = 6378137.0
        north = math.radians(lat - self.origin.latitude) * r
        east = (
            math.radians(lon - self.origin.longitude)
            * r
            * math.cos(math.radians(self.origin.latitude))
        )
        th = math.radians(self.origin.orientation_deg)
        x = east * math.cos(th) - north * math.sin(th)
        y = east * math.sin(th) + north * math.cos(th)
        return np.array([x, y, alt], dtype=np.float64)

    # -- commands ---------------------------------------------------------

    def set_origin(self, origin: ShowOrigin) -> None:
        """Push the show origin, orientation and AMSL reference to every drone.

        Sent as COMMAND_INT, never COMMAND_LONG: latitude and longitude go
        as integer 1e-7 degrees, and routing them through float params
        would throw away roughly a centimetre of precision, which is the
        same order as the RTK accuracy the whole show depends on.
        """
        from pymavlink import mavutil

        self.origin = origin
        amsl_mm = (
            -10_000_000 if origin.amsl is None else int(round(origin.amsl * 1000.0))
        )
        for link, s in zip(self.links, self.states):
            link.mav.command_int_send(
                link.target_system,
                link.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL,
                mavutil.mavlink.MAV_CMD_USER_2,
                0,
                0,
                0.0,
                0.0,
                0.0,
                float(origin.orientation_deg),
                int(round(origin.latitude * 1e7)),
                int(round(origin.longitude * 1e7)),
                float(amsl_mm),
            )

    def arm(self, indices: Sequence[int] | None = None) -> None:
        self._arm_command(1.0, indices)

    def disarm(self, indices: Sequence[int] | None = None) -> None:
        self._arm_command(0.0, indices)

    def _arm_command(self, value: float, indices: Sequence[int] | None) -> None:
        from pymavlink import mavutil

        for i in self._indices(indices):
            link = self.links[i]
            link.mav.command_long_send(
                link.target_system,
                link.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                value,
                0,
                0,
                0,
                0,
                0,
                0,
            )

    def set_mode(self, custom_mode: int, indices: Sequence[int] | None = None) -> None:
        from pymavlink import mavutil

        for i in self._indices(indices):
            link = self.links[i]
            link.mav.command_long_send(
                link.target_system,
                link.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                float(mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
                float(custom_mode),
                0,
                0,
                0,
                0,
                0,
            )

    def emergency(self, action: str, indices: Sequence[int] | None = None) -> None:
        """Operator-selected emergency action.

        There is deliberately no motor-cut here. An armed multirotor at 80
        metres with its motors stopped is a falling object aimed at
        whatever is below it. The safe options are to stop moving, to come
        home, or to land where it is; which one is right depends on what
        is under the drone and is the operator's call, not the software's.
        `disarm` exists only for an aircraft confirmed to be on the ground.
        """
        from pymavlink import mavutil

        commands = {
            "hold": mavutil.mavlink.MAV_CMD_DO_REPOSITION,
            "rth": mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            "land": mavutil.mavlink.MAV_CMD_NAV_LAND,
        }
        if action == "disarm":
            self.disarm(indices)
            return
        if action not in commands:
            raise ValueError(
                f"unknown emergency action {action!r}; expected one of "
                "hold, rth, land, disarm"
            )
        for i in self._indices(indices):
            link = self.links[i]
            if action == "hold":
                # Reposition to the current position = stop and hold.
                link.mav.command_long_send(
                    link.target_system,
                    link.target_component,
                    commands[action],
                    0,
                    -1,
                    0,
                    0,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                )
            else:
                link.mav.command_long_send(
                    link.target_system,
                    link.target_component,
                    commands[action],
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )

    def release_hold(self, indices: Sequence[int] | None = None) -> None:
        """Put the aircraft back into its show mode after a hold.

        BRAKE and LOITER ignore position targets, so without this a
        resumed show streams setpoints into an aircraft that is no longer
        listening to them.
        """
        mode = getattr(self, "GUIDED_MODE", SHOW_CUSTOM_MODE)
        self.set_mode(mode, indices)

    def set_lights(self, index: int, color: np.ndarray, channels=(9, 10, 11)) -> None:
        """Drive an RGB payload on three PWM channels, 1000-2000 us.

        Matches the common show-drone wiring where the LED controller reads
        servo outputs. Firmware with a native light-program channel should
        use that instead: 500 drones x 3 servo commands per colour change
        is a lot of packets for something the aircraft can play back
        locally from its own uploaded program.
        """
        from pymavlink import mavutil

        link = self.links[index]
        for ch, value in zip(channels, np.asarray(color, dtype=np.float64)):
            pwm = 1000.0 + 1000.0 * float(np.clip(value, 0, 255)) / 255.0
            link.mav.command_long_send(
                link.target_system,
                link.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                float(ch),
                pwm,
                0,
                0,
                0,
                0,
                0,
            )

    def set_show_start(
        self, at_unix_time: float, indices: Sequence[int] | None = None
    ) -> None:
        """Tell each aircraft the absolute instant the show begins.

        Sent as COMMAND_INT: whole seconds as an int32 in `x`, the
        millisecond remainder in `y`. Not as a float param - see
        SHOW_START_COMMAND above for why 128 seconds of error would
        otherwise be representable.

        This is what makes the fleet move together. The aircraft counts
        down to this instant from its own GNSS-derived clock, so the
        command can be sent to 500 drones one packet at a time without the
        start being smeared across the fleet.
        """
        from pymavlink import mavutil

        whole = int(math.floor(at_unix_time))
        millis = int(round((at_unix_time - whole) * 1000.0))
        if not (-(2**31) <= whole < 2**31):
            raise ValueError(
                f"start time {at_unix_time} does not fit in an int32 field; "
                "the show epoch is only valid until 2038"
            )
        for i in self._indices(indices):
            link = self.links[i]
            link.mav.command_int_send(
                link.target_system,
                link.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL,
                mavutil.mavlink.MAV_CMD_USER_3,
                0, 0,
                0.0, 0.0, 0.0, 0.0,
                whole,
                millis,
                0.0,
            )

    def show_control(self, action: str, indices: Sequence[int] | None = None) -> None:
        """Act on the stored show, via MAV_CMD_USER_1.

        reload and clear manage the uploaded plan. suspend and resume stop
        and restart show playback on the aircraft itself: a ground-side
        clock freeze cannot work when the aircraft owns the clock, and a
        drone that missed the freeze would keep flying while its neighbours
        stopped.
        """
        from pymavlink import mavutil

        codes = {"reload": 1, "clear": 2, "pyro_test": 3, "suspend": 4, "resume": 5}
        if action not in codes:
            raise ValueError(
                f"unknown show action {action!r}; expected one of {', '.join(codes)}"
            )
        for i in self._indices(indices):
            link = self.links[i]
            link.mav.command_long_send(
                link.target_system,
                link.target_component,
                mavutil.mavlink.MAV_CMD_USER_1,
                0,
                float(codes[action]),
                0, 0, 0, 0, 0, 0,
            )

    # -- MAVLink FTP client ------------------------------------------------

    def _ftp_transaction(
        self,
        index: int,
        opcode: int,
        *,
        session: int = 0,
        offset: int = 0,
        data: bytes = b"",
        timeout: float = 3.0,
        retries: int = 3,
    ) -> dict:
        """One FTP request, retried, with the reply matched by opcode.

        UDP drops packets, and an upload that silently loses a chunk is an
        aircraft flying a truncated trajectory. Every exchange is therefore
        acknowledged and the whole file is CRC-checked afterwards.
        """
        link = self.links[index]
        replies = self._ftp_replies[index]
        last_error = "no reply"

        for _ in range(max(1, retries)):
            self._ftp_seq = (self._ftp_seq + 1) & 0xFFFF
            seq = self._ftp_seq
            head = struct.pack(
                FTP_HEADER, seq, session, opcode, len(data), 0, 0, 0, offset
            )
            payload = head + data
            payload += b"\0" * (FTP_PAYLOAD_LEN - len(payload))
            while not replies.empty():      # discard stale replies
                replies.get_nowait()
            link.mav.file_transfer_protocol_send(0, link.target_system, 0, list(payload))

            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    raw = replies.get(timeout=max(0.01, deadline - time.time()))
                except queue.Empty:
                    break
                (
                    _seq, rsession, ropcode, rsize, req_opcode, _burst, _pad, roffset
                ) = struct.unpack(FTP_HEADER, raw[:FTP_HEADER_LEN])
                if req_opcode != opcode:
                    continue
                body = raw[FTP_HEADER_LEN : FTP_HEADER_LEN + rsize]
                if ropcode == FTP_OP_ACK:
                    return {"session": rsession, "offset": roffset, "data": body}
                if ropcode == FTP_OP_NAK:
                    code = body[0] if body else 0
                    last_error = f"NAK {code}"
                    break
        raise IOError(
            f"MAVLink FTP opcode {opcode} to drone {index} failed: {last_error}"
        )

    def ftp_write(
        self, index: int, data: bytes, path: str = SHOW_PLAN_PATH
    ) -> int:
        """Write a file to one aircraft and verify it by CRC32.

        Returns the number of bytes written. Raises if the aircraft's CRC
        disagrees with ours, because a plan that differs by one byte is a
        drone in the wrong place.
        """
        name = path.encode("utf-8") + b"\0"
        session = self._ftp_transaction(index, FTP_OP_CREATE, data=name)["session"]
        try:
            offset = 0
            while offset < len(data):
                chunk = data[offset : offset + FTP_MAX_DATA]
                self._ftp_transaction(
                    index, FTP_OP_WRITE, session=session, offset=offset, data=chunk
                )
                offset += len(chunk)

            reply = self._ftp_transaction(
                index, FTP_OP_CRC32, session=session, data=name
            )
            if len(reply["data"]) >= 4:
                remote = struct.unpack("<I", reply["data"][:4])[0]
                local = zlib.crc32(data) & 0xFFFFFFFF
                if remote != local:
                    raise IOError(
                        f"drone {index} stored a file whose CRC32 is {remote:08x} "
                        f"but we sent {local:08x}: the upload is corrupt, do not fly it"
                    )
        finally:
            try:
                self._ftp_transaction(
                    index, FTP_OP_TERMINATE, session=session, retries=1, timeout=1.0
                )
            except IOError:
                pass
        return len(data)

    def _indices(self, indices: Sequence[int] | None) -> list[int]:
        if indices is None:
            return list(range(len(self.links)))
        return [i for i in indices if 0 <= i < len(self.links)]


class MavlinkNativeDriver(_MavlinkBase):
    """Upload the show, then get out of the way.

    The aircraft holds its own trajectory and light program and plays them
    from a GNSS-derived clock. The ground station sets the origin, uploads,
    arms, names a start instant and then only watches. Nothing
    time-critical crosses the link once the show is running, which is why
    this is the only option at show scale.

    Plans go up over MAVLink FTP, each one CRC-verified against what the
    aircraft actually stored. `upload_hook` overrides that for firmware
    using a different transport; it receives (link, index, plan) and is
    expected to raise on failure, because reporting a successful upload
    that did not happen is a lie an operator could fly on.
    """

    name = "mavlink-native"
    streams_setpoints = False

    def __init__(
        self,
        endpoints: Sequence[str],
        *,
        origin: ShowOrigin | None = None,
        upload_hook: Callable[[object, int, dict], None] | None = None,
        plan_path: str = SHOW_PLAN_PATH,
        signing_key: bytes | None = None,
    ) -> None:
        super().__init__(endpoints, origin=origin, signing_key=signing_key)
        self.upload_hook = upload_hook
        self.plan_path = plan_path
        self._start_time: float | None = None

    def upload(self, mission: MissionPackage) -> None:
        body = mission.to_dict()
        failures: list[tuple[int, str]] = []
        for plan in body["plans"]:
            i = int(plan["drone"])
            if i >= len(self.links):
                continue
            try:
                if self.upload_hook is not None:
                    self.upload_hook(self.links[i], i, plan)
                else:
                    blob = json.dumps(plan, separators=(",", ":")).encode("utf-8")
                    self.ftp_write(i, blob, self.plan_path)
            except (IOError, OSError) as exc:
                failures.append((i, str(exc)))
                continue
            with self._lock:
                self.states[i].mission_uploaded = True
                self.states[i].mission_checksum = body["checksum"]

        if failures:
            # Partial success is the dangerous outcome: some aircraft hold
            # the plan and some do not, and the ones that do not will sit
            # on the pad while the rest take off. Fail the whole upload.
            names = ", ".join(f"{i} ({why})" for i, why in failures[:5])
            raise IOError(
                f"{len(failures)} of {len(body['plans'])} plan uploads failed: "
                f"{names}. The fleet is now in a mixed state - re-upload before "
                "arming, and check the preflight checksum report."
            )

    def start(self, at_unix_time: float) -> None:
        """Hand the show to the aircraft, then get out of the way.

        Order matters: the start instant goes out first, then show mode.
        A drone that enters show mode without knowing when to begin has to
        wait, which is safe; a drone told to begin before it is in show
        mode would ignore it.
        """
        self._start_time = at_unix_time
        self.set_show_start(at_unix_time)
        self.set_mode(SHOW_CUSTOM_MODE)

    def setpoint(self, index: int, position: np.ndarray, color: np.ndarray) -> None:
        raise NotImplementedError(
            "native show mode executes onboard; it takes no setpoints"
        )


class MavlinkGuidedDriver(_MavlinkBase):
    """Stream position setpoints to stock ArduPilot in GUIDED mode.

    Honest about its ceiling: every drone needs a setpoint every 100 ms, so
    a 50 drone fleet is already 500 packets per second and a lost or late
    packet is a drone that stops where it is. It is the right tool for a
    bench test, a SITL rehearsal or a small indoor show, and the wrong tool
    for a show with an audience.
    """

    name = "mavlink-guided"
    streams_setpoints = True

    GUIDED_MODE = 4  # ArduCopter GUIDED

    def __init__(
        self, endpoints: Sequence[str], *, origin: ShowOrigin | None = None,
        signing_key: bytes | None = None,
    ) -> None:
        if len(endpoints) > GUIDED_STREAM_MAX_DRONES:
            raise ValueError(
                f"{len(endpoints)} endpoints exceeds the {GUIDED_STREAM_MAX_DRONES} "
                "drone ceiling for ground-streamed setpoints. Use "
                "MavlinkNativeDriver with show-mode firmware: a show of this "
                "size must not depend on the radio staying up, packet by "
                "packet, for its entire duration."
            )
        super().__init__(endpoints, origin=origin, signing_key=signing_key)

    def upload(self, mission: MissionPackage) -> None:
        """Nothing is uploaded: the plan stays on the ground and is streamed.

        Recorded on each drone anyway so the operator can see which plan
        the ground station is streaming, and so a checksum mismatch after a
        re-plan is visible.
        """
        checksum = mission.to_dict()["checksum"]
        with self._lock:
            for s in self.states:
                s.mission_uploaded = True
                s.mission_checksum = checksum

    def start(self, at_unix_time: float) -> None:
        self.set_mode(self.GUIDED_MODE)

    def setpoint(self, index: int, position: np.ndarray, color: np.ndarray) -> None:
        from pymavlink import mavutil

        link = self.links[index]
        p = np.asarray(position, dtype=np.float64)
        # Show frame is ENU; MAVLink LOCAL_NED is north-east-down.
        north, east, down = float(p[1]), float(p[0]), float(-p[2])
        type_mask = (
            # ignore velocity, acceleration and yaw rate; position only
            0b0000_1111_1111_1000
        )
        link.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            link.target_system,
            link.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            north,
            east,
            down,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        self.set_lights(index, color)


DRIVERS = {
    "simulator": SimDriver,
    "mavlink-native": MavlinkNativeDriver,
    "mavlink-guided": MavlinkGuidedDriver,
}
