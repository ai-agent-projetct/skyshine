"""Software in the loop: aircraft that speak real MAVLink over real sockets.

Two things live here.

`VirtualFleet` is a fleet of MAVLink-2 aircraft, each on its own UDP port,
that behave the way show-mode firmware is supposed to: they stream
telemetry, acknowledge commands, accept a show plan over MAVLink FTP, and
fly it from their own clock once told the absolute start instant. It exists
to exercise the wire protocol in `drivers.py` - packet encoding, command
semantics, FTP transfer, ack handling - which the pure-Python SimDriver
cannot touch because it never serialises anything.

`ArduPilotSITL` launches real ArduPilot SITL instances when they are
installed, and says plainly what to install when they are not.

What this is NOT: ArduPilot. There is no flight dynamics model, no
controller, no EKF. A virtual aircraft tracks its commanded position with
first-order lag and noise. It will tell you that your MAVLink is correct
and your show executes; it will not tell you whether a real airframe can
fly the trajectory. For that, run the same ground station against
`ArduPilotSITL`.
"""

from __future__ import annotations

import json
import math
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .show import ShowOrigin, ease

# Where a show plan is written on the aircraft. Real firmware picks its own
# path; both ends only have to agree.
SHOW_PLAN_PATH = "@SHOW/plan.json"

# MAVLink FTP wire format: 12 byte header, then up to 239 bytes of data,
# all inside a fixed 251 byte FILE_TRANSFER_PROTOCOL payload.
FTP_HEADER = "<HBBBBBBI"
FTP_HEADER_LEN = 12
FTP_MAX_DATA = 239
FTP_PAYLOAD_LEN = 251

OP_TERMINATE = 1
OP_RESET = 2
OP_OPEN_RO = 4
OP_READ = 5
OP_CREATE_FILE = 6
OP_WRITE = 7
OP_OPEN_WO = 11
OP_CRC32 = 14
OP_ACK = 128
OP_NAK = 129

NAK_FAIL = 1
NAK_INVALID_SESSION = 2
NAK_NO_SESSIONS = 3
NAK_EOF = 6
NAK_UNKNOWN_COMMAND = 7

# ArduCopter custom mode numbers, for readable telemetry. 127 is the show
# mode added by show-capable firmware.
COPTER_MODES = {
    0: "STABILIZE", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED", 5: "LOITER",
    6: "RTL", 9: "LAND", 17: "BRAKE", 20: "GUIDED_NOGPS", 127: "SHOW",
}


class _SocketWriter:
    """Adapts a VirtualAircraft's socket to the file object MAVLink writes to."""

    def __init__(self, aircraft: "VirtualAircraft") -> None:
        self.aircraft = aircraft

    def write(self, buf) -> None:
        self.aircraft._write(bytes(buf))


def decode_ftp(payload: bytes) -> dict:
    seq, session, opcode, size, req_opcode, burst, _pad, offset = struct.unpack(
        FTP_HEADER, bytes(payload[:FTP_HEADER_LEN])
    )
    return {
        "seq": seq,
        "session": session,
        "opcode": opcode,
        "size": size,
        "req_opcode": req_opcode,
        "burst_complete": burst,
        "offset": offset,
        "data": bytes(payload[FTP_HEADER_LEN : FTP_HEADER_LEN + size]),
    }


def encode_ftp(
    seq: int,
    session: int,
    opcode: int,
    req_opcode: int = 0,
    offset: int = 0,
    data: bytes = b"",
    burst_complete: int = 0,
) -> bytes:
    if len(data) > FTP_MAX_DATA:
        raise ValueError(f"FTP payload of {len(data)} exceeds {FTP_MAX_DATA} bytes")
    head = struct.pack(
        FTP_HEADER, seq, session, opcode, len(data), req_opcode, burst_complete, 0, offset
    )
    body = head + data
    return body + b"\0" * (FTP_PAYLOAD_LEN - len(body))


# --------------------------------------------------------------------------
# One aircraft
# --------------------------------------------------------------------------


@dataclass
class AircraftFaults:
    """Faults a virtual aircraft can be told to exhibit."""

    # Stop transmitting entirely at this show time, seconds.
    go_silent_at: float | None = None
    # Refuse to arm, as a real aircraft does on a prearm failure.
    refuse_arm: bool = False
    # Reject an FTP write, so the uploader's retry path gets exercised.
    fail_upload: bool = False
    # Starting battery percentage.
    battery_percent: float = 100.0
    # GNSS fix type to report. 6 is RTK fixed, 3 is plain 3D.
    gps_fix: int = 6
    # Constant offset added to the reported position, metres, in show ENU.
    position_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Clock error in seconds: the aircraft believes the wrong time, so it
    # starts the show early or late. This is the fault that collides a
    # fleet, so it must be reproducible.
    clock_offset: float = 0.0


class VirtualAircraft:
    """A MAVLink-2 aircraft on one UDP port.

    Deliberately strict where real firmware is strict. It NAKs
    MAV_CMD_USER_2 arriving as a COMMAND_LONG, because the whole point of
    that command being COMMAND_INT is to carry latitude and longitude as
    integer 1e-7 degrees rather than through a float32 that would throw
    away a centimetre.
    """

    def __init__(
        self,
        index: int,
        gcs_port: int,
        *,
        host: str = "127.0.0.1",
        faults: AircraftFaults | None = None,
        tracking_tau: float = 0.35,
        noise: float = 0.02,
        seed: int | None = None,
        signing_key: bytes | None = None,
    ) -> None:
        from pymavlink.dialects.v20 import ardupilotmega as mavlink2

        self.index = index
        self.port = gcs_port
        self.sysid = index + 1
        self.faults = faults or AircraftFaults()
        self.tracking_tau = tracking_tau
        self.noise = noise
        self.rng = np.random.default_rng(index if seed is None else seed)

        # The aircraft owns a plain socket on an ephemeral port and streams
        # to the ground station's port, which is how a real autopilot's
        # telemetry output is configured. Going through mavutil here would
        # drag in its udpout behaviour, which on Windows binds the local
        # socket to the *destination* port and then competes with the
        # listener for the very datagrams it just sent.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, 0))
        self.sock.setblocking(False)
        self.gcs_addr = (host, gcs_port)
        self.rx_bytes = 0
        self.tx_bytes = 0

        self.mav = mavlink2.MAVLink(
            _SocketWriter(self), srcSystem=self.sysid, srcComponent=1
        )
        self.mav.robust_parsing = True
        self.use_key(signing_key)

        self.boot = time.time()
        self.armed = False
        self.custom_mode = 5  # LOITER
        self.origin = ShowOrigin()
        self.origin_set = False

        self.position = np.zeros(3)      # true position, show ENU metres
        self.target = np.zeros(3)        # commanded position
        self.velocity = np.zeros(3)
        self.leds = np.zeros(3, dtype=int)

        self.plan: dict | None = None
        self.plan_bytes = b""
        self.show_start_unix: float | None = None
        # Show time at which playback was suspended, or None when running.
        self.suspended_at: float | None = None
        self.silent = False
        # ArduPilot disarms by itself this long after landing (DISARM_DELAY).
        self.disarm_delay = 10.0
        self._has_flown = False
        self._landed_for = 0.0

        self.commands_seen: list[int] = []
        self.nacks_sent: list[tuple[int, int]] = []
        self._ftp_files: dict[str, bytearray] = {}
        self._ftp_sessions: dict[int, str] = {}
        self._next_stream = 0.0
        self._next_slow = 0.0

    # -- clock ------------------------------------------------------------

    def now(self) -> float:
        """The aircraft's own idea of the time, which may be wrong."""
        return time.time() + self.faults.clock_offset

    @property
    def show_time(self) -> float | None:
        if self.show_start_unix is None:
            return None
        if self.suspended_at is not None:
            return self.suspended_at
        return self.now() - self.show_start_unix

    # -- main step --------------------------------------------------------

    def step(self, dt: float) -> None:
        self.drain()
        self.fly(dt)
        self.stream()

    def drain(self) -> None:
        while True:
            try:
                data, _addr = self.sock.recvfrom(65535)
            except (BlockingIOError, OSError):
                return
            if not data:
                return
            self.rx_bytes += len(data)
            try:
                for msg in self.mav.parse_buffer(data) or ():
                    self.handle(msg)
            except Exception:
                # A malformed frame is a dropped frame, not a dead radio.
                continue

    def _write(self, buf: bytes) -> None:
        if self.silent:
            return
        try:
            self.sock.sendto(buf, self.gcs_addr)
            self.tx_bytes += len(buf)
        except OSError:
            pass

    def fly(self, dt: float) -> None:
        st = self.show_time
        if self.custom_mode == 127 and self.armed and self.plan and st is not None:
            self.target = self.evaluate_plan(max(st, 0.0))
            self.leds = self.evaluate_lights(max(st, 0.0))
        elif self.custom_mode == 9 and self.armed:  # LAND
            self.target = np.array([self.position[0], self.position[1], 0.0])
        elif self.custom_mode == 6 and self.armed:  # RTL
            self.target = np.array([0.0, 0.0, max(self.position[2], 15.0)])

        if not self.armed:
            self.position = np.array([self.target[0], self.target[1], 0.0])
            self.velocity = np.zeros(3)
            return

        # First-order lag toward the commanded position. Not a flight model,
        # just enough that telemetry is not a perfect echo of the command.
        alpha = 1.0 - math.exp(-dt / max(self.tracking_tau, 1e-3))
        step = (self.target - self.position) * alpha
        self.velocity = step / max(dt, 1e-6)
        self.position = self.position + step
        if self.position[2] < 0:
            self.position[2] = 0.0

        # Auto-disarm once landed, as ArduPilot does - but only after a
        # flight. Sitting armed on the pad waiting for the start is normal.
        if self.position[2] > 0.5:
            self._has_flown = True
        landed = self.position[2] <= 0.05 and self.target[2] <= 0.05
        self._landed_for = self._landed_for + dt if self._has_flown and landed else 0.0
        if self._landed_for >= self.disarm_delay:
            self.armed = False
            self.show_start_unix = None
            self._has_flown = self._landed_for = 0.0

    def evaluate_plan(self, t: float) -> np.ndarray:
        pts = self.plan["points"]
        easings = self.plan["easings"]
        if t <= pts[0][0]:
            return np.array(pts[0][1:], dtype=float)
        if t >= pts[-1][0]:
            return np.array(pts[-1][1:], dtype=float)
        k = 0
        for i in range(len(pts) - 1):
            if pts[i][0] <= t:
                k = i
        a, b = pts[k], pts[k + 1]
        span = b[0] - a[0]
        tau = 0.0 if span <= 0 else (t - a[0]) / span
        kind = easings[k] if k < len(easings) else "hold"
        s = 0.0 if kind == "hold" else float(ease(kind, tau))
        pa = np.array(a[1:], dtype=float)
        pb = np.array(b[1:], dtype=float)
        return pa + (pb - pa) * s

    def evaluate_lights(self, t: float) -> np.ndarray:
        keys = self.plan.get("lights") or []
        if not keys:
            return np.zeros(3, dtype=int)
        if t <= keys[0][0]:
            return np.array(keys[0][1:], dtype=int)
        for i in range(len(keys) - 1):
            if keys[i][0] <= t <= keys[i + 1][0]:
                a, b = keys[i], keys[i + 1]
                span = b[0] - a[0]
                f = 0.0 if span <= 0 else (t - a[0]) / span
                return np.array(
                    [round(a[j] + (b[j] - a[j]) * f) for j in (1, 2, 3)], dtype=int
                )
        return np.array(keys[-1][1:], dtype=int)

    # -- telemetry --------------------------------------------------------

    def stream(self) -> None:
        if self.silent:
            return
        st = self.show_time
        if (
            self.faults.go_silent_at is not None
            and st is not None
            and st >= self.faults.go_silent_at
        ):
            self.silent = True
            return

        now = time.time()
        boot_ms = int((now - self.boot) * 1000) & 0xFFFFFFFF
        if now >= self._next_stream:
            self._next_stream = now + 0.2
            self.send_position(boot_ms)
        if now >= self._next_slow:
            self._next_slow = now + 1.0
            self.send_heartbeat()
            self.send_gps()
            self.send_sys_status()
            self.send_system_time(boot_ms)

    def send_heartbeat(self) -> None:
        from pymavlink import mavutil

        base = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
        if self.armed:
            base |= mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        self.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_QUADROTOR,
            mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
            base,
            self.custom_mode,
            mavutil.mavlink.MAV_STATE_ACTIVE
            if self.armed
            else mavutil.mavlink.MAV_STATE_STANDBY,
        )

    def reported_position(self) -> np.ndarray:
        bias = np.asarray(self.faults.position_bias, dtype=float)
        return self.position + bias + self.rng.normal(0.0, self.noise, size=3)

    def send_position(self, boot_ms: int) -> None:
        p = self.reported_position()
        lat, lon = self._to_wgs84(p)
        amsl = (self.origin.amsl or 0.0) + p[2]
        self.mav.global_position_int_send(
            boot_ms,
            int(round(lat * 1e7)),
            int(round(lon * 1e7)),
            int(round(amsl * 1000)),
            int(round(p[2] * 1000)),
            int(round(self.velocity[1] * 100)),   # vx = north
            int(round(self.velocity[0] * 100)),   # vy = east
            int(round(-self.velocity[2] * 100)),  # vz = down
            0,
        )

    def send_gps(self) -> None:
        p = self.reported_position()
        lat, lon = self._to_wgs84(p)
        fix = self.faults.gps_fix
        self.mav.gps_raw_int_send(
            int(time.time() * 1e6),
            fix,
            int(round(lat * 1e7)),
            int(round(lon * 1e7)),
            int(round(((self.origin.amsl or 0.0) + p[2]) * 1000)),
            60 if fix >= 5 else 340,   # eph, centimetres of HDOP*100
            80 if fix >= 5 else 420,
            0,
            0,
            22 if fix >= 5 else 7,
        )

    def send_sys_status(self) -> None:
        pct = self.faults.battery_percent
        st = self.show_time
        if st is not None and st > 0:
            pct = max(0.0, pct - st / 900.0 * 100.0)
        voltage = int((13.2 + 3.6 * pct / 100.0) * 1000)
        self.mav.sys_status_send(
            0, 0, 0, 250, voltage, -1, int(pct), 0, 0, 0, 0, 0, 0
        )

    def send_system_time(self, boot_ms: int) -> None:
        self.mav.system_time_send(int(self.now() * 1e6), boot_ms)

    def _to_wgs84(self, p: np.ndarray) -> tuple[float, float]:
        th = math.radians(self.origin.orientation_deg)
        east = p[0] * math.cos(th) + p[1] * math.sin(th)
        north = -p[0] * math.sin(th) + p[1] * math.cos(th)
        r = 6378137.0
        lat = self.origin.latitude + math.degrees(north / r)
        lon = self.origin.longitude + math.degrees(
            east / (r * math.cos(math.radians(self.origin.latitude)))
        )
        return lat, lon

    # -- MAVLink 2 signing ----------------------------------------------

    def use_key(self, key: bytes | None, timestamp: int = 0) -> None:
        """Sign and check traffic with this key, as ArduPilot does.

        An all-zero key (or None) turns signing off. With a key, unsigned
        messages are dropped except RADIO_STATUS and RADIO, which ArduPilot
        also lets through because a telemetry radio injects them.
        """
        s = self.mav.signing
        if not key or not any(key):
            s.secret_key = None
            s.sign_outgoing = False
            s.allow_unsigned_callback = None
            return
        s.secret_key = bytes(key)
        s.sign_outgoing = True
        s.link_id = 0
        s.allow_unsigned_callback = lambda mav, msgid: msgid in (109, 166)
        now = int((time.time() - 1420070400) * 100_000)
        s.timestamp = max(s.timestamp, int(timestamp), now)

    @property
    def signed(self) -> bool:
        return self.mav.signing.secret_key is not None

    # -- command handling -------------------------------------------------

    def handle(self, msg) -> None:
        kind = msg.get_type()
        if kind == "SETUP_SIGNING":
            if self.armed:
                self.mav.statustext_send(4, b"ERROR: signing not set while armed")
            else:
                self.use_key(bytes(msg.secret_key), msg.initial_timestamp)
        elif kind == "COMMAND_LONG":
            self.handle_command(msg, is_int=False)
        elif kind == "COMMAND_INT":
            self.handle_command(msg, is_int=True)
        elif kind == "SET_POSITION_TARGET_LOCAL_NED":
            # MAVLink LOCAL_NED is north-east-down; the show frame is ENU.
            self.target = np.array([msg.y, msg.x, -msg.z], dtype=float)
        elif kind == "FILE_TRANSFER_PROTOCOL":
            self.handle_ftp(msg)

    def handle_command(self, msg, *, is_int: bool) -> None:
        from pymavlink import mavutil

        cmd = msg.command
        self.commands_seen.append(cmd)
        M = mavutil.mavlink
        result = M.MAV_RESULT_ACCEPTED

        if cmd == M.MAV_CMD_COMPONENT_ARM_DISARM:
            want = msg.param1 > 0.5
            if want and self.faults.refuse_arm:
                result = M.MAV_RESULT_TEMPORARILY_REJECTED
            else:
                self.armed = want
                self._has_flown = self._landed_for = 0.0
                if not want:
                    self.show_start_unix = None

        elif cmd == M.MAV_CMD_DO_SET_MODE:
            self.custom_mode = int(msg.param2)

        elif cmd == M.MAV_CMD_USER_1:
            action = int(msg.param1)
            if action == 2:  # clear the loaded show
                self.plan = None
                self.plan_bytes = b""
                self.show_start_unix = None
            elif action == 1:  # reload from storage
                self._load_plan_from_storage()
            elif action == 4:  # suspend playback, hold position
                if self.suspended_at is None and self.show_time is not None:
                    self.suspended_at = self.show_time
            elif action == 5:  # resume from where playback stopped
                if self.suspended_at is not None and self.show_start_unix is not None:
                    # Shift the start instant forward by however long we were
                    # held, so playback continues rather than jumping.
                    self.show_start_unix = self.now() - self.suspended_at
                    self.suspended_at = None

        elif cmd == M.MAV_CMD_USER_2:
            if not is_int:
                # Refused on purpose: as a COMMAND_LONG the latitude would
                # arrive through a float32 and lose about a centimetre,
                # which is the same order as the RTK accuracy the show
                # depends on.
                result = M.MAV_RESULT_DENIED
            else:
                self.origin = ShowOrigin(
                    latitude=msg.x / 1e7,
                    longitude=msg.y / 1e7,
                    amsl=None if msg.z < -9_999_999 else msg.z / 1000.0,
                    orientation_deg=float(msg.param4),
                )
                self.origin_set = True

        elif cmd == M.MAV_CMD_USER_3:
            # Show start instant. Seconds travel as an int32 in the x field:
            # a float32 cannot hold unix epoch seconds, where its spacing is
            # 128 s, and a show started 128 s early is a crash.
            if not is_int:
                result = M.MAV_RESULT_DENIED
            else:
                self.show_start_unix = float(msg.x) + float(msg.y) / 1000.0

        elif cmd == M.MAV_CMD_DO_SET_SERVO:
            channel = int(msg.param1)
            pwm = float(msg.param2)
            value = int(round((pwm - 1000.0) / 1000.0 * 255.0))
            if channel in (9, 10, 11):
                self.leds[channel - 9] = max(0, min(255, value))

        elif cmd == M.MAV_CMD_NAV_RETURN_TO_LAUNCH:
            self.custom_mode = 6
        elif cmd == M.MAV_CMD_NAV_LAND:
            self.custom_mode = 9
        elif cmd == M.MAV_CMD_DO_REPOSITION:
            self.custom_mode = 17  # BRAKE
            self.target = self.position.copy()
        else:
            result = M.MAV_RESULT_UNSUPPORTED

        if result != M.MAV_RESULT_ACCEPTED:
            self.nacks_sent.append((cmd, result))
        if not self.silent:
            self.mav.command_ack_send(cmd, result)

    def _load_plan_from_storage(self) -> None:
        blob = self._ftp_files.get(SHOW_PLAN_PATH)
        if not blob:
            return
        try:
            self.plan = json.loads(bytes(blob).decode("utf-8"))
            self.plan_bytes = bytes(blob)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.plan = None

    # -- MAVLink FTP server ------------------------------------------------

    def handle_ftp(self, msg) -> None:
        op = decode_ftp(bytes(msg.payload))
        opcode = op["opcode"]
        reply: bytes | None = None

        if opcode in (OP_CREATE_FILE, OP_OPEN_WO):
            path = op["data"].split(b"\0")[0].decode("utf-8", "replace")
            session = (max(self._ftp_sessions) + 1) if self._ftp_sessions else 1
            self._ftp_sessions[session] = path
            if opcode == OP_CREATE_FILE or path not in self._ftp_files:
                self._ftp_files[path] = bytearray()
            reply = encode_ftp(op["seq"] + 1, session, OP_ACK, opcode)

        elif opcode == OP_WRITE:
            path = self._ftp_sessions.get(op["session"])
            if path is None:
                reply = encode_ftp(
                    op["seq"] + 1, op["session"], OP_NAK, opcode,
                    data=bytes([NAK_INVALID_SESSION]),
                )
            elif self.faults.fail_upload:
                reply = encode_ftp(
                    op["seq"] + 1, op["session"], OP_NAK, opcode, data=bytes([NAK_FAIL])
                )
            else:
                buf = self._ftp_files[path]
                end = op["offset"] + len(op["data"])
                if len(buf) < end:
                    buf.extend(b"\0" * (end - len(buf)))
                buf[op["offset"] : end] = op["data"]
                reply = encode_ftp(
                    op["seq"] + 1, op["session"], OP_ACK, opcode,
                    offset=op["offset"], data=struct.pack("<I", len(op["data"])),
                )

        elif opcode == OP_CRC32:
            import zlib

            path = op["data"].split(b"\0")[0].decode("utf-8", "replace")
            blob = self._ftp_files.get(path)
            if blob is None:
                reply = encode_ftp(
                    op["seq"] + 1, op["session"], OP_NAK, opcode, data=bytes([NAK_FAIL])
                )
            else:
                crc = zlib.crc32(bytes(blob)) & 0xFFFFFFFF
                reply = encode_ftp(
                    op["seq"] + 1, op["session"], OP_ACK, opcode,
                    data=struct.pack("<I", crc),
                )

        elif opcode == OP_TERMINATE:
            path = self._ftp_sessions.pop(op["session"], None)
            if path == SHOW_PLAN_PATH:
                self._load_plan_from_storage()
            reply = encode_ftp(op["seq"] + 1, op["session"], OP_ACK, opcode)

        elif opcode == OP_RESET:
            self._ftp_sessions.clear()
            reply = encode_ftp(op["seq"] + 1, 0, OP_ACK, opcode)

        elif opcode == OP_OPEN_RO:
            path = op["data"].split(b"\0")[0].decode("utf-8", "replace")
            blob = self._ftp_files.get(path)
            if blob is None:
                reply = encode_ftp(
                    op["seq"] + 1, 0, OP_NAK, opcode, data=bytes([NAK_FAIL])
                )
            else:
                session = (max(self._ftp_sessions) + 1) if self._ftp_sessions else 1
                self._ftp_sessions[session] = path
                reply = encode_ftp(
                    op["seq"] + 1, session, OP_ACK, opcode,
                    data=struct.pack("<I", len(blob)),
                )

        elif opcode == OP_READ:
            path = self._ftp_sessions.get(op["session"])
            blob = self._ftp_files.get(path or "")
            if blob is None:
                reply = encode_ftp(
                    op["seq"] + 1, op["session"], OP_NAK, opcode,
                    data=bytes([NAK_INVALID_SESSION]),
                )
            else:
                chunk = bytes(blob[op["offset"] : op["offset"] + min(op["size"], FTP_MAX_DATA)])
                if not chunk:
                    reply = encode_ftp(
                        op["seq"] + 1, op["session"], OP_NAK, opcode,
                        data=bytes([NAK_EOF]),
                    )
                else:
                    reply = encode_ftp(
                        op["seq"] + 1, op["session"], OP_ACK, opcode,
                        offset=op["offset"], data=chunk,
                    )
        else:
            reply = encode_ftp(
                op["seq"] + 1, op["session"], OP_NAK, opcode,
                data=bytes([NAK_UNKNOWN_COMMAND]),
            )

        if reply is not None and not self.silent:
            self.mav.file_transfer_protocol_send(0, 0, 0, list(reply))

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# The fleet
# --------------------------------------------------------------------------


class VirtualFleet:
    """N virtual aircraft serviced by one thread.

    One thread, not one per aircraft: a hundred Python threads contending
    on the GIL would add scheduling jitter that looks exactly like the
    timing faults these tests are meant to detect.
    """

    def __init__(
        self,
        count: int,
        *,
        base_port: int = 14700,
        host: str = "127.0.0.1",
        faults: dict[int, AircraftFaults] | None = None,
        rate: float = 50.0,
        noise: float = 0.02,
        signing_key: bytes | None = None,
    ) -> None:
        self.host = host
        self.rate = rate
        faults = faults or {}
        self.aircraft = [
            VirtualAircraft(
                i,
                _free_port(host, base_port + i),
                host=host,
                faults=faults.get(i, AircraftFaults()),
                noise=noise,
                signing_key=signing_key,
            )
            for i in range(count)
        ]
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors: list[tuple[int, str]] = []

    @property
    def endpoints(self) -> list[str]:
        """What to hand a MAVLink driver as its endpoint list.

        The ground station listens on one port per aircraft and each
        aircraft streams to it. One port per aircraft, rather than 500
        system ids multiplexed onto one socket, so a single noisy drone
        cannot delay telemetry for the whole fleet.
        """
        return [f"udpin:{self.host}:{a.port}" for a in self.aircraft]

    def start(self) -> "VirtualFleet":
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        step = 1.0 / self.rate
        last = time.time()
        while not self._stop.is_set():
            now = time.time()
            dt = max(now - last, 1e-4)
            last = now
            for a in self.aircraft:
                try:
                    a.step(dt)
                except Exception as exc:
                    # One misbehaving aircraft must not stop the fleet - a
                    # real ground station sees a silent drone, not a dead
                    # telemetry thread - but a swallowed error is a bug that
                    # presents as "no heartbeat", so keep it.
                    self.errors.append((a.index, repr(exc)))
            time.sleep(max(0.0, step - (time.time() - now)))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        for a in self.aircraft:
            a.close()

    def __enter__(self) -> "VirtualFleet":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- inspection, for tests and for the operator -----------------------

    def positions(self) -> np.ndarray:
        return np.array([a.position for a in self.aircraft])

    def place(self, positions: np.ndarray) -> None:
        """Put the aircraft on their launch pads before anyone connects.

        Without this they all spawn at the show origin, which fails the
        `launch positions` preflight check for a reason that is an artefact
        of the harness rather than a real setup error - and a check that
        always fails is a check nobody reads.
        """
        p = np.asarray(positions, dtype=float)
        if len(p) != len(self.aircraft):
            raise ValueError(
                f"{len(p)} positions for {len(self.aircraft)} aircraft"
            )
        for aircraft, xyz in zip(self.aircraft, p):
            aircraft.position = np.array([xyz[0], xyz[1], 0.0])
            aircraft.target = aircraft.position.copy()

    def summary(self) -> dict:
        return {
            "count": len(self.aircraft),
            "armed": sum(a.armed for a in self.aircraft),
            "modes": sorted({COPTER_MODES.get(a.custom_mode, str(a.custom_mode))
                             for a in self.aircraft}),
            "origin_set": sum(a.origin_set for a in self.aircraft),
            "plans_loaded": sum(a.plan is not None for a in self.aircraft),
            "start_scheduled": sum(a.show_start_unix is not None for a in self.aircraft),
            "suspended": sum(a.suspended_at is not None for a in self.aircraft),
            "signed": sum(a.signed for a in self.aircraft),
            "silent": [a.index for a in self.aircraft if a.silent],
            "nacks": {a.index: a.nacks_sent for a in self.aircraft if a.nacks_sent},
            "errors": self.errors[:10],
            "rx_bytes": sum(a.rx_bytes for a in self.aircraft),
            "tx_bytes": sum(a.tx_bytes for a in self.aircraft),
        }


def _free_port(host: str, preferred: int) -> int:
    """Use `preferred` if it is free, otherwise let the OS choose.

    Tests must not fail because a previous run left a socket in TIME_WAIT.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind((host, preferred))
        return preferred
    except OSError:
        probe.close()
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind((host, 0))
        return probe.getsockname()[1]
    finally:
        probe.close()


# --------------------------------------------------------------------------
# Real ArduPilot SITL
# --------------------------------------------------------------------------


@dataclass
class ArduPilotSITL:
    """Launch real ArduPilot SITL instances, one per drone.

    Each instance gets an `--instance` number, which ArduPilot uses to
    offset its port block by 10, so instance N listens for a GCS on
    5760 + 10*N over TCP. Use `endpoints` to feed the driver.

    This gives real ArduCopter: a real controller, a real EKF, real prearm
    checks and real failsafes. It is slower and heavier than VirtualFleet
    and a dozen instances will saturate a laptop, so rehearse the protocol
    against VirtualFleet and the flight behaviour against this.

    Stock ArduCopter has no show mode, so use `mavlink-guided` against it
    unless you have built show-mode firmware.
    """

    count: int = 1
    vehicle: str = "ArduCopter"
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    heading: float = 0.0
    speedup: float = 1.0
    sim_vehicle: str | None = None
    spacing: float = 5.0
    processes: list = field(default_factory=list)
    workdir: Path | None = None

    def locate(self) -> str:
        """Find sim_vehicle.py, or explain precisely what is missing."""
        import shutil

        if self.sim_vehicle and Path(self.sim_vehicle).exists():
            return self.sim_vehicle
        found = shutil.which("sim_vehicle.py")
        if found:
            return found
        for base in (Path.home() / "ardupilot", Path("/opt/ardupilot")):
            candidate = base / "Tools" / "autotest" / "sim_vehicle.py"
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            "ArduPilot SITL was not found. Sky Shine does not bundle it, "
            "because SITL is the real autopilot and must be the real autopilot.\n"
            "\n"
            "  git clone --recurse-submodules https://github.com/ArduPilot/ardupilot\n"
            "  cd ardupilot && Tools/environment_install/install-prereqs-ubuntu.sh -y\n"
            "  ./waf configure --board sitl && ./waf copter\n"
            "\n"
            "On Windows, run that inside WSL2 (wsl --install) and point\n"
            "Sky Shine at the WSL host, or pass sim_vehicle= explicitly.\n"
            "Until then, use VirtualFleet, which exercises the same MAVLink\n"
            "wire protocol without a flight dynamics model."
        )

    @property
    def endpoints(self) -> list[str]:
        return [f"tcp:127.0.0.1:{5760 + 10 * i}" for i in range(self.count)]

    def start(self) -> "ArduPilotSITL":
        import subprocess
        import tempfile

        exe = self.locate()
        self.workdir = Path(self.workdir or tempfile.mkdtemp(prefix="skyshine-sitl-"))
        cols = max(1, int(math.sqrt(self.count)))
        for i in range(self.count):
            # Spread the instances on the ground so they do not all spawn on
            # top of each other, which upsets the EKF and is not how a real
            # launch grid looks anyway.
            row, col = divmod(i, cols)
            dlat = (row * self.spacing) / 111_320.0
            dlon = (col * self.spacing) / (
                111_320.0 * max(math.cos(math.radians(self.latitude)), 1e-6)
            )
            home = (
                f"{self.latitude + dlat},{self.longitude + dlon},"
                f"{self.altitude},{self.heading}"
            )
            run = self.workdir / f"i{i}"
            run.mkdir(parents=True, exist_ok=True)
            cmd = [
                exe, "-v", self.vehicle, "--instance", str(i),
                "--custom-location", home, "--speedup", str(self.speedup),
                "--no-mavproxy", "-N",
            ]
            self.processes.append(
                subprocess.Popen(
                    cmd, cwd=str(run),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                )
            )
        return self

    def stop(self) -> None:
        for p in self.processes:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        self.processes = []

    def __enter__(self) -> "ArduPilotSITL":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def sitl_available(sim_vehicle: str | None = None) -> bool:
    try:
        ArduPilotSITL(sim_vehicle=sim_vehicle).locate()
        return True
    except FileNotFoundError:
        return False
