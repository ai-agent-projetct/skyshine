"""Operations: the ground station's state machine, gates and watchdogs.

The design rule here is that unsafe states must be unreachable rather than
merely discouraged. A show cannot be uploaded unless the safety report
passed, cannot be armed unless preflight passed, and cannot be started
unless it was uploaded and armed. Every override is possible, requires a
written reason, and lands in the event log.

The watchdog compares each drone's measured position against the plan
while the show runs. Reacting to drift is a judgement call about what is
underneath the aircraft, so the watchdog raises graded alerts and performs
only the action the operator pre-authorised.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import safety
from .drivers import DroneState, SimDriver
from .show import Show
from .trajectory import MissionPackage, build_mission
from .show import Timeline


class Phase(str, Enum):
    IDLE = "idle"
    VALIDATED = "validated"
    UPLOADED = "uploaded"
    ARMED = "armed"
    RUNNING = "running"
    PAUSED = "paused"
    ABORTING = "aborting"
    FINISHED = "finished"
    FAULT = "fault"


EMERGENCY_ACTIONS = ("hold", "rth", "land", "disarm")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    blocking: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "blocking": self.blocking,
        }


@dataclass
class Event:
    time: float
    kind: str
    message: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(self.time)),
            "kind": self.kind,
            "message": self.message,
            "data": self.data,
        }


@dataclass
class PreflightPolicy:
    """What has to be true before the swarm is allowed to arm."""

    min_battery_percent: float = 90.0
    min_gps_fix: int = 5           # 5 = RTK float, 6 = RTK fixed
    min_satellites: int = 12
    max_hdop: float = 1.2
    max_clock_offset_s: float = 0.05
    require_all_connected: bool = True
    require_mission_uploaded: bool = True
    # Drift handling while running.
    warn_deviation_m: float = 1.5
    abort_deviation_m: float = 3.0
    # How far behind its commanded position a healthy aircraft sits while
    # moving, as a time constant. A position controller lags its setpoint
    # by roughly speed x this, so a threshold tighter than that will fire
    # on every fast transition and train the operator to ignore it.
    # Measure it in SITL or on the bench, do not guess it.
    expected_tracking_lag_s: float = 0.4
    # What to do automatically when a drone exceeds abort_deviation_m.
    # "none" means alert only and wait for the operator, which is the
    # right default: the software does not know what is under the drone.
    auto_action: str = "none"


class FleetManager:
    """One show, one fleet, one operator. Not thread-safe by design.

    Call `tick()` from a single loop. Driving it from several threads would
    let two abort decisions race, which is exactly the situation where you
    want one clear chain of command.
    """

    def __init__(
        self,
        show: Show,
        *,
        driver=None,
        timeline: Timeline | None = None,
        policy: PreflightPolicy | None = None,
        clock: Callable[[], float] = time.time,
        registry=None,
        assignment_key: str | None = None,
    ) -> None:
        from .trajectory import build_timeline

        self.show = show
        self.timeline = timeline if timeline is not None else build_timeline(show)
        self.driver = driver if driver is not None else SimDriver(
            battery_endurance_s=show.limits.battery_endurance_s
        )
        self.policy = policy or PreflightPolicy()
        self.clock = clock
        # Optional: the physical fleet. With it, preflight checks that the
        # airframe answering on each slot is the one assigned to it.
        self.registry = registry
        self.assignment_key = assignment_key or show.name

        self.phase = Phase.IDLE
        self.report: safety.SafetyReport | None = None
        self.mission: MissionPackage | None = None
        self.events: list[Event] = []
        self.start_unix: float | None = None
        self.show_time: float = 0.0
        self.alerts: list[dict] = []
        self._overrides: list[str] = []
        self._last_tick: float | None = None
        self._alert_state: set[tuple[int, str]] = set()
        self._auto_actioned: set[int] = set()

        # Connect only if the caller has not already done so. A driver
        # handed in already connected - which is how a real link gets its
        # heartbeat timeout and its endpoint checks applied - must not have
        # its sockets torn down and rebuilt here, which silently left the
        # whole fleet looking offline.
        if not self.driver.poll():
            self.driver.connect(show.drone_count)
            self.log(
                "fleet", f"connected {show.drone_count} drones via {self.driver.name}"
            )
        else:
            self.log(
                "fleet",
                f"adopted an existing {self.driver.name} link to "
                f"{len(self.driver.poll())} drones",
            )
        if isinstance(self.driver, SimDriver) and show.formations:
            # Park the simulated fleet on its launch pads so preflight has
            # real positions to check instead of silently passing on None.
            self.driver.set_origin(show.origin)
            self.driver.advance(0.0, show.formations[0].points)

    def wait_for_fleet(self, timeout: float = 15.0, *, require: float = 1.0) -> int:
        """Block until telemetry has arrived from the fleet.

        A real link reports nothing until the first packets land, so
        running preflight straight after connecting would fail every check
        for want of data and tell the operator their fleet is dead.
        `require` is the fraction of the fleet that must report.

        Returns the number of drones heard from.
        """
        need = max(1, int(math.ceil(self.show.drone_count * require)))
        deadline = self.clock() + timeout
        seen = 0
        while self.clock() < deadline:
            states = self.drones
            seen = sum(
                1 for d in states if d.connected and (d.position is not None or d.gps_fix)
            )
            if seen >= need:
                break
            time.sleep(0.1)
        self.log(
            "fleet",
            f"{seen} of {self.show.drone_count} drones reporting telemetry",
            heard=seen,
        )
        return seen

    # -- logging ----------------------------------------------------------

    def log(self, kind: str, message: str, **data) -> Event:
        ev = Event(time=self.clock(), kind=kind, message=message, data=data)
        self.events.append(ev)
        return ev

    @property
    def drones(self) -> list[DroneState]:
        return self.driver.poll()

    # -- gate 1: validation ----------------------------------------------

    def validate(self) -> safety.SafetyReport:
        """Run the safety engine. This is the only route to a flyable show."""
        self.report = safety.validate(self.show, self.timeline)
        self.phase = Phase.VALIDATED if self.report.passed else Phase.IDLE
        self.log(
            "validate",
            f"safety {'passed' if self.report.passed else 'FAILED'}: "
            f"{len(self.report.errors)} errors, {len(self.report.warnings)} warnings",
            metrics=self.report.metrics,
        )
        return self.report

    # -- gate 2: upload ---------------------------------------------------

    def upload(self, *, override_reason: str | None = None) -> MissionPackage:
        if self.report is None:
            raise RuntimeError("validate() the show before uploading it")
        if not self.report.passed:
            if not override_reason:
                raise PermissionError(
                    "safety validation failed, so this show cannot be uploaded. "
                    f"{len(self.report.errors)} error(s), the first being: "
                    f"{self.report.errors[0].message}. Fix the design. If you "
                    "are certain the finding does not apply, pass "
                    "override_reason and it will be recorded against your name "
                    "in the flight log."
                )
            self._overrides.append(override_reason)
            self.log(
                "override",
                "uploaded a show that failed safety validation",
                reason=override_reason,
                errors=[v.to_dict() for v in self.report.errors],
            )

        self.mission = build_mission(self.show, safety=self.report.to_dict())
        self.driver.set_origin(self.show.origin)
        self.driver.upload(self.mission)
        self.phase = Phase.UPLOADED
        self.log(
            "upload",
            f"uploaded {self.show.drone_count} plans, checksum "
            f"{self.mission.to_dict()['checksum'][:12]}",
        )
        return self.mission

    # -- gate 3: preflight and arm ---------------------------------------

    def preflight(self) -> list[Check]:
        p = self.policy
        drones = self.drones
        checks: list[Check] = []

        checks.append(
            Check(
                "safety validation",
                self.report is not None and self.report.passed,
                "show passed the safety engine"
                if self.report and self.report.passed
                else "show has not passed safety validation",
            )
        )
        checks.append(
            Check(
                "mission uploaded",
                all(d.mission_uploaded for d in drones) if drones else False,
                f"{sum(d.mission_uploaded for d in drones)}/{len(drones)} drones hold a plan",
                blocking=p.require_mission_uploaded,
            )
        )
        if self.mission is not None:
            want = self.mission.to_dict()["checksum"]
            mismatched = [d.index for d in drones if d.mission_checksum != want]
            checks.append(
                Check(
                    "plan checksums match",
                    not mismatched,
                    "every drone holds the current plan"
                    if not mismatched
                    else f"drones {mismatched[:8]} hold a different or older plan",
                )
            )

        offline = [d.index for d in drones if not d.connected]
        checks.append(
            Check(
                "all drones online",
                not offline,
                "every drone is reporting" if not offline else f"offline: {offline[:12]}",
                blocking=p.require_all_connected,
            )
        )

        weak = [(d.index, round(d.battery_percent, 1)) for d in drones if d.battery_percent < p.min_battery_percent]
        checks.append(
            Check(
                "batteries charged",
                not weak,
                f"all above {p.min_battery_percent:.0f}%"
                if not weak
                else f"below {p.min_battery_percent:.0f}%: {weak[:8]}",
            )
        )

        bad_gps = [
            (d.index, d.gps_fix, d.satellites, round(d.hdop, 2))
            for d in drones
            if d.gps_fix < p.min_gps_fix
            or d.satellites < p.min_satellites
            or d.hdop > p.max_hdop
        ]
        checks.append(
            Check(
                "GNSS quality",
                not bad_gps,
                f"all at fix>={p.min_gps_fix}, sats>={p.min_satellites}, hdop<={p.max_hdop}"
                if not bad_gps
                else f"insufficient: {bad_gps[:6]}",
            )
        )

        skewed = [
            (d.index, round(d.clock_offset, 4))
            for d in drones
            if abs(d.clock_offset) > p.max_clock_offset_s
        ]
        checks.append(
            Check(
                "clock synchronisation",
                not skewed,
                f"all within {p.max_clock_offset_s * 1000:.0f} ms"
                if not skewed
                else f"out of sync: {skewed[:8]} - a drone on the wrong clock "
                "flies the right path at the wrong time, which is what "
                "collides with its neighbours",
            )
        )

        placed = self._check_placement(drones)
        if placed is not None:
            checks.append(placed)

        checks.extend(self._check_airframes(drones))

        lag_check = self._check_deviation_thresholds()
        if lag_check is not None:
            checks.append(lag_check)

        faulted = [(d.index, d.faults) for d in drones if d.faults]
        checks.append(
            Check(
                "no reported faults",
                not faulted,
                "no drone is reporting a fault"
                if not faulted
                else f"faults: {faulted[:5]}",
                blocking=False,
            )
        )
        return checks

    def _check_placement(self, drones: Sequence[DroneState]) -> Check | None:
        """Is each drone physically on the pad its plan expects?

        A swapped pair on the launch grid produces two drones flying each
        other's trajectory. Separation is still respected - the paths are
        unchanged - but they fly through each other's slot on the way, and
        no amount of trajectory validation catches it. It has to be caught
        on the ground.
        """
        if not self.show.formations or not drones:
            return None
        expected = self.show.formations[0].points
        if len(expected) != len(drones):
            return None
        wrong: list[tuple[int, float]] = []
        for d in drones:
            if d.position is None:
                continue
            err = float(np.linalg.norm(d.position[:2] - expected[d.index][:2]))
            if err > self.show.limits.takeoff_grid_spacing / 2.0:
                wrong.append((d.index, round(err, 2)))
        tolerance = self.show.limits.takeoff_grid_spacing / 2.0
        return Check(
            "launch positions",
            not wrong,
            f"every drone is within {tolerance:.1f} m of its assigned pad"
            if not wrong
            else f"wrong pad (index, metres off): {wrong[:8]} - most likely two "
            "airframes were swapped during setup",
        )

    def _check_deviation_thresholds(self) -> Check | None:
        """Are the watchdog thresholds above normal tracking error?

        A position controller trails its setpoint while moving. If the
        abort threshold is below that, the watchdog cries wolf on every
        fast transition, and an operator who has learned to dismiss it will
        dismiss the one that matters. Non-blocking: it is a tuning problem,
        not a reason to stop a launch.
        """
        p = self.policy
        peak = float(np.max(np.linalg.norm(self.timeline.velocity(), axis=2)))
        expected = peak * p.expected_tracking_lag_s
        ok = p.abort_deviation_m > expected and p.warn_deviation_m > expected * 0.6
        return Check(
            "deviation thresholds",
            ok,
            f"abort at {p.abort_deviation_m:.1f} m against about "
            f"{expected:.2f} m of expected lag "
            f"({peak:.1f} m/s peak x {p.expected_tracking_lag_s:.2f} s)"
            if ok
            else f"abort threshold {p.abort_deviation_m:.1f} m is below the "
                 f"{expected:.2f} m a healthy aircraft will trail its setpoint by "
                 f"at {peak:.1f} m/s - raise it, or slow the transitions",
            blocking=False,
        )

    def arm(self, *, override_reason: str | None = None) -> list[Check]:
        checks = self.preflight()
        blocking = [c for c in checks if c.blocking and not c.ok]
        if blocking:
            if not override_reason:
                names = ", ".join(c.name for c in blocking)
                raise PermissionError(
                    f"preflight failed on: {names}. Resolve these before arming, "
                    "or pass override_reason to record a deliberate exception in "
                    "the flight log."
                )
            self._overrides.append(override_reason)
            self.log(
                "override",
                "armed with failing preflight checks",
                reason=override_reason,
                failed=[c.to_dict() for c in blocking],
            )
        self.driver.arm()
        self.phase = Phase.ARMED
        self.log("arm", f"armed {self.show.drone_count} drones")
        return checks

    # -- gate 4: run ------------------------------------------------------

    def start(self, *, at_unix_time: float | None = None, lead_in: float = 5.0) -> float:
        """Schedule the show against absolute time, not 'now'.

        A start instant in the near future is what makes the fleet move
        together: every drone counts down to the same UTC moment from its
        own GNSS clock. Telling 500 aircraft to "go" one packet at a time
        spreads the start over seconds.
        """
        if self.phase != Phase.ARMED:
            raise RuntimeError(
                f"cannot start from phase {self.phase.value}; arm the fleet first"
            )
        when = at_unix_time if at_unix_time is not None else self.clock() + lead_in
        if when < self.clock():
            raise ValueError("start time is in the past")
        self.start_unix = when
        self.driver.start(when)
        self.phase = Phase.RUNNING
        self.show_time = 0.0
        self.log(
            "start",
            f"show scheduled for {time.strftime('%H:%M:%S', time.gmtime(when))} UTC "
            f"({when - self.clock():.1f} s from now)",
            start_unix=when,
        )
        return when

    def tick(self) -> dict:
        """Advance the show clock, drive the link, run the watchdog.

        Returns a snapshot suitable for sending straight to the operator's
        display.
        """
        now = self.clock()
        if self.phase == Phase.RUNNING and self.start_unix is not None:
            self.show_time = max(0.0, now - self.start_unix)

        planned, colors = self.timeline.sample(self.show_time)

        if isinstance(self.driver, SimDriver):
            self.driver.advance(self.show_time, planned)
        elif getattr(self.driver, "streams_setpoints", False) and self.phase == Phase.RUNNING:
            for i in range(self.show.drone_count):
                self.driver.setpoint(i, planned[i], colors[i])

        self.alerts = self._watchdog(planned)

        if self.phase == Phase.RUNNING and self.show_time >= self.timeline.duration:
            self.phase = Phase.FINISHED
            self.log("finish", f"show complete after {self.show_time:.1f} s")

        self._last_tick = now
        return self.snapshot(planned, colors)

    def _watchdog(self, planned: np.ndarray) -> list[dict]:
        """Grade every drone against the plan, edge-triggered.

        Alerts describe the live state and are rebuilt each tick, so the
        display always shows what is wrong now. The event log only records
        the moment a condition appears or clears: logging a 1.5 m
        deviation ten times a second buries the one line that mattered
        under three thousand that did not.
        """
        if self.phase not in (Phase.RUNNING, Phase.PAUSED):
            return []
        p = self.policy
        alerts: list[dict] = []
        active: set[tuple[int, str]] = set()
        breached: list[int] = []

        def raise_alert(drone: int, kind: str, severity: str, detail: str) -> None:
            active.add((drone, kind))
            alerts.append(
                {"drone": drone, "kind": kind, "severity": severity, "detail": detail}
            )
            if (drone, kind) not in self._alert_state:
                self.log(
                    "alert",
                    f"drone {drone}: {kind} - {detail}",
                    drone=drone,
                    condition=kind,
                    severity=severity,
                    show_time=round(self.show_time, 2),
                )

        for d in self.drones:
            if not d.connected:
                raise_alert(d.index, "link_lost", "critical", "no telemetry")
                continue
            if d.position is not None:
                dev = float(np.linalg.norm(d.position - planned[d.index]))
                d.deviation = dev
                if dev >= p.abort_deviation_m:
                    breached.append(d.index)
                    raise_alert(
                        d.index,
                        "deviation_abort",
                        "critical",
                        f"{dev:.2f} m off plan, at or past the "
                        f"{p.abort_deviation_m:.1f} m abort threshold",
                    )
                elif dev >= p.warn_deviation_m:
                    raise_alert(
                        d.index, "deviation_warn", "warning", f"{dev:.2f} m off plan"
                    )
            if d.battery_percent < 15.0:
                raise_alert(
                    d.index,
                    "battery_low",
                    "critical",
                    f"battery at {d.battery_percent:.0f}%",
                )
            if d.gps_fix < p.min_gps_fix:
                raise_alert(
                    d.index,
                    "gnss_degraded",
                    "warning",
                    f"fix type {d.gps_fix}, {d.satellites} satellites, "
                    f"hdop {d.hdop:.1f}",
                )

        for drone, kind in self._alert_state - active:
            self.log(
                "alert_cleared",
                f"drone {drone}: {kind} cleared",
                drone=drone,
                condition=kind,
                show_time=round(self.show_time, 2),
            )
        self._alert_state = active

        if breached and p.auto_action != "none":
            new = [i for i in breached if i not in self._auto_actioned]
            if new:
                self._auto_actioned.update(new)
                self.emergency(p.auto_action, indices=new, reason="watchdog")
        return alerts

    def pause(self) -> None:
        """Hold every drone in place, show clock frozen.

        Only meaningful with a driver that streams setpoints. Native show
        mode runs from the aircraft's own clock, so a ground-side pause
        would desynchronise the fleet from the drones that did not receive
        it; there, use an emergency hold instead.
        """
        if self.phase != Phase.RUNNING:
            raise RuntimeError(f"cannot pause from phase {self.phase.value}")
        if not getattr(self.driver, "streams_setpoints", False) and not isinstance(
            self.driver, SimDriver
        ):
            raise RuntimeError(
                "this driver executes the show onboard, so the ground station "
                "cannot pause it. Use emergency('hold') to stop the fleet."
            )
        self.phase = Phase.PAUSED
        self.log("pause", f"paused at show time {self.show_time:.1f} s")

    def _release_hold(self, indices=None) -> None:
        release = getattr(self.driver, "release_hold", None)
        if release is not None:
            try:
                release(indices)
            except Exception as exc:
                self.log("error", f"could not release hold: {exc}")

    def resume(self) -> None:
        if self.phase != Phase.PAUSED:
            raise RuntimeError(f"cannot resume from phase {self.phase.value}")
        self._release_hold()
        self.start_unix = self.clock() - self.show_time
        self.phase = Phase.RUNNING
        self.log("resume", f"resumed at show time {self.show_time:.1f} s")

    def emergency(
        self,
        action: str,
        *,
        indices: Sequence[int] | None = None,
        reason: str = "operator",
    ) -> None:
        """Hold, return home, land, or disarm.

        There is no kill switch. Stopping the motors of an airborne
        multirotor converts it into a falling object, which is worse than
        almost anything it was about to do. `disarm` is accepted only for
        aircraft that telemetry shows on the ground.
        """
        if action not in EMERGENCY_ACTIONS:
            raise ValueError(
                f"unknown action {action!r}; expected one of "
                f"{', '.join(EMERGENCY_ACTIONS)}"
            )
        if action == "disarm":
            airborne = [
                d.index
                for d in self.drones
                if (indices is None or d.index in set(indices))
                and d.altitude_agl > 0.5
            ]
            if airborne:
                raise PermissionError(
                    f"refusing to disarm airborne drones {airborne[:10]}: cutting "
                    "motors at altitude drops the aircraft. Use 'land' or 'rth'."
                )
        self.driver.emergency(action, indices)
        if action == "disarm" and indices is None and self.phase not in (
            Phase.RUNNING, Phase.PAUSED
        ):
            # Every aircraft is on the ground (checked above) and the show is
            # not flying, so this stands the fleet down; it is not an abort.
            # Logging it as an emergency turned a completed show into
            # "aborted" in its post-show report. The phase is kept, except
            # that an armed fleet goes back to holding its uploaded plan.
            if self.phase == Phase.ARMED:
                self.phase = Phase.UPLOADED
            self.log("disarm", "fleet disarmed on the ground", reason=reason)
            return
        if indices is None:
            self.phase = Phase.ABORTING
        self.log(
            "emergency",
            f"{action} issued to {'all drones' if indices is None else f'drones {list(indices)[:10]}'}",
            action=action,
            reason=reason,
        )

    def suspend(self, *, reason: str = "operator") -> None:
        """Freeze the show with every drone holding where it is.

        Safe by construction and worth distinguishing from an abort: nobody
        moves, so whatever separation the fleet had is exactly the
        separation it keeps. Unlike `pause`, this works with onboard
        execution too, because it is a command to the aircraft rather than
        a ground-side clock trick.
        """
        if self.phase not in (Phase.RUNNING, Phase.PAUSED):
            raise RuntimeError(f"cannot suspend from phase {self.phase.value}")
        if hasattr(self.driver, "show_control"):
            try:
                self.driver.show_control("suspend")
            except (ValueError, NotImplementedError):
                self.driver.emergency("hold", None)
        else:
            self.driver.emergency("hold", None)
        self.phase = Phase.PAUSED
        self.log(
            "suspend",
            f"show suspended at {self.show_time:.1f}s, fleet holding position",
            reason=reason,
        )

    def unsuspend(self, *, reason: str = "operator") -> None:
        """Resume a suspended show from where it stopped."""
        if self.phase != Phase.PAUSED:
            raise RuntimeError(f"cannot resume from phase {self.phase.value}")
        if hasattr(self.driver, "show_control"):
            try:
                self.driver.show_control("resume")
            except (ValueError, NotImplementedError):
                pass
        self._release_hold()
        self.start_unix = self.clock() - self.show_time
        self.phase = Phase.RUNNING
        self.log("resume", f"show resumed at {self.show_time:.1f}s", reason=reason)

    def collective_rth(
        self,
        *,
        reason: str = "operator",
        grid: np.ndarray | None = None,
        limits=None,
    ) -> dict:
        """Bring the whole fleet home on one deconflicted plan.

        Per-drone RTL hands each autopilot a private straight line to its
        own pad, computed in ignorance of every other aircraft. At show
        density that is a collision. This instead computes one return plan
        for the fleet from current measured positions, verifies it with the
        same continuous-time sweep as a designed show, and only then flies
        it. If the verification does not clear - which can happen, because
        the fleet's current separation is whatever the show left it at -
        the fleet is held instead and the operator is told why.

        Returns a report of what was decided and done.
        """
        from . import planning
        from .trajectory import build_timeline

        drones = self.drones
        measured = []
        fallback = 0
        planned, _ = self.timeline.sample(self.show_time)
        for d in drones:
            if d.position is not None and d.connected:
                measured.append(d.position)
            else:
                # A drone we cannot see has to be assumed to be where the
                # plan put it. Noted, because it is an assumption.
                measured.append(planned[d.index])
                fallback += 1
        positions = np.asarray(measured, dtype=np.float64)

        escape = planning.build_escape(
            positions, limits or self.show.limits, grid=grid
        )
        timeline = build_timeline(escape)
        violations, worst = safety.check_separation(
            timeline,
            escape.limits.min_separation,
            ignore_below_altitude=escape.limits.min_nav_altitude,
        )

        report = {
            "action": "collective_rth",
            "drones": len(positions),
            "positions_assumed": fallback,
            "min_separation_m": round(worst, 3),
            "min_separation_limit_m": escape.limits.min_separation,
            "duration_s": round(escape.duration, 1),
            "breaches": len(violations),
            "executed": False,
            "reason": reason,
        }

        if violations:
            self.driver.emergency("hold", None)
            self.phase = Phase.PAUSED
            report["fallback"] = "hold"
            report["detail"] = (
                f"a deconflicted return from the fleet's current positions would "
                f"come within {worst:.2f} m, under the "
                f"{escape.limits.min_separation:.2f} m minimum. The fleet is "
                "holding position instead, which preserves the separation it "
                "already has. Thin the fleet out with individual land commands, "
                "or accept the risk deliberately."
            )
            self.log(
                "collective_rth",
                "refused: a deconflicted return could not be found, fleet holding",
                **report,
            )
            return report

        # Release any latched hold first. An aircraft in BRAKE ignores the
        # plan, so without this the return would be computed, verified,
        # reported as flying, and never actually happen.
        self._release_hold()

        # Swap the show for the return plan and fly it from now.
        self.show = escape
        self.timeline = timeline
        self.start_unix = self.clock()
        self.show_time = 0.0
        self._alert_state.clear()
        self._auto_actioned.clear()
        self.phase = Phase.RUNNING

        if getattr(self.driver, "streams_setpoints", False) or isinstance(
            self.driver, SimDriver
        ):
            # Setpoints come from tick(), which is already running.
            pass
        else:
            try:
                self.mission = build_mission(escape)
                self.driver.upload(self.mission)
                self.driver.start(self.clock() + 1.0)
            except (IOError, OSError, NotImplementedError) as exc:
                self.driver.emergency("rth", None)
                report["fallback"] = "per-drone rth"
                report["detail"] = (
                    f"could not upload the deconflicted return ({exc}); fell back "
                    "to per-drone RTL, which is NOT deconflicted - watch the fleet"
                )
                self.log("collective_rth", report["detail"], **report)
                return report

        report["executed"] = True
        report["detail"] = (
            f"flying a deconflicted return, closest approach {worst:.2f} m over "
            f"{escape.duration:.0f}s"
        )
        self.log("collective_rth", report["detail"], **report)
        return report

    # -- reporting --------------------------------------------------------

    def snapshot(
        self, planned: np.ndarray | None = None, colors: np.ndarray | None = None
    ) -> dict:
        if planned is None or colors is None:
            planned, colors = self.timeline.sample(self.show_time)
        drones = self.drones
        return {
            "phase": self.phase.value,
            "show": self.show.name,
            "show_time": round(self.show_time, 3),
            "duration": round(self.timeline.duration, 3),
            "start_unix": self.start_unix,
            "drone_count": self.show.drone_count,
            "connected": sum(d.connected for d in drones),
            "armed": sum(d.armed for d in drones),
            "max_deviation": round(
                max((d.deviation for d in drones), default=0.0), 3
            ),
            "min_battery": round(
                min((d.battery_percent for d in drones), default=0.0), 1
            ),
            "alerts": self.alerts,
            "planned": np.round(planned, 2).tolist(),
            "colors": colors.tolist(),
            "drones": [d.to_dict() for d in drones],
        }

    def _check_airframes(self, drones) -> list[Check]:
        """Is the right airframe answering on every slot?

        Only runs with a registry. Identity problems - an empty slot, a
        grounded drone, a system id that belongs to a different airframe -
        block arming; maintenance advisories such as an ageing battery do
        not, because they are judgements for the crew rather than errors.
        """
        if self.registry is None:
            return []
        problems = self.registry.verify(
            self.assignment_key, self.show.drone_count, drones
        )
        blocking = [p for p in problems if p["kind"] != "advisory"]
        advisory = [p for p in problems if p["kind"] == "advisory"]
        checks = [
            Check(
                "airframe identity",
                not blocking,
                "every slot has its assigned airframe answering"
                if not blocking
                else "; ".join(p["detail"] for p in blocking[:4])
                + (f" (+{len(blocking) - 4} more)" if len(blocking) > 4 else ""),
            )
        ]
        if advisory:
            checks.append(
                Check(
                    "fleet maintenance",
                    False,
                    "; ".join(p["detail"] for p in advisory[:4])
                    + (f" (+{len(advisory) - 4} more)" if len(advisory) > 4 else ""),
                    blocking=False,
                )
            )
        return checks

    def record_to_registry(self) -> list[str]:
        """Credit the airframes that flew with the time they actually flew.

        Uses the show time reached, not the designed duration, so an abort
        after 40 seconds adds 40 seconds. A show that never left the ground
        adds nothing and uses no battery cycle.
        """
        if self.registry is None or self.show_time <= 0:
            return []
        # If the show clock ran, every assigned airframe took off - a test
        # on current altitude would miss a fleet that already landed after
        # an abort, which flew just as much.
        flew = [d for d in self.registry.assignment(self.assignment_key) if d]
        touched = self.registry.record_flight(
            self.assignment_key, self.show_time, flew=flew
        )
        self.log(
            "registry",
            f"recorded {self.show_time:.0f}s and a battery cycle on "
            f"{len(touched)} airframes",
            drones=touched,
        )
        return touched

    def flight_log(self) -> dict:
        drones = self.drones
        return {
            "format": "skyshine.flightlog",
            "version": 1,
            "show": self.show.name,
            "drone_count": self.show.drone_count,
            "driver": self.driver.name,
            "phase": self.phase.value,
            "start_unix": self.start_unix,
            "show_time_reached": round(self.show_time, 3),
            "duration": round(self.timeline.duration, 3),
            "completed": self.phase == Phase.FINISHED,
            "mission_checksum": (
                self.mission.to_dict()["checksum"] if self.mission else None
            ),
            "safety": self.report.to_dict() if self.report else None,
            "overrides": list(self._overrides),
            "airframes": (
                self.registry.assignment(self.assignment_key)
                if self.registry is not None else None
            ),
            "final_state": [d.to_dict() for d in drones],
            "events": [e.to_dict() for e in self.events],
        }

    def save_flight_log(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.flight_log(), indent=1), encoding="utf-8")
        self.log("log", f"flight log written to {p}")
        return p


def rehearse(
    show: Show,
    *,
    driver=None,
    policy: PreflightPolicy | None = None,
    step: float = 0.1,
    override_reason: str | None = None,
) -> tuple[FleetManager, dict]:
    """Fly a whole show in simulated time as fast as the CPU allows.

    This is the dress rehearsal: validate, upload, arm, run to the end,
    and report what the watchdog saw. Running it with faults injected is
    how the ten scenarios in docs/SAFETY.md get exercised without putting
    anything in the air.
    """
    clock_state = {"t": 1_700_000_000.0}

    def fake_clock() -> float:
        return clock_state["t"]

    fleet = FleetManager(show, driver=driver, policy=policy, clock=fake_clock)
    report = fleet.validate()
    fleet.upload(override_reason=override_reason)
    fleet.arm(override_reason=override_reason)
    fleet.start(lead_in=0.0)

    worst_deviation = 0.0
    seen: dict[tuple[int, str], float] = {}
    while fleet.phase in (Phase.RUNNING, Phase.PAUSED):
        snap = fleet.tick()
        worst_deviation = max(worst_deviation, snap["max_deviation"])
        for a in snap["alerts"]:
            seen.setdefault((a["drone"], a["kind"]), round(fleet.show_time, 1))
        clock_state["t"] += step
        if fleet.show_time > fleet.timeline.duration + 1.0:
            break

    summary = {
        "completed": fleet.phase == Phase.FINISHED,
        "phase": fleet.phase.value,
        "show_time": round(fleet.show_time, 2),
        "safety_passed": report.passed,
        "worst_deviation_m": round(worst_deviation, 3),
        # One line per drone per condition, with the show time it first
        # appeared. That is the post-show debrief, not a 3000 line scroll.
        "issues": [
            {"drone": d, "kind": k, "first_seen_s": t}
            for (d, k), t in sorted(seen.items(), key=lambda kv: kv[1])
        ],
    }
    return fleet, summary
