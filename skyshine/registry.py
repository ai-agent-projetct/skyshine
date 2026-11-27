"""The fleet as physical objects, not array indices.

Everywhere else in Sky Shine a drone is an integer: the index of a row in
a formation. That is the right abstraction for geometry and completely
wrong for operations. On the night you are not holding 500 indices, you
are holding 500 airframes, each with a serial number, a firmware version,
a battery that has been charged a known number of times, and a history of
what has broken on it.

Three things fall out of tracking that properly:

**Slot assignment becomes explicit.** Which airframe stands on which pad
is currently implicit in the order of the endpoint list, which is a
configuration file nobody re-reads. Here it is a record, and preflight
checks the system id actually reporting on each link against the drone
that record says should be there.

**Battery cycles become visible.** A LiPo that has been through 250 cycles
does not hold the voltage the flight plan assumed. The endurance figure in
ShowLimits is a number somebody typed; this is the number the packs have
actually earned.

**Maintenance stops living in someone's head.** A drone grounded for a
cracked arm should not be assignable to a slot, and the refusal should
come from the software rather than from remembering.
"""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Lifecycle states. `ready` is the only one that may be assigned to a show.
READY = "ready"
MAINTENANCE = "maintenance"
RETIRED = "retired"
STATES = (READY, MAINTENANCE, RETIRED)

# Past this many charge cycles a pack's usable capacity has usually fallen
# far enough that the endurance assumed at design time is optimistic. Not a
# hard limit - it is a warning, because the real number depends on the
# chemistry, the charge rate and how hard the show works them.
BATTERY_CYCLE_WARNING = 200


class RegistryError(ValueError):
    """The requested change would leave the registry inconsistent."""


@dataclass
class Drone:
    """One physical aircraft."""

    id: str                       # DD-001, operator-facing
    serial: str = ""              # manufacturer serial
    model: str = ""               # airframe
    controller: str = ""          # flight controller
    firmware: str = ""
    gnss: str = ""
    led: str = ""
    battery_id: str = ""
    sysid: int = 0                # MAVLink system id, 0 = unknown
    status: str = READY
    flight_seconds: float = 0.0
    flight_count: int = 0
    battery_cycles: int = 0
    notes: str = ""
    maintenance: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise RegistryError("a drone needs an id")
        self.id = str(self.id).strip()
        if self.status not in STATES:
            raise RegistryError(
                f"{self.id}: status must be one of {', '.join(STATES)}"
            )
        if self.sysid and not 0 < int(self.sysid) < 256:
            raise RegistryError(
                f"{self.id}: MAVLink system id {self.sysid} is outside 1-255"
            )

    @property
    def flight_hours(self) -> float:
        return self.flight_seconds / 3600.0

    @property
    def assignable(self) -> bool:
        return self.status == READY

    def warnings(self) -> list[str]:
        out = []
        if self.battery_cycles >= BATTERY_CYCLE_WARNING:
            out.append(
                f"battery {self.battery_id or '(unnamed)'} has "
                f"{self.battery_cycles} cycles, past the "
                f"{BATTERY_CYCLE_WARNING} review point"
            )
        if not self.firmware:
            out.append("no firmware version recorded")
        if not self.sysid:
            out.append("no MAVLink system id recorded, so it cannot be verified on the link")
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["flight_hours"] = round(self.flight_hours, 2)
        d["assignable"] = self.assignable
        d["warnings"] = self.warnings()
        return d


@dataclass
class Registry:
    """Every airframe, plus which show slot each is standing on."""

    drones: dict[str, Drone] = field(default_factory=dict)
    # show name -> list of drone ids, index = formation slot
    assignments: dict[str, list[str]] = field(default_factory=dict)

    # -- drones -----------------------------------------------------------

    def add(self, drone: Drone) -> Drone:
        if drone.id in self.drones:
            raise RegistryError(f"{drone.id} is already registered")
        clash = self.by_serial(drone.serial)
        if drone.serial and clash:
            raise RegistryError(
                f"serial {drone.serial!r} is already registered as {clash.id}"
            )
        self.drones[drone.id] = drone
        return drone

    def get(self, drone_id: str) -> Drone:
        try:
            return self.drones[drone_id]
        except KeyError:
            raise RegistryError(f"no drone {drone_id!r}") from None

    def by_serial(self, serial: str) -> Drone | None:
        if not serial:
            return None
        return next((d for d in self.drones.values() if d.serial == serial), None)

    def by_sysid(self, sysid: int) -> Drone | None:
        if not sysid:
            return None
        return next((d for d in self.drones.values() if d.sysid == sysid), None)

    def remove(self, drone_id: str) -> None:
        drone = self.get(drone_id)
        holding = [s for s, ids in self.assignments.items() if drone.id in ids]
        if holding:
            raise RegistryError(
                f"{drone_id} is assigned to {', '.join(holding)}; unassign it first"
            )
        del self.drones[drone_id]

    def set_status(self, drone_id: str, status: str, reason: str = "") -> Drone:
        drone = self.get(drone_id)
        if status not in STATES:
            raise RegistryError(f"status must be one of {', '.join(STATES)}")
        if status != READY:
            # A grounded drone cannot stay on a pad. Leaving it assigned is
            # how an aircraft nobody meant to fly ends up in the air.
            for show, ids in self.assignments.items():
                if drone.id in ids:
                    ids[ids.index(drone.id)] = ""
        drone.status = status
        drone.maintenance.append({
            "at": time.time(), "event": f"status -> {status}", "detail": reason,
        })
        return drone

    def log_maintenance(self, drone_id: str, event: str, detail: str = "") -> Drone:
        drone = self.get(drone_id)
        drone.maintenance.append({"at": time.time(), "event": event, "detail": detail})
        return drone

    def ready(self) -> list[Drone]:
        return sorted(
            (d for d in self.drones.values() if d.assignable), key=lambda d: d.id
        )

    # -- assignment -------------------------------------------------------

    def assign(self, show: str, drone_ids: list[str]) -> list[str]:
        """Place drones on slots, slot index = position in the list.

        An empty string means the slot is unfilled, which is allowed while
        planning and refused by `verify` at preflight.
        """
        seen: set[str] = set()
        for i, drone_id in enumerate(drone_ids):
            if not drone_id:
                continue
            drone = self.get(drone_id)
            if not drone.assignable:
                raise RegistryError(
                    f"{drone_id} is {drone.status} and cannot be assigned to a slot"
                )
            if drone_id in seen:
                raise RegistryError(
                    f"{drone_id} is assigned to two slots; one airframe cannot "
                    "stand on two pads"
                )
            seen.add(drone_id)
        self.assignments[show] = list(drone_ids)
        return self.assignments[show]

    def auto_assign(self, show: str, count: int) -> list[str]:
        """Fill `count` slots from the ready pool, lowest flight time first.

        Spreading hours evenly keeps the fleet ageing together rather than
        wearing out a favoured subset, and it puts the freshest airframes
        on the show rather than whichever happened to be listed first.
        """
        pool = sorted(self.ready(), key=lambda d: (d.flight_seconds, d.id))
        if len(pool) < count:
            raise RegistryError(
                f"{count} slots but only {len(pool)} drones are ready "
                f"({len(self.drones)} registered). Return some from maintenance "
                "or adapt the show to the fleet you have."
            )
        return self.assign(show, [d.id for d in pool[:count]])

    def assignment(self, show: str) -> list[str]:
        return list(self.assignments.get(show, []))

    def unassign(self, show: str) -> None:
        self.assignments.pop(show, None)

    def verify(self, show: str, slots: int, states=None) -> list[dict]:
        """Check an assignment against the fleet, and against live telemetry.

        `states` are DroneState rows from a driver, in slot order. When they
        are given, each slot's reported MAVLink system id is compared with
        the id the registry says should be standing there - which is the
        check that catches two airframes swapped during setup. Comparing
        positions alone cannot: swap two drones and each reports the
        position the plan expects of the other only if the link order
        happens to follow the pads.
        """
        problems: list[dict] = []
        assigned = self.assignment(show)

        if not assigned:
            return [{
                "slot": None, "kind": "unassigned",
                "detail": f"no airframes assigned to {show!r}",
            }]
        if len(assigned) != slots:
            problems.append({
                "slot": None, "kind": "count",
                "detail": (
                    f"{len(assigned)} airframes assigned but the show has "
                    f"{slots} slots"
                ),
            })

        for i, drone_id in enumerate(assigned[:slots]):
            if not drone_id:
                problems.append({
                    "slot": i, "kind": "empty", "detail": f"slot {i} has no airframe"
                })
                continue
            try:
                drone = self.get(drone_id)
            except RegistryError:
                problems.append({
                    "slot": i, "kind": "missing",
                    "detail": f"slot {i} names {drone_id}, which is not registered",
                })
                continue
            if not drone.assignable:
                problems.append({
                    "slot": i, "kind": "status",
                    "detail": f"slot {i}: {drone_id} is {drone.status}",
                })
            for warning in drone.warnings():
                problems.append({
                    "slot": i, "kind": "advisory",
                    "detail": f"slot {i}: {drone_id} {warning}",
                })

            if states is not None and i < len(states):
                reported = int(getattr(states[i], "sysid", 0) or 0)
                if drone.sysid and reported and reported != drone.sysid:
                    actual = self.by_sysid(reported)
                    problems.append({
                        "slot": i, "kind": "wrong_airframe",
                        "detail": (
                            f"slot {i} should be {drone_id} (system id "
                            f"{drone.sysid}) but system id {reported} is "
                            f"answering"
                            + (f", which is {actual.id}" if actual else "")
                            + ". Two airframes have been swapped, and no "
                            "trajectory check can see that."
                        ),
                    })
        return problems

    # -- after the show ---------------------------------------------------

    def record_flight(
        self, show: str, seconds: float, *, flew: list[str] | None = None,
        battery_cycle: bool = True,
    ) -> list[str]:
        """Add flight time and a battery cycle to the drones that flew.

        Called from the flight log rather than from the plan, so a show
        that was aborted after 40 seconds adds 40 seconds, not its designed
        duration.
        """
        if seconds < 0:
            raise RegistryError("flight time cannot be negative")
        ids = flew if flew is not None else [d for d in self.assignment(show) if d]
        touched = []
        for drone_id in ids:
            if not drone_id or drone_id not in self.drones:
                continue
            drone = self.drones[drone_id]
            drone.flight_seconds += seconds
            drone.flight_count += 1
            if battery_cycle:
                drone.battery_cycles += 1
            touched.append(drone_id)
        return touched

    # -- storage ----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": "skyshine.registry",
            "version": 1,
            "drones": [asdict(d) for d in self.drones.values()],
            "assignments": self.assignments,
        }

    @staticmethod
    def from_dict(data: dict) -> "Registry":
        if data and data.get("format") not in (None, "skyshine.registry"):
            raise RegistryError("not a Sky Shine registry file")
        reg = Registry()
        for row in data.get("drones", []):
            row = dict(row)
            row.pop("flight_hours", None)
            row.pop("assignable", None)
            row.pop("warnings", None)
            drone = Drone(**row)
            reg.drones[drone.id] = drone
        reg.assignments = {k: list(v) for k, v in data.get("assignments", {}).items()}
        return reg

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return p

    @staticmethod
    def load(path: str | Path) -> "Registry":
        p = Path(path)
        if not p.exists():
            return Registry()
        try:
            return Registry.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise RegistryError(f"registry file is corrupt: {exc}") from None

    def summary(self) -> dict:
        by_status = {s: 0 for s in STATES}
        for d in self.drones.values():
            by_status[d.status] = by_status.get(d.status, 0) + 1
        hours = [d.flight_hours for d in self.drones.values()]
        cycles = [d.battery_cycles for d in self.drones.values()]
        return {
            "registered": len(self.drones),
            **by_status,
            "total_flight_hours": round(sum(hours), 2),
            "mean_flight_hours": round(sum(hours) / len(hours), 2) if hours else 0.0,
            "max_battery_cycles": max(cycles) if cycles else 0,
            "packs_past_review": sum(
                1 for c in cycles if c >= BATTERY_CYCLE_WARNING
            ),
            "assignments": {k: sum(1 for i in v if i) for k, v in self.assignments.items()},
        }


def bulk_register(
    registry: Registry, count: int, *, prefix: str = "DD", start: int = 1,
    model: str = "", sysid_from: int = 1,
) -> list[Drone]:
    """Register a block of identical airframes.

    Typing 500 records by hand is how a fleet ends up half-registered, so
    the common case - a batch of identical drones with sequential ids and
    system ids - is one call. Serials stay blank, because those are the one
    field that genuinely has to be read off each aircraft.
    """
    made = []
    for i in range(count):
        number = start + i
        drone = Drone(
            id=f"{prefix}-{number:03d}",
            model=model,
            sysid=sysid_from + i if sysid_from else 0,
        )
        made.append(registry.add(drone))
    return made


# -- CSV import and export -------------------------------------------------
#
# A fleet is filled in with a spreadsheet: export, type 500 serials and
# firmware versions, import. So the format is CSV, not the registry's JSON.

CSV_COLUMNS = (
    "id", "serial", "model", "controller", "firmware", "gnss", "led",
    "battery_id", "sysid", "status", "notes",
    "flight_seconds", "flight_count", "battery_cycles",
)
CSV_COUNTERS = ("flight_seconds", "flight_count", "battery_cycles")
_NUMERIC = {"sysid": int, "flight_count": int, "battery_cycles": int,
            "flight_seconds": float}
MAX_IMPORT_ROWS = 2000
# A cell starting with one of these runs as a formula when the file is
# opened in a spreadsheet, and `notes` is free text any operator can type.
_FORMULA = ("=", "+", "-", "@")


def export_csv(registry: Registry) -> str:
    buf = io.StringIO()
    out = csv.writer(buf, lineterminator="\n")
    out.writerow(CSV_COLUMNS)
    for drone in sorted(registry.drones.values(), key=lambda d: d.id):
        cells = [str(getattr(drone, col)) for col in CSV_COLUMNS]
        out.writerow(["'" + c if c.startswith(_FORMULA) else c for c in cells])
    return buf.getvalue()


def import_csv(registry: Registry, text: str) -> dict:
    """Add new airframes and update existing ones from an exported sheet.

    Matching is by id. For an airframe already registered, only the columns
    present in the file are written, a blank cell means blank, and the
    flight counters are ignored: the registry has credited flights since the
    sheet was exported and a stale number must not wind them back. Any bad
    row raises, naming the row, and the caller saves nothing.
    """
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    cols = [(c or "").strip() for c in (reader.fieldnames or [])]
    if "id" not in cols:
        raise RegistryError(
            "the file needs an 'id' column, and the first row must be the header"
        )
    unknown = sorted(set(cols) - set(CSV_COLUMNS))
    if unknown:
        raise RegistryError(
            f"unknown column(s): {', '.join(unknown)}. "
            f"Accepted: {', '.join(CSV_COLUMNS)}."
        )
    reader.fieldnames = cols

    added: list[str] = []
    updated: list[str] = []
    seen: set[str] = set()
    for line, raw in enumerate(reader, start=2):
        if line - 1 > MAX_IMPORT_ROWS:
            raise RegistryError(f"more than {MAX_IMPORT_ROWS} rows; split the file")
        if None in raw:
            raise RegistryError(f"row {line}: more cells than columns")
        row = {}
        for key, value in raw.items():
            value = (value or "").strip()
            if value[:1] == "'" and value[1:].startswith(_FORMULA):
                value = value[1:]
            row[key] = value
        if not any(row.values()):
            continue
        try:
            for key in row.keys() & _NUMERIC.keys():
                try:
                    row[key] = _NUMERIC[key](row[key] or 0)
                except ValueError:
                    raise RegistryError(
                        f"{key} must be a number, got {row[key]!r}"
                    ) from None
            drone_id = row["id"]
            if drone_id in seen:
                raise RegistryError(f"{drone_id} appears twice in the file")
            seen.add(drone_id)
            status = row.pop("status", "")
            if drone_id in registry.drones:
                drone = registry.drones[drone_id]
                clash = registry.by_serial(row.get("serial", ""))
                if row.get("serial") and clash and clash.id != drone_id:
                    raise RegistryError(
                        f"serial {row['serial']!r} is already registered as {clash.id}"
                    )
                for key, value in row.items():
                    if key != "id" and key not in CSV_COUNTERS:
                        setattr(drone, key, value)
                drone.__post_init__()  # re-run the sysid range check
                if status and status != drone.status:
                    registry.set_status(drone_id, status, "fleet import")
                updated.append(drone_id)
            else:
                if status:
                    row["status"] = status
                registry.add(Drone(**row))
                added.append(drone_id)
        except RegistryError as exc:
            raise RegistryError(f"row {line}: {exc}") from None
    return {"added": added, "updated": updated}
