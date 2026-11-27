"""Tests for the airframe registry and the post-show report.

The class to read is TestSwappedAirframes. Two drones swapped during setup
fly each other's trajectories; separation is unaffected - the paths do not
change - so no trajectory check can see it. The registry closes that gap
by comparing the system id answering on each slot with the airframe
assigned to it.
"""

from __future__ import annotations

import pathlib
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from skyshine import geometry, planning, reports
from skyshine.drivers import SimDriver, SimFaults
from skyshine.fleet import FleetManager, Phase
from skyshine.registry import (
    BATTERY_CYCLE_WARNING,
    Drone,
    Registry,
    RegistryError,
    bulk_register,
)
from skyshine.show import Formation, Show, ShowLimits


def fleet(n=8, extra=2) -> Registry:
    reg = Registry()
    bulk_register(reg, n + extra, model="X1")
    for d in reg.drones.values():
        d.firmware = "4.5.7"
    return reg


def telemetry(sysids):
    return [SimpleNamespace(sysid=s) for s in sysids]


class ManualClock:
    def __init__(self):
        self.t = 1_700_000_000.0

    def __call__(self):
        return self.t


def small_show(n=8, name="night") -> Show:
    show = Show(name=name, drone_count=n,
                limits=ShowLimits(max_altitude=200.0, battery_endurance_s=1800.0))
    show.origin.amsl = 920.0
    for label, pts in (("ball", geometry.ball(n, 14)), ("disc", geometry.disc(n, 16))):
        show.add(Formation(label, geometry.lift(pts, 40.0), np.array([255, 60, 20]),
                           transition=6.0, hold=3.0))
    return planning.prepare(show).show


# ==========================================================================
# Records
# ==========================================================================


class TestDrone:
    def test_needs_an_id(self):
        with pytest.raises(RegistryError, match="needs an id"):
            Drone(id="  ")

    def test_rejects_an_unknown_status(self):
        with pytest.raises(RegistryError, match="status"):
            Drone(id="DD-1", status="flying")

    @pytest.mark.parametrize("sysid", [0, 1, 128, 255])
    def test_accepts_valid_system_ids(self, sysid):
        Drone(id="DD-1", sysid=sysid)

    @pytest.mark.parametrize("sysid", [256, 300, -1])
    def test_rejects_out_of_range_system_ids(self, sysid):
        with pytest.raises(RegistryError, match="1-255"):
            Drone(id="DD-1", sysid=sysid)

    def test_battery_warning_threshold(self):
        d = Drone(id="DD-1", firmware="x", sysid=1,
                  battery_cycles=BATTERY_CYCLE_WARNING)
        assert any("cycles" in w for w in d.warnings())
        d.battery_cycles = BATTERY_CYCLE_WARNING - 1
        assert not any("cycles" in w for w in d.warnings())

    def test_missing_sysid_is_flagged(self):
        """Without one the airframe cannot be verified on the link at all."""
        assert any("system id" in w for w in Drone(id="DD-1", firmware="x").warnings())

    def test_only_ready_drones_are_assignable(self):
        assert Drone(id="a").assignable
        assert not Drone(id="a", status="maintenance").assignable
        assert not Drone(id="a", status="retired").assignable


class TestRegistry:
    def test_duplicate_id_is_refused(self):
        reg = Registry()
        reg.add(Drone(id="DD-1"))
        with pytest.raises(RegistryError, match="already registered"):
            reg.add(Drone(id="DD-1"))

    def test_duplicate_serial_is_refused(self):
        """One physical airframe, two records, is how hours go missing."""
        reg = Registry()
        reg.add(Drone(id="DD-1", serial="SN123"))
        with pytest.raises(RegistryError, match="SN123"):
            reg.add(Drone(id="DD-2", serial="SN123"))

    def test_blank_serials_do_not_clash(self):
        reg = Registry()
        reg.add(Drone(id="DD-1"))
        reg.add(Drone(id="DD-2"))

    def test_lookup_by_sysid_and_serial(self):
        reg = Registry()
        reg.add(Drone(id="DD-1", serial="A", sysid=7))
        assert reg.by_sysid(7).id == "DD-1"
        assert reg.by_serial("A").id == "DD-1"
        assert reg.by_sysid(0) is None and reg.by_serial("") is None

    def test_unknown_drone(self):
        with pytest.raises(RegistryError, match="no drone"):
            Registry().get("DD-404")

    def test_bulk_register(self):
        reg = Registry()
        made = bulk_register(reg, 5, prefix="SS", start=10, sysid_from=40)
        assert [d.id for d in made] == ["SS-010", "SS-011", "SS-012", "SS-013", "SS-014"]
        assert [d.sysid for d in made] == [40, 41, 42, 43, 44]

    def test_bulk_register_refuses_to_overlap(self):
        reg = Registry()
        bulk_register(reg, 3)
        with pytest.raises(RegistryError, match="already registered"):
            bulk_register(reg, 3)

    def test_removing_an_assigned_drone_is_refused(self):
        reg = fleet(4, 0)
        reg.auto_assign("s", 4)
        with pytest.raises(RegistryError, match="unassign it first"):
            reg.remove("DD-001")

    def test_maintenance_log(self):
        reg = fleet(2, 0)
        reg.log_maintenance("DD-001", "prop swap", "front left")
        assert reg.get("DD-001").maintenance[-1]["event"] == "prop swap"


class TestAssignment:
    def test_assign_by_slot(self):
        reg = fleet(4, 0)
        reg.assign("s", ["DD-003", "DD-001", "DD-004", "DD-002"])
        assert reg.assignment("s") == ["DD-003", "DD-001", "DD-004", "DD-002"]

    def test_one_airframe_on_two_pads_is_refused(self):
        reg = fleet(4, 0)
        with pytest.raises(RegistryError, match="two slots"):
            reg.assign("s", ["DD-001", "DD-001", "DD-002", "DD-003"])

    def test_grounded_drone_cannot_be_assigned(self):
        reg = fleet(4, 0)
        reg.set_status("DD-002", "maintenance", "cracked arm")
        with pytest.raises(RegistryError, match="maintenance"):
            reg.assign("s", ["DD-001", "DD-002"])

    def test_grounding_removes_the_drone_from_its_slot(self):
        """An aircraft nobody meant to fly must not stay on a pad."""
        reg = fleet(4, 0)
        reg.assign("s", ["DD-001", "DD-002", "DD-003", "DD-004"])
        reg.set_status("DD-002", "maintenance", "motor noise")
        assert reg.assignment("s")[1] == ""
        assert reg.get("DD-002").maintenance[-1]["detail"] == "motor noise"

    def test_auto_assign_prefers_the_freshest_airframes(self):
        reg = fleet(4, 2)
        reg.get("DD-001").flight_seconds = 9000
        reg.get("DD-002").flight_seconds = 8000
        picked = reg.auto_assign("s", 4)
        assert "DD-001" not in picked and "DD-002" not in picked

    def test_auto_assign_skips_grounded_drones(self):
        reg = fleet(4, 2)
        reg.set_status("DD-003", "retired")
        assert "DD-003" not in reg.auto_assign("s", 4)

    def test_auto_assign_with_too_few_ready_drones(self):
        reg = fleet(3, 0)
        reg.set_status("DD-001", "maintenance")
        with pytest.raises(RegistryError, match="only 2 drones are ready"):
            reg.auto_assign("s", 3)


# ==========================================================================
# Verification
# ==========================================================================


class TestVerify:
    def test_nothing_assigned(self):
        problems = fleet().verify("s", 8)
        assert problems[0]["kind"] == "unassigned"

    def test_clean_assignment_has_no_blocking_problems(self):
        reg = fleet(8, 0)
        reg.auto_assign("s", 8)
        assert not [p for p in reg.verify("s", 8) if p["kind"] != "advisory"]

    def test_slot_count_mismatch(self):
        reg = fleet(8, 0)
        reg.auto_assign("s", 6)
        assert "count" in {p["kind"] for p in reg.verify("s", 8)}

    def test_empty_slot(self):
        reg = fleet(4, 0)
        reg.assign("s", ["DD-001", "", "DD-003", "DD-004"])
        empties = [p for p in reg.verify("s", 4) if p["kind"] == "empty"]
        assert empties and empties[0]["slot"] == 1

    def test_unregistered_id_in_an_assignment(self):
        reg = fleet(2, 0)
        reg.assignments["s"] = ["DD-001", "GHOST"]
        assert "missing" in {p["kind"] for p in reg.verify("s", 2)}

    def test_advisories_are_separate_from_problems(self):
        reg = fleet(2, 0)
        reg.get("DD-001").battery_cycles = BATTERY_CYCLE_WARNING + 10
        reg.auto_assign("s", 2)
        kinds = {p["kind"] for p in reg.verify("s", 2)}
        assert kinds == {"advisory"}


class TestSwappedAirframes:
    """The failure no trajectory check can see."""

    def test_a_swap_is_caught_from_both_sides(self):
        reg = fleet(4, 0)
        reg.assign("s", ["DD-001", "DD-002", "DD-003", "DD-004"])
        live = telemetry([2, 1, 3, 4])        # DD-001 and DD-002 swapped pads
        wrong = [p for p in reg.verify("s", 4, live) if p["kind"] == "wrong_airframe"]
        assert {p["slot"] for p in wrong} == {0, 1}
        assert "DD-002" in wrong[0]["detail"], "the culprit must be named"

    def test_a_correct_fleet_passes(self):
        reg = fleet(4, 0)
        reg.assign("s", ["DD-001", "DD-002", "DD-003", "DD-004"])
        assert not [p for p in reg.verify("s", 4, telemetry([1, 2, 3, 4]))
                    if p["kind"] == "wrong_airframe"]

    def test_an_unregistered_airframe_answering(self):
        reg = fleet(2, 0)
        reg.assign("s", ["DD-001", "DD-002"])
        wrong = [p for p in reg.verify("s", 2, telemetry([1, 99]))
                 if p["kind"] == "wrong_airframe"]
        assert wrong and "99" in wrong[0]["detail"]

    def test_unknown_sysid_on_either_side_is_not_a_false_alarm(self):
        """No id recorded, or none heard yet, is a gap - not a swap."""
        reg = fleet(2, 0)
        reg.get("DD-002").sysid = 0
        reg.assign("s", ["DD-001", "DD-002"])
        assert not [p for p in reg.verify("s", 2, telemetry([0, 7]))
                    if p["kind"] == "wrong_airframe"]

    def test_preflight_blocks_arming_on_a_swap(self):
        show = small_show(4)
        reg = fleet(4, 0)
        reg.assign(show.name, ["DD-001", "DD-002", "DD-003", "DD-004"])
        driver = SimDriver()
        manager = FleetManager(show, driver=driver, clock=ManualClock(), registry=reg)
        # Simulated aircraft report sysid = index + 1; swap two on the link.
        driver.states[0].sysid, driver.states[1].sysid = 2, 1

        checks = {c.name: c for c in manager.preflight()}
        identity = checks["airframe identity"]
        assert not identity.ok and identity.blocking
        assert "swapped" in identity.detail

        manager.validate()
        manager.upload()
        with pytest.raises(PermissionError, match="airframe identity"):
            manager.arm()

    def test_without_a_registry_there_is_no_identity_check(self):
        manager = FleetManager(small_show(4), clock=ManualClock())
        assert "airframe identity" not in {c.name for c in manager.preflight()}

    def test_maintenance_advisories_do_not_block(self):
        show = small_show(4)
        reg = fleet(4, 0)
        reg.get("DD-001").firmware = ""
        reg.assign(show.name, ["DD-001", "DD-002", "DD-003", "DD-004"])
        manager = FleetManager(show, clock=ManualClock(), registry=reg)
        checks = {c.name: c for c in manager.preflight()}
        assert checks["airframe identity"].ok
        assert not checks["fleet maintenance"].ok
        assert not checks["fleet maintenance"].blocking


# ==========================================================================
# After the show
# ==========================================================================


def flown(show, reg, **faults):
    clock = ManualClock()
    manager = FleetManager(show, driver=SimDriver(faults=SimFaults(**faults)),
                           clock=clock, registry=reg)
    manager.validate()
    manager.upload()
    manager.arm(override_reason="test")
    manager.start(lead_in=0.0)
    return manager, clock


class TestFlightRecording:
    def test_completed_show_credits_its_real_duration(self):
        show = small_show(4)
        reg = fleet(4, 0)
        reg.auto_assign(show.name, 4)
        manager, clock = flown(show, reg)
        while manager.phase == Phase.RUNNING:
            manager.tick()
            clock.t += 0.1
        touched = manager.record_to_registry()
        assert len(touched) == 4
        d = reg.get(touched[0])
        assert d.flight_seconds == pytest.approx(manager.show_time, abs=0.2)
        assert d.flight_count == 1 and d.battery_cycles == 1

    def test_aborted_show_credits_only_what_it_flew(self):
        """40 seconds flown is 40 seconds, not the designed duration."""
        show = small_show(4)
        reg = fleet(4, 0)
        reg.auto_assign(show.name, 4)
        manager, clock = flown(show, reg)
        for _ in range(400):
            manager.tick()
            clock.t += 0.1
        manager.emergency("land")
        manager.record_to_registry()
        flew = reg.get(reg.assignment(show.name)[0]).flight_seconds
        assert flew == pytest.approx(40.0, abs=0.5)
        assert flew < show.duration

    def test_a_landed_fleet_after_an_abort_is_still_credited(self):
        """Regression: crediting by current altitude missed landed drones."""
        show = small_show(4)
        reg = fleet(4, 0)
        reg.auto_assign(show.name, 4)
        manager, clock = flown(show, reg)
        for _ in range(300):
            manager.tick()
            clock.t += 0.1
        for state in manager.drones:
            state.altitude_agl = 0.0          # all back on the ground
        assert len(manager.record_to_registry()) == 4

    def test_a_show_that_never_started_credits_nothing(self):
        show = small_show(4)
        reg = fleet(4, 0)
        reg.auto_assign(show.name, 4)
        manager = FleetManager(show, clock=ManualClock(), registry=reg)
        assert manager.record_to_registry() == []
        assert reg.get("DD-001").battery_cycles == 0

    def test_negative_time_is_refused(self):
        reg = fleet(2, 0)
        reg.auto_assign("s", 2)
        with pytest.raises(RegistryError, match="negative"):
            reg.record_flight("s", -5.0)

    def test_flight_log_names_the_airframes(self):
        show = small_show(4)
        reg = fleet(4, 0)
        reg.auto_assign(show.name, 4)
        manager, _ = flown(show, reg)
        assert manager.flight_log()["airframes"] == reg.assignment(show.name)


class TestPersistence:
    def test_round_trip(self, tmp_path):
        reg = fleet(4, 0)
        reg.get("DD-002").serial = "SN-9"
        reg.set_status("DD-003", "maintenance", "arm")
        reg.auto_assign("s", 3)
        reg.record_flight("s", 120.0)
        back = Registry.load(reg.save(tmp_path / "fleet.json"))
        assert set(back.drones) == set(reg.drones)
        assert back.get("DD-002").serial == "SN-9"
        assert back.get("DD-003").status == "maintenance"
        assert back.assignment("s") == reg.assignment("s")
        assert back.get(reg.assignment("s")[0]).flight_seconds == 120.0

    def test_missing_file_is_an_empty_registry(self, tmp_path):
        assert Registry.load(tmp_path / "nope.json").drones == {}

    def test_corrupt_file_is_reported(self, tmp_path):
        p = tmp_path / "fleet.json"
        p.write_text("{ broken", encoding="utf-8")
        with pytest.raises(RegistryError, match="corrupt"):
            Registry.load(p)

    def test_foreign_file_is_refused(self):
        with pytest.raises(RegistryError, match="not a Sky Shine registry"):
            Registry.from_dict({"format": "something.else"})

    def test_summary(self):
        reg = fleet(4, 0)
        reg.set_status("DD-004", "retired")
        reg.get("DD-001").battery_cycles = BATTERY_CYCLE_WARNING
        s = reg.summary()
        assert s["registered"] == 4 and s["ready"] == 3 and s["retired"] == 1
        assert s["packs_past_review"] == 1


# ==========================================================================
# Post-show report
# ==========================================================================


class TestPostShowReport:
    def _log(self, **faults):
        show = small_show(6)
        reg = fleet(6, 0)
        reg.auto_assign(show.name, 6)
        manager, clock = flown(show, reg, **faults)
        while manager.phase == Phase.RUNNING:
            manager.tick()
            clock.t += 0.1
        return manager.flight_log(), reg

    def test_completed_outcome(self):
        log, _ = self._log()
        s = reports.summarise(log)
        assert s["outcome"] == "completed"
        assert s["percent_flown"] == pytest.approx(100.0, abs=0.5)

    def test_issues_are_attributed_to_drones(self):
        log, _ = self._log(link_loss={2: 10.0}, position_drift={4: 0.15})
        s = reports.summarise(log)
        assert "link_lost" in s["issues_by_drone"][2]
        assert any("deviation" in k for k in s["issues_by_drone"][4])
        assert s["offline_at_end"] == 1

    def test_aborted_outcome(self):
        show = small_show(4)
        reg = fleet(4, 0)
        reg.auto_assign(show.name, 4)
        manager, clock = flown(show, reg)
        for _ in range(200):
            manager.tick()
            clock.t += 0.1
        manager.emergency("land")
        assert reports.summarise(manager.flight_log())["outcome"] == "aborted"

    def test_not_flown_outcome(self):
        manager = FleetManager(small_show(4), clock=ManualClock())
        assert reports.summarise(manager.flight_log())["outcome"] == "not flown"

    def test_html_names_the_airframe_that_had_trouble(self):
        log, reg = self._log(link_loss={2: 10.0})
        html = reports.post_show_html(log, registry_summary=reg.summary())
        assert reg.assignment(log["show"])[2] in html
        assert "link_lost" in html

    def test_operator_text_is_escaped(self):
        """Override reasons are typed by people; the report is HTML."""
        log, _ = self._log()
        log["overrides"] = ['<script>alert("x")</script>']
        log["events"].append({"iso": "2026-01-01T00:00:00", "kind": "override",
                              "message": "<img src=x onerror=alert(1)>", "data": {}})
        html = reports.post_show_html(log)
        assert "<script>alert" not in html
        assert "<img src=x" not in html
        assert "&lt;script&gt;" in html


# ==========================================================================
# API
# ==========================================================================


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from skyshine import server

    tmp = pathlib.Path(tempfile.mkdtemp())
    monkeypatch.setattr(server, "SHOW_DIR", tmp / "shows")
    monkeypatch.setattr(server, "LOG_DIR", tmp / "logs")
    monkeypatch.setattr(server, "FLEET_FILE", tmp / "fleet.json")
    monkeypatch.setattr(server, "_fleet", None)
    (tmp / "shows").mkdir()
    return TestClient(server.app)


class TestFleetApi:
    def test_bulk_register_and_list(self, client):
        r = client.post("/api/fleet/drones/bulk", json={"count": 5, "model": "X1"})
        assert r.json()["registered"][0] == "DD-001"
        listing = client.get("/api/fleet/drones").json()
        assert listing["summary"]["registered"] == 5

    def test_add_single_and_reject_duplicate(self, client):
        assert client.post("/api/fleet/drones", json={"id": "A-1"}).status_code == 200
        r = client.post("/api/fleet/drones", json={"id": "A-1"})
        assert r.status_code == 400 and "already registered" in r.text

    def test_patch_fields_and_status(self, client):
        client.post("/api/fleet/drones/bulk", json={"count": 2})
        r = client.patch("/api/fleet/drones/DD-001",
                         json={"firmware": "4.5.7", "status": "maintenance",
                               "reason": "arm"})
        d = r.json()["drone"]
        assert d["firmware"] == "4.5.7" and d["status"] == "maintenance"

    def test_patch_rejects_a_bad_sysid(self, client):
        client.post("/api/fleet/drones/bulk", json={"count": 1})
        r = client.patch("/api/fleet/drones/DD-001", json={"sysid": 999})
        assert r.status_code == 400

    def test_auto_assign_to_a_show(self, client):
        from skyshine import server
        from skyshine.show import save_show

        save_show(small_show(4, "api show"), server.SHOW_DIR / "api show.show.json")
        client.post("/api/fleet/drones/bulk", json={"count": 6})
        r = client.post("/api/fleet/assign/api show", json={"auto": True}).json()
        assert len([x for x in r["assignment"] if x]) == 4
        state = client.get("/api/fleet/assign/api show").json()
        assert state["ready"] is True

    def test_assign_with_too_few_drones_is_explained(self, client):
        from skyshine import server
        from skyshine.show import save_show

        save_show(small_show(4, "big"), server.SHOW_DIR / "big.show.json")
        client.post("/api/fleet/drones/bulk", json={"count": 2})
        r = client.post("/api/fleet/assign/big", json={"auto": True})
        assert r.status_code == 400 and "ready" in r.text

    def test_load_finish_and_report(self, client):
        from skyshine import server
        from skyshine.show import save_show

        save_show(small_show(4, "ops"), server.SHOW_DIR / "ops.show.json")
        client.post("/api/fleet/drones/bulk", json={"count": 4})
        client.post("/api/fleet/assign/ops", json={"auto": True})
        client.post("/api/ops/load", json={"show": "ops", "driver": "simulator"})
        pre = client.get("/api/ops/preflight").json()
        assert any(c["name"] == "airframe identity" for c in pre["checks"])

        client.post("/api/ops/validate")
        client.post("/api/ops/upload", json={})
        client.post("/api/ops/arm", json={})
        client.post("/api/ops/start", json={"lead_in": 0.0})
        import time

        time.sleep(1.5)
        server._fleet.tick()

        done = client.post("/api/ops/finish").json()
        assert done["airframes_credited"] == 4
        # Seconds, not the summary's hours: a 1.5 s test flight rounds to
        # 0.00 h at the two decimals the summary shows people.
        drones = client.get("/api/fleet/drones").json()["drones"]
        assert all(d["flight_seconds"] > 0 for d in drones)
        assert all(d["battery_cycles"] == 1 for d in drones)

        report = client.get("/api/ops/report")
        assert report.status_code == 200
        assert "Post-show report" in report.text


class TestFleetCsv:
    def test_export_edit_import_round_trip(self, client):
        client.post("/api/fleet/drones/bulk", json={"count": 3, "model": "X1"})
        r = client.get("/api/fleet/export")
        assert r.headers["content-type"].startswith("text/csv")
        assert "attachment" in r.headers["content-disposition"]
        lines = r.text.splitlines()
        assert lines[0].startswith("id,serial,model") and len(lines) == 4

        # Fill in a serial and firmware for DD-002, and add a new airframe.
        head = lines[0].split(",")
        row = lines[2].split(",")
        row[head.index("serial")] = "SN-2"
        row[head.index("firmware")] = "4.5.1"
        lines[2] = ",".join(row)
        lines.append("DD-900,SN-900,X2,,4.5.1,,,,90,maintenance,,,,")
        r = client.post("/api/fleet/import", json={"csv": "\n".join(lines)}).json()
        assert r["added"] == ["DD-900"] and len(r["updated"]) == 3

        drones = {d["id"]: d for d in client.get("/api/fleet/drones").json()["drones"]}
        assert drones["DD-002"]["serial"] == "SN-2"
        assert drones["DD-002"]["firmware"] == "4.5.1"
        assert drones["DD-900"]["status"] == "maintenance"
        assert drones["DD-900"]["sysid"] == 90

    def test_import_never_winds_back_flight_counters(self):
        from skyshine.registry import export_csv, import_csv

        reg = fleet(2, extra=0)
        sheet = export_csv(reg)
        first = next(iter(reg.drones.values()))
        first.flight_seconds, first.flight_count = 600.0, 3
        import_csv(reg, sheet)
        assert (first.flight_seconds, first.flight_count) == (600.0, 3)

    @pytest.mark.parametrize("sheet, complaint", [
        ("serial,model\nA,B", "'id' column"),
        ("id,colour\nDD-1,red", "unknown column"),
        ("id,sysid\nDD-1,abc", "row 2: sysid must be a number"),
        ("id,sysid\nDD-1,999", "row 2"),
        ("id\nDD-1\nDD-1", "row 3: DD-1 appears twice"),
        ("id,status\nDD-1,flying", "row 2"),
        ("id,serial\nDD-1,S\nDD-2,S", "row 3: serial 'S' is already registered"),
    ])
    def test_a_bad_file_is_refused_whole(self, client, sheet, complaint):
        client.post("/api/fleet/drones/bulk", json={"count": 2})
        r = client.post("/api/fleet/import", json={"csv": sheet})
        assert r.status_code == 400 and complaint in r.json()["detail"]
        # Nothing from the file landed, not even the rows before the bad one.
        ids = [d["id"] for d in client.get("/api/fleet/drones").json()["drones"]]
        assert ids == ["DD-001", "DD-002"]

    def test_formula_cells_are_defused_on_export_and_restored_on_import(self):
        from skyshine.registry import export_csv, import_csv

        reg = Registry()
        reg.add(Drone(id="DD-1", notes="=HYPERLINK(\"http://x\")"))
        sheet = export_csv(reg)
        assert ",'=HYPERLINK" in sheet.replace('"', "")
        fresh = Registry()
        import_csv(fresh, sheet)
        assert fresh.get("DD-1").notes == "=HYPERLINK(\"http://x\")"
