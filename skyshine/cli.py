"""Command line: `python -m skyshine <command>`.

Everything the GUI does, available without it, so that validation and
rehearsal can run in CI and a show can be signed off from a terminal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import geometry, planning, safety
from .drivers import SimDriver, SimFaults
from .fleet import rehearse
from .show import Formation, Show, ShowLimits, load_show, save_show
from .trajectory import (
    build_mission, build_timeline, evaluate_plan, mission_key, verify_mission,
)

ROOT = Path(__file__).resolve().parent.parent


def cmd_shapes(args) -> int:
    for name in sorted(geometry.SHAPES):
        doc = (geometry.SHAPES[name].__doc__ or "").strip().splitlines()
        print(f"  {name:<14} {doc[0] if doc else ''}")
    return 0


def cmd_demo(args) -> int:
    """Build a reference show: a real storyboard, not a single formation."""
    n = args.drones
    sep = args.separation
    limits = ShowLimits(
        min_separation=sep,
        max_altitude=args.ceiling,
        battery_endurance_s=args.endurance,
    )
    width, height = geometry.billboard_for(n, limits.formation_separation)
    print(
        f"{n} drones at {limits.formation_separation:.2f} m formation spacing need "
        f"roughly a {width:.0f} x {height:.0f} m billboard."
    )
    if height + limits.min_nav_altitude > limits.max_altitude:
        print(
            f"  warning: that is taller than the {limits.max_altitude:.0f} m ceiling. "
            "Shapes will be refused or the show will fail validation."
        )

    show = Show(name=args.name, drone_count=n, limits=limits)
    show.origin.latitude = args.latitude
    show.origin.longitude = args.longitude
    show.origin.amsl = args.amsl

    base = args.ceiling * 0.32
    radius = min(width, height) * 0.5
    if args.template:
        from . import templates

        show = templates.build(
            args.template, args.name, n, text=args.text, limits=limits,
            origin=show.origin,
        )
        storyboard = []
    else:
        storyboard = [
            ("sphere rise", geometry.ball(n, radius * 0.85), (20, 90, 255)),
            ("open heart", geometry.heart(n, solid=True), (255, 40, 90)),
            ("five point star", geometry.star(n, solid=True), (255, 205, 60)),
            ("spinning disc", geometry.disc(n, radius), (0, 230, 190)),
            ("cube", geometry.cube_shell(n, radius * 1.4), (190, 120, 255)),
            ("wide wave", geometry.wave(n, width, height * 0.22, 2.0), (90, 255, 140)),
        ]
    for label, pts, color in storyboard:
        show.add(
            Formation(
                name=label,
                points=geometry.lift(pts, base),
                colors=np.asarray(color, dtype=np.int32),
                transition=args.transition,
                hold=args.hold,
            )
        )

    out = Path(args.out) if args.out else ROOT / "shows" / f"{args.name}.show.json"
    save_show(show, out)
    print(f"wrote {out}  ({len(show.formations)} formations, {show.duration:.0f} s raw)")
    return 0


def _load(path: str) -> Show:
    p = Path(path)
    if not p.exists():
        candidate = ROOT / "shows" / f"{path}.show.json"
        if candidate.exists():
            p = candidate
        else:
            print(f"no such show file: {path}", file=sys.stderr)
            raise SystemExit(2)
    return load_show(p)


def cmd_prepare(args) -> int:
    show = _load(args.show)
    prepared = planning.prepare(show)
    for note in prepared.notes:
        print(f"  note: {note}")
    prepared.show.name = args.name or f"{show.name} prepared"
    out = Path(args.out) if args.out else ROOT / "shows" / f"{prepared.show.name}.show.json"
    save_show(prepared.show, out)
    print(
        f"prepared {len(prepared.show.formations)} formations, "
        f"{prepared.show.duration:.1f} s -> {out}"
    )
    print("launch grid corners (m):")
    g = prepared.grid
    print(
        f"  x {g[:, 0].min():.1f} .. {g[:, 0].max():.1f}   "
        f"y {g[:, 1].min():.1f} .. {g[:, 1].max():.1f}   "
        f"spacing {geometry.min_pair_distance(g):.2f}"
    )
    return 0


def cmd_validate(args) -> int:
    show = _load(args.show)
    timeline = build_timeline(show)
    report = safety.validate(show, timeline, check_takeoff_proximity=args.strict)
    print(report.summary())
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=1), encoding="utf-8")
        print(f"\nreport written to {args.json}")
    return 0 if report.passed else 1


def cmd_export(args) -> int:
    show = _load(args.show)
    report = safety.validate(show)
    if not report.passed and not args.force:
        print(report.summary())
        print(
            "\nrefusing to export a show that failed validation. "
            "Pass --force only if you intend to record a deliberate override."
        )
        return 1
    mission = build_mission(show, safety=report.to_dict())
    out = Path(args.out) if args.out else ROOT / "shows" / f"{show.name}.mission.json"
    import getpass

    signer = getpass.getuser()
    mission.save(out, key=mission_key(ROOT / "mission.key"), signer=signer)
    body = mission.to_dict()

    # Prove the compression is lossless before anyone uploads it: replay
    # each drone's compressed plan and compare against the sampled design.
    timeline = build_timeline(show)
    worst = 0.0
    for plan in mission.plans:
        for t in np.linspace(0.0, timeline.duration, 97):
            want = timeline.sample(float(t))[0][plan.drone_index]
            got = evaluate_plan(plan, float(t))
            worst = max(worst, float(np.linalg.norm(want - got)))
    print(f"wrote {out}")
    print(f"  checksum              {body['checksum']}")
    print(f"  signed by             {signer}")
    print(f"  drones                {mission.drone_count}")
    print(f"  duration              {mission.duration:.1f} s")
    # Sub-millimetre is the sampled timeline's own linear-interpolation
    # error, not the plan's: the compressed plan replays the exact easing,
    # so it is the more accurate of the two. A large number here would mean
    # the compression dropped a control point.
    print(f"  plan vs sampled design {worst * 1000:.3f} mm")
    if worst > 0.01:
        print("  WARNING: the uploaded plan does not reproduce the designed path")
        return 1
    return 0


def cmd_verify_mission(args) -> int:
    try:
        body = json.loads(Path(args.file).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"cannot read {args.file}: {exc}")
        return 1
    valid, detail = verify_mission(body, mission_key(ROOT / "mission.key"))
    print(f"{'VALID' if valid else 'INVALID'}: {detail}")
    return 0 if valid else 1


def _parse_map(text: str | None, cast=float) -> dict[int, float]:
    out: dict[int, float] = {}
    for pair in (text or "").replace(",", " ").split():
        key, _, value = pair.partition(":")
        if value:
            out[int(key)] = cast(value)
    return out


def cmd_rehearse(args) -> int:
    show = _load(args.show)
    faults = SimFaults(
        link_loss=_parse_map(args.link_loss),
        weak_battery=_parse_map(args.weak_battery),
        gps_degradation=_parse_map(args.gps_degraded),
        position_drift=_parse_map(args.drift),
        wind=tuple(float(v) for v in args.wind.split(",")) if args.wind else (0.0, 0.0, 0.0),
        position_noise=args.noise,
    )
    driver = SimDriver(faults=faults, battery_endurance_s=show.limits.battery_endurance_s)
    fleet, summary = rehearse(
        show, driver=driver, step=args.step, override_reason=args.override
    )
    print(f"rehearsal of {show.name!r} on the simulator")
    print(f"  safety passed         {summary['safety_passed']}")
    print(f"  completed             {summary['completed']} (phase {summary['phase']})")
    print(f"  show time reached     {summary['show_time']} s")
    print(f"  worst deviation       {summary['worst_deviation_m']} m")
    if summary["issues"]:
        print("  conditions raised:")
        for issue in summary["issues"]:
            print(
                f"    t={issue['first_seen_s']:>7.1f}s  drone {issue['drone']:>4}  "
                f"{issue['kind']}"
            )
    else:
        print("  conditions raised     none")
    if args.log:
        fleet.save_flight_log(args.log)
        print(f"  flight log            {args.log}")
    return 0 if summary["completed"] and summary["safety_passed"] else 1


def cmd_sitl(args) -> int:
    """Fly a show over real MAVLink against virtual or real aircraft."""
    import time

    from .drivers import MavlinkGuidedDriver, MavlinkNativeDriver
    from .fleet import FleetManager, Phase, PreflightPolicy
    from .sitl import AircraftFaults, ArduPilotSITL, VirtualFleet, sitl_available

    show = _load(args.show)
    n = show.drone_count
    driver_cls = MavlinkNativeDriver if args.mode == "native" else MavlinkGuidedDriver

    faults: dict[int, AircraftFaults] = {}
    for index, seconds in _parse_map(args.go_silent).items():
        faults.setdefault(index, AircraftFaults()).go_silent_at = seconds
    for index, pct in _parse_map(args.weak_battery).items():
        faults.setdefault(index, AircraftFaults()).battery_percent = pct
    for index, offset in _parse_map(args.clock_offset).items():
        faults.setdefault(index, AircraftFaults()).clock_offset = offset
    for index, _ in _parse_map(args.refuse_arm, cast=int).items():
        faults.setdefault(index, AircraftFaults()).refuse_arm = True

    # The real aircraft must already hold this key; virtual ones are given it.
    key = mission_key(ROOT / "link.key")[:32] if args.signed else None
    backend = None
    if args.ardupilot:
        if not sitl_available(args.sim_vehicle):
            print(ArduPilotSITL(sim_vehicle=args.sim_vehicle).locate.__doc__ or "")
            try:
                ArduPilotSITL(sim_vehicle=args.sim_vehicle).locate()
            except FileNotFoundError as exc:
                print(f"error: {exc}", file=sys.stderr)
            return 2
        backend = ArduPilotSITL(
            count=n,
            latitude=show.origin.latitude,
            longitude=show.origin.longitude,
            speedup=args.speedup,
            sim_vehicle=args.sim_vehicle,
        ).start()
        print(f"launched {n} ArduPilot SITL instances; waiting for boot")
        time.sleep(args.boot_wait)
        endpoints = backend.endpoints
    else:
        backend = VirtualFleet(n, faults=faults, noise=args.noise, signing_key=key)
        if show.formations:
            # Stand them on the launch grid, so the launch-position check
            # tests the plan rather than the harness.
            backend.place(show.formations[0].points)
        backend.start()
        endpoints = backend.endpoints
        print(f"started {n} virtual MAVLink aircraft on {endpoints[0]} ...")

    driver = driver_cls(endpoints, origin=show.origin, signing_key=key)
    rc = 1
    try:
        driver.connect(n, wait_heartbeat=args.connect_timeout)
        print(f"  heartbeats       {sum(s.connected for s in driver.poll())}/{n}")

        manager = FleetManager(
            show,
            driver=driver,
            policy=PreflightPolicy(auto_action=args.auto_action),
        )
        heard = manager.wait_for_fleet(args.connect_timeout)
        print(f"  telemetry        {heard}/{n}")

        report = manager.validate()
        print(f"  safety           {'passed' if report.passed else 'FAILED'}")
        if not report.passed:
            print(report.summary())
            return 1

        driver.set_origin(show.origin)
        started = time.time()
        manager.upload()
        print(f"  plans uploaded   {n} in {time.time() - started:.2f}s")

        failed = [c.name for c in manager.preflight() if c.blocking and not c.ok]
        if failed:
            print(f"  preflight        failing: {', '.join(failed)}")
        manager.arm(override_reason=args.override)
        time.sleep(1.5)
        print(f"  armed            {sum(s.armed for s in driver.poll())}/{n}")

        manager.start(lead_in=args.lead_in)
        print(f"  running          {manager.timeline.duration:.0f}s show")
        deadline = time.time() + manager.timeline.duration + args.lead_in + 5.0
        peak = 0.0
        while time.time() < deadline and manager.phase in (
            Phase.RUNNING, Phase.PAUSED
        ):
            snapshot = manager.tick()
            peak = max(peak, snapshot["max_deviation"])
            time.sleep(0.1)

        print(f"  phase            {manager.phase.value}")
        print(f"  peak deviation   {peak:.2f} m")
        if hasattr(backend, "summary"):
            summary = backend.summary()
            print(f"  aircraft modes   {', '.join(summary['modes'])}")
            print(f"  plans loaded     {summary['plans_loaded']}/{n}")
            print(f"  telemetry bytes  {summary['tx_bytes']:,} up, "
                  f"{summary['rx_bytes']:,} down")
            if summary["nacks"]:
                print(f"  command refusals {summary['nacks']}")
            if summary["errors"]:
                print(f"  aircraft errors  {summary['errors']}")
        issues = sorted({(a["drone"], a["kind"]) for a in manager.alerts})
        print(f"  active alerts    {issues if issues else 'none'}")
        if args.log:
            manager.save_flight_log(args.log)
            print(f"  flight log       {args.log}")
        rc = 0 if manager.phase == Phase.FINISHED else 1
    finally:
        driver.close()
        backend.stop()
    return rc


def cmd_user(args) -> int:
    """Manage accounts from the terminal - the only way to create the first."""
    import getpass

    from . import auth, server

    store = auth.AuthStore(server.USERS_FILE)

    def read_password(prompt: str = "password: ") -> str:
        # Never from the command line: argv lands in shell history and in
        # the process list where any other user on the machine can read it.
        if args.password_stdin:
            return sys.stdin.readline().rstrip("\n")
        first = getpass.getpass(prompt)
        if getpass.getpass("again: ") != first:
            raise auth.AuthError("passwords do not match")
        return first

    try:
        if args.action == "list":
            if not store.users:
                print("no accounts - the server accepts local requests only")
            for u in sorted(store.users.values(), key=lambda u: u.name):
                print(f"  {u.name:<20} {u.role:<9}{'  disabled' if u.disabled else ''}")
            return 0
        if not args.name:
            print("error: give a username", file=sys.stderr)
            return 2
        if args.action == "add":
            store.add_user(args.name, read_password(), args.role)
            print(f"added {args.name} ({args.role})")
            if args.role == "admin" and len(store.users) == 1:
                print("accounts now exist: the server requires sign-in from here on")
        elif args.action == "passwd":
            store.set_password(args.name, read_password("new password: "))
            print(f"password changed for {args.name}; their sessions were ended")
        elif args.action == "role":
            store.set_role(args.name, args.role)
            print(f"{args.name} is now {args.role}; their sessions were ended")
        elif args.action == "remove":
            store.remove_user(args.name)
            print(f"removed {args.name}")
    except auth.AuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_serve(args) -> int:
    from . import auth, server
    from .server import serve

    exposed = args.host not in ("127.0.0.1", "localhost", "::1")
    try:
        configured = auth.AuthStore(server.USERS_FILE).configured
    except auth.AuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if exposed and not configured:
        # Not fatal - the server refuses remote requests on its own in this
        # state - but almost certainly not what was intended.
        print(
            f"warning: serving on {args.host} with no user accounts. Remote "
            "requests will be refused until an admin exists:\n"
            "  python -m skyshine user add <name> --role admin",
            file=sys.stderr,
        )
    print(f"Sky Shine on http://{args.host}:{args.port}"
          + ("  (sign-in required)" if configured else "  (local only, no accounts)"))
    serve(host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skyshine",
        description="Sky Shine: drone light show design, validation and swarm control.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("shapes", help="list the formation shapes").set_defaults(func=cmd_shapes)

    d = sub.add_parser("demo", help="generate a reference show")
    d.add_argument("--name", default="sky shine demo")
    d.add_argument("--drones", type=int, default=200)
    d.add_argument("--separation", type=float, default=3.0)
    d.add_argument("--ceiling", type=float, default=120.0)
    d.add_argument("--endurance", type=float, default=900.0)
    d.add_argument("--transition", type=float, default=10.0)
    d.add_argument("--hold", type=float, default=6.0)
    d.add_argument("--latitude", type=float, default=0.0)
    d.add_argument("--longitude", type=float, default=0.0)
    d.add_argument("--amsl", type=float, default=None)
    d.add_argument("--template", help="build a built-in template instead "
                   "(showcase, wedding, new year, brand reveal, festival)")
    d.add_argument("--text", default="", help="words for the template's text formation")
    d.add_argument("--out")
    d.set_defaults(func=cmd_demo)

    pr = sub.add_parser("prepare", help="scale, wrap with takeoff and landing, time it")
    pr.add_argument("show")
    pr.add_argument("--name")
    pr.add_argument("--out")
    pr.set_defaults(func=cmd_prepare)

    v = sub.add_parser("validate", help="run the safety engine; exit 1 on failure")
    v.add_argument("show")
    v.add_argument("--json", help="also write the report to this path")
    v.add_argument(
        "--strict",
        action="store_true",
        help="also flag drones parked close together on the launch pads",
    )
    v.set_defaults(func=cmd_validate)

    e = sub.add_parser("export", help="build the upload bundle")
    e.add_argument("show")
    e.add_argument("--out")
    e.add_argument("--force", action="store_true", help="export despite a failed report")
    e.set_defaults(func=cmd_export)

    vm = sub.add_parser(
        "verify-mission", help="check a mission file's signature; exit 1 if it fails"
    )
    vm.add_argument("file")
    vm.set_defaults(func=cmd_verify_mission)

    r = sub.add_parser("rehearse", help="fly the show on the simulator, with faults")
    r.add_argument("show")
    r.add_argument("--step", type=float, default=0.1)
    r.add_argument("--wind", help="east,north,up in m/s")
    r.add_argument("--noise", type=float, default=0.03, help="position noise sigma, m")
    r.add_argument("--link-loss", help="drone:second, e.g. 3:40")
    r.add_argument("--weak-battery", help="drone:percent, e.g. 7:55")
    r.add_argument("--gps-degraded", help="drone:second, e.g. 11:60")
    r.add_argument("--drift", help="drone:metres_per_second, e.g. 5:0.08")
    r.add_argument("--override", help="reason, if the show must run despite failures")
    r.add_argument("--log", help="write a flight log here")
    r.set_defaults(func=cmd_rehearse)

    sl = sub.add_parser(
        "sitl", help="fly a show over real MAVLink against virtual or real aircraft"
    )
    sl.add_argument("show")
    sl.add_argument(
        "--mode", choices=["native", "guided"], default="native",
        help="native uploads the plan and the aircraft flies it from its own "
             "clock; guided streams setpoints from the ground",
    )
    sl.add_argument(
        "--ardupilot", action="store_true",
        help="use real ArduPilot SITL instead of the virtual MAVLink fleet",
    )
    sl.add_argument("--sim-vehicle", help="path to sim_vehicle.py")
    sl.add_argument("--speedup", type=float, default=1.0, help="ArduPilot SITL speedup")
    sl.add_argument("--boot-wait", type=float, default=25.0, help="seconds to let SITL boot")
    sl.add_argument("--connect-timeout", type=float, default=20.0)
    sl.add_argument("--lead-in", type=float, default=3.0)
    sl.add_argument("--noise", type=float, default=0.02, help="position noise sigma, m")
    sl.add_argument("--go-silent", help="drone:second, drop the link at that show time")
    sl.add_argument("--weak-battery", help="drone:percent")
    sl.add_argument(
        "--clock-offset",
        help="drone:seconds - the aircraft believes the wrong time, so it starts "
             "the show early or late. This is the fault that collides a fleet.",
    )
    sl.add_argument("--refuse-arm", help="drone:1, aircraft fails its prearm checks")
    sl.add_argument("--auto-action", default="none",
                    choices=["none", "hold", "rth", "land"])
    sl.add_argument("--override", help="reason, if preflight must be overridden")
    sl.add_argument("--log", help="write a flight log here")
    sl.add_argument("--signed", action="store_true",
                    help="MAVLink 2 signing with link.key; unsigned traffic is dropped")
    sl.set_defaults(func=cmd_sitl)

    u = sub.add_parser("user", help="manage sign-in accounts")
    u.add_argument("action", choices=["list", "add", "passwd", "role", "remove"])
    u.add_argument("name", nargs="?")
    u.add_argument("--role", default="viewer",
                   choices=["viewer", "designer", "operator", "admin"])
    u.add_argument("--password-stdin", action="store_true",
                   help="read the password from stdin, for scripted setup")
    u.set_defaults(func=cmd_user)

    s = sub.add_parser("serve", help="run the designer and ground station UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, PermissionError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
