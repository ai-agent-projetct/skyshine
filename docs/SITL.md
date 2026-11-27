# Software in the loop

How to fly a Sky Shine show without an aircraft, and what each rung of the
ladder actually proves.

```bash
python -m skyshine sitl "my show prepared"                  # virtual MAVLink fleet
python -m skyshine sitl "my show prepared" --mode guided    # streamed setpoints
python -m skyshine sitl "my show prepared" --ardupilot      # real ArduPilot SITL
```

Or pick **SITL: virtual aircraft over real MAVLink** as the driver in the UI.

---

## Three rungs, and what each one is worth

| rung | driver | what it proves | what it cannot tell you |
|---|---|---|---|
| 1 | `simulator` | the plan, the gates, the watchdog, the whole operational state machine | nothing about MAVLink: it never serialises a byte |
| 2 | `sitl` / `sitl-guided` | the wire protocol: command encoding, frame conventions, FTP transfer, ack handling, clock discipline | nothing about flight dynamics: there is no controller and no EKF |
| 3 | `--ardupilot` | real ArduCopter: real controller, real EKF, real prearm checks, real failsafes | nothing about your airframe, your radio or the weather |

Climb all three before you fly. Rung 2 is the one most likely to catch a bug
that would otherwise show up on the flight line, because it is the first rung
where the ground station and the aircraft are genuinely separate programs
talking over a socket.

---

## Rung 2: the virtual MAVLink fleet

`skyshine.sitl.VirtualFleet` runs N aircraft, each on its own UDP port, serviced
by one thread. They are real MAVLink 2 endpoints: they stream `HEARTBEAT`,
`GLOBAL_POSITION_INT`, `GPS_RAW_INT`, `SYS_STATUS` and `SYSTEM_TIME`, answer
`COMMAND_LONG` and `COMMAND_INT` with `COMMAND_ACK`, serve MAVLink FTP, and fly
an uploaded plan from their own clock.

```python
from skyshine.sitl import VirtualFleet
from skyshine.drivers import MavlinkNativeDriver
from skyshine.fleet import FleetManager

fleet = VirtualFleet(show.drone_count)
fleet.place(show.formations[0].points)        # stand them on the launch pads
fleet.start()

driver = MavlinkNativeDriver(fleet.endpoints, origin=show.origin)
driver.connect(show.drone_count, wait_heartbeat=20.0)

manager = FleetManager(show, driver=driver)
manager.wait_for_fleet()
```

One thread services the whole fleet rather than one thread per aircraft: a
hundred Python threads contending on the GIL would add scheduling jitter that
looks exactly like the timing faults this is meant to detect.

### Ground truth

`GET /api/ops/sitl` and `VirtualFleet.summary()` report what the aircraft
*actually* are — plans loaded, start instants scheduled, origins accepted,
modes, command refusals, bytes moved. The ground station only knows what
telemetry tells it, and the gap between the two is the entire reason to run
SITL. The UI shows both side by side.

### Faults you can inject

| fault | what it reproduces |
|---|---|
| `--go-silent 3:60` | drone 3 stops transmitting 60 s into the show |
| `--weak-battery 7:55` | drone 7 boots at 55% |
| `--refuse-arm 0:1` | drone 0 fails its prearm checks and NAKs the arm command |
| `--clock-offset 2:0.4` | drone 2 believes the time is 0.4 s off |
| `--noise 0.5` | GNSS noise at non-RTK levels |

`--clock-offset` is the important one. See below.

---

## What this exercise actually found

These are not hypothetical. Each was a real defect in the code before SITL
existed, and each would have passed every simulator test.

### 1. A unix timestamp will not fit in a MAVLink float param

`start()` originally only set the flight mode. The aircraft was never told
*when* to begin. Adding the start instant then ran into the real trap: at
1.79e9, the gap between representable `float32` values is **128 seconds**. A
start time sent through a float param could land over two minutes out, with
every drone rounding to a slightly different instant.

The show start is therefore `MAV_CMD_USER_3` as a **`COMMAND_INT`**, with whole
seconds in the int32 `x` field and the millisecond remainder in `y`.
`test_show_start_survives_the_float32_trap` asserts both that the aircraft
receives it to the millisecond and that the float32 error really is over a
second, so nobody can "tidy" it back into a float.

The same reasoning is why `MAV_CMD_USER_2` carries the show origin as a
`COMMAND_INT`: latitude as integer 1e-7 degrees, not through a float that would
discard about a centimetre — the same order as the RTK accuracy the show rests
on. The virtual aircraft **NAKs** that command when it arrives as a
`COMMAND_LONG`, and the test checks the refusal surfaces to the operator.

### 2. The ground station never sent a heartbeat

A real GCS heartbeats at 1 Hz. Ours did not, and a drone reached over `udpin`
has no return address until it has received a packet — so the ground station
waited for a heartbeat the aircraft could not send. Nothing connected, with no
error to explain why. Firmware also uses the GCS heartbeat to detect a lost
link, so its absence disables GCS failsafe.

### 3. `FleetManager` tore down a live link

It called `driver.connect()` unconditionally, so a driver handed in already
connected — with its heartbeat timeout already satisfied — had its sockets
rebuilt and every drone went to "offline". It now adopts an existing link, and
`wait_for_fleet()` blocks until telemetry actually arrives, because running
preflight immediately after connecting fails every check for want of data.

### 4. A partial upload is worse than a failed one

If some aircraft hold the plan and some do not, the ones that do not sit on the
pad while the rest take off. `upload()` now fails the whole operation and names
which drones are loaded.

### 5. The deviation watchdog cried wolf

A position controller trails its setpoint while moving. Against aircraft with
realistic tracking lag, the default 3 m abort threshold fired on every fast
transition — and an operator who learns to dismiss an alert will dismiss the one
that matters. Preflight now checks `abort_deviation_m` against
`peak_speed × expected_tracking_lag_s` and says so. Non-blocking: it is a tuning
problem, not a reason to stop a launch.

### 6. On Windows, pymavlink's `udpout` fights its own destination

`mavlink_connection("udpout:host:port")` binds the *local* socket to the
destination port on Windows, so a listener on that port and the sender compete
for the same datagrams. Sky Shine's links are therefore `udpin:` — the ground
station listens on one port per aircraft and each aircraft streams to it, which
is also how ArduPilot's `--out` is configured in the field.

---

## The clock fault

> Every drone flies the correct path. One flies it four seconds early. It is in
> exactly the right place at exactly the wrong time, and **nothing in trajectory
> validation can see it.**

`docs/SAFETY.md` §5 lists clock error as outside the separation guarantee. SITL
is where that gap gets tested:

```bash
python -m skyshine sitl "my show prepared" --clock-offset 2:4.0
```

`test_a_skewed_clock_desynchronises_that_aircraft` confirms the drone ends up
closer to where its own plan puts it four seconds ahead than to where it should
be now. The only defence is on the ground: the `clock synchronisation` preflight
check reads `SYSTEM_TIME` from every aircraft and **blocks arming** when any of
them is more than 50 ms out.

Verified end to end: a 0.35 s offset injected into one of eight aircraft fails
preflight and arming returns `409`.

---

## Rung 3: real ArduPilot SITL

Sky Shine does not bundle ArduPilot, because SITL is the real autopilot and has
to be the real autopilot.

```bash
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot
cd ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y
./waf configure --board sitl && ./waf copter
```

On Windows, run that inside WSL2 (`wsl --install`) and point Sky Shine at the
WSL host, or pass `--sim-vehicle /path/to/sim_vehicle.py`.

```bash
python -m skyshine sitl "my show prepared" --ardupilot --mode guided --speedup 5
```

Each instance gets an `--instance` number; ArduPilot offsets its port block by
10, so instance N serves a GCS on TCP `5760 + 10*N`. The launcher spreads homes
across a grid so the instances do not all spawn on top of each other, which
upsets the EKF and is not what a launch grid looks like anyway.

**Use `--mode guided` against stock ArduCopter.** There is no show mode in stock
firmware; `--mode native` needs a build that adds one. A dozen instances will
saturate a laptop, so rehearse the protocol on rung 2 and the flight behaviour
here.

`test_real_sitl_boots_and_heartbeats` runs automatically when `sim_vehicle.py`
is on the path and skips otherwise.

---

## What SITL still does not cover

Rung 2 has no flight dynamics: a virtual aircraft tracks its commanded position
with first-order lag and noise, nothing more. Rung 3 has a real controller and a
real EKF, but still no real airframe, no real radio, no real wind and no real
GNSS multipath.

A clean SITL run means your plan is sound and your protocol is correct. It is
not a flight test, and the checklist in `docs/SAFETY.md` §9 still applies in
full.
