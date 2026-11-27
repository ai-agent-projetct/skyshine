# The Sky Shine safety case

Why 500 drones flying this software do not hit each other, what that claim
actually covers, and where it stops.

Read this before flying anything. The claim is narrow and precise, and most
accidents live in the gap between a narrow claim and a broad assumption.

---

## 1. The claim

> For any show that `skyshine.safety.validate` passes, no two drones come
> within `min_separation` of each other, centre to centre, at any instant
> between the start of the climb and touchdown.

Two independent arguments support it:

- **by construction** — transitions are built so that breaching the minimum is
  geometrically impossible (§2, §3)
- **by measurement** — the finished trajectories are re-checked in continuous
  time by code that shares no assumptions with the planner (§4)

Either alone would be weaker than it looks. A proof only constrains the code
that matches it; a measurement only covers the instants it measures. Together
they cover each other's blind spot.

---

## 2. The transition bound

Let formation **A** have positions `a_i` and formation **B** have `b_i`.

### Step 1: the matching

Choose the assignment σ minimising `Σ |a_i − b_σ(i)|²`. Squared distance, not
distance — this matters, and §2.4 says why. Sky Shine uses
`scipy.optimize.linear_sum_assignment`, a true optimal assignment, never a
greedy or nearest-neighbour heuristic.

Relabel so σ is the identity. Optimality means no swap of any pair `i, j`
improves the total:

```
|a_i − b_i|² + |a_j − b_j|²  ≤  |a_i − b_j|² + |a_j − b_i|²
```

Expanding and cancelling the `|a|²` and `|b|²` terms:

```
−2 a_i·b_i − 2 a_j·b_j  ≤  −2 a_i·b_j − 2 a_j·b_i
     a_i·b_i + a_j·b_j  ≥   a_i·b_j + a_j·b_i
```

which rearranges to the one fact everything else depends on:

```
(a_i − a_j) · (b_i − b_j)  ≥  0            for every pair i, j        (*)
```

Geometrically: the vector separating any two drones at the start and the vector
separating the same two at the end never point in opposing directions. They
cannot "swap sides".

### Step 2: the motion

Every drone flies a straight line under **one shared** easing `s(t)` running
0 → 1 over the transition:

```
p_i(t) = (1 − s) a_i + s b_i
```

Write `u = a_i − a_j` and `v = b_i − b_j`. Then:

```
|p_i − p_j|² = (1−s)² |u|²  +  2 s(1−s) (u·v)  +  s² |v|²
```

By (*) the middle term is ≥ 0, and `s ∈ [0,1]` makes its coefficient ≥ 0, so:

```
|p_i − p_j|²  ≥  (1−s)² |u|² + s² |v|²
              ≥  min(|u|², |v|²) · ((1−s)² + s²)
              ≥  min(|u|², |v|²) / 2            since (1−s)² + s² ≥ ½
```

Taking the minimum over all pairs:

```
separation during the transition  ≥  min(d_A, d_B) / √2
```

where `d_A`, `d_B` are the closest-pair distances in each formation.

### Step 3: the consequence

Scale every formation so its closest pair is at least `min_separation × √2`
and the transition can never breach `min_separation`. That is
`ShowLimits.formation_separation`, and it defaults to exactly `√2 ×
min_separation`.

No search. No iterative de-confliction. No "try again with a different seed".

### 2.4 Why the two conditions are load-bearing

**Squared distance.** Minimising plain distance does not yield (*). The
inequality falls directly out of expanding squares; without it the middle term
can be negative and the bound collapses.

**One shared easing.** This is not a stylistic choice. `test_per_drone_timing_
destroys_the_bound` takes an optimally matched pair whose paths stay 4.00 m
apart for the entire transition under a shared easing — comfortably above the
2.91 m the bound promises — and shows that with the two drones on independent
clocks there is a pair of progress values, both perfectly legal, at which they
occupy the same point in space.

So the easing is applied to the transition, never per drone, and
`safety.check_design` warns on a show that mixes easings. Any *shared* easing
preserves the bound — the proof never touches the shape of `s(t)`, only that it
is common and lands in [0,1] — which is why min-jerk, smooth-step and linear are
all safe here.

A drone that does not move during a transition (identical start and end) is
fine. A drone on a different clock is not.

### 2.5 Tightness

The bound is approached, not merely satisfied. A 200-drone show measures a
closest approach of **3.017 m** against a 3.000 m limit, from formations held at
4.243 m. The factor √2 is real, and treating it as conservative padding would
be a mistake.

---

## 3. Takeoff and landing

A launch grid is the one place the bound cannot be satisfied directly. Drones
sit 2–4 m apart on the ground, so `d_A` for the first transition is the grid
spacing, and 3 m / √2 = 2.12 m — below the minimum.

### The climb

The climb is **purely vertical**. Horizontal separation is identical to grid
spacing throughout and the vertical motion is monotone and shared, so no pair
can close at all. `test_vertical_climb_preserves_horizontal_separation` enforces
that the lateral coordinates do not change.

### The layers

The climb ends in **altitude-separated layers**, assigned as a 2×2 checkerboard
over the grid (`grid_layers`). With four layers:

- same-layer neighbours are two grid cells apart → `2 × spacing`
- cross-layer neighbours are at least one cell plus one layer gap apart →
  `√(spacing² + gap²)`

A 3 m grid with a 3 m layer gap gives a cloud whose closest pair is 4.24 m,
which satisfies the bound exactly. `test_layered_launch_separates_more_than_
the_flat_grid` asserts `layered / √2 ≥ min_separation`.

### Landing

Return-to-home deliberately does **not** send each drone back to its own pad.
The matching is free to choose any pad, which is both shorter and better
separated. The descent is then purely vertical from wherever RTH parked each
drone — `test_landing_descends_vertically` enforces that the land formation
shares the RTH formation's lateral coordinates, because a drone drifting
sideways while sinking would pass through another layer's airspace.

If you must recover each airframe from its own pad, pass an ordered grid and
accept a longer, more crowded transit.

### What the validator ignores, and why

Drones parked on their pads are closer than `min_separation`. The sweep
suppresses findings where **both** drones are below `min_nav_altitude`,
otherwise the report is nothing but launch-pad noise and the one finding that
matters is buried. Grid spacing is instead checked statically and the climb is
refused outright if the grid is tighter than the minimum
(`test_takeoff_refuses_a_grid_tighter_than_the_minimum`).

`validate(..., check_takeoff_proximity=True)` and `validate --strict` check
everything, pads included.

---

## 4. The validator

Independent of §2 and §3. It reads finished trajectories and knows nothing
about how they were produced.

### 4.1 Continuous-time separation

The important part. Sampling a show at 10 Hz and checking only the samples
would miss a pair that dips below the minimum between two frames.
`test_catches_a_collision_that_happens_between_samples` is exactly that case:
two drones swap ends of a line in one interval, 10 m apart at both samples,
**zero** in the middle. A frame-by-frame validator reports a comfortable pass on
a show that destroys two aircraft.

Instead, for each sample interval:

1. treat the motion as a straight line and solve the pair distance minimum in
   closed form — a convex quadratic in the interval parameter, so the exact
   minimum is `s* = clip(−r₀·Δ / |Δ|², 0, 1)`
2. draw candidate pairs from a KD-tree query at radius
   `threshold + 2 × max_displacement`

Step 2 is complete, not a heuristic. If a pair breaches at any point in the
interval, then at the start of it their distance is at most
`threshold + |Δ_i| + |Δ_j| ≤ threshold + 2 × max_displacement`, so the query
cannot miss it.

The straight-line assumption between samples is itself covered: the real eased
path bows away from the chord by at most the sagitta, estimated from the
second difference of the sampled positions (`curvature_margin`) and **doubled**,
since two drones can bow toward each other. That margin is added to the
threshold and reported in every report (`curvature_margin_m`, typically ~3 mm).

### 4.2 The rest

| check | failure mode it catches |
|---|---|
| altitude ceiling | formation scaled up past the regulatory limit |
| altitude floor | a trajectory commanded below ground |
| geofence | lateral excursion outside the authorised box |
| horizontal speed, climb and descent rate | a transition too short for its distance |
| acceleration | a move the airframe cannot track |
| endurance | show longer than battery minus reserve |
| formation separation | artwork too dense for the drone count |
| zero-time move | a formation asked to move with no transition |
| drone count | a formation with the wrong number of positions |
| mixed easings | the §2.4 condition |
| missing AMSL | AGL-only operation on a dense show |
| yaw rate | a heading change faster than the aircraft can hold position through |
| yaw coverage | heading set on some formations and not others, leaving it undefined between them |
| pyro altitude | firing low enough that the spray reaches the ground still burning |
| pyro separation | two firing drones closer than the pyro minimum - a gerb throws sparks well past the airframe |
| pyro heading | firing with no heading defined, or facing the wrong way |
| pyro settle | firing while the airframe is still rotating into place |

Errors block. Warnings do not. `SafetyReport.passed` is true only when there
are no errors.

### 4.3 What the report tells you

Not just pass or fail, but how much room was left:
`separation_headroom_m`, `pair_seconds_checked`, peak speeds and accelerations
against their limits, total path length, and the curvature margin used.
A show that passes with 0.017 m of headroom passed; a show that passes with
8 m of headroom is a different kind of safe.

---

## 5. What this does **not** cover

The honest part. Everything here concerns the **plan**. None of it is a claim
about aircraft.

- **Position error.** The bound is about commanded positions. A drone with
  1 m of GNSS error consumes a third of a 3 m budget before anything else
  happens. This is why RTK is not optional for a dense show, and why the
  preflight gate requires fix type ≥ 5 and HDOP ≤ 1.2.
- **Airframe size.** Distances are centre to centre. A machine with 450 mm arms
  plus props occupies most of a metre. Set `min_separation` from the airframe,
  not from taste.
- **Wind.** A gust displaces the whole fleet, mostly coherently — which is
  largely benign — but not uniformly. Headroom is your margin here; the
  simulator injects wind so you can see the effect.
- **Tracking error.** A drone that cannot track its commanded path is not
  where the plan says. The watchdog measures this live; it cannot prevent it.
- **Clock error.** A drone on the wrong clock flies the right path at the
  wrong time, which is precisely what collides with its neighbours. Preflight
  requires every clock within 50 ms, and the firmware must run the show from a
  GNSS-derived clock, not from when a packet arrived. This is tested over real
  MAVLink in [SITL.md](SITL.md#the-clock-fault); a 0.35 s offset on one of
  eight aircraft fails preflight and arming is refused.
- **The wrong drone on the wrong pad.** Two airframes swapped during setup fly
  each other's trajectories. Separation is unaffected — the paths are
  unchanged — but each flies through the other's slot, and **no amount of
  trajectory validation catches it.** It is caught on the ground, twice:
  the `launch positions` check compares telemetry against the pads, and with
  airframes assigned in the registry the `airframe identity` check compares
  the MAVLink system id answering on each slot against the airframe that
  should be standing there, and names the one that is.
- **Birds, aircraft, failures, weather.** Out of scope entirely.
- **Airworthiness and law.** Validation says the plan is sound. Nothing more.

---

## 6. Rehearsal scenarios

Each of these should be flown on the simulator before a show. All are
reproducible from the CLI.

| # | scenario | how | expected |
|---|---|---|---|
| 1 | drone loses its link mid-show | `--link-loss 3:60` | `link_lost` raised, show continues, logged once |
| 2 | drone starts on a weak battery | `--weak-battery 7:55` | **arming refused** before anything flies |
| 3 | GNSS degrades in flight | `--gps-degraded 11:80` | `gnss_degraded` raised, position noise grows |
| 4 | two trajectories conflict | hand-edit a formation | `validate` fails, **upload refused** |
| 5 | drone drifts off plan | `--drift 5:0.1` | `deviation_warn` then `deviation_abort` |
| 6 | wind across the box | `--wind 6,3,0` | whole fleet offset, deviation reported |
| 7 | plan changed but not re-uploaded | re-plan, skip upload | `plan checksums match` fails preflight |
| 8 | drone on the wrong pad | swap two in the sim | `launch positions` fails preflight |
| 9 | operator tries to kill motors | `emergency('kill')` | **rejected**, no such action exists |
| 10 | operator disarms in flight | `emergency('disarm')` | **rejected**, names the airborne drones |
| 11 | a drone's clock is wrong | `sitl --clock-offset 2:0.4` | `clock synchronisation` fails, **arming refused** |
| 12 | a drone fails its prearm checks | `sitl --refuse-arm 0:1` | arm NAKed; the GCS reports it rather than assuming |
| 13 | a plan upload is corrupted | FTP CRC mismatch | **upload refused**, fleet reported as mixed |
| 14 | abort mid-show | `collective_rth()` | one verified deconflicted return, or a hold if none clears |
| 15 | suspend then resume | `suspend()` / `unsuspend()` | clock frozen, fleet motionless, then continues from the same instant |
| 16 | two drones fail preflight | `adapt(show, n-2)` | the show is re-cut for the smaller fleet and passes in its own right |

Scenarios 11 to 13 need the MAVLink layer, so they run under
`python -m skyshine sitl` rather than `rehearse`. Scenario 11 is the one that
cannot be caught any other way: see [SITL.md](SITL.md#the-clock-fault).

```bash
python -m skyshine rehearse "my show prepared" \
  --wind 6,3,0 --link-loss 3:60 --gps-degraded 11:80 --drift 5:0.1 \
  --log logs/rehearsal.json
```

Every scenario is also a test in `tests/test_control.py` or
`tests/test_sitl.py`, so a regression in any of these fails the build rather
than surfacing on show night.

---

## 7. Aborting a running show

Three mechanisms, and the difference between them matters.

### Suspend

Every drone holds position; the show clock freezes. Safe by construction - no
aircraft moves, so the separation the fleet had is the separation it keeps.
It is a command to the aircraft, not a ground-side clock trick, so it works
with onboard execution where a ground pause could not.

One failure mode found and fixed while building it: a hold is a *mode*, not a
momentary action. Until it is released the aircraft ignores the plan, and the
first implementation never released it - so resume, and collective return
after it, were accepted, logged and reported as working while the fleet sat
motionless. An abort path that silently does nothing is worse than one that
errors, because it looks like it worked. `release_hold` now runs on resume and
before any escape, and `TestHoldRelease` fails if it stops.

### Collective return home

Per-drone RTL hands each autopilot a private straight line to its own pad,
computed in ignorance of every other aircraft. At show density that is a plan
for a mid-air, not a failsafe.

Collective return instead builds **one** plan for the fleet: measured
positions, matched optimally onto a layered cloud above the launch grid, then
a vertical descent. Same matching and same shared easing as the main show, so
§2 applies.

One caveat cannot be engineered away and is therefore *verified* instead. The
bound gives `min(d_start, d_end)/sqrt(2)`, and `d_start` here is whatever
separation the fleet happens to have at the moment of the abort, which may be
as low as `min_separation` itself. So the return is swept before it flies, and
if it does not clear:

- the fleet is **held** rather than returned, preserving the separation it has
- the operator is told the measured figure and why it was refused
- nothing is flown on an unverified plan

Drones with no telemetry are assumed to be where the plan put them, and that
assumption is **counted and reported** rather than hidden.

### Per-drone hold, RTL and land

Still available, still not deconflicted. The UI says so at the point of use.

## 8. Overrides

Every gate can be overridden. None can be overridden silently.

`upload(override_reason=...)` and `arm(override_reason=...)` require a written
reason, which goes into the event stream and into
`flight_log()["overrides"]`. `test_override_is_possible_but_recorded` asserts
the log entry exists, on the grounds that an override which is not in the log
is not an override, it is a loophole.

There is no override for disarming an airborne drone, and no action that cuts
motors in flight.

### Who may press the buttons

The server authenticates every request once any account exists. Roles are
ordered viewer < designer < operator < admin: viewers read, designers edit
shows, only operators reach `/api/ops` and `/api/fleet` (upload, preflight,
arm, start, abort), and only admins manage accounts or read the audit trail.
The voice endpoint checks the role of the *parsed intent*, not just the URL,
so a viewer cannot say "upload the plans" around the route check.

With no accounts the server answers only requests from its own machine; a
second laptop on the field network gets nothing until an admin exists. A
corrupt user file fails closed (503) rather than reading as "no users".

Every state-changing request, every login attempt and every refusal is
appended to `logs/audit.jsonl` with the user and role. Passwords never reach
it. Together with `flight_log()["overrides"]` this answers both "what was
overridden" and "who did it".

---

## 9. Before you fly

- [ ] `min_separation` set from the airframe's real dimensions, not a default
- [ ] RTK base surveyed, every drone at fix 6, HDOP under 1.2
- [ ] Show origin, orientation and **AMSL reference** set for the actual site
- [ ] Geofence drawn with margin, from the authorisation, not from the artwork
- [ ] Ceiling matches the authorisation
- [ ] Battery endurance measured on these packs in this weather, with reserve
- [ ] `validate` passes with headroom you are comfortable with, not just > 0
- [ ] `rehearse` run clean, and run again with §6 faults injected
- [ ] `skyshine sitl` run clean over real MAVLink, and again with a clock offset
- [ ] Clock synchronisation verified on the real fleet, not just in SITL
- [ ] Mission checksum on the ground station matches every aircraft
- [ ] Launch positions verified against assigned pads
- [ ] Emergency action chosen for what is actually under the fleet tonight
- [ ] Collective return rehearsed from mid-show, not just assumed to work
- [ ] If flying pyro: altitude, separation, heading and settle all validated,
      and the fallout zone cleared by someone who walked it
- [ ] If the fleet shrank: the show re-adapted and re-validated, not flown
      with holes in the formations
- [ ] If the server is reachable from the network: an admin account exists,
      and only the people flying tonight hold operator
- [ ] Authorisation, licences and insurance in hand

A green report is a statement about a plan. The rest of this list is the
difference between a plan and a show.
