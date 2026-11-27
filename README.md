# Sky Shine

Drone light show design and swarm control. One application: storyboard a show
in 3D, prove it cannot collide, and fly it from the same ground station.

Built for 100–500 aircraft. The collision-freedom argument is the centre of the
design, not a feature bolted on afterwards — see [docs/SAFETY.md](docs/SAFETY.md)
for the argument in full.

```bash
pip install -r requirements.txt
python -m skyshine demo --drones 200 --ceiling 150 --amsl 920
python -m skyshine prepare "sky shine demo"
python -m skyshine validate "sky shine demo prepared"
python -m skyshine rehearse "sky shine demo prepared"
python -m skyshine sitl "sky shine demo prepared"   # real MAVLink, virtual aircraft
python -m skyshine serve          # designer + ground station on :8765
```

## What it does

**Design.** A storyboard of formations. Sixteen generators (sphere, filled
ball, heart, star, disc, cube shell, helix, double helix, torus, wave, grid,
ring stack, line, polygon, plus text and arbitrary images sampled into evenly
spread point clouds). Twelve light effects, a music timeline with beat
locking, per-drone yaw, and pyro events. 3D preview, 2D map, scrubbing
timeline with formation blocks, beat grid and pyro markers.

**Prove.** Formations are scaled so no two drones can come within the minimum
separation at any point in the show — not checked after the fact, but made
impossible by the way transitions are constructed. An independent validator
then re-measures the finished trajectories in continuous time and refuses
anything that does not hold up.

**Fly.** Preflight checklist across the whole fleet, plan upload with
checksums, absolute-time show start, live telemetry, a deviation watchdog,
suspend and resume, and graded emergency actions including a **verified
collective return home**. Every override is possible and every override is
written into the flight log.

**Rehearse.** Three rungs. A pure simulator that flies the show with wind, GNSS
noise, clock jitter, link loss, weak batteries and position drift injected. A
fleet of virtual MAVLink aircraft on real sockets, which exercises the actual
wire protocol. And real ArduPilot SITL when it is installed. See
[docs/SITL.md](docs/SITL.md).

## How collisions are prevented

The whole design rests on one result. Match two formations with the assignment
that minimises total **squared** distance flown, and move every drone in a
straight line under **one shared** time-easing. Then for any pair of drones the
separation during the move never drops below

```
min(separation in A, separation in B) / sqrt(2)
```

So Sky Shine scales every formation to `min_separation × sqrt(2)` — 4.24 m for
a 3 m minimum — and transitions are then safe by construction, with no search,
no re-planning and no iteration. The planner adds a few centimetres on top
(2.8 cm at the default 10 Hz), because the validator checks against the
minimum plus a path-curvature margin, and a formation packed to exactly the
bound would otherwise fail it by millimetres. Both conditions are enforced in code rather
than assumed: the matching is a true optimal assignment, and a single easing is
applied across the whole transition.

Takeoff is the one place a flat grid cannot satisfy that, so the climb is
purely vertical into **altitude-separated layers** — a 2×2 checkerboard over the
launch grid. A 3 m grid becomes a cloud with 4.24 m of real separation, which
feeds the first formation safely.

Measured on a 200-drone show: closest approach **3.017 m** against a 3.000 m
limit — exactly the bound, independently verified.

## Architecture

```
skyshine/
  show.py         data model, show file format, easings, ENU <-> WGS84
  geometry.py     formation generators, separation fitting, capacity sizing
  planning.py     optimal matching, takeoff/landing, transition timing
  trajectory.py   timeline sampling, mission package, lossless compression
  safety.py       the validator - the only gate that authorises a flight
  voice.py        spoken command grammar, risk model, show composer
  assistant.py    fix it for me, music to show, Claude helpers
  assets.py       uploaded artwork: validation, thresholding, mask preview
  registry.py     airframes, slot assignment, flight hours, battery cycles
  reports.py      post-show report, built only from the flight log
  effects.py      light effects, baked to keyframes so playback is onboard
  interop.py      vviz and CSV, out and in
  drivers.py      simulator, MAVLink onboard show mode, MAVLink guided stream
  sitl.py         virtual MAVLink aircraft, ArduPilot SITL launcher
  fleet.py        ground station state machine, preflight gates, watchdog
  auth.py         accounts, roles, sessions, login lockout, audit trail
  server.py       REST + WebSocket API
  cli.py          command line
web/index.html    designer and ground station UI (three.js, no build step)
tests/            758 tests
```

`planning.py` builds shows that should be safe. `safety.py` assumes none of
that and measures the result. They are kept independent on purpose: a validator
that shares assumptions with the planner validates nothing.

## Design features

| | |
|---|---|
| **Logos &amp; text** | Upload a PNG/JPEG/GIF/BMP/WEBP and it becomes a formation. Otsu picks the starting threshold; the preview shows the mask **and the drone positions that would be sampled from it**, at the real drone count, so you tune before committing. Text renders from any installed font. |
| **Shapes** | 16 generators. Filled variants (`solid=True`, `disc`, `ball`) grow as the square root of the drone count instead of linearly - a 500 drone *outline* needs two kilometres of perimeter and fits under no legal ceiling. |
| **Light effects** | solid, fade, strobe, pulse, gradient, rainbow, wipe, chase, ripple, twinkle, meteor, alternate. Spatial effects project each drone onto an axis of the formation, so a gradient across a heart follows the heart. |
| **Music** | BPM, downbeat phase, named cues, beat snapping. Effect periods can be written in beats, so retiming to a different tempo does not silently desynchronise them. |
| **Yaw** | Optional per-drone heading. Left unset for a light-only show: a drone with an omnidirectional LED has no reason to burn energy pointing anywhere. |
| **Pyro** | Timed events with a required heading and settle time, checked for altitude, extra separation, correct heading and settle. |
| **Show adapt** | Re-cut a show for a smaller fleet when aircraft fail preflight. Farthest-point resampling keeps the silhouette, then the whole plan is rescaled, rematched and retimed, so the result passes validation in its own right. |
| **Auto geofence** | Convex hull of what the show actually does, plus margin. A starting point, not an authorisation. |
| **Templates** | Start from a built-in storyboard instead of a blank show: Showcase, Wedding, New Year, Brand reveal, Festival. Shapes, colours and light effects are sized to the drone count, and three of them spell your own text. See [Templates](#templates). |
| **Import design** | Bring in a design made in another program: 3D models (OBJ, STL, PLY, GLB) or SVG drawings. Drones are spread evenly over the model's surface or points, or the drawing's lines or fill, sized to fit. From Blender, Maya or 3ds Max export OBJ or GLB; FBX cannot be read. The result is an ordinary formation. |
| **Draw by hand** | Draw on a pad with a mouse, pen or finger (editor and Simple mode); drones spread evenly along the lines. Lines are thin, so many drones need a big drawing: 200 drones at legal spacing need about 850 m of line. |
| **Pick, nudge, swap, lock** | Click drones in the 3D view (Shift+click for more) while a formation is held, then move them, or swap two so they trade places. Swapping locks that formation's pairing with the one before it, so preparing keeps it; the lock can also be set by hand. Swapped paths can cross, so the safety engine has to pass the show again. |
| **Colour tools** | Paint picked drones one colour; render any light effect onto the picked drones only ("only the picked drones"); colour a formation from an uploaded picture, as the crowd sees it or from above; and the **volume** effect, a ball of colour sweeping through the formation along x, y or z that can leave its colour behind like a brush, over the formation's current colours or a background. A small ball on a straight line paints a stripe; radius 1 paints the whole shape. |
| **Generators** | **QR code** (any text or link, up to about 150 characters; checked with a real QR reader), lit on its dark squares (fewest drones, read inverted by current phone cameras) or its light squares (standard, about twice the drones). At legal spacing an upright code is 90 m or taller, so by default it lies flat and is read from below. **Countdown** from up to 99 on seven-segment digits sized to the height available. **Scrolling text** across an LED-sign grid. Countdown and text change lights only, so nothing moves; spare drones wait unlit behind. |
| **Keep-out zones** | No-fly cylinders around buildings, trees, cranes or a stage, drawn in red in the 3D view. The safety engine refuses a show that brings any drone within half the minimum separation of one, at any time, take-off and landing included. Paths are not rerouted around zones: move the formations instead. |
| **Flight limits** | Minimum separation, ceiling, lowest flight height, sideways, climb and descent speed, acceleration, the moving-formation speed share and battery time, edited in the app. Changing them means preparing again. |
| **Fewer drones in a formation** | A formation can use only some of the drones ("Drones to use"); the rest wait, unlit, on a grid just behind it. |
| **Song** | Upload the show's song (MP3, WAV, OGG, FLAC or M4A) in the Music panel. Its waveform appears on the timeline, it plays in step with the 3D preview and with live rehearsals, and **Detect tempo** sets the BPM and downbeat from the song itself (within 0.5 BPM on test tracks from 72 to 140 BPM). Preparing adds take-off at the start, so it moves the song, beat grid and cues later to keep them on the first formation; if preparing has to lengthen a later move, the music after it drifts and the moment should be re-checked. Replacing or deleting a song leaves the old file in `assets/audio`. |
| **Flight charts** | Height (highest and lowest drone), sideways speed, climb and descent, acceleration, and the closest two drones in the air, over the whole show, with each limit dashed and the worst moment marked. Taken from the samples the safety check uses, and each point is the worst value in its slice of time, so a short spike is never averaged away. Click a chart to jump the 3D view there. |
| **Venue** | Lay an aerial photo or site plan on the ground (width in metres, turn, position) and add a 3D model of the site (OBJ, STL, PLY or GLB, metres per unit, up axis, turn, position; very detailed models are thinned for display). Both are only shown, in the 3D view, not the 2D map. **Keep drones away from it** adds a keep-out zone: one cylinder over the whole model plus 3 m, rounded outwards. Generous for an L-shaped site; draw tighter zones by hand if the show must fly over part of it. Everything stays offline, from your own files. |
| **Video** | **Record video** plays the show from the start and saves the 3D view as a video, in the browser, in real time. MP4 where the browser can actually write it, otherwise WebM; a format that fails is skipped. |
| **Skybrush export (beta)** | `.skyc`, written from Skybrush's published format: flight paths within 2 cm of the validated show, light programs, limits, cues and the site location and orientation. The outer packaging of a .skyc is not documented, so this one is a ZIP holding show.json, which is unverified: load it into Skybrush before relying on it. Skybrush's code is GPL-3.0 and none of it is used. UgCS export is not built: no public format description was found. |
| **Move, turn and scale** | Shift any formation east, north or up, turn it, tilt it toward the crowd, or scale it, about its centre. Undoable like every other edit. |
| **Moving formations** | A formation can spin, travel and grow while it is held (Motion panel in the editor; a Spin button in Simple mode). It starts and ends at rest. Prepare lengthens the hold until the move uses no more than `motion_speed_share` (default 50 %) of the speed and acceleration limits: a spin holds that speed for its whole hold, and in simulation a spin at the full limit trailed its plan by 3.5 m. The safety engine, the uploaded mission and the 3D preview all use the same motion; shrinking starts wide enough to end at legal spacing; pyro is not allowed on a moving formation. |
| **Undo, redo, duplicate** | Every design edit is undoable, 30 steps deep per show (Ctrl+Z / Ctrl+Y). History lives in memory, so restarting the server clears it. Duplicate saves a copy under a new name and never overwrites. |

Effects are **baked into keyframes**, never evaluated live. A light program the
ground station computed each frame would make the show depend on the radio; a
baked one plays from the drone's own clock, same as the trajectory. Fast
effects raise their own sample rate - a 10 Hz strobe rendered at 10 Hz aliases
into a solid colour.

### Simple mode

Open http://localhost:8765/simple, or press **Simple mode** at the top of the
full editor. It is three big steps, made for someone who has never designed a
show:

1. **Pick a show**: choose a template, set how many drones with + and -, and
   type the words to write in the sky where the template has some.
2. **Make it yours**: tap a colour on any picture to change it, take pictures
   out, or press Undo.
3. **Check and play**: prepares the show, runs the full safety engine, says in
   plain words whether it is safe to fly, and plays it in 3D.

Simple mode makes ordinary shows. Open the same show in the full editor for
every other tool, and to fly it. Changing a colour replaces that picture's
light effect with a solid colour.

### Templates

Library → **New from template…**, or from the command line:

```
python -m skyshine demo --template wedding --text "R & S" --drones 120 --name "r and s"
```

A template gives an ordinary unprepared show. It still has to be prepared and
pass the safety engine, and every formation can be edited or removed afterwards.

- Formations are drawn at legal spacing and placed to fit under the ceiling:
  at the usual show height when there is room, lower when a big fleet needs
  the space.
- With the default limits (3 m separation, 120 m ceiling) all five validate up
  to about 200 drones. Above that some shapes are taller than the ceiling at
  legal spacing; the show is still created and the response lists which ones.
- Text is drawn in strokes, so it is wide: 200 drones spelling nine letters
  span roughly 400 m. Keep it short.

## The fleet as airframes

Everywhere else a drone is an integer - a row in a formation. That is right
for geometry and wrong for operations: on the night you are holding 500
airframes, each with a serial, a firmware version, a battery that has been
charged a known number of times, and a history of what broke on it.

- **Register** airframes singly or in a batch (sequential ids and MAVLink
  system ids in one call; serials stay blank because those have to be read
  off each aircraft).
- **Assign** them to a show's slots - automatically picks the ready
  airframes with the fewest flight hours, so the fleet ages together.
- **Ground** one for maintenance and it is removed from any slot it holds.
  An aircraft nobody meant to fly must not stay on a pad.
- **Preflight checks identity**: the system id answering on each slot must
  be the airframe assigned to it. This is what catches two drones swapped
  during setup - they fly each other's trajectories, separation is
  unchanged, and no trajectory check can see it. Advisories such as a pack
  past 200 cycles or a missing firmware version are reported but do not
  block.
- **Finish & credit** records the time actually flown - an abort after 40
  seconds adds 40 seconds, not the designed duration - plus a battery
  cycle per airframe.

The registry is opt-in per show: a show with no airframes assigned flies
exactly as before, without an identity check it never asked for.

### Import and export

**Export CSV** downloads every airframe as a spreadsheet; **Import CSV** reads
an edited one back, which is the practical way to enter 500 serials and
firmware versions. Rows match on `id`: new ids are added, existing ones are
updated from the columns present in the file. Flight counters are never wound
back by an import, and one bad row refuses the whole file, naming the row.
Maintenance history is not in the sheet.

### Post-show report

Separate from the validation report on purpose. That one says what the plan
promised; this one says what the night delivered - how far the show got,
which drones raised which conditions and when, every override in full, and
the state each aircraft finished in, by airframe. It is built only from the
flight log, so it cannot quietly describe the design instead of the flight.
Operator-typed text is escaped: an override reason is rendered into HTML.

## Artwork

A logo is mostly a thresholding problem, and the threshold is the one thing
nobody can guess from the filename: the same mark light-on-dark and
dark-on-light need opposite settings, and getting it wrong yields a cloud
of drones in the shape of nothing.

So the preview draws the mask *and the drones sampled from it* at the real
count. A logo that thresholds beautifully can still read as noise once it
is only 120 points, and that is worth finding out before the show is built
rather than after.

- **Otsu** picks the opening threshold, and flags when the subject looks
  dark-on-light so the first preview is not the background.
- The **drone ceiling** is reported up front: a mask cannot hold more
  drones than it has filled pixels, so the UI says so before the attempt.
- Every sampled drone is **snapped onto an actual filled pixel**. k-means
  centroids are cluster means, so on a non-convex mark - a disc beside a
  bar - clusters spanning both put their centre in the gap. 42% of drones
  landed off the artwork before this was fixed.
- **SVG is refused** with instructions, not a decode error: Pillow cannot
  rasterise it, so export a PNG at around 1000 px on the long edge.

Uploads are untrusted input. They are decoded by Pillow (which rejects
anything that is not an image), stored under a content hash rather than
the supplied filename, capped in bytes and in pixels against
decompression bombs, and never executed or passed to a shell.

## Voice and the AI assistant

Say it instead of clicking it. Speech recognition and speech are the
browser's own Web Speech API, so there is no key and nothing leaves the
machine but the recognised text. There is also a text box for the same
grammar, because a blocked microphone should not lock you out.

```
"create a show called summer gala with two hundred drones"
"add a red heart at 60 metres"
"apply a chase effect in cyan"      "strobe at 4 hertz"
"set the tempo to one hundred and twenty eight"
"prepare"      "validate"      "auto geofence"      "export vviz"
"suspend"      "collective return home"      "status"
```

**Understanding is a grammar, not a language model.** Speech recognition
already mishears; feeding its output to something that also guesses
compounds two sources of error on a surface that commands aircraft -
"start" and "stop" differ by one phoneme. The grammar can be enumerated,
unit tested, and behaves identically on the flight line and in the office.

**Voice can never arm, start or disarm.** Refused outright, at any
confidence, by both the parser and the server independently. A microphone
is not an authenticated control surface: it hears bystanders, two-way
radios, and the show's own audio track. Every intent carries a risk level:

| risk | what happens | examples |
|---|---|---|
| `auto` | runs immediately | design edits, navigation, validate, preflight, **suspend** |
| `confirm` | parsed, then waits for a click | resume, land, collective return, upload, adapt, load fleet |
| `refused` | never, by any route | **arm, start, disarm** |

Suspend is deliberately `auto`: it stops motion, so the fleet keeps
whatever separation it already had, and adding a click would add latency
to the one command that is always safe to obey.

### Hands-free voice mode

**Start listening** keeps the microphone on until you stop it. Spoken
switches work at any time:

| Say | What happens |
|---|---|
| "go to sleep", "stop listening", "stand by" | standby: only the wake phrase is acted on |
| "Sky Shine, wake up", "activate voice" | back to listening |
| "voice off" | microphone off |

The switches are recognised in the browser, so in standby nothing except
the wake phrase reaches Sky Shine. Chrome and Edge do send microphone audio
to their own speech service whenever the microphone is on, standby included.

### The assistant

Describe a show and get a storyboard back:

> *"something romantic for a wedding, with a heart and a star"*

It picks a mood, palette, pacing and effects, sizes each shape to the
drone count so the stack fits under the ceiling, and presents a **draft for
review**. Building it runs the same generators, scaling, matching and
validation as a hand-built show - there is no special path, and the draft
is not flyable until it has been prepared and has passed the safety engine.

With `ANTHROPIC_API_KEY` set, the draft comes from Claude; without one, a
rule-based composer runs. The fallback is the default, not a degraded
mode, and it is what the tests exercise. Either way the output passes
through `_sanitise_spec`, which rejects or corrects anything the
generators cannot build - an unknown shape, an out-of-range altitude, a
malformed colour - and reports every correction rather than hiding it.

### AI helpers

Four more buttons sit in the AI assistant panel. Each one is an ordinary
design edit: one undo puts it back, and nothing flies until prepare and
the safety check pass.

| Helper | Needs a key? | What it does |
|---|---|---|
| **Fix it for me** | no | Lowers or raises formations to fit the ceiling, slides them inside the geofence or out of keep-out zones, shortens holds to fit the battery, and unlocks a locked drone order that causes a collision. It never squashes a shape, since that is what keeps drones apart. It says what it changed, and what is left for a person to do. |
| **Match music** | no | Retimes the formations so each one lands every 2, 4, 8 or 16 beats of the song. Transitions are never made shorter than the flight limits allow, and prepare keeps the alignment. Extra time goes into the holds. It needs the song's tempo first. |
| **Add a shape from words** | optional | "a red heart" uses the built-in heart. Anything else ("a dolphin") is drawn by Claude as an outline, then spread over with drones like an imported SVG. Without a key, the words are written in the sky. |
| **Free-form commands** | yes | If the typed or spoken command grammar does not understand a request, Claude rewrites it as up to four commands the grammar does understand. Each one is then parsed and checked as if it had been typed: arming and starting stay refused, and risky commands still need a click. |

Claude calls use `claude-opus-5-5` with structured JSON output, so the app
only ever reads answers of the shape it asked for. If Claude declines, is
offline or the key is wrong, the helper says so and changes nothing.

## Interop

```bash
python -m skyshine serve     # then use the .vviz / CSV buttons
```

- **vviz out** - the format Finale 3D imports, for hybrid shows where pyro is
  scripted against the drones. Decoded from a real sample file: `ogl` frame
  (y up, +z toward the audience), per-sample position deltas, run-length
  encoded colour, one `Pyro` payload per event on the firing drones only.
- **CSV out** - per-drone sampled trajectories, the lowest common denominator.
  Lossy by construction: for handing a show to another tool, not for
  round-tripping your own.
- **CSV in** - somebody else's show, pointed at this validator. An imported
  trajectory has no formations, no matching and no shared easing, so none of
  the design-time guarantees apply to it - but `safety.py` does not care where
  a trajectory came from, so it still gets the full continuous-time sweep.

One honest limit the import path surfaces: the interchange sample rate caps
what can be *proven*. A prepared show clears the minimum by about 17 mm, and a
5 Hz export leaves roughly 19 mm of chord-versus-curve uncertainty between
samples, so the sweep reports the pair rather than assuming it is fine. Export
denser if you need the clearance demonstrated.

## The safety gate

A show reaches the aircraft only through this sequence, and each step refuses
until the one before it passed:

1. **validate** — continuous-time separation sweep, geofence, altitude ceiling,
   velocity and acceleration envelope, battery endurance, design sanity
2. **upload** — refused outright if validation failed; an override needs a
   written reason that lands in the flight log
3. **preflight** — nine fleet-wide checks: plan checksums, link, batteries,
   GNSS fix and HDOP, clock sync, correct launch pad, faults
4. **arm** — refused on any failing blocking check, same override rule
5. **start** — scheduled against an absolute UTC instant, never "now"

### Mission approval

Before a plan goes to **real aircraft** an operator has to approve it: one
named sign-off, for one exact plan. The record carries a fingerprint of the
show, so any edit or re-prepare voids it without anyone remembering to revoke
it. A show that fails validation cannot be approved, and unlike the safety
override there is no written-reason way round a missing approval. The sign-off
is copied into the flight log at upload.

Rehearsal on the simulator and SITL needs no approval. The control is the
Approve button above the launch sequence, or `POST /api/ops/approvals/{show}`.
Approver and designer may be the same person: nothing enforces two people.

### Signed mission packages

A built mission file carries a signature as well as a checksum. The checksum
only proves the file is internally consistent, and anyone who edits a plan can
recompute it. The signature needs this ground station's key, so an edited
file fails verification even with its checksum repaired, and the signer's
name cannot be rewritten either.

```bash
python -m skyshine export "my show prepared"          # builds and signs
python -m skyshine verify-mission "shows/my show prepared.mission.json"
```

In the UI, **Mission** builds and signs and **Verify** checks the built file.
The key is `mission.key` in the project folder, made on first use: back it up
and keep it private. It is an HMAC key, so whoever can verify can also sign;
the signature covers the saved file, not the radio link to the aircraft.

### Signed MAVLink link

MAVLink 2 signing covers the radio link. With it on, every message the ground
station sends is signed, and every message it receives must carry a valid
signature or it is dropped. Anyone on the frequency can still listen, but they
cannot arm, steer or land the fleet.

1. Connect the disarmed fleet over USB or a closed bench network, not the show
   radio.
2. Load it unsigned, then press **Load signing key onto aircraft**. The key
   goes to every aircraft and the link switches to signed. The response lists
   any aircraft that did not take it; those are now dropped as unsigned.
3. From then on, tick **Sign the MAVLink link** when loading, or pass
   `--signed` to `python -m skyshine sitl`.

The key is `link.key` in the project folder, made on first use. It is not
`mission.key`: this one is copied onto every aircraft, so it leaves the ground
station. Keep it private and back it up. Deleting it means loading a new key
onto every aircraft.

**Warnings**

- The key travels unencrypted while it is being loaded. That is why step 1
  says bench or USB.
- ArduPilot refuses a new key while armed, and so does Sky Shine.
- ArduPilot always accepts unsigned messages on its USB port, and RADIO_STATUS
  from a telemetry radio. Sky Shine is stricter and accepts nothing unsigned.
- Tested against the virtual aircraft, which follow ArduPilot's rules as read
  from its source (`GCS_Signing.cpp`). Not yet tested against real firmware.
- To turn signing off on an aircraft, send it an all-zero key from your usual
  ground station. Sky Shine has no button for that.

## Aborting

Three levels, in order of preference:

1. **Suspend** - every drone holds where it is, show clock frozen. Safe by
   construction: nobody moves, so whatever separation the fleet had is exactly
   the separation it keeps. Works with onboard execution, because it is a
   command to the aircraft rather than a ground-side clock trick.
2. **Collective return home** - one deconflicted plan for the whole fleet,
   computed from measured positions, matched and eased like any other show,
   and **verified with the same continuous-time sweep before it flies**. If no
   safe return exists it holds the fleet instead and says why.
3. **Per-drone hold / RTL / land** - the blunt instruments. Note that RTL is
   **not** deconflicted: each autopilot flies its own straight line home with
   no knowledge of the other 499. Prefer collective return.

### There is no kill switch

Stopping an airborne multirotor's motors turns it into a falling object aimed
at whatever is underneath. The emergency actions are **hold**, **return to
home** and **land where you are**; `disarm` is refused for any drone telemetry
shows off the ground. Which action is right depends on what is below the fleet,
which is the operator's judgement, not the software's — so the watchdog raises
graded alerts and performs only the action pre-authorised in the policy
(default: alert and wait).

## Accounts and roles

With no accounts the server answers only the machine it runs on. Before
serving it to a field network, create an admin:

```bash
python -m skyshine user add chief --role admin
python -m skyshine serve --host 0.0.0.0
```

| role | can |
|---|---|
| viewer | watch the 3D view, telemetry and reports |
| designer | edit, prepare, validate and export shows |
| operator | everything above, plus upload, preflight, arm, start and abort |
| admin | everything above, plus accounts and the audit trail |

Passwords are PBKDF2-SHA256 at 600,000 rounds. Sessions live in server
memory, expire after 12 hours, travel in an HttpOnly SameSite=Strict cookie
and are revoked the moment a password or role changes. Five wrong passwords
lock the name for a minute. Every change and every login attempt lands in
`logs/audit.jsonl`. `python -m skyshine user --help` lists, re-roles,
resets and removes accounts; the last admin cannot be removed or demoted.

Voice follows the same rules: the role is checked against what the phrase
*does*, so a viewer saying "upload the plans" is refused.

## Scale and performance

A 500-drone, 347-second show on one desktop core:

| stage | time |
|---|---|
| prepare: scale, wrap, match 10 formations, time | 1.4 s |
| sample the timeline (3 468 frames) | 0.2 s |
| validate (432 million pair-seconds, continuous) | 2.0 s |

Mission packages compress a drone's whole show to its formation boundaries
plus the shared easing, which is lossless: replaying the uploaded plan
reproduces the designed path to under a millimetre, and `export` verifies that
every time.

## Sizing a show before designing it

Drone count, separation and sky are not independent. 500 drones at 4.24 m
spacing need roughly a 112 × 70 m billboard; a 500-drone *outline* needs over
two kilometres of perimeter and will not fit under any legal ceiling. The UI
and `demo` both report the billboard a given count requires, and generators
offer filled variants (`solid=True`, `disc`, `ball`) whose size grows as the
square root of the drone count instead of linearly.

Adding a formation that will end up over the ceiling once scaled warns at
once, with the reason and the fix, rather than after `prepare`. The show is
still saved: the warning is advice, and the validator remains the gate.

## Hardware

**Verified here:** the design engine, the validator, the mission format, the
full operational state machine, and the complete MAVLink wire protocol —
command encoding, frame conventions, FTP plan transfer, ack handling and clock
discipline — against aircraft on real sockets, under 310 tests.

**Not flown:** no real airframe has ever executed a Sky Shine show. SITL proves
the plan and the protocol. It is not a flight test.

- `simulator` — no hardware, no MAVLink. Fastest rehearsal, and where the
  operational gates get exercised.
- `sitl` / `sitl-guided` — virtual aircraft over real MAVLink on real sockets.
  Run this before any hardware session; it is the rung that catches protocol
  bugs. [docs/SITL.md](docs/SITL.md)
- `mavlink-native` — upload the trajectory over MAVLink FTP, CRC-verified, then
  the aircraft plays it from its own GNSS clock while the ground station only
  supervises. The only approach that scales to hundreds of drones, because
  nothing time-critical crosses the radio during the show. Needs show-mode
  firmware (custom mode 127); `upload_hook` overrides the transport.
- `mavlink-guided` — stock ArduPilot in GUIDED with setpoints streamed at
  10 Hz. Honest about its ceiling and hard-capped at 50 drones: a show with an
  audience must not depend on the radio staying up packet by packet.

Protocol details that are load-bearing rather than incidental:

- The show start instant is `MAV_CMD_USER_3` as a **`COMMAND_INT`**, seconds as
  an int32. A `float32` cannot hold unix epoch seconds — its spacing at 1.79e9
  is 128 s.
- The show origin is `MAV_CMD_USER_2` as a **`COMMAND_INT`**, latitude and
  longitude as integer 1e-7 degrees, so they keep the precision RTK provides.
  Firmware refuses the `COMMAND_LONG` form, and so does our SITL.
- Links are `udpin:` — the ground station listens on one port per aircraft. One
  noisy drone then cannot delay telemetry for the fleet.

Recommended fleet hardware: Pixhawk-class controller, **RTK GNSS** with a base
station, 5 GHz Wi-Fi rather than 433 MHz telemetry, addressable RGB payload.
Plain GPS has metre-scale error, which is a third of a 3 m separation budget
before anything else goes wrong.

## Regulatory

Sky Shine does not make a flight legal. A multi-drone show needs an
authorisation wherever it flies — in India via Digital Sky / DGCA, in the US via
a Part 107 waiver. Validation passing means the *plan* is sound; airworthiness,
licensing, airspace and insurance are separate and still yours.

## Testing

```bash
python -m pytest tests/ -q               # 930 tests, about four minutes
python -m pytest tests/test_sitl.py -q   # the MAVLink wire path alone
```

The tests that matter are `TestCollisionGuarantee` and `TestSeparationSweep`.
The first checks the inequality the proof rests on and then verifies the bound
numerically on random geometry, plus a case proving that per-drone timing
destroys it. The second plants a collision that happens *between* two sample
frames — 10 m apart at both samples, zero in the middle — which a frame-by-frame
validator would pass with room to spare.

`tests/test_sitl.py` covers what neither can: real MAVLink frames over real
sockets. Six genuine defects came out of writing it, each listed in
[docs/SITL.md](docs/SITL.md), and every one of them would have passed the
simulator-only suite.

`tests/test_registry.py` covers the fleet; `TestSwappedAirframes` checks
that a swap is caught from both sides with the culprit named, and that it
blocks arming.

`tests/test_assets.py` covers uploads, and its `TestUploadRejects` class
is the one to read: empty files, renamed executables, truncated PNGs,
decompression bombs, SVG, and path traversal through both the filename and
the asset id.

`tests/test_voice.py` covers the command grammar, and its `TestRefusals`
class is the one to read: it asserts that every phrasing of arm, start,
launch, take off and disarm is refused, that a refusal beats every other
rule in the same sentence, and that a client claiming the operator already
confirmed still gets nothing.

`tests/test_features.py` covers the later additions. Two worth reading:
`TestHoldRelease`, for a bug where a suspended fleet latched its driver into
hold and every later command - resume, collective return - was accepted,
logged and reported as working while the fleet sat motionless; and
`TestCollectiveRth`, which checks that an unsafe return is refused and the
fleet held, rather than flown.
