"""Voice and natural-language control.

Two deliberate design decisions, both about failure modes rather than
features.

**The command grammar is deterministic, not a language model.** Speech
recognition already mishears; feeding its output to something that also
guesses compounds two sources of error on a surface that commands
aircraft. "start" and "stop" differ by one phoneme. So intents are matched
by an explicit grammar: it can be enumerated, unit tested, and it behaves
the same on the flight line as it did in the office. The language model is
used for the one thing it is actually better at - turning a free-form
description into a draft storyboard - and even that output is validated
and executed through the same deterministic path.

**Voice can never arm, start or disarm.** Those three verbs are refused
outright, however clearly they are heard. A microphone is not an
authenticated control surface: it picks up bystanders, radios and the show
audio itself. Anything that commits aircraft to motion is either refused
or requires a deliberate click, and the risk level is part of the intent
rather than a check bolted on at the call site.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

# Risk levels. The parser assigns one to every intent; the server refuses
# to auto-execute anything above AUTO.
AUTO = "auto"          # runs immediately: design edits, navigation, queries
CONFIRM = "confirm"    # parsed and offered, needs a click to execute
REFUSED = "refused"    # never available by voice, at any confidence

# Verbs that commit aircraft to flight. Spoken by anyone in earshot, these
# would launch a show; they are not reachable from this surface.
REFUSED_INTENTS = {"arm", "start", "disarm"}


@dataclass
class Intent:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    risk: str = AUTO
    speech: str = ""
    transcript: str = ""
    confidence: float = 1.0
    # What the operator should be shown when risk is CONFIRM.
    confirm_prompt: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "args": self.args,
            "risk": self.risk,
            "speech": self.speech,
            "transcript": self.transcript,
            "confidence": round(self.confidence, 3),
            "confirm_prompt": self.confirm_prompt,
        }


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

# Spoken shape words mapped onto generator calls. Several map to a filled
# variant on purpose: say "heart" with 300 drones and an outline would need
# half a kilometre of perimeter, so the spoken word means the thing that
# can actually fly.
SHAPE_WORDS: dict[str, tuple[str, dict]] = {
    "sphere": ("ball", {}),
    "ball": ("ball", {}),
    "globe": ("ball", {}),
    "orb": ("ball", {}),
    "shell": ("sphere", {}),
    "heart": ("heart", {"solid": True}),
    "love heart": ("heart", {"solid": True}),
    "star": ("star", {"solid": True}),
    "circle": ("circle", {}),
    "ring": ("circle", {}),
    "disc": ("disc", {}),
    "disk": ("disc", {}),
    "cube": ("cube", {}),
    "box": ("cube", {}),
    "square": ("grid", {}),
    "grid": ("grid", {}),
    "helix": ("helix", {}),
    "spiral": ("helix", {}),
    "double helix": ("double_helix", {}),
    "dna": ("double_helix", {}),
    "torus": ("torus", {}),
    "donut": ("torus", {}),
    "doughnut": ("torus", {}),
    "wave": ("wave", {}),
    "line": ("line", {}),
    "rings": ("ring_stack", {}),
    "ring stack": ("ring_stack", {}),
}

COLOUR_WORDS: dict[str, tuple[int, int, int]] = {
    "red": (255, 30, 30), "green": (40, 230, 90), "blue": (40, 120, 255),
    "white": (255, 255, 255), "yellow": (255, 220, 50), "orange": (255, 140, 20),
    "purple": (170, 80, 255), "violet": (170, 80, 255), "pink": (255, 90, 190),
    "magenta": (255, 0, 200), "cyan": (0, 230, 255), "turquoise": (0, 230, 200),
    "gold": (255, 190, 60), "amber": (255, 170, 40), "teal": (0, 180, 180),
    "lime": (160, 255, 60), "crimson": (220, 20, 60), "indigo": (80, 60, 220),
    "warm white": (255, 230, 200), "ice blue": (170, 220, 255),
}

EFFECT_WORDS: dict[str, str] = {
    "solid": "solid", "plain": "solid",
    "fade": "fade", "crossfade": "fade",
    "strobe": "strobe", "flash": "strobe", "flashing": "strobe",
    "pulse": "pulse", "breathe": "pulse", "breathing": "pulse",
    "gradient": "gradient", "ramp": "gradient",
    "rainbow": "rainbow",
    "wipe": "wipe", "sweep": "wipe",
    "chase": "chase", "chaser": "chase",
    "ripple": "ripple", "wave effect": "ripple",
    "twinkle": "twinkle", "sparkle": "twinkle", "glitter": "twinkle",
    "meteor": "meteor", "shooting star": "meteor",
    "alternate": "alternate", "alternating": "alternate",
}

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}

# Speech recognisers reliably mangle a few of these; map the common
# mishearings rather than pretending the audio is clean.
MISHEARD = {
    "sky shine": "skyshine", "sky-shine": "skyshine", "skyshine": "skyshine",
    "pre flight": "preflight", "pre-flight": "preflight",
    "geo fence": "geofence", "geo-fence": "geofence",
    "v viz": "vviz", "v-viz": "vviz", "vis": "vviz",
    "c s v": "csv", "see s v": "csv",
    "b p m": "bpm", "beats per minute": "bpm",
    "r t h": "rth", "r t l": "rth", "return to launch": "return home",
    "formation's": "formations", "dr ones": "drones", "drone's": "drones",
}


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, fix common mishearings, fold numbers."""
    t = (text or "").lower().strip()
    t = re.sub(r"[^\w\s.\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    for wrong, right in MISHEARD.items():
        t = re.sub(rf"\b{re.escape(wrong)}\b", right, t)
    return words_to_number(t)


def words_to_number(text: str) -> str:
    """Replace spoken numbers with digits: 'two hundred and fifty' -> '250'.

    Needed because the count in "a heart with two hundred drones" has to
    reach the generator as an integer, and recognisers hand back words as
    often as digits depending on phrasing.
    """
    tokens = text.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        value, used = _consume_number(tokens, i)
        if used:
            out.append(str(value))
            i += used
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def _consume_number(tokens: list[str], start: int) -> tuple[int, int]:
    total = 0
    current = 0
    used = 0
    i = start
    saw = False
    while i < len(tokens):
        word = tokens[i]
        if word in NUMBER_WORDS:
            current += NUMBER_WORDS[word]
            saw = True
        elif word == "hundred":
            # "a hundred and fifty" has no number word before "hundred",
            # so a bare "hundred" has to count as one hundred.
            current = max(current, 1) * 100
            saw = True
        elif word == "thousand":
            total += max(current, 1) * 1000
            current = 0
            saw = True
        elif word == "and" and saw and i + 1 < len(tokens) and tokens[i + 1] in NUMBER_WORDS:
            pass  # "a hundred and fifty"
        elif word.isdigit() and not saw:
            return int(word), 1
        else:
            break
        i += 1
        used = i - start
    if not saw:
        return 0, 0
    return total + current, used


def _fuzzy(word: str, options, cutoff: float = 0.78) -> str | None:
    hit = difflib.get_close_matches(word, list(options), n=1, cutoff=cutoff)
    return hit[0] if hit else None


def find_shape(text: str) -> tuple[str, dict, float] | None:
    """Longest spoken shape phrase in the text, with a fuzzy fallback."""
    for phrase in sorted(SHAPE_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            name, extra = SHAPE_WORDS[phrase]
            return name, dict(extra), 1.0
    for token in text.split():
        hit = _fuzzy(token, SHAPE_WORDS)
        if hit:
            name, extra = SHAPE_WORDS[hit]
            return name, dict(extra), 0.75
    return None


def find_colour(text: str) -> tuple[int, int, int] | None:
    for phrase in sorted(COLOUR_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            return COLOUR_WORDS[phrase]
    return None


def find_effect(text: str) -> str | None:
    for phrase in sorted(EFFECT_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            return EFFECT_WORDS[phrase]
    return None


def find_number(text: str, *, after: str | None = None) -> int | None:
    """First integer in the text, optionally only after a keyword."""
    hay = text
    if after:
        m = re.search(rf"\b{re.escape(after)}\b(.*)$", text)
        if not m:
            return None
        hay = m.group(1)
    m = re.search(r"\b(\d+(?:\.\d+)?)\b", hay)
    return int(float(m.group(1))) if m else None


def find_float(text: str, *, after: str | None = None) -> float | None:
    hay = text
    if after:
        m = re.search(rf"\b{re.escape(after)}\b(.*)$", text)
        if not m:
            return None
        hay = m.group(1)
    m = re.search(r"\b(\d+(?:\.\d+)?)\b", hay)
    return float(m.group(1)) if m else None


# --------------------------------------------------------------------------
# The grammar
# --------------------------------------------------------------------------

# Ordered: the first rule whose pattern matches wins, so put the specific
# and the dangerous ones first. Refusals are matched BEFORE anything that
# could be mistaken for them.
_RULES: list[tuple[str, str]] = [
    # --- refused, matched first and unconditionally -------------------
    ("arm", r"\b(arm|arming|armed)\b(?!\s*ed\b)"),
    ("start", r"\b(start|launch|take ?off|go for show|begin the show)\b"),
    ("disarm", r"\b(disarm|dis arm)\b"),

    # --- emergencies ---------------------------------------------------
    ("suspend", r"\b(suspend|freeze|hold position|all hold|stop moving|pause the fleet)\b"),
    ("resume", r"\b(resume|unsuspend|continue the show|carry on)\b"),
    ("collective_rth", r"\b(collective return|bring (everyone|them all|the fleet) (home|back)|all return home|return home)\b"),
    ("land", r"\b(land( now| the fleet| everyone| them)?)\b"),
    ("hold", r"\b(hold|brake)\b"),

    # --- design ---------------------------------------------------------
    ("create_show", r"\b(new|create|make) (a )?(show|project)\b"),
    ("add_formation", r"\b(add|make|create|show|give me|put up|draw)\b.*\b(formation|shape|sphere|ball|globe|orb|shell|heart|star|circle|ring|disc|disk|cube|box|square|grid|helix|spiral|dna|torus|donut|doughnut|wave|line|rings)\b"),
    ("apply_effect", r"\b(effect|strobe|flash|pulse|breathe|rainbow|chase|chaser|twinkle|sparkle|glitter|meteor|ripple|wipe|sweep|gradient|alternate|alternating|fade)\b"),
    ("set_colour", r"\b(colour|color|paint|make it)\b"),
    ("set_music", r"\b(bpm|tempo|beat|music)\b"),
    ("compose", r"\b(design|compose|build|generate|come up with)\b.*\b(show|sequence|storyboard)\b"),

    # --- pipeline --------------------------------------------------------
    ("prepare", r"\b(prepare|prep)\b"),
    ("validate", r"\b(validate|check safety|safety check|is it safe|run the checks?)\b"),
    ("adapt", r"\b(adapt|re ?cut|resize|scale down|fewer drones)\b"),
    ("auto_geofence", r"\b(geofence|fence)\b"),
    ("export", r"\b(export|save as|write out)\b"),
    ("report", r"\b(report|paperwork|print the)\b"),

    # --- operations ------------------------------------------------------
    ("ops_load", r"\b(load|connect)\b.*\b(fleet|drones|simulator|sitl)\b"),
    ("preflight", r"\b(preflight|pre flight|run the checks on the fleet)\b"),
    ("upload", r"\b(upload|send the plans?)\b"),
    ("save_log", r"\b(save the log|flight log)\b"),

    # --- preview navigation ---------------------------------------------
    ("play", r"\b(play|run the preview)\b"),
    ("pause", r"\b(pause|stop the preview)\b"),
    ("seek", r"\b(go to|jump to|seek|skip to|rewind|back to)\b"),
    ("next_formation", r"\b(next formation|forward one)\b"),
    ("previous_formation", r"\b(previous formation|back one|last formation)\b"),
    ("camera", r"\b(top view|from the top|crowd view|audience view|from the front|iso|isometric|map view|show the map|3d view)\b"),

    # --- queries ----------------------------------------------------------
    ("status", r"\b(status|how (is|are) (it|we|things)|what.s happening|sit ?rep)\b"),
    ("help", r"\b(help|what can (you|i) (do|say)|commands)\b"),
]

_COMPILED = [(name, re.compile(pattern)) for name, pattern in _RULES]


def parse(text: str, context: dict | None = None) -> Intent:
    """Turn a spoken phrase into an Intent, with its risk level.

    `context` carries what the UI already knows - the open show, the
    selected formation, the fleet phase - so that "make it red" and "add a
    strobe" resolve without the operator repeating themselves.
    """
    context = context or {}
    raw = text or ""
    t = normalise(raw)
    if not t:
        return Intent("none", risk=AUTO, speech="I didn't catch that.", transcript=raw,
                      confidence=0.0)

    for name, pattern in _COMPILED:
        if pattern.search(t):
            intent = _build(name, t, context)
            intent.transcript = raw
            return intent

    # Nothing matched. Suggest the nearest command rather than failing mute.
    guess = _fuzzy(t.split()[0], [n for n, _ in _RULES], cutoff=0.7)
    hint = f" Did you mean {guess.replace('_', ' ')}?" if guess else ""
    return Intent(
        "unknown", risk=AUTO, transcript=raw, confidence=0.0,
        speech=f"I didn't understand that.{hint} Say 'help' for what I can do.",
    )


def _build(name: str, t: str, ctx: dict) -> Intent:
    # ctx may also carry "show"; only the selection is needed for now.
    selected = int(ctx.get("selected", 0) or 0)

    # ---- refused -------------------------------------------------------
    if name in REFUSED_INTENTS:
        verb = {"arm": "Arming", "start": "Starting a show",
                "disarm": "Disarming"}[name]
        return Intent(
            name, risk=REFUSED, confidence=1.0,
            speech=(
                f"{verb} is not available by voice. A microphone picks up "
                "bystanders, radios and the show audio, so anything that "
                "commits aircraft to flight stays a deliberate click on the "
                "launch panel."
            ),
        )

    # ---- emergencies ---------------------------------------------------
    if name == "suspend":
        return Intent(name, risk=AUTO,
                      speech="Suspending. Every drone holds where it is.")
    if name == "hold":
        return Intent("suspend", risk=AUTO,
                      speech="Holding the fleet in place.")
    if name == "resume":
        return Intent(name, risk=CONFIRM,
                      confirm_prompt="Resume the show? The fleet starts moving again.",
                      speech="Resuming puts the fleet back in motion. Confirm on screen.")
    if name == "collective_rth":
        return Intent(
            name, risk=CONFIRM,
            confirm_prompt=(
                "Collective return home? One deconflicted plan is computed from "
                "measured positions and verified before it flies."
            ),
            speech="Collective return home needs confirmation on screen.")
    if name == "land":
        return Intent(
            name, risk=CONFIRM,
            confirm_prompt="Land the whole fleet where it is?",
            speech="Landing the fleet needs confirmation on screen.")

    # ---- design --------------------------------------------------------
    if name == "create_show":
        count = find_number(t, after="with") or find_number(t) or 100
        label = _quoted_name(t) or "voice show"
        return Intent(name, {"name": label, "drone_count": count}, AUTO,
                      speech=f"Creating {label} with {count} drones.")

    if name == "add_formation":
        found = find_shape(t)
        if not found:
            return Intent("unknown", risk=AUTO, confidence=0.3,
                          speech="Which shape? Try sphere, heart, star, cube or helix.")
        shape, params, conf = found
        colour = find_colour(t)
        altitude = find_float(t, after="at") or find_float(t, after="altitude") or 50.0
        args = {"shape": shape, "params": params, "altitude": altitude}
        if colour:
            args["color"] = list(colour)
        speech = f"Adding a {shape.replace('_', ' ')}"
        if colour:
            speech += f" in {_colour_name(colour)}"
        return Intent(name, args, AUTO, speech=speech + ".", confidence=conf)

    if name == "apply_effect":
        effect = find_effect(t)
        if not effect:
            return Intent("unknown", risk=AUTO, confidence=0.3,
                          speech="Which effect? Try chase, strobe, rainbow or twinkle.")
        args: dict[str, Any] = {"effect": effect, "formation": selected, "params": {}}
        colour = find_colour(t)
        if colour and effect not in ("rainbow",):
            key = "color_a" if effect == "alternate" else (
                "start" if effect in ("fade", "gradient", "wipe") else "color")
            args["params"][key] = list(colour)
        hz = find_float(t, after="at")
        if effect == "strobe" and hz:
            args["params"]["hz"] = hz
        beats = find_float(t, after="every")
        if beats:
            args["beats_per_cycle"] = beats
        return Intent(name, args, AUTO,
                      speech=f"Applying {effect} to formation {selected + 1}.")

    if name == "set_colour":
        colour = find_colour(t)
        if not colour:
            return Intent("unknown", risk=AUTO, confidence=0.3,
                          speech="Which colour?")
        return Intent("apply_effect",
                      {"effect": "solid", "formation": selected,
                       "params": {"color": list(colour)}},
                      AUTO, speech=f"Formation {selected + 1} is now {_colour_name(colour)}.")

    if name == "set_music":
        bpm = _find_bpm(t) or find_number(t)
        if not bpm:
            return Intent("unknown", risk=AUTO, confidence=0.3,
                          speech="What tempo, in beats per minute?")
        return Intent(name, {"bpm": float(bpm)}, AUTO,
                      speech=f"Tempo set to {bpm} beats per minute.")

    if name == "compose":
        return Intent(name, {"description": t}, AUTO,
                      speech="Composing a storyboard. I'll show it for review before anything flies.")

    # ---- pipeline -------------------------------------------------------
    if name == "prepare":
        return Intent(name, {}, AUTO,
                      speech="Preparing: scaling formations, adding takeoff and landing, timing it.")
    if name == "validate":
        return Intent(name, {}, AUTO, speech="Running the safety engine.")
    if name == "adapt":
        count = find_number(t, after="to") or find_number(t)
        if not count:
            return Intent("unknown", risk=AUTO, confidence=0.3,
                          speech="Adapt to how many drones?")
        return Intent(name, {"count": count}, CONFIRM,
                      confirm_prompt=f"Re-cut this show for {count} drones? It is saved as a new show.",
                      speech=f"Adapting to {count} drones needs confirmation.")
    if name == "auto_geofence":
        return Intent(name, {"margin": find_float(t, after="margin") or 15.0}, AUTO,
                      speech="Drawing a geofence around the show, plus margin. Check it against your permit.")
    if name == "export":
        fmt = "vviz" if "vviz" in t else ("csv" if "csv" in t else None)
        if not fmt:
            return Intent("unknown", risk=AUTO, confidence=0.3,
                          speech="Export as vviz or CSV?")
        return Intent(name, {"format": fmt}, AUTO, speech=f"Exporting {fmt}.")
    if name == "report":
        return Intent(name, {}, AUTO, speech="Opening the validation report.")

    # ---- operations ------------------------------------------------------
    if name == "ops_load":
        driver = "sitl" if "sitl" in t else "simulator"
        return Intent(name, {"driver": driver}, CONFIRM,
                      confirm_prompt=f"Load this show onto the {driver} fleet?",
                      speech=f"Loading onto the {driver} fleet needs confirmation.")
    if name == "preflight":
        return Intent(name, {}, AUTO, speech="Running preflight checks.")
    if name == "upload":
        return Intent(name, {}, CONFIRM,
                      confirm_prompt="Upload plans to every drone?",
                      speech="Uploading plans needs confirmation on screen.")
    if name == "save_log":
        return Intent(name, {}, AUTO, speech="Saving the flight log.")

    # ---- navigation -------------------------------------------------------
    if name == "seek":
        seconds = find_float(t, after="to") or find_float(t)
        return Intent(name, {"seconds": seconds or 0.0}, AUTO,
                      speech=f"Jumping to {seconds or 0:.0f} seconds.")
    if name == "camera":
        view = ("top" if "top" in t else
                "map" if "map" in t else
                "crowd" if ("crowd" in t or "audience" in t or "front" in t) else "iso")
        return Intent(name, {"view": view}, AUTO, speech=f"{view.title()} view.")
    if name in ("play", "pause", "next_formation", "previous_formation"):
        return Intent(name, {}, AUTO, speech=name.replace("_", " ").title() + ".")

    # ---- queries -----------------------------------------------------------
    if name == "status":
        return Intent(name, {}, AUTO, speech="")   # server fills in the answer
    if name == "help":
        return Intent(name, {}, AUTO, speech=help_text())

    return Intent(name, {}, AUTO, speech="")


def _quoted_name(t: str) -> str | None:
    m = re.search(r"\b(?:called|named)\s+(.+?)(?:\s+with\b|$)", t)
    return m.group(1).strip() if m else None


def _colour_name(rgb) -> str:
    for name, value in COLOUR_WORDS.items():
        if tuple(value) == tuple(rgb):
            return name
    return "that colour"


def help_text() -> str:
    return (
        "You can say: create a show with 200 drones; add a red heart at 60 metres; "
        "apply a chase effect; set the tempo to 128; prepare; validate; "
        "auto geofence; export vviz; load the fleet; run preflight; "
        "suspend; resume; collective return home. "
        "Arming, starting and disarming are not available by voice."
    )


def vocabulary() -> dict:
    """What the UI shows as 'things you can say'."""
    return {
        "shapes": sorted(set(SHAPE_WORDS)),
        "colours": sorted(COLOUR_WORDS),
        "effects": sorted(set(EFFECT_WORDS)),
        "refused": sorted(REFUSED_INTENTS),
        "examples": [
            "create a show called summer gala with two hundred drones",
            "add a red heart at 60 metres",
            "add a sphere",
            "apply a chase effect in cyan",
            "strobe at 4 hertz",
            "set the tempo to 128",
            "prepare",
            "validate",
            "auto geofence",
            "export vviz",
            "adapt to 180 drones",
            "load the fleet on the simulator",
            "run preflight",
            "suspend",
            "collective return home",
            "top view",
            "go to 45 seconds",
            "status",
        ],
    }


# --------------------------------------------------------------------------
# Composing a storyboard from a description
# --------------------------------------------------------------------------

# Moods map onto a palette and a pace. Deliberately small and legible: the
# point is a reviewable draft, not a surprise.
MOODS: dict[str, dict] = {
    "calm": {"palette": [(40, 120, 255), (0, 230, 255), (255, 255, 255)],
             "transition": 12.0, "hold": 7.0, "effects": ["pulse", "gradient"]},
    "slow": {"palette": [(80, 60, 220), (170, 80, 255)],
             "transition": 14.0, "hold": 8.0, "effects": ["pulse", "fade"]},
    "energetic": {"palette": [(255, 30, 30), (255, 140, 20), (255, 220, 50)],
                  "transition": 7.0, "hold": 4.0, "effects": ["chase", "strobe"]},
    "festive": {"palette": [(255, 190, 60), (255, 30, 30), (40, 230, 90)],
                "transition": 8.0, "hold": 5.0, "effects": ["twinkle", "chase"]},
    "romantic": {"palette": [(255, 90, 190), (255, 30, 30), (255, 230, 200)],
                 "transition": 11.0, "hold": 7.0, "effects": ["pulse", "gradient"]},
    "patriotic": {"palette": [(255, 30, 30), (255, 255, 255), (40, 120, 255)],
                  "transition": 9.0, "hold": 6.0, "effects": ["wipe", "solid"]},
}
DEFAULT_MOOD = "festive"


def compose(description: str, drone_count: int = 100, *, use_llm: bool | None = None) -> dict:
    """Draft a storyboard from a free-form description.

    Returns a spec the caller validates and executes; it is never flown
    directly. This is the one place a language model earns its keep, since
    "something romantic for a wedding" has no deterministic right answer -
    but the output is still only a list of shapes, colours and timings that
    goes through exactly the same generators, scaling and validation as a
    hand-built show.

    With no API key, or when the call fails, a rule-based composer runs
    instead and the result says which produced it. The fallback is not a
    degraded mode: it is the default, and it is what the tests exercise.
    """
    if use_llm is None:
        from .assistant import available

        use_llm = available()
    if use_llm:
        try:
            spec = _compose_with_llm(description, drone_count)
            spec["source"] = "llm"
            return _sanitise_spec(spec, drone_count)
        except Exception as exc:  # any failure falls back, loudly
            spec = _compose_rules(description, drone_count)
            spec["source"] = "rules"
            spec["llm_error"] = str(exc)[:200]
            return _sanitise_spec(spec, drone_count)
    spec = _compose_rules(description, drone_count)
    spec["source"] = "rules"
    return _sanitise_spec(spec, drone_count)


def _compose_rules(description: str, drone_count: int) -> dict:
    t = normalise(description)
    mood = next((m for m in MOODS if m in t), None)
    if mood is None:
        if any(w in t for w in ("wedding", "love", "valentine", "anniversary")):
            mood = "romantic"
        elif any(w in t for w in ("party", "new year", "celebration", "diwali", "festival")):
            mood = "festive"
        elif any(w in t for w in ("fast", "punchy", "rock", "drop", "hype")):
            mood = "energetic"
        elif any(w in t for w in ("gentle", "ambient", "peaceful", "serene")):
            mood = "calm"
        elif any(w in t for w in ("flag", "national", "independence")):
            mood = "patriotic"
        else:
            mood = DEFAULT_MOOD
    recipe = MOODS[mood]

    # Any shape the description actually named comes first, so "a show with
    # a heart and a star" produces those, in that order.
    named: list[str] = []
    for phrase in sorted(SHAPE_WORDS, key=len, reverse=True):
        for m in re.finditer(rf"\b{re.escape(phrase)}\b", t):
            named.append((m.start(), SHAPE_WORDS[phrase][0]))
    ordered = [s for _, s in sorted(set(named))]

    filler = ["ball", "disc", "helix", "cube", "wave", "torus"]
    sequence = list(dict.fromkeys(ordered + filler))[: max(3, min(6, len(ordered) + 3))]

    # Size every shape to what this many drones actually need, and keep the
    # stack under a default 120 m ceiling with room for the climb and RTH.
    from . import geometry
    from .show import ShowLimits

    limits = ShowLimits()
    width, height = geometry.billboard_for(drone_count, limits.formation_separation)
    extent = max(min(height, 45.0), 12.0)
    base = max(limits.min_nav_altitude + 25.0, 35.0)

    palette = recipe["palette"]
    formations = []
    for i, shape in enumerate(sequence):
        params: dict = _size_params(shape, extent)
        for phrase, (gen, extra) in SHAPE_WORDS.items():
            if gen == shape and extra:
                params.update(extra)
                break
        formations.append({
            "name": f"{shape.replace('_', ' ')}",
            "shape": shape,
            "params": params,
            "color": list(palette[i % len(palette)]),
            "altitude": base + (i % 3) * 6.0,
            "transition": recipe["transition"],
            "hold": recipe["hold"],
            "effect": recipe["effects"][i % len(recipe["effects"])],
            "effect_params": effect_params(
                recipe["effects"][i % len(recipe["effects"])],
                palette[i % len(palette)],
                palette[(i + 1) % len(palette)],
            ),
        })

    bpm = _find_bpm(t)
    return {
        "description": description,
        "mood": mood,
        "drone_count": drone_count,
        "bpm": float(bpm) if bpm else None,
        "formations": formations,
        "note": (
            "A draft for review. It is not flyable until it has been prepared "
            "and has passed the safety engine."
        ),
    }


def _find_bpm(t: str) -> int | None:
    """Tempo, spoken either way round: '128 bpm' or 'tempo 128'."""
    m = re.search(r"(\d{2,3})\s*bpm", t)
    if m:
        return int(m.group(1))
    return find_number(t, after="bpm") or find_number(t, after="tempo")


def effect_params(effect: str, colour, accent=None, duration: float = 4.0) -> dict:
    """Build an effect's keyword arguments around a chosen colour.

    Effects do not share a parameter name: `fade`, `gradient` and `wipe`
    take start/end, `alternate` takes two colours, `rainbow` takes none,
    and the rest take `color`. Passing an empty dict instead - which the
    composer did - raises "missing 1 required positional argument" from
    deep inside the effect factory, so every composed formation was
    silently dropped. `fade` and `wipe` also need a duration, and for the
    same reason: without one they failed at render time.
    """
    rgb = [int(c) for c in (colour or (255, 255, 255))][:3]
    other = [int(c) for c in (accent or (0, 0, 0))][:3]
    if effect == "rainbow":
        return {}
    if effect == "gradient":
        return {"start": rgb, "end": other}
    if effect in ("fade", "wipe"):
        # These run once, so they need to know how long the formation holds.
        return {"start": rgb, "end": other, "duration": float(duration)}
    if effect == "alternate":
        return {"color_a": rgb, "color_b": other}
    return {"color": rgb}


def _size_params(shape: str, extent: float) -> dict:
    """Size a generator to a target extent in metres.

    Without this the composer takes each generator's default size, which
    is tuned for looking right on its own rather than for fitting under a
    ceiling alongside a takeoff and an RTH. A 60 drone show then tops out
    at 121 m against a 120 m limit and fails validation for a reason the
    operator did not choose.
    """
    r = max(extent / 2.0, 6.0)
    return {
        "ball": {"radius": r},
        "sphere": {"radius": r},
        "disc": {"radius": r},
        "circle": {"radius": r},
        "torus": {"major": r, "minor": max(r / 3.0, 3.0)},
        "ring_stack": {"radius": r, "gap": max(extent / 5.0, 4.0)},
        "cube": {"size": extent},
        "grid": {"spacing": max(extent / 10.0, 3.0)},
        "helix": {"radius": max(r * 0.7, 5.0), "height": extent},
        "double_helix": {"radius": max(r * 0.7, 5.0), "height": extent},
        "wave": {"width": extent, "amplitude": max(extent / 6.0, 4.0)},
        "line": {"length": extent},
        "star": {"outer": r},
        "heart": {"scale": max(extent / 30.0, 0.5)},
    }.get(shape, {})


def _compose_with_llm(description: str, drone_count: int) -> dict:
    """Ask Claude for a storyboard, constrained to the vocabulary we have.

    NOT exercised by the test suite - there is no API key in CI, and a test
    that needs the network is a test that fails for reasons unrelated to
    the code. The fallback path is what is tested; this branch is held to
    the same output contract by _sanitise_spec, which rejects anything the
    generators cannot build.
    """
    from .assistant import ask_json

    shapes = sorted({v[0] for v in SHAPE_WORDS.values()})
    effect_names = sorted(set(EFFECT_WORDS.values()))
    system = (
        "You design drone light shows as a storyboard. "
        f"Shapes available: {', '.join(shapes)}. "
        f"Effects available: {', '.join(effect_names)}. "
        "3 to 7 formations. altitude 30-90 metres, transition 6-15 seconds, "
        "hold 3-10 seconds. Use only the listed shapes and effects."
    )
    formation = {"type": "object", "additionalProperties": False,
                 "required": ["name", "shape", "color", "altitude", "transition",
                              "hold", "effect"],
                 "properties": {"name": {"type": "string"}, "shape": {"type": "string"},
                                "color": {"type": "array", "items": {"type": "integer"}},
                                "altitude": {"type": "number"},
                                "transition": {"type": "number"},
                                "hold": {"type": "number"}, "effect": {"type": "string"}}}
    schema = {"type": "object", "additionalProperties": False,
              "required": ["mood", "bpm", "formations"],
              "properties": {"mood": {"type": "string"},
                             "bpm": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                             "formations": {"type": "array", "items": formation}}}
    spec = ask_json(system, f"Drone count: {drone_count}. Brief: {description}", schema)
    spec["description"] = description
    spec["drone_count"] = drone_count
    spec.setdefault("note", "LLM draft for review. Not flyable until validated.")
    return spec


def _sanitise_spec(spec: dict, drone_count: int) -> dict:
    """Force a composed spec back inside what the generators can build.

    Applies to the language model's output and the rule composer's alike.
    An unknown shape, an out-of-range altitude or a malformed colour is
    corrected here rather than surfacing as a traceback three layers down,
    and anything corrected is recorded so the draft is honest about it.
    """
    from . import effects as fx
    from . import geometry

    fixes: list[str] = []
    out = []
    for f in spec.get("formations", [])[:8]:
        shape = str(f.get("shape", "")).strip()
        if shape not in geometry.SHAPES:
            alt = _fuzzy(shape, geometry.SHAPES, cutoff=0.6) or "ball"
            fixes.append(f"shape {shape!r} -> {alt}")
            shape = alt
        effect = str(f.get("effect", "solid")).strip()
        if effect not in fx.EFFECTS:
            alt = _fuzzy(effect, fx.EFFECTS, cutoff=0.6) or "solid"
            fixes.append(f"effect {effect!r} -> {alt}")
            effect = alt
        colour = f.get("color") or [255, 255, 255]
        try:
            colour = [max(0, min(255, int(c))) for c in colour][:3]
            if len(colour) != 3:
                raise ValueError
        except (TypeError, ValueError):
            fixes.append(f"colour {f.get('color')!r} -> white")
            colour = [255, 255, 255]
        altitude = _clamp(f.get("altitude", 50.0), 20.0, 110.0, "altitude", fixes)
        transition = _clamp(f.get("transition", 9.0), 4.0, 25.0, "transition", fixes)
        hold = _clamp(f.get("hold", 5.0), 1.0, 20.0, "hold", fixes)
        out.append({
            "name": str(f.get("name") or shape)[:40],
            "shape": shape,
            "params": dict(f.get("params") or {}),
            "color": colour,
            "altitude": altitude,
            "transition": transition,
            "hold": hold,
            "effect": effect,
            # Rebuilt rather than trusted: the effect or the colour may
            # have just been corrected above, and a stale parameter set
            # would fail at render time instead of here.
            "effect_params": effect_params(effect, colour, duration=hold),
        })

    if not out:
        fixes.append("no usable formations; substituted a default sphere")
        out = [{"name": "sphere", "shape": "ball", "params": {},
                "color": [255, 255, 255], "altitude": 50.0,
                "transition": 9.0, "hold": 5.0, "effect": "solid",
                "effect_params": effect_params("solid", [255, 255, 255])}]

    spec["formations"] = out
    spec["drone_count"] = int(drone_count)
    spec["corrections"] = fixes
    return spec


def _clamp(value, low: float, high: float, label: str, fixes: list[str]) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        fixes.append(f"{label} {value!r} -> {low}")
        return low
    if v < low or v > high:
        fixes.append(f"{label} {v:g} -> {min(max(v, low), high):g}")
    return min(max(v, low), high)
