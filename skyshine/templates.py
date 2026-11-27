"""Built-in show templates.

A template is a fixed storyboard - shapes, colours, light effects, pacing -
sized to whatever drone count it is built for. It produces an ordinary,
unprepared show: it still has to be prepared and pass the safety engine,
exactly like one built by hand.
"""

from __future__ import annotations

import numpy as np

from . import effects, geometry, planning
from .show import Formation, Show, ShowLimits, ShowOrigin
from .voice import _size_params, effect_params

# Metres kept clear between the top of a formation and the ceiling.
HEADROOM = 5.0

# A step with this shape spells the caller's words (or the template default).
TEXT = "text"

GOLD, WHITE, RED = (255, 190, 60), (255, 255, 255), (255, 30, 30)
# A filled star with fat arms. The default thin-armed star packs badly: 200
# drones at legal spacing make it 135 m tall, over a 120 m ceiling.
STAR = {"solid": True, "outer": 20.0, "inner": 12.0}

# step: (name, shape, extra shape parameters, colour, effect, accent colour)
TEMPLATES: dict[str, dict] = {
    "showcase": {
        "title": "Showcase",
        "summary": "Six classic shapes, one after another. A fleet demo.",
        "transition": 10.0, "hold": 6.0,
        "steps": [
            ("sphere rise", "ball", {}, (20, 90, 255), "pulse", None),
            ("open heart", "heart", {"solid": True}, (255, 40, 90), "solid", None),
            ("five point star", "star", STAR, (255, 205, 60), "twinkle", None),
            ("disc", "disc", {}, (0, 230, 190), "ripple", None),
            ("cube", "cube", {}, (190, 120, 255), "gradient", (40, 120, 255)),
            ("wide wave", "wave", {}, (90, 255, 140), "rainbow", None),
        ],
    },
    "wedding": {
        "title": "Wedding",
        "summary": "Two rings, a heart, the couple's initials, a star to finish.",
        "text": "A & B",
        "transition": 11.0, "hold": 7.0,
        "steps": [
            ("two rings", "ring_stack", {"rings": 2}, GOLD, "chase", None),
            ("heart", "heart", {"solid": True}, (255, 30, 60), "pulse", None),
            ("initials", TEXT, {}, (255, 230, 200), "solid", None),
            ("veil", "wave", {}, (255, 90, 190), "gradient", WHITE),
            ("finale star", "star", STAR, GOLD, "twinkle", None),
        ],
    },
    "new year": {
        "title": "New Year",
        "summary": "Clock face, the year in lights, firework and confetti.",
        "text": "2027",
        "transition": 8.0, "hold": 5.0,
        "steps": [
            ("clock face", "disc", {}, GOLD, "chase", None),
            ("the year", TEXT, {}, WHITE, "pulse", None),
            ("firework", "ball", {}, RED, "ripple", None),
            ("star burst", "star", STAR, (255, 220, 50), "twinkle", None),
            ("confetti", "wave", {}, (40, 230, 90), "rainbow", None),
        ],
    },
    "brand reveal": {
        "title": "Brand reveal",
        "summary": "Gather, a curtain of light, then the brand name.",
        "text": "SKY SHINE",
        "transition": 9.0, "hold": 6.0,
        "steps": [
            ("gather", "ball", {}, (40, 120, 255), "pulse", None),
            ("curtain", "grid", {"upright": True}, WHITE, "wipe", (0, 230, 255)),
            ("the name", TEXT, {}, WHITE, "gradient", (0, 230, 255)),
            ("sign off", "wave", {}, (0, 230, 255), "chase", None),
        ],
    },
    "festival": {
        "title": "Festival",
        "summary": "Fast and bright: rings, cube, star, disc, sphere.",
        "transition": 8.0, "hold": 5.0,
        "steps": [
            ("stacked rings", "ring_stack", {}, (255, 140, 20), "meteor", None),
            ("cube", "cube", {}, (40, 230, 90), "chase", None),
            ("star", "star", STAR, GOLD, "twinkle", None),
            ("colour wheel", "disc", {}, WHITE, "rainbow", None),
            ("lantern", "ball", {}, RED, "alternate", WHITE),
        ],
    },
}


def catalogue() -> list[dict]:
    """What the UI needs to offer the templates. `text` is the default words
    of a template that spells something, or None if it spells nothing."""
    return [
        {"id": key, "title": t["title"], "summary": t["summary"],
         "formations": len(t["steps"]), "text": t.get("text")}
        for key, t in TEMPLATES.items()
    ]


def build(
    key: str,
    name: str,
    drone_count: int,
    *,
    text: str = "",
    limits: ShowLimits | None = None,
    origin: ShowOrigin | None = None,
) -> Show:
    """Build an unprepared show from a template.

    Every formation is scaled to legal spacing here, not left for `prepare`,
    so its real height is known and it can be placed to fit: at the usual
    show height when there is room, lower when a big fleet needs the space.
    A shape too tall for the ceiling at any height is left over it, and the
    validator will say so. Raises ValueError if a shape cannot be built.
    """
    try:
        tpl = TEMPLATES[key]
    except KeyError:
        raise ValueError(
            f"unknown template {key!r}; available: {', '.join(TEMPLATES)}"
        ) from None
    show = Show(
        name=name, drone_count=drone_count, limits=limits or ShowLimits(),
        origin=origin or ShowOrigin(), notes=f"From the {tpl['title']} template.",
    )
    lim = show.limits
    width, height = geometry.billboard_for(drone_count, lim.formation_separation)
    base = max(lim.min_nav_altitude + 25.0, 35.0)
    extent = max(min(height, 45.0, lim.max_altitude - base - 12.0), 12.0)
    message = text.strip() or tpl.get("text", "")

    for i, (label, shape, extra, colour, effect, accent) in enumerate(tpl["steps"]):
        if shape == TEXT:
            params = {"message": message, "width": width}
        else:
            params = {**_size_params(shape, extent), **extra}
        points = geometry.fit_separation(
            geometry.make(shape, drone_count, **params),
            planning.design_separation(lim, show.fps),
        )
        floor = min(
            base + (i % 3) * 6.0,
            lim.max_altitude - HEADROOM - float(np.ptp(points[:, 2])),
        )
        formation = Formation(
            name=label,
            points=geometry.lift(points, max(floor, lim.min_nav_altitude)),
            colors=np.asarray(colour, dtype=np.int32),
            transition=tpl["transition"],
            hold=tpl["hold"],
        )
        show.add(formation)
        if effect != "solid":
            effects.render_named(
                formation, effect,
                **effect_params(effect, colour, accent, duration=tpl["hold"]),
            )
    return show
