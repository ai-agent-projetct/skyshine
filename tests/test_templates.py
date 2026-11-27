"""Templates must produce shows that actually fly.

A template that builds but fails the safety engine is worse than no
template: the designer starts from something they then have to debug.
"""

from __future__ import annotations

import pytest

from skyshine import planning, safety, templates


@pytest.mark.parametrize("count", [5, 120])
@pytest.mark.parametrize("key", list(templates.TEMPLATES))
def test_every_template_passes_the_safety_engine(key, count):
    show = templates.build(key, "t", count)
    assert all(f.count == count for f in show.formations)
    report = safety.validate(planning.prepare(show).show)
    assert report.passed, report.summary()


def test_text_is_used_and_falls_back_to_the_default():
    custom = templates.build("new year", "t", 40, text="2031")
    default = templates.build("new year", "t", 40, text="   ")
    year = [f.name for f in custom.formations].index("the year")
    assert (custom.formations[year].points != default.formations[year].points).any()


def test_unknown_template_is_a_clear_error():
    with pytest.raises(ValueError, match="unknown template"):
        templates.build("nope", "t", 10)
