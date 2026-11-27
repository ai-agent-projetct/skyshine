"""Tests for the features added after the first release.

Light effects, music and beat locking, pyro and yaw with their safety
checks, vviz and CSV interop, show adaptation, auto geofence, and the two
operational additions: suspend/resume and collective return-to-home.

The ones worth reading are TestCollectiveRth (per-drone RTL is not
deconflicted and this is what replaces it) and TestHoldRelease (a latched
hold silently swallowed every later command, which is the worst way for an
abort path to fail - it looks like it worked).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from skyshine import effects, geometry, interop, planning, safety
from skyshine.drivers import SimDriver, SimFaults
from skyshine.fleet import FleetManager, Phase
from skyshine.show import (
    Formation,
    Music,
    PyroEvent,
    Show,
    ShowLimits,
    load_show,
    save_show,
)
from skyshine.trajectory import build_timeline


def base_show(n=24, hold=6.0, yaw=None, name="feat") -> Show:
    show = Show(
        name=name,
        drone_count=n,
        limits=ShowLimits(max_altitude=280.0, battery_endurance_s=2400.0),
    )
    show.origin.latitude, show.origin.longitude, show.origin.amsl = 12.9716, 77.5946, 920.0
    for label, pts in (
        ("ball", geometry.ball(n, 20)),
        ("disc", geometry.disc(n, 24)),
        ("cube", geometry.cube_shell(n, 28)),
    ):
        show.add(
            Formation(
                name=label,
                points=geometry.lift(pts, 70.0),
                colors=np.array([255, 60, 20]),
                transition=8.0,
                hold=hold,
                yaw=yaw,
            )
        )
    return show


class ManualClock:
    def __init__(self, start=1_700_000_000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# ==========================================================================
# Light effects
# ==========================================================================


EFFECT_ARGS = {
    "solid": {"color": (255, 120, 0)},
    "fade": {"start": (255, 0, 0), "end": (0, 0, 255), "duration": 3.0},
    "strobe": {"color": (255, 255, 255), "hz": 5.0},
    "pulse": {"color": (0, 180, 255)},
    "gradient": {"start": (255, 0, 0), "end": (0, 0, 255)},
    "rainbow": {},
    "wipe": {"start": (255, 0, 0), "end": (0, 0, 255), "duration": 3.0},
    "chase": {"color": (0, 255, 180)},
    "ripple": {"color": (255, 200, 0)},
    "twinkle": {"color": (255, 255, 255)},
    "meteor": {"color": (255, 140, 0)},
    "alternate": {"color_a": (255, 0, 0), "color_b": (0, 255, 0)},
    "volume": {"color": (255, 0, 0), "radius": 0.5, "duration": 2.0},
}


class TestEffects:
    def test_every_effect_is_covered_by_these_tests(self):
        assert set(EFFECT_ARGS) == set(effects.EFFECTS), (
            "an effect was added without a test"
        )

    @pytest.mark.parametrize("name", sorted(EFFECT_ARGS))
    def test_effect_returns_valid_colours(self, name):
        fn = effects.make(name, **EFFECT_ARGS[name])
        pts = geometry.lift(geometry.disc(40, 20), 50.0)
        for t in (0.0, 0.37, 1.0, 2.5, 7.9):
            out = np.asarray(fn(pts, t))
            assert out.shape == (40, 3), f"{name} returned {out.shape}"
            assert out.dtype == np.uint8
            assert out.min() >= 0 and out.max() <= 255

    @pytest.mark.parametrize("name", sorted(EFFECT_ARGS))
    def test_render_bakes_into_keyframes(self, name):
        f = Formation("f", geometry.lift(geometry.disc(30, 18), 50.0),
                      np.array([0, 0, 0]), transition=4.0, hold=4.0)
        effects.render_named(f, name, **EFFECT_ARGS[name])
        assert f.colors.shape == (30, 3)
        assert f.color_keys, f"{name} produced no keyframes"
        for t, c in f.color_keys:
            assert 0 < t <= f.hold + 1e-9
            assert c.shape == (30, 3)
        # Keys must be ordered, or colors_at interpolates backwards.
        times = [t for t, _ in f.color_keys]
        assert times == sorted(times)

    def test_fast_effects_raise_their_own_sample_rate(self):
        """A 10 Hz strobe rendered at 10 Hz aliases into a constant colour."""
        slow = effects.recommended_fps(effects.make("solid", color=(1, 2, 3)))
        fast = effects.recommended_fps(effects.make("strobe", color=(255,) * 3, hz=10))
        assert slow == 10.0
        assert fast >= 40.0

        f = Formation("f", geometry.lift(geometry.disc(10, 8), 50.0),
                      np.array([0, 0, 0]), transition=1.0, hold=2.0)
        effects.render_named(f, "strobe", color=(255, 255, 255), hz=10.0)
        seen = {tuple(c[0]) for _, c in f.color_keys}
        assert len(seen) > 1, "the strobe aliased into a single colour"

    def test_twinkle_is_deterministic(self):
        """A show that renders differently each time cannot be reviewed."""
        pts = geometry.lift(geometry.disc(50, 20), 50.0)
        a = effects.make("twinkle", color=(255, 255, 255), seed=7)
        b = effects.make("twinkle", color=(255, 255, 255), seed=7)
        c = effects.make("twinkle", color=(255, 255, 255), seed=8)
        assert np.array_equal(a(pts, 1.3), b(pts, 1.3))
        assert not np.array_equal(a(pts, 1.3), c(pts, 1.3))

    def test_gradient_follows_the_shape_not_the_sky(self):
        pts = geometry.lift(geometry.disc(40, 20), 50.0)
        out = effects.make("gradient", start=(0, 0, 0), end=(255, 255, 255),
                           axis="z")(pts, 0.0)
        lowest = int(np.argmin(pts[:, 2]))
        highest = int(np.argmax(pts[:, 2]))
        assert out[lowest].max() < 40
        assert out[highest].min() > 215

    def test_wipe_runs_from_one_colour_to_the_other(self):
        pts = geometry.lift(geometry.disc(40, 20), 50.0)
        fn = effects.make("wipe", start=(255, 0, 0), end=(0, 0, 255), duration=4.0)
        assert fn(pts, 0.0)[:, 0].mean() > 200
        assert fn(pts, 4.0)[:, 2].mean() > 200

    def test_beat_locked_period(self):
        music = Music(bpm=120.0)
        assert effects.beat_locked(music, 1.0) == pytest.approx(0.5)
        assert effects.beat_locked(music, 4.0) == pytest.approx(2.0)
        # No tempo: fall back rather than divide by zero.
        assert effects.beat_locked(Music(), 1.0, fallback=3.0) == 3.0

    def test_unknown_effect_is_rejected(self):
        with pytest.raises(ValueError, match="unknown effect"):
            effects.make("disco_inferno")

    def test_invalid_parameters_are_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            effects.make("strobe", color=(1, 1, 1), hz=0)
        with pytest.raises(ValueError, match="unknown axis"):
            effects.make("chase", color=(1, 1, 1), axis="sideways")(
                np.zeros((4, 3)), 0.0
            )

    def test_describe_lists_parameters_for_the_ui(self):
        d = effects.describe("chase")
        names = {p["name"] for p in d["params"]}
        assert {"color", "period", "axis", "width", "background"} <= names

    def test_effects_survive_the_show_file(self, tmp_path):
        show = base_show(n=16)
        effects.render_named(show.formations[1], "rainbow", period=4.0)
        keys = len(show.formations[1].color_keys)
        back = load_show(save_show(show, tmp_path / "fx.show.json"))
        assert len(back.formations[1].color_keys) == keys
        for (ta, ca), (tb, cb) in zip(
            show.formations[1].color_keys, back.formations[1].color_keys
        ):
            assert ta == pytest.approx(tb)
            assert np.array_equal(ca, cb)


# ==========================================================================
# Music
# ==========================================================================


class TestMusic:
    def test_beat_times(self):
        m = Music(bpm=120.0)
        beats = m.beat_times(2.0)
        assert beats == [0.0, 0.5, 1.0, 1.5, 2.0]

    def test_beat_phase_offsets_the_grid(self):
        m = Music(bpm=60.0, beat_phase=0.25)
        assert m.beat_times(2.0) == [0.25, 1.25]

    def test_snap(self):
        m = Music(bpm=120.0)          # 0.5 s per beat
        assert m.snap(0.61) == pytest.approx(0.5)
        assert m.snap(0.80) == pytest.approx(1.0)
        assert m.snap(0.61, division=2) == pytest.approx(0.5)   # step 0.25
        assert m.snap(0.70, division=2) == pytest.approx(0.75)
        assert m.snap(0.61, division=4) == pytest.approx(0.625)  # step 0.125

    def test_snap_without_tempo_is_a_no_op(self):
        assert Music().snap(1.234) == 1.234

    def test_round_trip(self, tmp_path):
        show = base_show(n=8)
        show.music = Music(path="track.mp3", bpm=128.0, offset=1.5,
                           beat_phase=0.2, cues=[(4.0, "drop"), (20.0, "bridge")])
        back = load_show(save_show(show, tmp_path / "m.show.json"))
        assert back.music.bpm == 128.0
        assert back.music.path == "track.mp3"
        assert back.music.cues == [(4.0, "drop"), (20.0, "bridge")]


# ==========================================================================
# Yaw and pyro
# ==========================================================================


class TestYaw:
    def test_scalar_yaw_broadcasts(self):
        f = Formation("f", geometry.ball(10, 8), np.array([1, 2, 3]), yaw=45.0)
        assert f.yaw.shape == (10,)
        assert np.allclose(f.yaw, 45.0)

    def test_yaw_is_wrapped_into_range(self):
        f = Formation("f", geometry.ball(4, 8), np.array([1, 2, 3]), yaw=[-90, 450, 0, 360])
        assert np.allclose(f.yaw, [270, 90, 0, 0])

    def test_wrong_length_yaw_is_rejected(self):
        with pytest.raises(ValueError, match="one angle per drone"):
            Formation("f", geometry.ball(10, 8), np.array([1, 2, 3]), yaw=[0, 1, 2])

    def test_fast_turn_is_an_error(self):
        show = base_show(n=12, yaw=0.0)
        show.formations[1].transition = 1.0
        show.formations[1].yaw = np.full(12, 170.0)
        kinds = {v.kind for v in safety.check_yaw(show)}
        assert "yaw_rate" in kinds

    def test_shortest_way_round_is_used(self):
        """0 to 350 degrees is a 10 degree turn, not a 350 degree one."""
        show = base_show(n=12, yaw=0.0)
        show.formations[1].transition = 1.0
        show.formations[1].yaw = np.full(12, 350.0)
        assert not [v for v in safety.check_yaw(show) if v.kind == "yaw_rate"]

    def test_partial_yaw_is_warned_about(self):
        show = base_show(n=12, yaw=0.0)
        show.formations[1].yaw = None
        warnings = [v for v in safety.check_yaw(show) if v.severity == safety.WARNING]
        assert warnings and "heading through those is undefined" in warnings[0].message

    def test_no_yaw_anywhere_is_fine(self):
        assert safety.check_yaw(base_show(n=12)) == []


class TestPyro:
    def test_event_outside_the_hold_is_rejected(self):
        with pytest.raises(ValueError, match="outside the"):
            Formation(
                "f", geometry.ball(8, 10), np.array([1, 2, 3]), hold=2.0,
                pyro=[PyroEvent(at=5.0)],
            )

    def test_event_targeting_a_missing_drone_is_rejected(self):
        with pytest.raises(ValueError, match="does not exist"):
            Formation(
                "f", geometry.ball(8, 10), np.array([1, 2, 3]), hold=4.0,
                pyro=[PyroEvent(at=1.0, drones=[99])],
            )

    def test_absolute_event_times(self):
        show = base_show(n=10)
        show.formations[2].pyro = [PyroEvent(at=2.0, description="gerb")]
        events = show.pyro_events()
        assert len(events) == 1
        when, index, event = events[0]
        # Two formations of 8+6 before it, then this one's 8 s transition.
        assert when == pytest.approx(2 * 14.0 + 8.0 + 2.0)
        assert index == 2

    def test_low_pyro_is_an_error(self):
        show = base_show(n=12, yaw=90.0)
        show.formations[2].pyro = [PyroEvent(at=3.0, heading_deg=90.0)]
        show.formations[2].points[:, 2] = 15.0
        assert "pyro_altitude" in {v.kind for v in safety.check_pyro(show)}

    def test_crowded_pyro_is_an_error(self):
        show = base_show(n=12, yaw=90.0)
        f = show.formations[2]
        f.points[0] = [0.0, 0.0, 80.0]
        f.points[1] = [2.0, 0.0, 80.0]
        f.pyro = [PyroEvent(at=3.0, drones=[0, 1], heading_deg=90.0)]
        assert "pyro_separation" in {v.kind for v in safety.check_pyro(show)}

    def test_pyro_without_yaw_is_an_error(self):
        show = base_show(n=12)
        show.formations[2].pyro = [PyroEvent(at=3.0, heading_deg=90.0)]
        found = [v for v in safety.check_pyro(show) if v.kind == "pyro_heading"]
        assert found and "sets no yaw" in found[0].message

    def test_wrong_heading_is_an_error(self):
        show = base_show(n=12, yaw=0.0)
        show.formations[2].pyro = [PyroEvent(at=3.0, drones=[0], heading_deg=90.0)]
        found = [v for v in safety.check_pyro(show) if v.kind == "pyro_heading"]
        assert found and found[0].value == pytest.approx(90.0)

    def test_firing_before_the_heading_settles_is_an_error(self):
        show = base_show(n=12, yaw=90.0)
        show.formations[2].pyro = [
            PyroEvent(at=0.5, drones=[0], heading_deg=90.0, settle_s=2.0)
        ]
        assert "pyro_settle" in {v.kind for v in safety.check_pyro(show)}

    def test_correct_pyro_passes(self):
        show = base_show(n=12, yaw=90.0)
        show.formations[2].pyro = [
            PyroEvent(at=3.0, drones=[0], heading_deg=90.0, settle_s=2.0,
                      description="10s Gold Gerb")
        ]
        assert safety.check_pyro(show) == []

    def test_pyro_reaches_the_validation_report(self):
        show = planning.prepare(base_show(n=12, yaw=90.0)).show
        show.formations[3].pyro = [PyroEvent(at=1.0, heading_deg=90.0, settle_s=0.5)]
        show.formations[3].points[:, 2] = 10.0  # too low on purpose
        report = safety.validate(show)
        assert "pyro_altitude" in {v.kind for v in report.errors}
        assert report.metrics["pyro_events"] == 1

    def test_round_trip(self, tmp_path):
        show = base_show(n=12, yaw=90.0)
        show.formations[2].pyro = [
            PyroEvent(at=3.0, drones=[0, 2], channel=2, description="comet",
                      heading_deg=90.0, settle_s=1.5)
        ]
        back = load_show(save_show(show, tmp_path / "p.show.json"))
        e = back.formations[2].pyro[0]
        assert e.drones == [0, 2] and e.channel == 2
        assert e.description == "comet" and e.heading_deg == 90.0
        assert np.allclose(back.formations[2].yaw, 90.0)


# ==========================================================================
# Interop
# ==========================================================================


class TestVviz:
    def test_frame_conversion_round_trips(self):
        p = np.array([[3.0, 5.0, 7.0], [-1.0, 2.0, 0.5]])
        assert np.allclose(interop.ogl_to_enu(interop.enu_to_ogl(p)), p)

    def test_up_axis_is_y_in_ogl(self):
        """Getting this wrong lays the whole show on its side."""
        out = interop.enu_to_ogl(np.array([0.0, 0.0, 50.0]))
        assert out[1] == pytest.approx(50.0)
        assert out[0] == pytest.approx(0.0)

    def test_north_maps_to_negative_z(self):
        """With the audience to the south, +z is toward them."""
        out = interop.enu_to_ogl(np.array([0.0, 10.0, 0.0]))
        assert out[2] == pytest.approx(-10.0)

    def test_export_structure(self, tmp_path):
        show = planning.prepare(base_show(n=10)).show
        body = json.loads(interop.export_vviz(show, tmp_path / "s.vviz").read_text())
        assert body["version"] == "1.0"
        assert body["coordinateFrame"] == "ogl"
        assert body["globalReferenceFrame"]["lat"] == pytest.approx(12.9716)
        assert len(body["performances"]) == 10
        agent = body["performances"][0]["agentDescription"]
        assert {"homeX", "homeY", "homeZ", "airframe", "agentTraversal"} <= set(agent)
        assert len(agent["agentTraversal"]) > 1

    def test_traversal_deltas_reconstruct_the_path(self):
        show = planning.prepare(base_show(n=6)).show
        timeline = build_timeline(show, fps=interop.VVIZ_POSITION_RATE)
        body = json.loads(
            interop.export_vviz(
                show, "shows/_vviz_tmp.vviz", timeline=timeline
            ).read_text()
        )
        agent = body["performances"][0]
        home = np.array([agent["agentDescription"][k]
                         for k in ("homeX", "homeY", "homeZ")])
        # Deltas are cumulative from home; the first is zero by construction.
        pos = home.copy()
        track = [pos.copy()]
        for step in agent["agentDescription"]["agentTraversal"][1:]:
            pos = pos + np.array([step["dx"], step["dy"], step["dz"]])
            track.append(pos.copy())
        rebuilt = interop.ogl_to_enu(np.array(track))
        want = np.stack([
            timeline.sample(float(t))[0][0]
            for t in np.arange(0, len(track)) / interop.VVIZ_POSITION_RATE
        ])
        assert np.allclose(rebuilt, want[: len(rebuilt)], atol=0.01)
        import os

        os.remove("shows/_vviz_tmp.vviz")

    def test_light_run_length_covers_every_frame(self):
        """Frame counts must sum to the track length or colour drifts."""
        show = planning.prepare(base_show(n=4)).show
        body = json.loads(
            interop.export_vviz(show, "shows/_vviz_rle.vviz").read_text()
        )
        actions = body["performances"][0]["payloadDescription"][0]["payloadActions"]
        total = sum(a.get("frames", 1) for a in actions)
        expected = int(show.duration * interop.VVIZ_COLOR_RATE) + 1
        assert total == expected
        import os

        os.remove("shows/_vviz_rle.vviz")

    def test_pyro_only_on_firing_drones(self, tmp_path):
        show = planning.prepare(base_show(n=10, yaw=90.0)).show
        show.formations[3].pyro = [
            PyroEvent(at=2.0, drones=[0, 4], heading_deg=90.0, description="gerb")
        ]
        body = json.loads(interop.export_vviz(show, tmp_path / "p.vviz").read_text())
        firing = {
            p["id"] for p in body["performances"]
            if any(x["type"] == "Pyro" for x in p["payloadDescription"])
        }
        assert firing == {0, 4}
        event = next(
            x for x in body["performances"][0]["payloadDescription"]
            if x["type"] == "Pyro"
        )
        assert event["vdl"] == "gerb"
        assert event["eventTime"] > 0


class TestCsv:
    def test_export_then_import_round_trips(self, tmp_path):
        show = planning.prepare(base_show(n=8)).show
        files = interop.export_csv(show, tmp_path / "csv", fps=5.0)
        assert len(files) == 8

        imported = interop.import_csv(tmp_path / "csv", limits=show.limits,
                                     origin=show.origin)
        assert imported.drone_count == 8
        assert imported.duration == pytest.approx(show.duration, abs=0.25)

        timeline = build_timeline(show)
        for t in np.linspace(0, imported.duration, 20):
            want = timeline.sample(float(t))[0]
            got = imported.timeline.sample(float(t))[0]
            assert np.allclose(want, got, atol=0.3)

    def test_imported_show_is_validated(self, tmp_path):
        show = planning.prepare(base_show(n=8)).show
        interop.export_csv(show, tmp_path / "dense", fps=20.0)
        report = interop.validate_imported(
            interop.import_csv(tmp_path / "dense", limits=show.limits)
        )
        assert report["passed"], [v["message"] for v in report["violations"]]
        assert report["metrics"]["min_separation_m"] >= 3.0
        assert "design-time collision guarantees" in report["coverage"]

    def test_sparse_sampling_cannot_prove_a_tight_clearance(self, tmp_path):
        """The interchange rate limits what the sweep can prove, and it says so.

        A prepared show clears the minimum by about 17 mm - the sqrt(2)
        bound is tight. Exported at 5 Hz, the chord-versus-curve uncertainty
        between samples is around 19 mm, which is more than the headroom, so
        the sweep can no longer demonstrate clearance and reports the pair
        rather than assuming it is fine. Being conservative about a
        trajectory whose origin is unknown is the correct behaviour.
        """
        show = planning.prepare(base_show(n=8)).show
        interop.export_csv(show, tmp_path / "sparse", fps=5.0)
        report = interop.validate_imported(
            interop.import_csv(tmp_path / "sparse", limits=show.limits)
        )
        breaches = [v for v in report["violations"] if v["kind"] == "separation"]
        assert breaches, "a 5 Hz export of a 17 mm-headroom show should be flagged"
        assert "curvature margin" in breaches[0]["message"]
        # And the measured distance is still the true one, above the limit:
        # it is the uncertainty that fails, not the geometry.
        assert report["metrics"]["min_separation_m"] > 3.0

    def test_imported_collision_is_caught(self, tmp_path):
        """The reason the import path exists: check somebody else's show."""
        folder = tmp_path / "bad"
        folder.mkdir()
        for i, (x0, x1) in enumerate(((-5.0, 5.0), (5.0, -5.0))):
            (folder / f"drone_{i:03d}.csv").write_text(
                "Time [msec],x [m],y [m],z [m],Red,Green,Blue\n"
                f"0,{x0},0,50,255,255,255\n"
                f"1000,{x1},0,50,255,255,255\n",
                encoding="utf-8",
            )
        report = interop.validate_imported(interop.import_csv(folder))
        assert not report["passed"]
        assert report["metrics"]["min_separation_m"] < 0.5, (
            "the mid-interval crossing was missed"
        )

    def test_mismatched_time_bases_are_refused(self, tmp_path):
        folder = tmp_path / "skew"
        folder.mkdir()
        (folder / "a.csv").write_text(
            "Time [msec],x [m],y [m],z [m]\n0,0,0,50\n1000,10,0,50\n", encoding="utf-8")
        (folder / "b.csv").write_text(
            "Time [msec],x [m],y [m],z [m]\n0,20,0,50\n2000,30,0,50\n", encoding="utf-8")
        with pytest.raises(ValueError, match="different time base"):
            interop.import_csv(folder)

    def test_empty_directory_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="no CSV files"):
            interop.import_csv(tmp_path)

    def test_sparse_sampling_is_flagged(self, tmp_path):
        folder = tmp_path / "sparse"
        folder.mkdir()
        for i in range(2):
            rows = "\n".join(
                f"{t * 1000},{i * 20},0,50,255,255,255" for t in range(0, 10, 2)
            )
            (folder / f"d{i}.csv").write_text(
                "Time [msec],x [m],y [m],z [m],Red,Green,Blue\n" + rows, encoding="utf-8")
        imported = interop.import_csv(folder)
        assert any("denser export" in n for n in imported.notes)


# ==========================================================================
# Show adaptation and geofence
# ==========================================================================


class TestAdapt:
    def test_adapt_down_produces_a_valid_show(self):
        show = base_show(n=40)
        adapted = planning.adapt(show, 28)
        assert adapted.show.drone_count == 28
        for f in adapted.show.formations:
            assert f.count == 28
        report = safety.validate(adapted.show)
        assert report.passed, report.summary()
        assert report.metrics["min_separation_m"] >= 3.0

    def test_adapt_keeps_the_silhouette(self):
        """Taking the first N points would delete one side of the shape."""
        pts = geometry.lift(geometry.disc(200, 40), 60.0)
        keep = planning._farthest_point_subset(pts, 60)
        kept = pts[keep]
        # The subset must still span nearly the full extent of the original.
        for axis in (0, 2):
            assert np.ptp(kept[:, axis]) > 0.9 * np.ptp(pts[:, axis])
        naive = pts[:60]
        assert np.ptp(kept[:, 0]) > np.ptp(naive[:, 0])

    def test_adapt_up_is_refused(self):
        with pytest.raises(ValueError, match="cannot adapt"):
            planning.adapt(base_show(n=10), 20)

    def test_adapt_renumbers_pyro(self):
        show = base_show(n=30, yaw=90.0)
        show.formations[2].pyro = [
            PyroEvent(at=2.0, drones=[0, 1, 2], heading_deg=90.0)
        ]
        adapted = planning.adapt(show, 20)
        for f in adapted.show.formations:
            for event in f.pyro:
                assert all(0 <= d < 20 for d in event.drones), (
                    "a pyro event still points at a drone that no longer exists"
                )

    def test_adapt_keeps_fleetwide_pyro(self):
        show = base_show(n=30, yaw=90.0)
        show.formations[2].pyro = [PyroEvent(at=2.0, drones=[], heading_deg=90.0)]
        adapted = planning.adapt(show, 20)
        assert any(f.pyro for f in adapted.show.formations)

    def test_adapt_records_what_it_did(self):
        adapted = planning.adapt(base_show(n=30), 22)
        assert "adapted from 30 to 22" in adapted.show.notes


class TestAutoGeofence:
    def test_hull_contains_the_show(self):
        show = planning.prepare(base_show(n=30)).show
        show.limits.geofence = planning.auto_geofence(show, margin=12.0)
        assert len(show.limits.geofence) >= 3
        assert "geofence" not in {v.kind for v in safety.validate(show).errors}

    def test_margin_widens_the_hull(self):
        show = planning.prepare(base_show(n=20)).show
        tight = np.array(planning.auto_geofence(show, margin=1.0))
        wide = np.array(planning.auto_geofence(show, margin=40.0))
        assert np.ptp(wide[:, 0]) > np.ptp(tight[:, 0]) + 50.0

    def test_vertex_count_is_capped(self):
        show = planning.prepare(base_show(n=60)).show
        assert len(planning.auto_geofence(show, simplify=8)) <= 8

    def test_empty_show_is_refused(self):
        with pytest.raises(ValueError, match="no formations"):
            planning.auto_geofence(Show(name="x", drone_count=4))


# ==========================================================================
# Operations
# ==========================================================================


def running_fleet(n=30, **fault_kwargs):
    clock = ManualClock()
    show = planning.prepare(base_show(n=n)).show
    fleet = FleetManager(
        show, driver=SimDriver(faults=SimFaults(**fault_kwargs)), clock=clock
    )
    fleet.validate()
    fleet.upload()
    fleet.arm()
    fleet.start(lead_in=0.0)
    for _ in range(500):
        fleet.tick()
        clock.advance(0.1)
    return fleet, clock


class TestSuspendResume:
    def test_suspend_freezes_the_clock_and_the_fleet(self):
        fleet, clock = running_fleet(position_noise=0.0)
        fleet.suspend()
        held = fleet.show_time
        before = [d.position.copy() for d in fleet.drones]
        for _ in range(100):
            fleet.tick()
            clock.advance(0.1)
        assert fleet.show_time == pytest.approx(held)
        moved = max(
            float(np.linalg.norm(d.position - b))
            for d, b in zip(fleet.drones, before)
        )
        assert moved < 0.1, f"a suspended fleet moved {moved:.2f} m"
        assert fleet.phase == Phase.PAUSED

    def test_resume_continues_from_where_it_stopped(self):
        fleet, clock = running_fleet(position_noise=0.0)
        fleet.suspend()
        held = fleet.show_time
        for _ in range(60):
            fleet.tick()
            clock.advance(0.1)
        fleet.unsuspend()
        for _ in range(30):
            fleet.tick()
            clock.advance(0.1)
        assert fleet.show_time == pytest.approx(held + 3.0, abs=0.15)
        assert fleet.phase == Phase.RUNNING

    def test_suspend_requires_a_running_show(self):
        show = planning.prepare(base_show(n=8)).show
        fleet = FleetManager(show, clock=ManualClock())
        with pytest.raises(RuntimeError, match="cannot suspend"):
            fleet.suspend()


class TestHoldRelease:
    def test_a_latched_hold_is_released_on_resume(self):
        """Regression: suspend latched the driver and nothing cleared it.

        Every later command - resume, collective RTH - was accepted, logged
        and reported as working while the fleet sat motionless. An abort
        path that silently does nothing is worse than one that errors.
        """
        fleet, clock = running_fleet(position_noise=0.0)
        fleet.suspend()
        assert fleet.driver._emergency, "the test needs the hold to actually latch"
        fleet.unsuspend()
        assert not fleet.driver._emergency, "the hold was never released"

        start = [d.position.copy() for d in fleet.drones]
        for _ in range(120):
            fleet.tick()
            clock.advance(0.1)
        moved = max(
            float(np.linalg.norm(d.position - b))
            for d, b in zip(fleet.drones, start)
        )
        assert moved > 1.0, "the fleet did not move after resuming"

    def test_release_hold_restores_show_mode(self):
        driver = SimDriver()
        driver.connect(4)
        driver.arm()
        driver.emergency("hold")
        assert all(d.mode == "BRAKE" for d in driver.poll())
        driver.release_hold()
        assert all(d.mode == "SHOW" for d in driver.poll())


class TestCollectiveRth:
    def test_returns_the_fleet_on_one_verified_plan(self):
        fleet, clock = running_fleet(n=30, position_noise=0.02)
        report = fleet.collective_rth(reason="test")
        assert report["executed"]
        assert report["breaches"] == 0
        assert report["min_separation_m"] >= report["min_separation_limit_m"]
        assert report["positions_assumed"] == 0

        steps = 0
        while fleet.phase == Phase.RUNNING and steps < 4000:
            fleet.tick()
            clock.advance(0.1)
            steps += 1
        assert fleet.phase == Phase.FINISHED
        assert max(d.altitude_agl for d in fleet.drones) < 0.5, (
            "the fleet did not actually land"
        )

    def test_the_return_is_verified_not_assumed(self):
        """The plan is swept before it flies, like any other show."""
        fleet, _ = running_fleet(n=20)
        report = fleet.collective_rth()
        assert "min_separation_m" in report
        timeline = build_timeline(fleet.show)
        violations, worst = safety.check_separation(
            timeline, fleet.show.limits.min_separation,
            ignore_below_altitude=fleet.show.limits.min_nav_altitude,
        )
        assert not violations
        assert worst == pytest.approx(report["min_separation_m"], abs=0.01)

    def test_unseen_drones_are_assumed_to_be_on_plan_and_counted(self):
        fleet, clock = running_fleet(n=20, link_loss={3: 5.0})
        report = fleet.collective_rth()
        assert report["positions_assumed"] >= 1, (
            "a drone with no telemetry must be counted as an assumption, "
            "not silently treated as known"
        )

    def test_it_holds_rather_than_flying_an_unsafe_return(self):
        """Refusing is the correct outcome, and must be reported as one."""
        fleet, _ = running_fleet(n=20)
        # Force an impossible return by demanding separation the fleet's
        # current positions cannot support.
        tight = ShowLimits(
            min_separation=60.0,
            formation_separation=85.0,
            max_altitude=400.0,
            battery_endurance_s=2400.0,
        )
        report = fleet.collective_rth(limits=tight)
        assert not report["executed"]
        assert report["fallback"] == "hold"
        assert report["breaches"] > 0
        assert "holding position" in report["detail"]
        assert fleet.phase == Phase.PAUSED

    def test_it_is_logged(self):
        fleet, _ = running_fleet(n=16)
        fleet.collective_rth(reason="audit")
        entries = [e for e in fleet.events if e.kind == "collective_rth"]
        assert entries and entries[-1].data["reason"] == "audit"
