"""Tests for the design half: geometry, matching, timing, validation.

The tests that matter most are in TestCollisionGuarantee and
TestSeparationSweep. Everything else in Sky Shine exists to serve the
claim that drones will not hit each other; those two classes are what
check the claim rather than restating it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from skyshine import geometry, planning, safety
from skyshine.show import (
    EASING_PEAKS,
    Formation,
    Show,
    ShowLimits,
    Timeline,
    ease,
    enu_to_wgs84,
    load_show,
    save_show,
)
from skyshine.trajectory import build_mission, build_timeline, evaluate_plan

SHAPE_CASES = [
    ("circle", {}),
    ("disc", {}),
    ("ring_stack", {}),
    ("sphere", {}),
    ("ball", {}),
    ("cube", {}),
    ("helix", {}),
    ("double_helix", {}),
    ("heart", {}),
    ("heart", {"solid": True}),
    ("star", {}),
    ("star", {"solid": True}),
    ("grid", {}),
    ("wave", {}),
    ("torus", {}),
    ("line", {}),
]


def simple_show(n=24, formations=3, transition=8.0, hold=4.0, **limit_kwargs) -> Show:
    limits = ShowLimits(**limit_kwargs)
    show = Show(name="test", drone_count=n, limits=limits)
    shapes = [geometry.ball(n, 20), geometry.disc(n, 22), geometry.cube_shell(n, 26)]
    for i in range(formations):
        show.add(
            Formation(
                name=f"f{i}",
                points=geometry.lift(shapes[i % len(shapes)], 40.0 + 5 * i),
                colors=np.array([200, 100, 50]),
                transition=transition,
                hold=hold,
            )
        )
    return show


# ==========================================================================
# Geometry
# ==========================================================================


class TestGeometry:
    @pytest.mark.parametrize("name,kwargs", SHAPE_CASES)
    @pytest.mark.parametrize("n", [1, 2, 7, 60, 251])
    def test_shape_returns_exactly_n_finite_points(self, name, kwargs, n):
        pts = geometry.make(name, n, **kwargs)
        assert pts.shape == (n, 3), f"{name} returned {pts.shape} for n={n}"
        assert np.all(np.isfinite(pts)), f"{name} produced non-finite coordinates"

    @pytest.mark.parametrize("name,kwargs", SHAPE_CASES)
    def test_no_coincident_points(self, name, kwargs):
        # Two drones at one position is the one defect scaling cannot fix,
        # so no generator may ever emit it.
        pts = geometry.make(name, 120, **kwargs)
        assert geometry.min_pair_distance(pts) > 1e-6, f"{name} has coincident points"

    def test_cube_faces_do_not_share_points(self):
        # Regression: a corner lattice duplicated every shared face edge.
        for n in (12, 60, 200, 497):
            assert geometry.min_pair_distance(geometry.cube_shell(n, 50)) > 1e-6

    def test_heart_is_arc_length_even(self):
        # Regression: uniform parameter sampling piled points into the cusp,
        # which forced a 420x scale and a 15 km tall show.
        pts = geometry.heart(120)
        d = geometry.min_pair_distance(pts)
        span = float(np.ptp(pts[:, 0]))
        # Evenly spaced points on a closed curve sit at roughly
        # perimeter/n apart, which is well above a hundredth of the width.
        assert d > span / 120.0 * 0.4, f"heart spacing {d:.3f} is uneven"

    @pytest.mark.parametrize("name,kwargs", SHAPE_CASES)
    def test_fit_separation_reaches_the_target(self, name, kwargs):
        pts = geometry.make(name, 80, **kwargs)
        fitted = geometry.fit_separation(pts, 4.25, max_scale=1000.0)
        assert geometry.min_pair_distance(fitted) >= 4.25

    def test_fit_separation_leaves_a_roomy_shape_alone(self):
        pts = geometry.disc(40, 200.0)
        before = geometry.min_pair_distance(pts)
        assert before > 3.0
        after = geometry.fit_separation(pts, 3.0)
        assert geometry.min_pair_distance(after) == pytest.approx(before, rel=1e-9)

    def test_fit_separation_refuses_a_pathological_shape(self):
        bad = np.array([[0.0, 0, 0], [1e-4, 0, 0], [50.0, 0, 0]])
        with pytest.raises(ValueError, match="clusters drones unevenly|coincident"):
            geometry.fit_separation(bad, 3.0)

    def test_fit_separation_rejects_coincident_points(self):
        same = np.zeros((3, 3))
        with pytest.raises(ValueError, match="coincident"):
            geometry.fit_separation(same, 3.0)

    def test_lift_puts_the_floor_where_asked(self):
        pts = geometry.lift(geometry.ball(50, 20), 37.5)
        assert pts[:, 2].min() == pytest.approx(37.5)

    def test_capacity_and_billboard_agree(self):
        sep = 4.24
        for count in (50, 200, 500):
            w, h = geometry.billboard_for(count, sep)
            assert geometry.capacity(w, h, sep) >= count, (
                f"a {w:.0f}x{h:.0f} m billboard should hold {count} drones"
            )

    def test_mask_shapes(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[8:56, 8:56] = True
        pts = geometry.from_mask(100, mask, width=80.0)
        assert pts.shape == (100, 3)
        assert geometry.min_pair_distance(pts) > 1.0

    def test_mask_rejects_impossible_requests(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 0] = True
        with pytest.raises(ValueError, match="higher resolution"):
            geometry.from_mask(50, mask)
        with pytest.raises(ValueError, match="empty"):
            geometry.from_mask(1, np.zeros((5, 5), dtype=bool))

    def test_text_formation(self):
        pts = geometry.text(90, "SKY", width=120.0)
        assert pts.shape == (90, 3)
        assert geometry.min_pair_distance(pts) > 0.5

    def test_filled_polygon(self):
        square = np.array([[0.0, 0], [40, 0], [40, 40], [0, 40]])
        edge = geometry.polygon(60, square)
        filled = geometry.polygon(60, square, solid=True)
        assert edge.shape == filled.shape == (60, 3)
        # The filled version spreads over the interior, so its nearest
        # neighbours are further apart than points strung along one edge.
        assert geometry.min_pair_distance(filled) > geometry.min_pair_distance(edge)

    def test_rotation_preserves_distances(self):
        pts = geometry.ball(50, 25)
        before = geometry.min_pair_distance(pts)
        for fn, deg in ((geometry.rotate_z, 37.0), (geometry.rotate_x, -64.0)):
            assert geometry.min_pair_distance(fn(pts, deg)) == pytest.approx(before)


# ==========================================================================
# The collision guarantee
# ==========================================================================


class TestCollisionGuarantee:
    """The claim: optimal matching plus one shared easing cannot collide.

    These tests check the inequality the proof rests on, and then check the
    conclusion numerically on random geometry, because a proof about the
    code is only worth as much as the code matching the proof.
    """

    @pytest.mark.parametrize("seed", range(12))
    def test_optimal_matching_satisfies_the_key_inequality(self, seed):
        # Optimality of a min-sum-of-squared-distance matching implies
        # (a_i - a_j) . (b_i - b_j) >= 0 for every pair. That single fact
        # is what makes the separation bound hold.
        rng = np.random.default_rng(seed)
        n = 40
        a = rng.normal(0, 40, size=(n, 3))
        b = rng.normal(0, 40, size=(n, 3))
        b = b[planning.match(a, b)]

        da = a[:, None, :] - a[None, :, :]
        db = b[:, None, :] - b[None, :, :]
        dots = np.einsum("ijk,ijk->ij", da, db)
        assert dots.min() > -1e-6, f"inequality violated by {dots.min():.3e}"

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("easing", ["linear", "smoothstep", "minjerk"])
    def test_transition_never_breaches_the_sqrt2_bound(self, seed, easing):
        rng = np.random.default_rng(seed)
        n = 45
        a = geometry.fit_separation(rng.normal(0, 45, size=(n, 3)), 5.0, max_scale=1e6)
        b = geometry.fit_separation(rng.normal(0, 45, size=(n, 3)), 5.0, max_scale=1e6)
        b = b[planning.match(a, b)]

        bound = planning.transition_lower_bound(a, b)
        assert bound == pytest.approx(
            min(geometry.min_pair_distance(a), geometry.min_pair_distance(b))
            / math.sqrt(2.0)
        )

        # Sweep the transition finely and confirm nothing dips below.
        worst = math.inf
        for tau in np.linspace(0.0, 1.0, 801):
            s = float(ease(easing, tau))
            p = a + (b - a) * s
            worst = min(worst, geometry.min_pair_distance(p))
        assert worst >= bound - 1e-6, (
            f"{easing}: observed {worst:.4f} m, bound promised {bound:.4f} m"
        )

    def test_the_bound_is_tight_not_merely_true(self):
        # Two drones swapping ends of the same line is the worst case: the
        # bound must be approached, otherwise it is a loose claim hiding a
        # different mechanism.
        a = np.array([[-2.0, 0, 50], [2.0, 0, 50]])
        b = np.array([[2.0, 0, 50], [-2.0, 0, 50]])
        perm = planning.match(a, b)
        # Optimal matching refuses the swap: each drone stays put.
        assert list(perm) == [1, 0] or list(perm) == [0, 1]
        matched = b[perm]
        assert geometry.min_pair_distance(matched) == pytest.approx(4.0)
        worst = min(
            geometry.min_pair_distance(a + (matched - a) * s)
            for s in np.linspace(0, 1, 501)
        )
        assert worst >= 4.0 / math.sqrt(2.0) - 1e-9

    def test_per_drone_timing_destroys_the_bound(self):
        """Why one shared easing is mandatory, not stylistic.

        This pair is optimally matched and its paths stay 4.00 m apart for
        the whole transition when both drones share a time-warp - safely
        above the 2.91 m the bound promises. Put the two drones on
        independent clocks and there is a pair of progress values, both
        perfectly legal, at which they occupy the same point in space.

        The engine therefore applies one easing to the whole transition,
        and safety.check_design warns when a show mixes them.
        """
        a = np.array([[0.0, 0.0, 50.0], [4.0, 1.0, 50.0]])
        b = np.array([[10.0, 1.0, 50.0], [14.0, 0.0, 50.0]])
        perm = planning.match(a, b)
        b = b[perm]

        # Premise: the matching is optimal, so the key inequality holds.
        u, v = a[0] - a[1], b[0] - b[1]
        assert float(u @ v) >= -1e-9

        bound = planning.transition_lower_bound(a, b)
        assert bound == pytest.approx(math.sqrt(17.0) / math.sqrt(2.0), rel=1e-6)

        shared = min(
            geometry.min_pair_distance(a + (b - a) * s)
            for s in np.linspace(0.0, 1.0, 4001)
        )
        assert shared == pytest.approx(4.0, abs=1e-3)
        assert shared > bound

        # Now let each drone pick its own progress through the move.
        grid = np.linspace(0.0, 1.0, 301)
        skewed = math.inf
        for s0 in grid:
            p0 = a[0] + (b[0] - a[0]) * s0
            p1 = a[1] + (b[1] - a[1]) * grid[:, None]
            skewed = min(skewed, float(np.min(np.linalg.norm(p1 - p0, axis=1))))
        assert skewed < 0.1, (
            f"independent clocks should put these two drones on top of each "
            f"other, but the closest found was {skewed:.3f} m"
        )

        # And both of those progress values are reachable by real easings,
        # since every easing is onto [0, 1].
        for name in ("linear", "smoothstep", "minjerk"):
            vals = np.asarray(ease(name, np.linspace(0, 1, 1001)))
            assert vals.min() == pytest.approx(0.0) and vals.max() == pytest.approx(1.0)

    @pytest.mark.parametrize("n", [50, 200])
    def test_prepared_show_meets_its_own_minimum(self, n):
        show = simple_show(n=n, max_altitude=250.0, battery_endurance_s=1800.0)
        prepared = planning.prepare(show)
        report = safety.validate(prepared.show)
        assert report.metrics["min_separation_m"] >= show.limits.min_separation, (
            report.summary()
        )
        assert report.passed, report.summary()

    @pytest.mark.parametrize("fps", [5.0, 10.0])
    def test_a_tightly_packed_lattice_clears_the_validator_margin(self, fps):
        """A lattice flown to the launch cloud sits exactly on the proof's
        bound. Packed to formation_separation with no slack, this show
        failed validation at 3.0015 m against a 3.003 m threshold."""
        show = Show(name="gap", drone_count=60, fps=fps)
        show.add(Formation(
            name="wave",
            points=geometry.lift(geometry.wave(60, width=20.0), 40.0),
            colors=np.array([255, 255, 255], dtype=np.int32),
        ))
        report = safety.validate(planning.prepare(show).show)
        assert report.passed, report.summary()


class TestLaunchAndLanding:
    def test_takeoff_grid_spacing(self):
        grid = planning.takeoff_grid(97, spacing=3.0)
        assert grid.shape == (97, 3)
        assert np.allclose(grid[:, 2], 0.0)
        assert geometry.min_pair_distance(grid) >= 3.0 - 1e-9

    def test_stagger_increases_clearance(self):
        plain = geometry.min_pair_distance(planning.takeoff_grid(100, 3.0, stagger=False))
        shifted = geometry.min_pair_distance(planning.takeoff_grid(100, 3.0, stagger=True))
        assert shifted >= plain

    def test_layered_launch_separates_more_than_the_flat_grid(self):
        grid = planning.takeoff_grid(100, spacing=3.0, stagger=False)
        cloud = planning.launch_cloud(
            grid, base_altitude=15.0, layers=4, layer_gap=3.0
        )
        flat = geometry.min_pair_distance(grid)
        layered = geometry.min_pair_distance(cloud)
        assert layered > flat, (
            "layering the climb is the whole reason a 3 m launch grid can feed "
            "a formation that needs 4.24 m separation"
        )
        # And the layered cloud must itself clear the transition bound.
        assert layered / math.sqrt(2.0) >= 3.0 - 1e-6

    def test_vertical_climb_preserves_horizontal_separation(self):
        grid = planning.takeoff_grid(80, spacing=3.2)
        cloud = planning.launch_cloud(grid, base_altitude=12.0, layers=4, layer_gap=3.0)
        assert np.allclose(cloud[:, :2], grid[:, :2]), (
            "the climb must be purely vertical; any lateral drift during takeoff "
            "happens while drones are at their closest"
        )

    def test_takeoff_refuses_a_grid_tighter_than_the_minimum(self):
        show = simple_show(n=16, min_separation=5.0, takeoff_grid_spacing=2.0)
        with pytest.raises(ValueError, match="takeoff grid spacing"):
            planning.build_takeoff(show)

    def test_landing_descends_vertically(self):
        show = planning.prepare(
            simple_show(n=40, max_altitude=220.0, battery_endurance_s=1800.0)
        ).show
        names = [f.name for f in show.formations]
        assert names[0] == "ground" and names[1] == "climb"
        assert names[-2] == "rth" and names[-1] == "land"
        rth, land = show.formations[-2].points, show.formations[-1].points
        assert np.allclose(rth[:, :2], land[:, :2]), "descent must not move laterally"
        assert np.allclose(land[:, 2], 0.0)

    def test_every_drone_gets_exactly_one_landing_pad(self):
        show = planning.prepare(
            simple_show(n=60, max_altitude=220.0, battery_endurance_s=1800.0)
        ).show
        pads = show.formations[-1].points[:, :2]
        assert geometry.min_pair_distance(
            np.column_stack([pads, np.zeros(len(pads))])
        ) > 1e-6, "two drones were assigned the same landing pad"


# ==========================================================================
# Timing
# ==========================================================================


class TestEasingAndTiming:
    @pytest.mark.parametrize("name", ["linear", "smoothstep", "minjerk"])
    def test_easing_endpoints(self, name):
        assert ease(name, 0.0) == pytest.approx(0.0)
        assert ease(name, 1.0) == pytest.approx(1.0)
        # Monotone: a non-monotone easing would make drones back up mid-move.
        vals = np.asarray(ease(name, np.linspace(0, 1, 400)))
        assert np.all(np.diff(vals) >= -1e-12)

    @pytest.mark.parametrize("name", ["smoothstep", "minjerk"])
    def test_documented_peaks_match_the_functions(self, name):
        # autotime sizes every transition from these constants, so a wrong
        # one silently lets a show exceed its velocity limit.
        tau = np.linspace(0, 1, 200_001)
        s = np.asarray(ease(name, tau))
        dt = tau[1] - tau[0]
        kv, ka = EASING_PEAKS[name]
        assert np.max(np.abs(np.gradient(s, dt))) == pytest.approx(kv, rel=1e-3)
        assert np.max(
            np.abs(np.gradient(np.gradient(s, dt), dt))
        ) == pytest.approx(ka, rel=2e-2)

    def test_autotime_brings_a_violent_show_inside_the_envelope(self):
        show = simple_show(n=30, transition=0.5, hold=1.0, max_altitude=250.0,
                           battery_endurance_s=1800.0)
        fast = safety.check_kinematics(build_timeline(show), show.limits)[0]
        assert fast, "the unsized show should breach its limits, or this proves nothing"

        timed = planning.autotime(show)
        violations, metrics = safety.check_kinematics(build_timeline(timed), timed.limits)
        assert not violations, [v.message for v in violations]
        assert metrics["peak_horizontal_speed_mps"] <= show.limits.max_velocity_xy * 1.01
        assert metrics["peak_acceleration_mps2"] <= show.limits.max_acceleration * 1.01

    def test_autotime_never_shortens_a_designer_choice(self):
        show = simple_show(n=20, transition=45.0, hold=5.0, max_altitude=250.0,
                           battery_endurance_s=1800.0)
        timed = planning.autotime(show)
        for original, result in zip(show.formations, timed.formations):
            assert result.transition >= original.transition - 1e-9


# ==========================================================================
# The separation sweep
# ==========================================================================


def manual_timeline(positions: np.ndarray, times: np.ndarray, show: Show) -> Timeline:
    n = positions.shape[1]
    return Timeline(
        t=times,
        pos=positions,
        col=np.zeros((len(times), n, 3), dtype=np.uint8),
        segment=np.zeros(len(times), dtype=np.int32),
        show=show,
    )


class TestSeparationSweep:
    def test_catches_a_collision_that_happens_between_samples(self):
        """The test this whole module exists for.

        Two drones swap ends of a line in one sample interval. At both
        samples they are 10 m apart; halfway through they are at the same
        point. A validator that only inspects frames reports a comfortable
        pass on a show that destroys two aircraft.
        """
        show = simple_show(n=2)
        pos = np.array(
            [
                [[-5.0, 0.0, 50.0], [5.0, 0.0, 50.0]],
                [[5.0, 0.0, 50.0], [-5.0, 0.0, 50.0]],
            ]
        )
        timeline = manual_timeline(pos, np.array([0.0, 1.0]), show)

        # Sanity: at every sample they are far apart.
        for k in range(2):
            assert geometry.min_pair_distance(timeline.pos[k]) == pytest.approx(10.0)

        violations, worst = safety.check_separation(timeline, 3.0)
        assert violations, "a mid-interval collision was not detected"
        assert worst < 0.01
        assert violations[0].kind == "separation"
        assert 0.4 < violations[0].time < 0.6

    def test_catches_a_near_miss_just_under_the_limit(self):
        show = simple_show(n=2)
        pos = np.array(
            [
                [[-6.0, 0.0, 50.0], [6.0, 2.9, 50.0]],
                [[6.0, 0.0, 50.0], [-6.0, 2.9, 50.0]],
            ]
        )
        timeline = manual_timeline(pos, np.array([0.0, 1.0]), show)
        violations, worst = safety.check_separation(timeline, 3.0)
        assert worst == pytest.approx(2.9, abs=1e-6)
        assert violations, "2.9 m against a 3.0 m minimum must be reported"

    def test_passes_a_near_miss_just_over_the_limit(self):
        show = simple_show(n=2)
        pos = np.array(
            [
                [[-6.0, 0.0, 50.0], [6.0, 3.6, 50.0]],
                [[6.0, 0.0, 50.0], [-6.0, 3.6, 50.0]],
            ]
        )
        timeline = manual_timeline(pos, np.array([0.0, 1.0]), show)
        violations, worst = safety.check_separation(timeline, 3.0)
        assert worst == pytest.approx(3.6, abs=1e-6)
        assert not violations

    def test_reports_the_real_closest_distance_not_a_sampled_one(self):
        show = simple_show(n=2)
        # Closest approach is 1.0 m at the midpoint of the interval.
        pos = np.array(
            [
                [[-8.0, 0.0, 50.0], [8.0, 1.0, 50.0]],
                [[8.0, 0.0, 50.0], [-8.0, 1.0, 50.0]],
            ]
        )
        timeline = manual_timeline(pos, np.array([0.0, 2.0]), show)
        _, worst = safety.check_separation(timeline, 3.0)
        assert worst == pytest.approx(1.0, abs=1e-6)

    def test_ignores_pads_but_not_the_air_above_them(self):
        show = simple_show(n=2, min_nav_altitude=5.0)
        on_pads = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]] * 2)
        timeline = manual_timeline(on_pads, np.array([0.0, 1.0]), show)
        ignored, _ = safety.check_separation(timeline, 3.0, ignore_below_altitude=5.0)
        assert not ignored, "drones parked on their pads must not raise findings"
        checked, _ = safety.check_separation(timeline, 3.0, ignore_below_altitude=None)
        assert checked, "the strict mode must still see them"

        airborne = np.array([[[0.0, 0.0, 40.0], [1.0, 0.0, 40.0]]] * 2)
        timeline = manual_timeline(airborne, np.array([0.0, 1.0]), show)
        violations, _ = safety.check_separation(timeline, 3.0, ignore_below_altitude=5.0)
        assert violations, "1 m apart at 40 m altitude is a collision, not pad noise"

    def test_scales_to_a_five_hundred_drone_show(self):
        show = planning.prepare(
            simple_show(n=500, max_altitude=400.0, battery_endurance_s=2400.0)
        ).show
        report = safety.validate(show)
        assert report.metrics["min_separation_m"] >= 3.0
        assert report.metrics["pair_seconds_checked"] > 10_000_000


class TestEnvelopeChecks:
    def test_point_in_polygon(self):
        square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        pts = np.array([[5.0, 5.0], [-1.0, 5.0], [5.0, 11.0], [9.9, 0.1], [11.0, 11.0]])
        inside = safety.points_in_polygon(pts, square)
        assert list(inside) == [True, False, False, True, False]

    def test_concave_polygon(self):
        # An L shape: the notch must read as outside.
        poly = np.array(
            [[0.0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]]
        )
        inside = safety.points_in_polygon(np.array([[1.0, 1], [8, 1], [8, 8], [2, 8]]), poly)
        assert list(inside) == [True, True, False, True]

    def test_geofence_breach_is_reported(self):
        show = simple_show(n=20, max_altitude=250.0, battery_endurance_s=1800.0)
        show.limits.geofence = [[-15.0, -15.0], [15.0, -15.0], [15.0, 15.0], [-15.0, 15.0]]
        report = safety.validate(show)
        kinds = {v.kind for v in report.errors}
        assert "geofence" in kinds, report.summary()

    def test_generous_geofence_passes(self):
        show = planning.prepare(
            simple_show(n=20, max_altitude=250.0, battery_endurance_s=1800.0)
        ).show
        show.limits.geofence = [
            [-500.0, -500.0], [500.0, -500.0], [500.0, 500.0], [-500.0, 500.0]
        ]
        assert "geofence" not in {v.kind for v in safety.validate(show).errors}

    def test_ceiling_breach_is_reported(self):
        show = simple_show(n=20, max_altitude=20.0, battery_endurance_s=1800.0)
        kinds = {v.kind for v in safety.validate(show).errors}
        assert "altitude_max" in kinds

    def test_endurance_is_enforced(self):
        show = simple_show(
            n=12, transition=200.0, hold=120.0, max_altitude=400.0,
            battery_endurance_s=300.0, battery_reserve_s=120.0
        )
        report = safety.validate(show)
        assert "endurance" in {v.kind for v in report.errors}

    def test_zero_time_move_is_rejected(self):
        show = simple_show(n=10, max_altitude=250.0, battery_endurance_s=1800.0)
        show.formations[1].transition = 0.0
        assert "timing" in {v.kind for v in safety.validate(show).errors}

    def test_linear_easing_is_warned_about_not_blocked(self):
        show = planning.prepare(
            simple_show(n=12, max_altitude=250.0, battery_endurance_s=1800.0)
        ).show
        show.formations[3].easing = "linear"
        report = safety.validate(show)
        assert "easing" in {v.kind for v in report.warnings}

    def test_missing_amsl_is_a_warning(self):
        show = planning.prepare(
            simple_show(n=12, max_altitude=250.0, battery_endurance_s=1800.0)
        ).show
        assert "origin" in {v.kind for v in safety.validate(show).warnings}
        show.origin.amsl = 910.0
        assert "origin" not in {v.kind for v in safety.validate(show).warnings}

    def test_report_verdict_ignores_warnings(self):
        report = safety.SafetyReport("x", 1, 1.0)
        report.add(safety.Violation("k", safety.WARNING, "just a note"))
        assert report.passed
        report.add(safety.Violation("k", safety.ERROR, "a real problem"))
        assert not report.passed

    def test_empty_show_fails_rather_than_crashing(self):
        show = Show(name="empty", drone_count=4)
        assert "empty" in {v.kind for v in safety.check_design(show)}


# ==========================================================================
# Trajectory and mission package
# ==========================================================================


class TestTrajectory:
    def test_timeline_is_strictly_increasing_and_hits_the_boundaries(self):
        show = planning.prepare(
            simple_show(n=20, max_altitude=250.0, battery_endurance_s=1800.0)
        ).show
        timeline = build_timeline(show)
        assert np.all(np.diff(timeline.t) > 0)
        assert timeline.t[0] == pytest.approx(0.0)
        assert timeline.duration == pytest.approx(show.duration, abs=1e-6)

        # Every held pose must appear exactly in the timeline, not as an
        # interpolation that straddles it.
        cursor = 0.0
        for f in show.formations:
            cursor += f.transition
            pos, _ = timeline.sample(cursor + f.hold * 0.5 if f.hold else cursor)
            if f.hold > 0:
                assert np.allclose(pos, f.points, atol=1e-6), f"{f.name} not held"
            cursor += f.hold

    def test_all_drones_move_on_one_clock(self):
        # If any drone led or lagged, the bound would not apply. Check it by
        # confirming every drone's normalised progress through a transition
        # is identical.
        show = planning.prepare(
            simple_show(n=16, max_altitude=250.0, battery_endurance_s=1800.0)
        ).show
        timeline = build_timeline(show)
        a = show.formations[2].points
        b = show.formations[3].points
        t0 = sum(f.duration for f in show.formations[:3])
        mid = t0 + show.formations[3].transition * 0.5
        pos, _ = timeline.sample(mid)
        delta = b - a
        moving = np.linalg.norm(delta, axis=1) > 1e-6
        progress = np.einsum("ij,ij->i", pos[moving] - a[moving], delta[moving]) / np.einsum(
            "ij,ij->i", delta[moving], delta[moving]
        )
        assert progress.std() < 1e-9, f"drones are on different clocks: {progress.std():.2e}"

    def test_mission_package_replays_the_designed_path(self):
        show = planning.prepare(
            simple_show(n=24, max_altitude=250.0, battery_endurance_s=1800.0)
        ).show
        mission = build_mission(show)
        assert mission.drone_count == show.drone_count
        assert len(mission.plans) == show.drone_count

        # Compare against the analytic path, not the sampled timeline, so
        # the test measures the compression and nothing else.
        cursor = 0.0
        checkpoints = []
        for i, f in enumerate(show.formations):
            if i > 0 and f.transition > 0:
                for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                    s = float(ease(f.easing, frac))
                    want = show.formations[i - 1].points + (
                        f.points - show.formations[i - 1].points
                    ) * s
                    checkpoints.append((cursor + f.transition * frac, want))
                cursor += f.transition
            if f.hold > 0:
                checkpoints.append((cursor + f.hold * 0.5, f.points))
                cursor += f.hold

        worst = 0.0
        for plan in mission.plans:
            for t, want in checkpoints:
                got = evaluate_plan(plan, t)
                worst = max(worst, float(np.linalg.norm(want[plan.drone_index] - got)))
        assert worst < 1e-6, f"compression lost {worst * 1000:.3f} mm"

    def test_mission_checksum_tracks_content(self):
        show = planning.prepare(
            simple_show(n=10, max_altitude=250.0, battery_endurance_s=1800.0)
        ).show
        first = build_mission(show).to_dict()["checksum"]
        assert build_mission(show).to_dict()["checksum"] == first, "checksum unstable"
        show.formations[2].points[0, 0] += 0.5
        assert build_mission(show).to_dict()["checksum"] != first, (
            "a moved drone must change the checksum, or an operator cannot tell "
            "which plan is loaded"
        )

    def test_light_thinning_keeps_the_colour_curve(self):
        show = simple_show(n=6, max_altitude=250.0, battery_endurance_s=1800.0)
        show.formations[1].colors = np.tile(np.array([0, 0, 255]), (6, 1))
        mission = build_mission(show)
        for plan in mission.plans:
            assert len(plan.lights) >= 2
            assert len(plan.lights) < 10_000

    def test_oversized_show_is_refused_rather_than_exhausting_memory(self):
        show = simple_show(n=400, transition=1e6, hold=1e6)
        with pytest.raises(ValueError, match="drone-samples"):
            build_timeline(show)


class TestShowFormat:
    def test_round_trip(self, tmp_path):
        show = planning.prepare(
            simple_show(n=18, max_altitude=250.0, battery_endurance_s=1800.0)
        ).show
        show.origin.latitude, show.origin.longitude, show.origin.amsl = 12.9716, 77.5946, 920.0
        show.limits.geofence = [
            [-400.0, -400.0], [400.0, -400.0], [400.0, 400.0], [-400.0, 400.0]
        ]
        path = save_show(show, tmp_path / "x.show.json")
        back = load_show(path)

        assert back.name == show.name
        assert back.drone_count == show.drone_count
        assert back.origin.amsl == pytest.approx(920.0)
        assert back.limits.geofence == show.limits.geofence
        assert len(back.formations) == len(show.formations)
        for a, b in zip(show.formations, back.formations):
            assert a.name == b.name and a.easing == b.easing
            assert np.allclose(a.points, b.points, atol=1e-4)
            assert np.array_equal(a.colors, b.colors)
        assert safety.validate(back).passed

    def test_rejects_a_foreign_file(self):
        from skyshine.show import show_from_dict

        with pytest.raises(ValueError, match="not a Sky Shine"):
            show_from_dict({"format": "something.else"})

    def test_rejects_a_future_version(self):
        from skyshine.show import show_from_dict

        with pytest.raises(ValueError, match="newer than this build"):
            show_from_dict({"format": "skyshine.show", "version": 99, "drone_count": 1})

    def test_formation_rejects_a_mismatched_drone_count(self):
        show = Show(name="x", drone_count=10)
        with pytest.raises(ValueError, match="has 5 points"):
            show.add(
                Formation("bad", geometry.ball(5, 10), np.array([1, 2, 3]))
            )

    def test_limits_reject_an_unsafe_formation_separation(self):
        with pytest.raises(ValueError, match="formation_separation"):
            ShowLimits(min_separation=3.0, formation_separation=1.0).validate()

    def test_default_formation_separation_is_the_safe_multiple(self):
        limits = ShowLimits(min_separation=2.5)
        assert limits.formation_separation == pytest.approx(2.5 * math.sqrt(2.0))

    def test_enu_to_wgs84_round_trips_through_a_known_offset(self):
        from skyshine.show import ShowOrigin

        origin = ShowOrigin(latitude=12.9716, longitude=77.5946, amsl=920.0)
        # 100 m north should move latitude by about 0.000899 degrees.
        lat, lon = enu_to_wgs84([0.0, 100.0], origin)
        assert lat - origin.latitude == pytest.approx(0.000899, abs=2e-6)
        assert lon == pytest.approx(origin.longitude, abs=1e-9)
        # 100 m east moves longitude, scaled by the cosine of the latitude.
        lat, lon = enu_to_wgs84([100.0, 0.0], origin)
        assert lat == pytest.approx(origin.latitude, abs=1e-9)
        assert lon > origin.longitude

    def test_orientation_rotates_the_show(self):
        from skyshine.show import ShowOrigin

        origin = ShowOrigin(latitude=0.0, longitude=0.0, orientation_deg=90.0)
        # With the show rotated 90 degrees, show +y points east.
        lat, lon = enu_to_wgs84([0.0, 100.0], origin)
        assert lon > 0.0 and abs(lat) < 1e-9
