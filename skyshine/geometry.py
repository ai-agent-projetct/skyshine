"""Formation shape generators.

Every generator returns exactly n points as an (n, 3) array in show ENU
metres, roughly centred on the origin at z=0. Callers lift and scale them;
`fit_separation` is what makes a shape legal to fly.

Two families:
  parametric  closed-form point placement (circle, sphere, heart, ...)
  raster      sample n evenly spread points out of a 2D mask (text, image,
              arbitrary polygon), via Lloyd relaxation on the mask pixels
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------
# Separation fitting - the step that makes a pretty picture flyable
# --------------------------------------------------------------------------


def min_pair_distance(points: np.ndarray) -> float:
    """Smallest centre-to-centre distance in a point set. inf for n < 2."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return math.inf
    tree = cKDTree(p)
    # k=2 -> nearest neighbour other than self
    d, _ = tree.query(p, k=2)
    return float(np.min(d[:, 1]))


def fit_separation(
    points: np.ndarray,
    min_sep: float,
    *,
    center: bool = True,
    max_scale: float = 20.0,
) -> np.ndarray:
    """Uniformly scale a formation until no two drones are closer than min_sep.

    Uniform scaling is deliberate: it preserves the artwork. Scaling only
    the crowded axis would distort the shape and, worse, would break the
    transition bound in safety.py, which assumes nothing about shape but
    everything about endpoint separation.

    Raises if two points are coincident, since no finite scale fixes that.
    """
    p = np.array(points, dtype=np.float64, copy=True)
    if center:
        p -= p.mean(axis=0)
    if len(p) < 2:
        return p
    d = min_pair_distance(p)
    if d <= 1e-9:
        raise ValueError(
            "formation contains coincident points; two drones cannot share "
            "a position - regenerate the shape with fewer points or more detail"
        )
    if d >= min_sep:
        return p
    scale = (min_sep / d) * 1.001  # 0.1% so float error cannot land us under
    if scale > max_scale:
        raise ValueError(
            f"formation needs a {scale:.1f}x scale to reach {min_sep:.2f} m "
            f"separation (closest pair is {d:.3f} m apart). That means the "
            "shape clusters drones unevenly rather than being merely small - "
            "scaling it would produce an absurdly large show. Re-space the "
            "shape, or use fewer drones on it."
        )
    return p * scale


def capacity(width: float, height: float, min_sep: float) -> int:
    """How many drones fit in a width x height billboard at this spacing.

    Hexagonal packing, which is what an evenly relaxed formation
    approximates. Use it to size a show against the site before designing
    it: a drone count that does not fit will be scaled up by
    `fit_separation` until it breaks the altitude ceiling instead.
    """
    if min_sep <= 0:
        raise ValueError("min_sep must be positive")
    row_pitch = min_sep * math.sqrt(3.0) / 2.0
    rows = int(width // min_sep) + 1 if width > 0 else 0
    cols = int(height // row_pitch) + 1 if height > 0 else 0
    return max(0, rows * cols)


def billboard_for(count: int, min_sep: float, aspect: float = 1.6) -> tuple[float, float]:
    """Width and height of the smallest `aspect`-ratio billboard holding `count`.

    Shows run wide, not tall - the ceiling is a hard regulatory number and
    the audience is on the ground. aspect 1.6 is a reasonable default.
    """
    area = count * min_sep * min_sep * math.sqrt(3.0) / 2.0
    height = math.sqrt(area / max(aspect, 1e-6))
    return height * aspect, height


def lift(points: np.ndarray, altitude: float) -> np.ndarray:
    """Translate a formation so its lowest point sits at `altitude` AGL."""
    p = np.array(points, dtype=np.float64, copy=True)
    p[:, 2] += altitude - p[:, 2].min()
    return p


def center_at(points: np.ndarray, x: float, y: float, z: float) -> np.ndarray:
    p = np.array(points, dtype=np.float64, copy=True)
    p -= p.mean(axis=0)
    p += np.array([x, y, z], dtype=np.float64)
    return p


def rotate_z(points: np.ndarray, degrees: float) -> np.ndarray:
    th = math.radians(degrees)
    c, s = math.cos(th), math.sin(th)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(points, dtype=np.float64) @ r.T


def rotate_x(points: np.ndarray, degrees: float) -> np.ndarray:
    th = math.radians(degrees)
    c, s = math.cos(th), math.sin(th)
    r = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return np.asarray(points, dtype=np.float64) @ r.T


def _upright(xy: np.ndarray) -> np.ndarray:
    """A 2D (n,2) shape becomes a vertical billboard facing south (-y)."""
    n = len(xy)
    out = np.zeros((n, 3), dtype=np.float64)
    out[:, 0] = xy[:, 0]
    out[:, 2] = xy[:, 1]
    return out


# --------------------------------------------------------------------------
# Parametric shapes
# --------------------------------------------------------------------------


def circle(n: int, radius: float = 30.0, *, upright: bool = True) -> np.ndarray:
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    xy = np.stack([np.cos(th), np.sin(th)], axis=1) * radius
    return _upright(xy) if upright else np.column_stack([xy, np.zeros(n)])


def ring_stack(
    n: int, radius: float = 30.0, rings: int = 3, gap: float = 10.0
) -> np.ndarray:
    """Concentric horizontal rings stacked in altitude."""
    rings = max(1, min(rings, n))
    per = _split_evenly(n, rings)
    out = []
    for i, k in enumerate(per):
        th = np.linspace(0.0, 2.0 * math.pi, k, endpoint=False) + i * 0.3
        r = radius * (1.0 - 0.15 * i)
        out.append(
            np.column_stack(
                [r * np.cos(th), r * np.sin(th), np.full(k, i * gap, dtype=float)]
            )
        )
    return np.concatenate(out, axis=0)


def sphere(n: int, radius: float = 30.0) -> np.ndarray:
    """Fibonacci sphere - near-uniform, no clustering at the poles."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    golden = math.pi * (1.0 + math.sqrt(5.0))
    theta = golden * i
    return (
        np.stack(
            [
                np.cos(theta) * np.sin(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(phi),
            ],
            axis=1,
        )
        * radius
    )


def cube_shell(n: int, size: float = 50.0) -> np.ndarray:
    """Points spread over the six faces of a hollow cube."""
    per_face = _split_evenly(n, 6)
    half = size / 2.0
    out = []
    for face, k in enumerate(per_face):
        # Cell centres, not corners: a corner lattice puts a point on every
        # shared face edge twice, and two drones at one position is the one
        # formation defect no amount of scaling can fix.
        a, b = _square_lattice(k, centred=True)
        u = (a - 0.5) * size
        v = (b - 0.5) * size
        zeros = np.zeros(k)
        if face == 0:
            out.append(np.column_stack([u, v, zeros + half]))
        elif face == 1:
            out.append(np.column_stack([u, v, zeros - half]))
        elif face == 2:
            out.append(np.column_stack([u, zeros + half, v]))
        elif face == 3:
            out.append(np.column_stack([u, zeros - half, v]))
        elif face == 4:
            out.append(np.column_stack([zeros + half, u, v]))
        else:
            out.append(np.column_stack([zeros - half, u, v]))
    return np.concatenate(out, axis=0)[:n]


def helix(
    n: int, radius: float = 25.0, height: float = 60.0, turns: float = 3.0
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    th = 2.0 * math.pi * turns * t
    return np.column_stack([radius * np.cos(th), radius * np.sin(th), height * t])


def double_helix(
    n: int, radius: float = 25.0, height: float = 60.0, turns: float = 2.5
) -> np.ndarray:
    a = helix(n // 2 + n % 2, radius, height, turns)
    b = helix(n // 2, radius, height, turns)
    b = rotate_z(b, 180.0)
    return np.concatenate([a, b], axis=0)[:n]


def heart(n: int, scale: float = 2.2, *, solid: bool = False) -> np.ndarray:
    """Classic heart curve, as a vertical billboard.

    Sampled densely and then re-spaced by arc length. Sampling the
    parameter uniformly instead piles points into the bottom cusp, which
    leaves two drones centimetres apart and forces `fit_separation` to
    inflate the whole shape by hundreds of times to compensate.

    solid=True fills the interior instead of tracing the outline. Prefer it
    above roughly 60 drones: an outline needs perimeter proportional to the
    drone count, so 120 drones at 4 m spacing demand a half-kilometre of
    perimeter and a heart 150 m tall. A filled shape only grows as the
    square root of the count.
    """

    def curve(t: np.ndarray) -> np.ndarray:
        x = 16.0 * np.sin(t) ** 3
        y = (
            13.0 * np.cos(t)
            - 5.0 * np.cos(2.0 * t)
            - 2.0 * np.cos(3.0 * t)
            - np.cos(4.0 * t)
        )
        return np.stack([x, y], axis=1) * scale

    outline = _even_curve(curve, max(n, 512))
    if solid:
        return from_mask(n, rasterise_polygon(outline), width=float(np.ptp(outline[:, 0])))
    return _upright(_resample_polyline(outline, n, closed=True))


def star(
    n: int,
    points: int = 5,
    outer: float = 35.0,
    inner: float = 0.0,
    *,
    solid: bool = False,
) -> np.ndarray:
    """Star as a vertical billboard: outline by default, filled on request."""
    if inner <= 0:
        inner = outer * 0.42
    verts = []
    for i in range(points * 2):
        r = outer if i % 2 == 0 else inner
        a = math.pi / 2.0 + i * math.pi / points
        verts.append((r * math.cos(a), r * math.sin(a)))
    poly = np.array(verts, dtype=np.float64)
    if solid:
        return from_mask(n, rasterise_polygon(poly), width=outer * 2.0)
    return _upright(_resample_polyline(poly, n, closed=True))


def polygon(
    n: int,
    vertices: np.ndarray,
    *,
    upright: bool = True,
    solid: bool = False,
) -> np.ndarray:
    """n points on an arbitrary closed 2D polygon: its edge, or its interior."""
    v = np.asarray(vertices, dtype=np.float64)
    if solid:
        return from_mask(
            n, rasterise_polygon(v), width=float(np.ptp(v[:, 0])), upright=upright
        )
    xy = _resample_polyline(v, n, closed=True)
    return _upright(xy) if upright else np.column_stack([xy, np.zeros(n)])


def disc(n: int, radius: float = 35.0, *, upright: bool = True) -> np.ndarray:
    """Filled circle, evenly packed - a sunflower (Vogel) spiral."""
    i = np.arange(n, dtype=np.float64) + 0.5
    r = radius * np.sqrt(i / n)
    th = i * math.pi * (3.0 - math.sqrt(5.0))
    xy = np.column_stack([r * np.cos(th), r * np.sin(th)])
    return _upright(xy) if upright else np.column_stack([xy, np.zeros(n)])


def ball(n: int, radius: float = 30.0) -> np.ndarray:
    """Filled sphere: nested Fibonacci shells, so the volume is populated."""
    i = np.arange(n, dtype=np.float64) + 0.5
    # r ~ n^(1/3) keeps the radial density uniform through the volume.
    r = radius * np.cbrt(i / n)
    phi = np.arccos(1.0 - 2.0 * ((i * 0.7548776662) % 1.0))
    theta = i * math.pi * (1.0 + math.sqrt(5.0))
    return np.column_stack(
        [
            r * np.cos(theta) * np.sin(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(phi),
        ]
    )


def grid(n: int, spacing: float = 5.0, *, upright: bool = False) -> np.ndarray:
    """Compact rectangular lattice, as close to square as n allows."""
    a, b = _square_lattice(n)
    cols = math.ceil(math.sqrt(n))
    xy = np.column_stack([a * (cols - 1), b * (cols - 1)]) * spacing
    return _upright(xy) if upright else np.column_stack([xy, np.zeros(n)])


def wave(
    n: int, width: float = 80.0, amplitude: float = 12.0, cycles: float = 2.0
) -> np.ndarray:
    """A horizontal sheet with a travelling-wave profile in z."""
    a, b = _square_lattice(n)
    x = (a - 0.5) * width
    y = (b - 0.5) * width
    z = amplitude * np.sin(2.0 * math.pi * cycles * a)
    return np.column_stack([x, y, z])


def torus(
    n: int, major: float = 30.0, minor: float = 10.0, *, upright: bool = False
) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    golden = math.pi * (1.0 + math.sqrt(5.0))
    u = 2.0 * math.pi * i / n
    v = golden * i
    x = (major + minor * np.cos(v)) * np.cos(u)
    y = (major + minor * np.cos(v)) * np.sin(u)
    z = minor * np.sin(v)
    p = np.column_stack([x, y, z])
    return rotate_x(p, 90.0) if upright else p


def line(n: int, length: float = 80.0, *, vertical: bool = False) -> np.ndarray:
    t = np.linspace(-0.5, 0.5, n) * length
    zeros = np.zeros(n)
    return (
        np.column_stack([zeros, zeros, t])
        if vertical
        else np.column_stack([t, zeros, zeros])
    )


# --------------------------------------------------------------------------
# Raster shapes - text, images, anything you can rasterise to a mask
# --------------------------------------------------------------------------


def from_mask(
    n: int,
    mask: np.ndarray,
    *,
    width: float = 80.0,
    upright: bool = True,
    iterations: int = 40,
    seed: int = 0,
) -> np.ndarray:
    """Spread n points evenly over the True pixels of a 2D boolean mask.

    k-means on the filled pixels (Lloyd relaxation) gives a centroidal
    Voronoi layout: points land where the ink is, spaced as evenly as the
    shape allows. That even spacing is what lets `fit_separation` scale to
    the legal minimum without blowing the shape up.
    """
    m = np.asarray(mask)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        raise ValueError("mask is empty - nothing to place drones on")
    if n > len(xs):
        raise ValueError(
            f"mask has only {len(xs)} filled pixels for {n} drones; "
            "rasterise the shape at a higher resolution"
        )
    h, w = m.shape
    # Image row 0 is the top, so flip y to get a right-handed shape.
    pix = np.column_stack([xs.astype(np.float64), (h - 1 - ys).astype(np.float64)])

    rng = np.random.default_rng(seed)
    if n == len(pix):
        pts = pix
    else:
        start = pix[rng.choice(len(pix), size=n, replace=False)]
        pts, _ = kmeans2(
            pix, start, iter=max(1, iterations), minit="matrix", missing="raise"
        )
        pts = _snap_to_mask(pts, pix)

    pts = pts - pts.mean(axis=0)
    span = max(float(np.ptp(pix[:, 0])), 1.0)
    pts *= width / span
    return _upright(pts) if upright else np.column_stack([pts, np.zeros(n)])


def _snap_to_mask(centroids: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Move each centroid onto a filled pixel, keeping them distinct.

    A k-means centroid is the mean of its cluster, which for a non-convex
    shape can sit in a hole. On a logo of a disc beside a bar, roughly 40%
    of the drones ended up in the gap between them - the formation read as
    a rectangle of noise rather than the mark. Snapping each one to the
    nearest ink pixel fixes that.

    Snapping must stay injective: two drones sharing a pixel are two
    drones at the same position, which fit_separation refuses outright.
    Each centroid therefore takes its nearest *unused* pixel out of the
    few closest, and keeps its original position if all of them are taken
    - which only happens where the mask is thinner than the drone
    spacing, and is better than a duplicate.
    """
    out = np.array(centroids, dtype=np.float64, copy=True)
    if len(centroids) > len(pixels):
        # from_mask refuses this earlier; nothing sensible to do here.
        return out

    tree = cKDTree(pixels)
    k = min(len(pixels), 12)
    idx = np.atleast_2d(tree.query(centroids, k=k)[1].reshape(len(centroids), -1))

    taken: set[int] = set()
    unresolved: list[int] = []
    for i in range(len(centroids)):
        for candidate in idx[i]:
            c = int(candidate)
            if c not in taken:
                taken.add(c)
                out[i] = pixels[c]
                break
        else:
            unresolved.append(i)

    # Second pass for anything whose nearest dozen pixels were all claimed -
    # which happens wherever centroids pile up on one spot. Searching only
    # the unused pixels, and asking each centroid for at least as many
    # candidates as there are centroids left, makes the greedy assignment
    # always succeed, so no two drones can end up on one pixel.
    if unresolved:
        free = np.array(sorted(set(range(len(pixels))) - taken), dtype=np.int64)
        sub = cKDTree(pixels[free])
        want = min(len(free), len(unresolved))
        choices = np.atleast_2d(
            sub.query(centroids[unresolved], k=want)[1].reshape(len(unresolved), -1)
        )
        for row, i in enumerate(unresolved):
            for candidate in choices[row]:
                c = int(free[int(candidate)])
                if c not in taken:
                    taken.add(c)
                    out[i] = pixels[c]
                    break
    return out


def text(
    n: int,
    message: str,
    *,
    width: float = 100.0,
    font_path: str | None = None,
    font_size: int = 160,
    upright: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Render text to a mask, then place n drones on it."""
    return from_mask(
        n,
        rasterise_text(message, font_path=font_path, font_size=font_size),
        width=width,
        upright=upright,
        seed=seed,
    )


def image(
    n: int,
    path: str,
    *,
    width: float = 100.0,
    threshold: int = 128,
    invert: bool = False,
    resolution: int = 512,
    upright: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Place n drones on the bright (or dark) region of an image file."""
    return from_mask(
        n,
        rasterise_image(
            path, threshold=threshold, invert=invert, resolution=resolution
        ),
        width=width,
        upright=upright,
        seed=seed,
    )


def rasterise_text(
    message: str, *, font_path: str | None = None, font_size: int = 160
) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = _default_font(font_size)

    probe = Image.new("L", (16, 16))
    box = ImageDraw.Draw(probe).textbbox((0, 0), message, font=font)
    pad = max(8, font_size // 8)
    size = (box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad)
    img = Image.new("L", size, color=0)
    ImageDraw.Draw(img).text((pad - box[0], pad - box[1]), message, fill=255, font=font)
    return np.asarray(img) > 96


def _default_font(size: int):
    from PIL import ImageFont

    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    # The bitmap fallback ignores size, so text formations come out coarse.
    # Pass font_path explicitly for anything going in front of an audience.
    return ImageFont.load_default()


def rasterise_polygon(
    vertices: np.ndarray, *, resolution: int = 512
) -> np.ndarray:
    """Fill a closed 2D polygon into a boolean mask."""
    from PIL import Image, ImageDraw

    v = np.asarray(vertices, dtype=np.float64)
    if len(v) < 3:
        raise ValueError("need at least 3 vertices to fill a polygon")
    lo = v.min(axis=0)
    span = np.maximum(np.ptp(v, axis=0), 1e-9)
    scale = (resolution - 4) / float(span.max())
    px = (v - lo) * scale + 2.0
    size = (
        int(math.ceil(span[0] * scale)) + 4,
        int(math.ceil(span[1] * scale)) + 4,
    )
    img = Image.new("L", size, color=0)
    # Image y grows downwards; from_mask flips it back.
    flipped = [(float(x), float(size[1] - 1 - y)) for x, y in px]
    ImageDraw.Draw(img).polygon(flipped, fill=255)
    return np.asarray(img) > 128


def rasterise_image(
    path: str, *, threshold: int = 128, invert: bool = False, resolution: int = 512
) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGBA")
    # Composite onto black so a transparent logo keeps its silhouette.
    flat = Image.new("RGBA", img.size, (0, 0, 0, 255))
    flat.alpha_composite(img)
    grey = flat.convert("L")
    if max(grey.size) > resolution:
        scale = resolution / max(grey.size)
        grey = grey.resize(
            (max(1, int(grey.width * scale)), max(1, int(grey.height * scale))),
            Image.LANCZOS,
        )
    arr = np.asarray(grey)
    return (arr < threshold) if invert else (arr >= threshold)


# --------------------------------------------------------------------------
# Shape registry, so the UI and CLI can offer shapes without a big switch
# --------------------------------------------------------------------------

SHAPES: dict[str, Callable[..., np.ndarray]] = {
    "circle": circle,
    "disc": disc,
    "ring_stack": ring_stack,
    "sphere": sphere,
    "ball": ball,
    "cube": cube_shell,
    "polygon": polygon,
    "helix": helix,
    "double_helix": double_helix,
    "heart": heart,
    "star": star,
    "grid": grid,
    "wave": wave,
    "torus": torus,
    "line": line,
    "text": text,
    "image": image,
}


def make(name: str, n: int, **kwargs) -> np.ndarray:
    """Build a shape by name. Unknown keyword arguments are rejected loudly."""
    try:
        fn = SHAPES[name]
    except KeyError:
        raise ValueError(
            f"unknown shape {name!r}; available: {', '.join(sorted(SHAPES))}"
        ) from None
    return fn(n, **kwargs)


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


def _split_evenly(n: int, parts: int) -> list[int]:
    """Split n into `parts` counts differing by at most 1, all >= 1 if possible."""
    parts = max(1, min(parts, max(1, n)))
    base, extra = divmod(n, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def _square_lattice(
    n: int, *, centred: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """n lattice coordinates on the unit square, near-square grid.

    centred=False spans [0,1] corner to corner; centred=True returns cell
    centres, strictly inside (0,1), so adjoining patches never share a point.
    """
    if n <= 0:
        return np.zeros(0), np.zeros(0)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    idx = np.arange(n)
    c = idx % cols
    r = idx // cols
    if centred:
        return (c + 0.5) / cols, (r + 0.5) / rows
    return c / max(cols - 1, 1), r / max(rows - 1, 1)


def _even_curve(
    fn: Callable[[np.ndarray], np.ndarray], n: int, *, samples: int = 4096
) -> np.ndarray:
    """n points spaced by equal arc length along a closed parametric curve."""
    t = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    return _resample_polyline(fn(t), n, closed=True)


def _resample_polyline(
    verts: np.ndarray, n: int, *, closed: bool = True
) -> np.ndarray:
    """n points spaced by equal arc length along a polyline."""
    v = np.asarray(verts, dtype=np.float64)
    if closed:
        v = np.vstack([v, v[0]])
    seg = np.linalg.norm(np.diff(v, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 0:
        raise ValueError("degenerate polyline")
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    # endpoint=False on a closed curve avoids placing two drones on the seam
    targets = np.linspace(0.0, total, n, endpoint=closed is False) % total
    if closed:
        targets = np.linspace(0.0, total, n, endpoint=False)
    out = np.empty((n, 2), dtype=np.float64)
    for i, s in enumerate(targets):
        j = int(np.searchsorted(cum, s, side="right")) - 1
        j = max(0, min(j, len(seg) - 1))
        local = 0.0 if seg[j] <= 0 else (s - cum[j]) / seg[j]
        out[i] = v[j] + (v[j + 1] - v[j]) * local
    return out
