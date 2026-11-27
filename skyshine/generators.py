"""Formations made by generators: a QR code, a countdown, scrolling text.

The countdown and the ticker hold the drones still and change only their
lights, like an LED sign. Nothing moves, so the timing is exact and the
show is as safe as a static picture. Every generator lays drones on a grid
at the spacing it is given, so preparing does not have to rescale it.
"""

from __future__ import annotations

import math

import numpy as np

from .show import Formation

OFF = np.array([0, 0, 0], dtype=np.int32)


# --------------------------------------------------------------------------
# QR code (ISO/IEC 18004), byte mode, versions 1-10
# --------------------------------------------------------------------------

# Index 0 is unused; levels are L and M. From the standard's tables.
_ECC_PER_BLOCK = {"L": (-1, 7, 10, 15, 20, 26, 18, 20, 24, 30, 18),
                  "M": (-1, 10, 16, 26, 18, 24, 16, 18, 22, 22, 26)}
_NUM_BLOCKS = {"L": (-1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4),
               "M": (-1, 1, 1, 1, 2, 2, 4, 4, 4, 5, 5)}
_FORMAT_LEVEL = {"L": 1, "M": 0}
MAX_VERSION = 10


def _gf_mul(x: int, y: int) -> int:
    z = 0
    for i in reversed(range(8)):
        z = (z << 1) ^ ((z >> 7) * 0x11D)
        z ^= ((y >> i) & 1) * x
    return z


def _rs_divisor(degree: int) -> list[int]:
    result = [0] * (degree - 1) + [1]
    root = 1
    for _ in range(degree):
        for j in range(degree):
            result[j] = _gf_mul(result[j], root)
            if j + 1 < degree:
                result[j] ^= result[j + 1]
        root = _gf_mul(root, 0x02)
    return result


def _rs_remainder(data: list[int], divisor: list[int]) -> list[int]:
    result = [0] * len(divisor)
    for b in data:
        factor = b ^ result.pop(0)
        result.append(0)
        for i, coef in enumerate(divisor):
            result[i] ^= _gf_mul(coef, factor)
    return result


def _raw_modules(ver: int) -> int:
    result = (16 * ver + 128) * ver + 64
    if ver >= 2:
        n = ver // 7 + 2
        result -= (25 * n - 10) * n - 55
        if ver >= 7:
            result -= 36
    return result


def _data_codewords(ver: int, level: str) -> int:
    return _raw_modules(ver) // 8 - _ECC_PER_BLOCK[level][ver] * _NUM_BLOCKS[level][ver]


def _alignment(ver: int, size: int) -> list[int]:
    if ver == 1:
        return []
    n = ver // 7 + 2
    step = (ver * 4 + n * 2 + 1) // (n * 2 - 2) * 2
    out = [6]
    pos = size - 7
    for _ in range(n - 1):
        out.insert(1, pos)
        pos -= step
    return out


_MASKS = (
    lambda x, y: (x + y) % 2 == 0, lambda x, y: y % 2 == 0,
    lambda x, y: x % 3 == 0, lambda x, y: (x + y) % 3 == 0,
    lambda x, y: (x // 3 + y // 2) % 2 == 0, lambda x, y: x * y % 2 + x * y % 3 == 0,
    lambda x, y: (x * y % 2 + x * y % 3) % 2 == 0,
    lambda x, y: ((x + y) % 2 + x * y % 3) % 2 == 0,
)


def qr_matrix(text: str, level: str = "L") -> np.ndarray:
    """The QR code for `text` as a bool matrix, True for a dark module."""
    if level not in _ECC_PER_BLOCK:
        raise ValueError("error correction must be L or M")
    data = text.encode("utf-8")
    if not data:
        raise ValueError("type the text or link for the QR code")
    for ver in range(1, MAX_VERSION + 1):
        count_bits = 8 if ver < 10 else 16
        if 4 + count_bits + 8 * len(data) <= _data_codewords(ver, level) * 8:
            break
    else:
        raise ValueError(f"{len(data)} bytes is too long for a QR code drones can "
                         "fly; keep it under about 150 characters, a short link works best")

    # Bit stream: byte mode, length, data, terminator, padding.
    bits = [0, 1, 0, 0] + [(len(data) >> i) & 1 for i in reversed(range(count_bits))]
    for b in data:
        bits += [(b >> i) & 1 for i in reversed(range(8))]
    capacity = _data_codewords(ver, level) * 8
    bits += [0] * min(4, capacity - len(bits))
    bits += [0] * (-len(bits) % 8)
    codewords = [int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8)]
    pad = 0xEC
    while len(codewords) < capacity // 8:
        codewords.append(pad)
        pad ^= 0xEC ^ 0x11

    # Error correction, split into blocks and interleaved.
    blocks_n, ecc_len = _NUM_BLOCKS[level][ver], _ECC_PER_BLOCK[level][ver]
    raw = _raw_modules(ver) // 8
    short_n = blocks_n - raw % blocks_n
    short_len = raw // blocks_n
    divisor = _rs_divisor(ecc_len)
    blocks, k = [], 0
    for i in range(blocks_n):
        take = short_len - ecc_len + (0 if i < short_n else 1)
        dat = codewords[k:k + take]
        k += take
        ecc = _rs_remainder(dat, divisor)
        if i < short_n:
            dat = dat + [0]
        blocks.append(dat + ecc)
    stream = [blk[i] for i in range(len(blocks[0])) for j, blk in enumerate(blocks)
              if i != short_len - ecc_len or j >= short_n]

    size = ver * 4 + 17
    mod = np.zeros((size, size), dtype=bool)
    fixed = np.zeros((size, size), dtype=bool)

    def put(x, y, dark):
        mod[y, x] = dark
        fixed[y, x] = True

    for i in range(size):
        put(6, i, i % 2 == 0)
        put(i, 6, i % 2 == 0)
    for cx, cy in ((3, 3), (size - 4, 3), (3, size - 4)):
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                x, y = cx + dx, cy + dy
                if 0 <= x < size and 0 <= y < size:
                    put(x, y, max(abs(dx), abs(dy)) not in (2, 4))
    al = _alignment(ver, size)
    for i, ax in enumerate(al):
        for j, ay in enumerate(al):
            if (i, j) in ((0, 0), (0, len(al) - 1), (len(al) - 1, 0)):
                continue
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    put(ax + dx, ay + dy, max(abs(dx), abs(dy)) != 1)
    _draw_format(put, size, level, 0)        # reserve the format area
    if ver >= 7:
        rem = ver
        for _ in range(12):
            rem = (rem << 1) ^ ((rem >> 11) * 0x1F25)
        vbits = ver << 12 | rem
        for i in range(18):
            a, b = size - 11 + i % 3, i // 3
            put(a, b, (vbits >> i) & 1 == 1)
            put(b, a, (vbits >> i) & 1 == 1)

    i = 0
    right = size - 1
    while right >= 1:
        if right == 6:
            right = 5
        for vert in range(size):
            for j in range(2):
                x = right - j
                y = size - 1 - vert if ((right + 1) & 2) == 0 else vert
                if not fixed[y, x] and i < len(stream) * 8:
                    mod[y, x] = (stream[i >> 3] >> (7 - (i & 7))) & 1 == 1
                    i += 1
        right -= 2

    best, best_score = None, None
    for m, fn in enumerate(_MASKS):
        trial = mod.copy()
        for y in range(size):
            for x in range(size):
                if not fixed[y, x] and fn(x, y):
                    trial[y, x] = not trial[y, x]
        _draw_format(lambda x, y, d: trial.__setitem__((y, x), d), size, level, m)
        score = _penalty(trial)
        if best_score is None or score < best_score:
            best, best_score = trial, score
    return best


def _draw_format(put, size: int, level: str, mask: int) -> None:
    data = _FORMAT_LEVEL[level] << 3 | mask
    rem = data
    for _ in range(10):
        rem = (rem << 1) ^ ((rem >> 9) * 0x537)
    bits = (data << 10 | rem) ^ 0x5412
    bit = lambda i: (bits >> i) & 1 == 1  # noqa: E731
    for i in range(6):
        put(8, i, bit(i))
    put(8, 7, bit(6))
    put(8, 8, bit(7))
    put(7, 8, bit(8))
    for i in range(9, 15):
        put(14 - i, 8, bit(i))
    for i in range(8):
        put(size - 1 - i, 8, bit(i))
    for i in range(8, 15):
        put(8, size - 15 + i, bit(i))
    put(8, size - 8, True)


def _penalty(m: np.ndarray) -> int:
    size = len(m)
    score = 0
    for line in list(m) + list(m.T):
        run, prev = 0, None
        for v in line:
            if v == prev:
                run += 1
            else:
                if run >= 5:
                    score += 3 + run - 5
                run, prev = 1, v
        if run >= 5:
            score += 3 + run - 5
        s = "0000" + "".join("1" if v else "0" for v in line) + "0000"
        score += 40 * (s.count("10111010000") + s.count("00001011101"))
    same = (m[:-1, :-1] == m[1:, :-1]) & (m[:-1, :-1] == m[:-1, 1:]) & (m[:-1, :-1] == m[1:, 1:])
    score += 3 * int(same.sum())
    dark = int(m.sum())
    k = math.ceil(abs(dark * 20 - size * size * 10) / (size * size)) - 1
    return score + 10 * max(k, 0)


def qr_formation(text: str, count: int, spacing: float, *, color=(255, 255, 255),
                 lit: str = "dark", hold: float = 10.0, flat: bool = False) -> Formation:
    """A QR code in the sky, one drone per module, as an upright billboard,
    or with flat=True lying level to be read from below, top edge to the
    north. Lying flat it adds no height: at legal spacing an upright code
    is 90 m or more tall.

    lit="dark" lights the code's dark modules: fewest drones, and an
    inverted code, which current iPhone and Android camera apps read.
    lit="light" lights the light modules plus a two-module border, the
    standard polarity, for older scanners; it takes about twice the drones.
    Drones beyond what the code needs stay dark on the unlit modules, then
    on a row behind the code.
    """
    m = qr_matrix(text)
    if lit == "light":
        m = np.pad(~m, 2, constant_values=True)
    elif lit != "dark":
        raise ValueError("lit must be 'dark' or 'light'")
    size = len(m)
    rows, cols = np.nonzero(m)
    need = len(rows)
    if count < need:
        raise ValueError(f"this QR code needs {need} drones, the show has {count}. "
                         "Shorten the text, use a short link, or add drones.")
    grid = lambda r, c: np.column_stack(  # noqa: E731
        [(c - (size - 1) / 2) * spacing, np.zeros(len(r)), ((size - 1) / 2 - r) * spacing])
    lit_pts = grid(rows, cols)
    spare = count - need
    off_r, off_c = np.nonzero(~m)
    take = min(spare, len(off_r))
    spare_pts = grid(off_r, off_c)
    if take < len(off_r):
        from .planning import _farthest_point_subset

        spare_pts = spare_pts[_farthest_point_subset(spare_pts, take)] if take else spare_pts[:0]
    extra = spare - take
    behind = _park(extra, spacing)
    points = np.vstack([lit_pts, spare_pts, behind])
    if flat:
        # Tip it back: up becomes north, and the parked drones end up below.
        # Facing north and looking up, east stays on the right, so the code
        # reads the right way round from the ground.
        points = np.column_stack([points[:, 0], points[:, 2], -points[:, 1]])
    colors = np.vstack([np.tile(np.asarray(color, dtype=np.int32), (need, 1)),
                        np.tile(OFF, (len(points) - need, 1))])
    return Formation(name="QR code", points=points, colors=colors, hold=hold)


def _park(count: int, spacing: float) -> np.ndarray:
    """Spare drones, unlit, on a square grid one spacing behind (north of)
    the billboard and centred on it, so they add no height."""
    if count <= 0:
        return np.zeros((0, 3))
    side = math.ceil(math.sqrt(count))
    i = np.arange(count)
    r, c = i // side, i % side
    return np.column_stack([(c - (side - 1) / 2) * spacing, np.full(count, spacing),
                            ((side - 1) / 2 - r) * spacing])


# --------------------------------------------------------------------------
# Countdown: seven-segment digits, lights only
# --------------------------------------------------------------------------

_SEGMENTS = {d: set(s) for d, s in {
    0: "abcdef", 1: "bc", 2: "abged", 3: "abgcd", 4: "fgbc",
    5: "afgcd", 6: "afgedc", 7: "abc", 8: "abcdefg", 9: "abcdfg"}.items()}


def countdown_formation(start: int, count: int, spacing: float, *, color=(255, 255, 255),
                        seconds: float = 1.0, max_height: float = 80.0) -> Formation:
    """Numbers from `start` down to 1, one per `seconds`, on seven-segment
    digits made of drones that stay still while their lights change.

    Digits are as tall as max_height allows; drones they do not need wait,
    unlit, on a grid behind them."""
    if not 1 <= start <= 99:
        raise ValueError("a countdown starts from 1 to 99")
    if seconds <= 0.2:
        raise ValueError("each number needs more than 0.2 s")
    digits = 1 if start < 10 else 2
    per_seg = min(count // (7 * digits), int(max_height / (2 * spacing)) - 1)
    if per_seg < 2:
        raise ValueError(f"a {digits}-digit countdown needs at least {14 * digits} drones "
                         f"and {6 * spacing:.0f} m of height")
    seg_len = per_seg * spacing             # every segment holds up to per_seg+1 drones
    width = seg_len + spacing
    pitch = width + 3 * spacing
    h = seg_len + spacing                   # top row to middle row
    lines = {"a": ((-1, 1), (1, 1)), "g": ((-1, 0), (1, 0)), "d": ((-1, -1), (1, -1)),
             "f": ((-1, 1), (-1, 0)), "b": ((1, 1), (1, 0)),
             "e": ((-1, 0), (-1, -1)), "c": ((1, 0), (1, -1))}
    # Every segment carries per_seg drones, exactly `spacing` apart.
    slots = [(d, s) for d in range(digits) for s in "abcdefg"]
    sizes = [per_seg] * len(slots)
    pts, owner = [], []
    for (d, s), k in zip(slots, sizes):
        (x0, y0), (x1, y1) = lines[s]
        horizontal = y0 == y1
        cx = (d - (digits - 1) / 2) * pitch
        if horizontal:
            xs = np.linspace(-(seg_len - spacing) / 2, (seg_len - spacing) / 2, k) if k > 1 else np.zeros(1)
            pts += [(cx + x, 0.0, y0 * h) for x in xs]
        else:
            x = cx + x0 * ((seg_len - spacing) / 2 + spacing)
            mid = (y0 + y1) / 2 * h
            ys = np.linspace(mid - (seg_len - spacing) / 2, mid + (seg_len - spacing) / 2, k) if k > 1 else np.array([mid])
            pts += [(x, 0.0, y) for y in ys]
        owner += [(d, s)] * k
    spare = count - len(pts)
    pts = np.vstack([np.asarray(pts), _park(spare, spacing)])
    owner += [None] * spare
    on = np.asarray(color, dtype=np.int32)

    def frame(number):
        text = str(number).rjust(digits)
        lit = {(d, s) for d, ch in enumerate(text) if ch != " " for s in _SEGMENTS[int(ch)]}
        return np.array([on if o in lit else OFF for o in owner], dtype=np.int32)

    keys = [(i * seconds, frame(start - i)) for i in range(1, start)]
    return Formation(name=f"countdown {start}", points=pts, colors=frame(start),
                     color_keys=keys, hold=start * seconds)


# --------------------------------------------------------------------------
# Scrolling text: an LED matrix of drones
# --------------------------------------------------------------------------


def ticker_formation(text: str, count: int, spacing: float, *, rows: int = 7,
                     color=(255, 255, 255), speed: float = 4.0) -> Formation:
    """Text scrolling right to left across a grid of still drones.

    rows is the height of the letters in drones (5 to 15). speed is in
    columns per second. The grid is as wide as the drones allow.
    """
    if not text.strip():
        raise ValueError("type the text to scroll")
    if not 5 <= rows <= 15:
        raise ValueError("letters must be 5 to 15 drones tall")
    if speed <= 0:
        raise ValueError("speed must be positive")
    cols = count // rows
    if cols < rows:
        raise ValueError(f"a {rows}-row ticker needs at least {rows * rows} drones")
    bitmap = _text_bitmap(text, rows)
    width = bitmap.shape[1]
    r, c = np.mgrid[0:rows, 0:cols]
    grid = np.column_stack([(c.ravel() - (cols - 1) / 2) * spacing, np.zeros(rows * cols),
                            ((rows - 1) / 2 - r.ravel()) * spacing])
    extra = count - rows * cols
    parked = np.column_stack([(np.arange(extra) - (extra - 1) / 2) * spacing,
                              np.zeros(extra), np.full(extra, -((rows - 1) / 2 + 1) * spacing)])
    points = np.vstack([grid, parked])
    on = np.asarray(color, dtype=np.int32)

    def frame(step):
        # Text starts just off the right edge and leaves off the left.
        src = c.ravel() - cols + step
        inside = (src >= 0) & (src < width)
        lit = np.zeros(rows * cols, dtype=bool)
        lit[inside] = bitmap[r.ravel()[inside], src[inside]]
        cols_ = np.where(lit[:, None], on, OFF)
        return np.vstack([cols_, np.tile(OFF, (extra, 1))]).astype(np.int32)

    steps = width + cols
    keys = [(k / speed, frame(k)) for k in range(1, steps + 1)]
    return Formation(name="scrolling text", points=points, colors=frame(0),
                     color_keys=keys, hold=steps / speed)


def _text_bitmap(text: str, rows: int) -> np.ndarray:
    """Text as a rows-tall bool bitmap, one column of space between letters."""
    from PIL import Image

    from .geometry import rasterise_text

    mask = rasterise_text(text, font_size=rows * 12)
    ys, xs = np.nonzero(mask)
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = mask.shape
    img = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (max(1, round(w * rows / h)), rows), Image.LANCZOS)
    return np.asarray(img) > 100


if __name__ == "__main__":
    q = qr_matrix("https://example.com")
    assert q.shape == (25, 25), q.shape            # version 2 holds 19 bytes at level L
    f = countdown_formation(10, 140, 4.0)
    from .geometry import min_pair_distance
    assert f.count == 140 and min_pair_distance(f.points) >= 4.0 - 1e-9
    t = ticker_formation("HELLO", 140, 4.0)
    assert t.count == 140 and min_pair_distance(t.points) >= 4.0 - 1e-9
    print("generators.py self-check ok")
