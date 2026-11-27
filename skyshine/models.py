"""Import designs made in other programs.

3D models (OBJ, STL, PLY, GLB) and SVG drawings become formations: n drones
spread evenly over the model's surface or the drawing's lines or fill. The
result is an ordinary formation, so it can be moved, coloured, animated and
validated like any other.

Standard library and numpy only. FBX is a closed format: export OBJ or GLB
from the program that made it instead.
"""

from __future__ import annotations

import json
import math
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

MAX_BYTES = 40 * 1024 * 1024
MAX_TRIANGLES = 3_000_000
MODEL_SUFFIXES = (".obj", ".stl", ".ply", ".glb")
DRAWING_SUFFIXES = (".svg",)


class ImportError_(ValueError):
    """The file cannot be turned into drone positions; the message says why."""


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def place(
    data: bytes,
    filename: str,
    count: int,
    *,
    mode: str = "auto",
    up: str = "y",
    size: float = 60.0,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """n drone positions (metres, centred, z up) and a description of the file.

    mode: "surface" or "vertices" for a model, "outline" or "fill" for a
    drawing, "auto" for surface / outline. up: which axis of a model is up
    ("y" for glTF and most OBJ exports, "z" for CAD and STL from slicers).
    size: length of the longest side, metres.
    """
    if not data:
        raise ImportError_("the file is empty")
    if len(data) > MAX_BYTES:
        raise ImportError_(f"file is {len(data) / 1e6:.0f} MB, over the "
                           f"{MAX_BYTES / 1e6:.0f} MB limit")
    if count < 1:
        raise ImportError_("need at least one drone")
    if not 1.0 <= size <= 2000.0:
        raise ImportError_("size must be between 1 and 2000 metres")
    suffix = Path(filename or "").suffix.lower()
    rng = np.random.default_rng(seed)

    if suffix in DRAWING_SUFFIXES or _looks_like_svg(data):
        paths = svg_paths(data.decode("utf-8", errors="replace"))
        if not paths:
            raise ImportError_("the drawing has no lines or shapes in it")
        mode = "outline" if mode == "auto" else mode
        if mode == "outline":
            pts2 = _even(_along(paths, count), count)
        elif mode == "fill":
            pts2 = _fill(paths, count, seed)
        else:
            raise ImportError_("a drawing can be placed on its outline or its fill")
        pts2 = pts2 * [1.0, -1.0]           # SVG y grows downwards
        pts = np.column_stack([pts2[:, 0], np.zeros(len(pts2)), pts2[:, 1]])
        info = {"kind": "drawing", "paths": len(paths), "mode": mode}
    elif suffix in MODEL_SUFFIXES:
        verts, tris = load_mesh(data, suffix)
        if up == "y":
            # Y-up (glTF, most OBJ): x right, y up, z toward the viewer.
            verts = np.column_stack([verts[:, 0], -verts[:, 2], verts[:, 1]])
        elif up != "z":
            raise ImportError_("up must be 'y' or 'z'")
        mode = "surface" if mode == "auto" else mode
        if mode == "surface" and len(tris):
            candidates = _surface(verts, tris, max(30 * count, 6000), rng)
        elif mode in ("surface", "vertices"):
            candidates = np.unique(np.round(verts, 9), axis=0)
            if len(candidates) < count:
                raise ImportError_(
                    f"the model has only {len(candidates)} distinct points for "
                    f"{count} drones; place them on the surface instead")
        else:
            raise ImportError_("a model can be placed on its surface or its vertices")
        pts = _even(candidates, count)
        info = {"kind": "model", "vertices": int(len(verts)),
                "triangles": int(len(tris)), "mode": mode}
    else:
        raise ImportError_(
            f"cannot read {suffix or 'this file'}. Use a 3D model "
            f"({', '.join(MODEL_SUFFIXES)}) or a drawing (.svg). From Blender, "
            "Maya or 3ds Max export OBJ or GLB; FBX is not readable here.")

    pts = pts - (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    span = float(np.ptp(pts, axis=0).max())
    if span <= 1e-9:
        raise ImportError_("every point in the file is in the same place")
    pts *= size / span
    info["size_m"] = [round(float(v), 1) for v in np.ptp(pts, axis=0)]
    return pts, info


def place_strokes(strokes, count: int, *, size: float = 60.0) -> np.ndarray:
    """n drones spread evenly along a freehand drawing, as an upright
    billboard. Strokes are lists of (x, y) in screen pixels, y downwards."""
    paths = [(np.asarray(s, dtype=np.float64).reshape(-1, 2), False)
             for s in strokes if len(s) >= 2]
    if not paths:
        raise ImportError_("draw at least one line first")
    if not 1.0 <= size <= 2000.0:
        raise ImportError_("size must be between 1 and 2000 metres")
    pts2 = _even(_along(paths, count), count) * [1.0, -1.0]
    pts = np.column_stack([pts2[:, 0], np.zeros(len(pts2)), pts2[:, 1]])
    pts -= (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    span = float(np.ptp(pts, axis=0).max())
    if span <= 1e-9:
        raise ImportError_("the drawing is a single dot; draw a line")
    return pts * (size / span)


def venue_mesh(data: bytes, suffix: str, *, scale: float = 1.0, up: str = "y",
               rotation: float = 0.0, offset=(0.0, 0.0, 0.0),
               max_triangles: int = 60_000) -> tuple[np.ndarray, np.ndarray]:
    """A site model in show metres: z up, scaled, turned and moved.

    Unlike place(), the model keeps its own origin, so a survey or CAD file
    that is already in local metres lands where it should. Very detailed
    models are thinned for display by keeping every k-th triangle.
    """
    verts, tris = load_mesh(data, suffix)
    if up == "y":
        verts = np.column_stack([verts[:, 0], -verts[:, 2], verts[:, 1]])
    elif up != "z":
        raise ImportError_("up must be 'y' or 'z'")
    if not 1e-6 < scale < 1e6:
        raise ImportError_("scale must be a positive number of metres per model unit")
    th = math.radians(rotation)
    c, s = math.cos(th), math.sin(th)
    v = verts * scale
    v = np.column_stack([v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c, v[:, 2]])
    v += np.asarray(offset, dtype=np.float64)
    if len(tris) > max_triangles:
        tris = tris[:: math.ceil(len(tris) / max_triangles)]
    return v, tris


def _looks_like_svg(data: bytes) -> bool:
    head = data[:2000].lower()
    return b"<svg" in head


def _even(candidates: np.ndarray, count: int) -> np.ndarray:
    """count well-spread points out of many candidates."""
    from .planning import _farthest_point_subset

    if len(candidates) < count:
        raise ImportError_(f"only {len(candidates)} usable points for {count} drones")
    return candidates[_farthest_point_subset(candidates, count)]


# --------------------------------------------------------------------------
# 3D models
# --------------------------------------------------------------------------


def load_mesh(data: bytes, suffix: str) -> tuple[np.ndarray, np.ndarray]:
    """(vertices (V,3) float, triangles (T,3) int; T may be 0)."""
    try:
        if suffix == ".obj":
            v, t = _obj(data.decode("utf-8", errors="replace"))
        elif suffix == ".stl":
            v, t = _stl(data)
        elif suffix == ".ply":
            v, t = _ply(data)
        elif suffix == ".glb":
            v, t = _glb(data)
        else:
            raise ImportError_(f"unknown model type {suffix}")
    except ImportError_:
        raise
    except (ValueError, IndexError, KeyError, struct.error, UnicodeDecodeError) as exc:
        raise ImportError_(f"the {suffix[1:].upper()} file is damaged or "
                           f"unsupported ({exc})") from None
    v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(t, dtype=np.int64).reshape(-1, 3)
    if not len(v):
        raise ImportError_("the model has no vertices")
    if len(t) > MAX_TRIANGLES:
        raise ImportError_(f"the model has {len(t):,} triangles; simplify it below "
                           f"{MAX_TRIANGLES:,} first")
    if len(t) and (t.min() < 0 or t.max() >= len(v)):
        raise ImportError_("the model's faces point at vertices that do not exist")
    if not np.all(np.isfinite(v)):
        raise ImportError_("the model has vertices that are not numbers")
    return v, t


def _obj(text: str):
    verts, tris = [], []
    for line in text.splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = []
            for tok in line.split()[1:]:
                i = int(tok.split("/")[0])
                idx.append(i - 1 if i > 0 else len(verts) + i)
            tris.extend([idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1))
    return verts, tris


def _stl(data: bytes):
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([
                ("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")]),
                count=n, offset=84)
            v = rec["v"].reshape(-1, 3).astype(np.float64)
            return v, np.arange(len(v)).reshape(-1, 3)
    text = data.decode("ascii", errors="replace")
    v = [[float(x) for x in m.split()] for m in
         re.findall(r"vertex\s+([-+\deE.]+\s+[-+\deE.]+\s+[-+\deE.]+)", text)]
    if not v or len(v) % 3:
        raise ImportError_("the STL file has no complete triangles")
    return v, np.arange(len(v)).reshape(-1, 3)


_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
              "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
              "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def _ply(data: bytes):
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise ImportError_("not a PLY file")
    header = data[:end].decode("ascii", errors="replace").splitlines()
    body = data[data.index(b"\n", end) + 1:]
    fmt, elements = "ascii", []
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            elements.append({"name": parts[1], "count": int(parts[2]), "props": []})
        elif parts[0] == "property" and elements:
            elements[-1]["props"].append(parts[1:])
    if fmt not in ("ascii", "binary_little_endian"):
        raise ImportError_(f"PLY format {fmt} is not supported; save as ASCII "
                           "or binary little endian")
    verts, tris = None, []
    if fmt == "ascii":
        rows = iter(body.decode("ascii", errors="replace").split("\n"))
        for el in elements:
            for _ in range(el["count"]):
                vals = next(rows).split()
                if el["name"] == "vertex":
                    names = [p[-1] for p in el["props"]]
                    verts = verts if verts is not None else []
                    verts.append([float(vals[names.index(a)]) for a in "xyz"])
                elif el["name"] == "face":
                    k = int(vals[0])
                    idx = [int(x) for x in vals[1:1 + k]]
                    tris.extend([idx[0], idx[j], idx[j + 1]] for j in range(1, k - 1))
        return verts or [], tris
    offset = 0
    for el in elements:
        if all(p[0] != "list" for p in el["props"]):
            dt = np.dtype([(p[-1], "<" + _PLY_TYPES[p[0]]) for p in el["props"]])
            arr = np.frombuffer(body, dtype=dt, count=el["count"], offset=offset)
            offset += dt.itemsize * el["count"]
            if el["name"] == "vertex":
                verts = np.column_stack([arr["x"], arr["y"], arr["z"]])
            continue
        # A face element: one length-prefixed list per face, read one by one.
        for _ in range(el["count"]):
            for p in el["props"]:
                if p[0] == "list":
                    ct, it = "<" + _PLY_TYPES[p[1]], "<" + _PLY_TYPES[p[2]]
                    k = int(np.frombuffer(body, ct, 1, offset)[0])
                    offset += np.dtype(ct).itemsize
                    idx = np.frombuffer(body, it, k, offset).tolist()
                    offset += np.dtype(it).itemsize * k
                    if el["name"] == "face":
                        tris.extend([idx[0], idx[j], idx[j + 1]] for j in range(1, k - 1))
                else:
                    offset += np.dtype("<" + _PLY_TYPES[p[0]]).itemsize
    return (verts if verts is not None else []), tris


_GLTF_COMPONENT = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
_GLTF_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _glb(data: bytes):
    magic, version, _length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ImportError_("not a glTF 2.0 binary (.glb) file")
    jlen, jtype = struct.unpack_from("<I4s", data, 12)
    doc = json.loads(data[20:20 + jlen].decode("utf-8"))
    pos = 20 + jlen
    blob = b""
    if pos + 8 <= len(data):
        blen, _btype = struct.unpack_from("<I4s", data, pos)
        blob = data[pos + 8:pos + 8 + blen]

    def accessor(i):
        a = doc["accessors"][i]
        view = doc["bufferViews"][a["bufferView"]]
        dt = np.dtype(_GLTF_COMPONENT[a["componentType"]])
        width = _GLTF_WIDTH[a["type"]]
        start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = view.get("byteStride", 0) or dt.itemsize * width
        raw = np.frombuffer(blob, np.uint8, stride * (a["count"] - 1) + dt.itemsize * width, start)
        rows = np.lib.stride_tricks.as_strided(raw, (a["count"], dt.itemsize * width), (stride, 1))
        return np.frombuffer(rows.copy().tobytes(), dt).reshape(a["count"], width)

    def node_matrix(node):
        if "matrix" in node:
            return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
        m = np.eye(4)
        x, y, z, w = node.get("rotation", [0, 0, 0, 1])
        m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        m[:3, :3] *= np.asarray(node.get("scale", [1, 1, 1]))[None, :]
        m[:3, 3] = node.get("translation", [0, 0, 0])
        return m

    verts, tris = [], []

    def walk(index, parent):
        node = doc["nodes"][index]
        world = parent @ node_matrix(node)
        if "mesh" in node:
            for prim in doc["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4 or "POSITION" not in prim["attributes"]:
                    continue
                p = accessor(prim["attributes"]["POSITION"]).astype(np.float64)
                p = (np.column_stack([p, np.ones(len(p))]) @ world.T)[:, :3]
                base = sum(len(x) for x in verts)
                idx = (accessor(prim["indices"]).ravel().astype(np.int64)
                       if "indices" in prim else np.arange(len(p)))
                verts.append(p)
                tris.append(idx[: len(idx) // 3 * 3].reshape(-1, 3) + base)
        for child in node.get("children", []):
            walk(child, world)

    scene = doc.get("scenes", [{}])[doc.get("scene", 0)] if doc.get("scenes") else {}
    roots = scene.get("nodes", list(range(len(doc.get("nodes", [])))))
    for r in roots:
        walk(r, np.eye(4))
    if not verts:
        raise ImportError_("the GLB file has no triangle meshes")
    return np.concatenate(verts), np.concatenate(tris)


def _surface(verts: np.ndarray, tris: np.ndarray, count: int, rng) -> np.ndarray:
    """count points spread over the triangles in proportion to their area."""
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if area.sum() <= 0:
        raise ImportError_("the model's faces have no area")
    pick = rng.choice(len(tris), size=count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    return a[pick] + (b[pick] - a[pick]) * u[:, None] + (c[pick] - a[pick]) * v[:, None]


# --------------------------------------------------------------------------
# SVG drawings
# --------------------------------------------------------------------------


def svg_paths(text: str) -> list[tuple[np.ndarray, bool]]:
    """Every line in a drawing as (points (k,2), closed), transforms applied."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ImportError_(f"the SVG file is not valid XML ({exc})") from None
    out: list[tuple[np.ndarray, bool]] = []

    def walk(el, parent):
        m = parent @ _svg_transform(el.get("transform", ""))
        tag = el.tag.rsplit("}", 1)[-1]
        shapes = []
        f = lambda k, d=0.0: float(re.sub(r"[a-z%]+$", "", el.get(k, str(d))) or d)  # noqa: E731
        if tag == "path":
            shapes = _svg_path(el.get("d", ""))
        elif tag in ("polygon", "polyline"):
            nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?",
                                                  el.get("points", ""))]
            if len(nums) >= 4:
                shapes = [(np.asarray(nums[: len(nums) // 2 * 2]).reshape(-1, 2), tag == "polygon")]
        elif tag == "line":
            shapes = [(np.array([[f("x1"), f("y1")], [f("x2"), f("y2")]]), False)]
        elif tag == "rect":
            x, y, w, h = f("x"), f("y"), f("width"), f("height")
            if w > 0 and h > 0:
                shapes = [(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]), True)]
        elif tag in ("circle", "ellipse"):
            rx = f("r") if tag == "circle" else f("rx")
            ry = f("r") if tag == "circle" else f("ry")
            if rx > 0 and ry > 0:
                th = np.linspace(0, 2 * math.pi, 96, endpoint=False)
                shapes = [(np.column_stack([f("cx") + rx * np.cos(th), f("cy") + ry * np.sin(th)]), True)]
        for pts, closed in shapes:
            if len(pts) >= 2:
                h = np.column_stack([pts, np.ones(len(pts))]) @ m.T
                out.append((h[:, :2], closed))
        if tag not in ("defs", "clipPath", "mask", "symbol", "title", "desc"):
            for child in el:
                walk(child, m)

    walk(root, np.eye(3))
    return out


def _svg_transform(spec: str) -> np.ndarray:
    m = np.eye(3)
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", spec):
        a = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", args)]
        t = np.eye(3)
        if name == "matrix" and len(a) == 6:
            t[:2, :] = [[a[0], a[2], a[4]], [a[1], a[3], a[5]]]
        elif name == "translate":
            t[:2, 2] = [a[0], a[1] if len(a) > 1 else 0.0]
        elif name == "scale":
            t[0, 0], t[1, 1] = a[0], a[1] if len(a) > 1 else a[0]
        elif name == "rotate":
            r = math.radians(a[0])
            rot = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
            if len(a) == 3:
                pre, post = np.eye(3), np.eye(3)
                pre[:2, 2], post[:2, 2] = [a[1], a[2]], [-a[1], -a[2]]
                rot = pre @ rot @ post
            t = rot
        elif name in ("skewX", "skewY"):
            k = math.tan(math.radians(a[0]))
            t[0 if name == "skewX" else 1, 1 if name == "skewX" else 0] = k
        m = m @ t
    return m


def _svg_path(d: str) -> list[tuple[np.ndarray, bool]]:
    tokens = re.findall(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", d)
    shapes, pts = [], []
    cur = np.zeros(2)
    start = np.zeros(2)
    last_ctrl = None
    cmd = None
    i = 0

    def num():
        nonlocal i
        v = float(tokens[i])
        i += 1
        return v

    def flush(closed):
        nonlocal pts
        if len(pts) >= 2:
            shapes.append((np.asarray(pts), closed))
        pts = []

    while i < len(tokens):
        if re.match(r"[A-Za-z]", tokens[i]):
            cmd = tokens[i]
            i += 1
            if cmd in "Zz":
                flush(True)
                cur = start.copy()
                continue
        elif cmd is None:
            break
        rel = cmd.islower()
        c = cmd.upper()
        base = cur if rel else np.zeros(2)
        if c == "M":
            flush(False)
            cur = base + [num(), num()]
            start = cur.copy()
            pts = [cur.copy()]
            cmd = "l" if rel else "L"
            last_ctrl = None
        elif c == "L":
            cur = base + [num(), num()]
            pts.append(cur.copy())
            last_ctrl = None
        elif c == "H":
            cur = np.array([(cur[0] if rel else 0.0) + num(), cur[1]])
            pts.append(cur.copy())
            last_ctrl = None
        elif c == "V":
            cur = np.array([cur[0], (cur[1] if rel else 0.0) + num()])
            pts.append(cur.copy())
            last_ctrl = None
        elif c in "CS":
            p1 = (2 * cur - last_ctrl) if (c == "S" and last_ctrl is not None) else (
                cur.copy() if c == "S" else base + [num(), num()])
            p2 = base + [num(), num()]
            p3 = base + [num(), num()]
            t = np.linspace(0, 1, 17)[1:, None]
            pts.extend((1 - t) ** 3 * cur + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)
            cur, last_ctrl = p3, p2
        elif c in "QT":
            p1 = (2 * cur - last_ctrl) if (c == "T" and last_ctrl is not None) else (
                cur.copy() if c == "T" else base + [num(), num()])
            p2 = base + [num(), num()]
            t = np.linspace(0, 1, 13)[1:, None]
            pts.extend((1 - t) ** 2 * cur + 2 * (1 - t) * t * p1 + t ** 2 * p2)
            cur, last_ctrl = p2, p1
        elif c == "A":
            rx, ry, phi, large, sweep = num(), num(), num(), num(), num()
            end = base + [num(), num()]
            pts.extend(_arc(cur, end, rx, ry, phi, bool(large), bool(sweep)))
            cur = end
            last_ctrl = None
        else:
            break
    flush(False)
    return shapes


def _arc(p0, p1, rx, ry, phi_deg, large, sweep):
    """Points along an SVG elliptical arc (endpoint parameterisation)."""
    if rx == 0 or ry == 0 or np.allclose(p0, p1):
        return [p1.copy()]
    rx, ry = abs(rx), abs(ry)
    phi = math.radians(phi_deg)
    cs, sn = math.cos(phi), math.sin(phi)
    dx, dy = (p0 - p1) / 2.0
    x1, y1 = cs * dx + sn * dy, -sn * dx + cs * dy
    lam = (x1 / rx) ** 2 + (y1 / ry) ** 2
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num_ = rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1
    den = rx * rx * y1 * y1 + ry * ry * x1 * x1
    k = math.sqrt(max(num_ / den, 0.0)) * (-1 if large == sweep else 1)
    cx1, cy1 = k * rx * y1 / ry, -k * ry * x1 / rx
    centre = np.array([cs * cx1 - sn * cy1, sn * cx1 + cs * cy1]) + (p0 + p1) / 2.0
    ang = lambda ux, uy: math.atan2(uy, ux)  # noqa: E731
    t1 = ang((x1 - cx1) / rx, (y1 - cy1) / ry)
    dt = ang((-x1 - cx1) / rx, (-y1 - cy1) / ry) - t1
    if sweep and dt < 0:
        dt += 2 * math.pi
    elif not sweep and dt > 0:
        dt -= 2 * math.pi
    out = []
    for t in np.linspace(t1, t1 + dt, max(4, int(abs(dt) / (math.pi / 24))) + 1)[1:]:
        x, y = rx * math.cos(t), ry * math.sin(t)
        out.append(centre + [cs * x - sn * y, sn * x + cs * y])
    return out


def _along(paths, count: int) -> np.ndarray:
    """Dense, even samples along every line of a drawing."""
    segs = []
    for pts, closed in paths:
        p = np.vstack([pts, pts[:1]]) if closed else pts
        segs.append(p)
    total = sum(float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) for p in segs)
    if total <= 0:
        raise ImportError_("the drawing's lines have no length")
    step = total / max(20 * count, 2000)
    out = []
    for p in segs:
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])
        if s[-1] <= 0:
            continue
        at = np.arange(0.0, s[-1], step)
        out.append(np.column_stack([np.interp(at, s, p[:, 0]), np.interp(at, s, p[:, 1])]))
    return np.concatenate(out)


def _fill(paths, count: int, seed: int) -> np.ndarray:
    """Points spread over the filled areas, holes (even-odd) left empty."""
    from PIL import Image, ImageChops, ImageDraw

    from .geometry import from_mask

    closed = [p for p, c in paths if c and len(p) >= 3] or [p for p, _ in paths if len(p) >= 3]
    if not closed:
        raise ImportError_("the drawing has no closed shapes to fill; use its outline")
    allp = np.concatenate(closed)
    lo, span = allp.min(axis=0), max(float(np.ptp(allp, axis=0).max()), 1e-9)
    res = 768
    scale = (res - 4) / span
    size = tuple(int(math.ceil(v * scale)) + 4 for v in np.ptp(allp, axis=0))
    mask = Image.new("1", size, 0)
    for p in closed:
        layer = Image.new("1", size, 0)
        ImageDraw.Draw(layer).polygon([tuple(v) for v in ((p - lo) * scale + 2.0)], fill=1)
        mask = ImageChops.logical_xor(mask, layer)
    # Rows reversed so that, after from_mask's own flip, y grows downwards
    # like the outline path's SVG coordinates; place() flips both together.
    m = np.asarray(mask, dtype=bool)[::-1]
    pts = from_mask(count, m, width=float(np.ptp(allp[:, 0])), upright=False, seed=seed)
    return pts[:, :2]


if __name__ == "__main__":
    # Smallest end-to-end check: a cube as OBJ, and a ring as SVG.
    obj = "\n".join(["v 0 0 0", "v 1 0 0", "v 1 1 0", "v 0 1 0", "v 0 0 1", "v 1 0 1",
                     "v 1 1 1", "v 0 1 1", "f 1 2 3 4", "f 5 6 7 8", "f 1 2 6 5",
                     "f 2 3 7 6", "f 3 4 8 7", "f 4 1 5 8"]).encode()
    p, info = place(obj, "cube.obj", 50, size=20.0)
    assert p.shape == (50, 3) and abs(float(np.ptp(p, axis=0).max()) - 20.0) < 1e-6, info
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="40"/></svg>'
    q, info = place(svg, "ring.svg", 30, size=30.0)
    assert q.shape == (30, 3) and np.allclose(q[:, 1], 0.0), info
    # An arrow pointing up on the page must point up in the sky, both ways.
    tri = b'<svg><path d="M50 0 L100 100 L0 100 Z"/></svg>'
    for how in ("outline", "fill"):
        r, _ = place(tri, "arrow.svg", 40, mode=how)
        top = r[np.argmax(r[:, 2])]
        assert abs(top[0]) < 0.25 * np.ptp(r[:, 0]), (how, top)
    print("models.py self-check ok")
