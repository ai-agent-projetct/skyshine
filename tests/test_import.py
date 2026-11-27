"""Designs imported from other programs, and editing them afterwards."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from skyshine import geometry, models

# A unit cube, in every format the importer reads.
CUBE_V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                   [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
CUBE_T = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                   [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
                  dtype=np.uint32)


def obj_bytes():
    lines = [f"v {x} {y} {z}" for x, y, z in CUBE_V]
    lines += [f"f {a + 1}/1 {b + 1}/1 {c + 1}/1" for a, b, c in CUBE_T]
    return "\n".join(lines).encode()


def stl_bytes():
    out = bytearray(80) + struct.pack("<I", len(CUBE_T))
    for tri in CUBE_T:
        out += struct.pack("<3f", 0, 0, 0) + CUBE_V[tri].astype("<f4").tobytes() + b"\0\0"
    return bytes(out)


def ply_bytes():
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex 8\n"
            "property float x\nproperty float y\nproperty float z\n"
            "element face 12\nproperty list uchar int vertex_indices\nend_header\n")
    body = CUBE_V.astype("<f4").tobytes()
    for tri in CUBE_T:
        body += struct.pack("<B3i", 3, *map(int, tri))
    return head.encode() + body


def glb_bytes(translate=(0, 0, 0)):
    pos, idx = CUBE_V.astype("<f4").tobytes(), CUBE_T.astype("<u4").tobytes()
    doc = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "translation": list(translate)}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(pos) + len(idx)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos)},
                        {"buffer": 0, "byteOffset": len(pos), "byteLength": len(idx)}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3"},
                      {"bufferView": 1, "componentType": 5125, "count": 36, "type": "SCALAR"}],
    }
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    bin_ = pos + idx
    total = 12 + 8 + len(js) + 8 + len(bin_)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON")
            + js + struct.pack("<I4s", len(bin_), b"BIN\0") + bin_)


SVG_LOGO = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">
  <g transform="translate(10 10)">
    <rect x="0" y="0" width="80" height="80"/>
    <path d="M120 0 h60 v80 h-60 z M135 15 h30 v50 h-30 z"/>
    <circle cx="40" cy="40" r="20"/>
    <path d="M100 90 A 20 10 0 0 1 140 90"/>
  </g>
</svg>"""


class TestFormats:
    @pytest.mark.parametrize("name,data", [
        ("cube.obj", obj_bytes()), ("cube.stl", stl_bytes()),
        ("cube.ply", ply_bytes()), ("cube.glb", glb_bytes()),
    ])
    def test_every_model_format_gives_the_same_cube(self, name, data):
        pts, info = models.place(data, name, 60, size=20.0, up="z")
        assert pts.shape == (60, 3) and info["triangles"] == 12
        assert np.allclose(np.ptp(pts, axis=0), 20.0, atol=1.5)  # a cube stays a cube
        # Spread out, not clumped: nearest neighbours are a sensible distance apart.
        assert geometry.min_pair_distance(pts) > 1.0

    def test_y_up_models_stand_upright(self):
        tall = np.array([[0, 0, 0], [1, 0, 0], [0, 5, 0]], dtype=np.float32)  # tall in Y
        obj = "\n".join(f"v {x} {y} {z}" for x, y, z in tall) + "\nf 1 2 3"
        pts, _ = models.place(obj.encode(), "t.obj", 20, up="y", size=10.0)
        assert np.ptp(pts[:, 2]) == pytest.approx(10.0)  # Y became height

    def test_glb_node_transforms_are_applied(self):
        a, _ = models.place(glb_bytes(), "a.glb", 30, size=10.0, up="z")
        b, _ = models.place(glb_bytes(translate=(5, 0, 0)), "b.glb", 30, size=10.0, up="z")
        assert np.allclose(a, b)  # centred either way: the shift moved, not distorted

    def test_svg_outline_and_fill(self):
        out, info = models.place(SVG_LOGO, "logo.svg", 80, mode="outline", size=50.0)
        assert info["paths"] == 5 and np.allclose(out[:, 1], 0.0)
        fill, _ = models.place(SVG_LOGO, "logo.svg", 80, mode="fill", size=50.0)
        assert fill.shape == (80, 3)
        assert geometry.min_pair_distance(fill) > 0.5

    @pytest.mark.parametrize("name,data,match", [
        ("x.fbx", b"Kaydara FBX Binary", "FBX is not readable"),
        ("x.obj", b"", "empty"),
        ("x.obj", b"v 0 0 0\nv 0 0 0\nf 1 2 2", "no area|distinct"),
        ("x.glb", b"glTF" + b"\0" * 20, "damaged|glTF"),
        ("x.svg", b"<svg><g></g></svg>", "no lines"),
        ("x.svg", b"<svg><unclosed", "not valid XML"),
    ])
    def test_bad_files_are_refused_with_a_reason(self, name, data, match):
        with pytest.raises(ValueError, match=match):
            models.place(data, name, 10)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from skyshine import server

    monkeypatch.setattr(server, "SHOW_DIR", tmp_path / "shows")
    (tmp_path / "shows").mkdir()
    c = TestClient(server.app)
    c.post("/api/design/shows", json={"name": "imp", "drone_count": 40})
    return c


class TestApi:
    def test_import_then_edit_then_prepare(self, client):
        r = client.post("/api/design/shows/imp/import-design",
                        files={"file": ("cube.glb", glb_bytes(), "model/gltf-binary")},
                        data={"size": "25", "altitude": "40", "color": "0,200,255", "up": "z"})
        assert r.status_code == 200, r.text
        assert r.json()["source"]["kind"] == "model"
        show = client.get("/api/design/shows/imp").json()
        f = show["formations"][0]
        assert f["name"] == "cube" and f["colors"][0] == [0, 200, 255]
        assert min(p[2] for p in f["points"]) == pytest.approx(40.0)

        before = np.array(f["points"])
        r = client.put("/api/design/shows/imp/formations/0/transform",
                       json={"move": [10, 0, 5], "turn": 90, "scale": 1.5})
        assert r.status_code == 200, r.text
        after = np.array(client.get("/api/design/shows/imp").json()["formations"][0]["points"])
        # Show files keep positions to 0.1 mm, so compare to the millimetre.
        assert np.allclose(after.mean(axis=0), before.mean(axis=0) + [10, 0, 5], atol=1e-3)
        assert np.ptp(after[:, 2]) == pytest.approx(1.5 * np.ptp(before[:, 2]), rel=1e-3)

        # Undo takes the edit back, as for any other design change.
        client.post("/api/design/shows/imp/undo")
        again = np.array(client.get("/api/design/shows/imp").json()["formations"][0]["points"])
        assert np.allclose(again, before, atol=1e-3)

        prep = client.post("/api/design/shows/imp/prepare").json()
        report = client.post(f"/api/design/shows/{prep['saved_as']}/validate").json()
        assert report["passed"], report["violations"][:3]

    def test_svg_import_and_refusals(self, client):
        r = client.post("/api/design/shows/imp/import-design",
                        files={"file": ("logo.svg", SVG_LOGO, "image/svg+xml")},
                        data={"mode": "fill"})
        assert r.status_code == 200 and r.json()["source"]["kind"] == "drawing"
        bad = client.post("/api/design/shows/imp/import-design",
                          files={"file": ("x.fbx", b"Kaydara", "application/octet-stream")})
        assert bad.status_code == 400 and "OBJ or GLB" in bad.text
        assert client.put("/api/design/shows/imp/formations/9/transform",
                          json={"turn": 10}).status_code == 404
        assert client.put("/api/design/shows/imp/formations/0/transform",
                          json={"scale": 0}).status_code == 400
