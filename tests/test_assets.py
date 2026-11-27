"""Tests for uploaded artwork: logos and text becoming formations.

Uploads are untrusted input, so the half of this file that matters is
TestUploadRejects. The rest covers thresholding, which is the step that
decides whether a logo reads as itself or as a cloud of noise.
"""

from __future__ import annotations

import io
import pathlib
import tempfile

import numpy as np
import pytest
from PIL import Image, ImageDraw

from skyshine import assets, geometry


def png_bytes(draw_fn, size=(600, 300), background=(0, 0, 0)) -> bytes:
    img = Image.new("RGB", size, background)
    draw_fn(ImageDraw.Draw(img))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def logo() -> bytes:
    """A white disc and bar on black: unambiguous to threshold."""
    return png_bytes(lambda d: (
        d.ellipse([60, 60, 240, 240], fill=(255, 255, 255)),
        d.rectangle([300, 80, 540, 220], fill=(255, 255, 255)),
    ))


def inverted_logo() -> bytes:
    """The same marks, dark on light: must want inverting."""
    return png_bytes(lambda d: (
        d.ellipse([60, 60, 240, 240], fill=(0, 0, 0)),
        d.rectangle([300, 80, 540, 220], fill=(0, 0, 0)),
    ), background=(255, 255, 255))


@pytest.fixture()
def store():
    return pathlib.Path(tempfile.mkdtemp())


# ==========================================================================
# Upload validation
# ==========================================================================


class TestUploadRejects:
    def test_empty_file(self, store):
        with pytest.raises(assets.AssetError, match="empty"):
            assets.save_upload(b"", "x.png", store)

    def test_not_an_image(self, store):
        with pytest.raises(assets.AssetError, match="not a readable image"):
            assets.save_upload(b"this is just text", "x.png", store)

    def test_a_renamed_executable(self, store):
        """A .png extension proves nothing; the decoder decides."""
        with pytest.raises(assets.AssetError, match="not a readable image"):
            assets.save_upload(b"MZ\x90\x00\x03" + b"\x00" * 200, "logo.png", store)

    def test_svg_is_refused_with_an_actionable_message(self, store):
        svg = (b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg">'
               b'<rect width="10" height="10"/></svg>')
        with pytest.raises(assets.AssetError) as exc:
            assets.save_upload(svg, "logo.svg", store)
        assert "SVG" in str(exc.value)
        assert "PNG" in str(exc.value), "the message must say what to do instead"

    def test_bare_svg_without_xml_declaration(self, store):
        with pytest.raises(assets.AssetError, match="SVG"):
            assets.save_upload(b'<svg xmlns="http://www.w3.org/2000/svg"/>', "a.svg", store)

    def test_oversized_file(self, store):
        with pytest.raises(assets.AssetError, match="over the"):
            assets.save_upload(b"x" * (assets.MAX_BYTES + 1), "big.png", store)

    def test_tiny_image(self, store):
        data = png_bytes(lambda d: d.point((0, 0)), size=(4, 4))
        with pytest.raises(assets.AssetError, match="too small"):
            assets.save_upload(data, "tiny.png", store)

    def test_a_truncated_png_is_caught(self, store):
        data = logo()[: len(logo()) // 2]
        with pytest.raises(assets.AssetError, match="not a readable image"):
            assets.save_upload(data, "half.png", store)


class TestUploadStorage:
    def test_stored_under_a_generated_name(self, store):
        a = assets.save_upload(logo(), "../../etc/passwd", store)
        assert a.path.parent == store
        assert "passwd" not in a.path.name
        assert a.path.name.startswith(a.id)

    def test_display_name_is_kept_but_stripped_of_its_path(self, store):
        a = assets.save_upload(logo(), "../../secret/brand.png", store)
        assert a.name == "brand.png"

    def test_same_content_is_stored_once(self, store):
        a = assets.save_upload(logo(), "one.png", store)
        b = assets.save_upload(logo(), "two.png", store)
        assert a.id == b.id
        assert len(list(store.glob("*.png"))) == 1

    def test_listing_and_resolving(self, store):
        a = assets.save_upload(logo(), "logo.png", store)
        rows = assets.list_assets(store)
        assert [r["id"] for r in rows] == [a.id]
        assert assets.resolve(store, a.id) == a.path

    def test_resolving_an_unknown_id(self, store):
        with pytest.raises(assets.AssetError, match="no asset"):
            assets.resolve(store, "deadbeef")

    def test_resolving_a_traversal_id(self, store):
        for evil in ("../../etc/passwd", "..\\..\\windows\\system32\\config\\sam",
                     "/etc/shadow"):
            with pytest.raises(assets.AssetError):
                assets.resolve(store, evil)

    def test_delete_removes_file_and_index_entry(self, store):
        a = assets.save_upload(logo(), "logo.png", store)
        assets.delete(store, a.id)
        assert not a.path.exists()
        assert assets.list_assets(store) == []
        with pytest.raises(assets.AssetError):
            assets.resolve(store, a.id)

    def test_listing_skips_entries_whose_file_vanished(self, store):
        a = assets.save_upload(logo(), "logo.png", store)
        a.path.unlink()
        assert assets.list_assets(store) == []

    def test_a_corrupt_index_does_not_crash_the_listing(self, store):
        assets.save_upload(logo(), "logo.png", store)
        (store / "index.json").write_text("{ not json", encoding="utf-8")
        assert assets.list_assets(store) == []


# ==========================================================================
# Thresholding
# ==========================================================================


class TestThresholding:
    def test_otsu_splits_a_two_tone_image(self, store):
        a = assets.save_upload(logo(), "logo.png", store)
        grey = assets.grey_array(a.path)
        thr = assets.otsu(grey)
        assert 60 < thr < 200, f"otsu chose {thr}"

    def test_otsu_on_a_flat_image_does_not_divide_by_zero(self):
        assert assets.otsu(np.full((40, 40), 128, dtype=np.uint8)) in range(256)
        assert assets.otsu(np.zeros((40, 40), dtype=np.uint8)) in range(256)

    def test_describe_suggests_no_invert_for_light_on_dark(self, store):
        a = assets.save_upload(logo(), "logo.png", store)
        info = assets.describe(a.path)
        assert info["suggested_invert"] is False
        assert 0.05 < info["fill_ratio"] < 0.6

    def test_describe_suggests_invert_for_dark_on_light(self, store):
        a = assets.save_upload(inverted_logo(), "inv.png", store)
        info = assets.describe(a.path)
        assert info["suggested_invert"] is True, (
            "a dark subject on a light ground must be flagged for inverting, "
            "or the first preview selects the background"
        )
        assert info["fill_ratio"] < 0.6

    def test_max_drones_matches_the_filled_pixels(self, store):
        a = assets.save_upload(logo(), "logo.png", store)
        grey = assets.grey_array(a.path)
        thr = assets.otsu(grey)
        assert assets.max_drones(a.path, thr, False) == int((grey >= thr).sum())

    def test_invert_flips_the_count(self, store):
        a = assets.save_upload(logo(), "logo.png", store)
        normal = assets.max_drones(a.path, 128, False)
        inverted = assets.max_drones(a.path, 128, True)
        total = normal + inverted
        assert total > 0 and normal != inverted

    def test_an_extreme_threshold_selects_nothing(self, store):
        a = assets.save_upload(logo(), "logo.png", store)
        assert assets.max_drones(a.path, 256, False) == 0


class TestPreview:
    def test_preview_is_a_valid_png(self, store):
        a = assets.save_upload(logo(), "logo.png", store)
        png = assets.preview_png(a.path, drones=120, threshold=128)
        img = Image.open(io.BytesIO(png))
        assert img.format == "PNG"
        assert max(img.size) <= assets.PREVIEW_SIZE

    def test_preview_draws_the_drones_not_just_the_mask(self, store):
        """The dots are the point: a mask alone hides how coarse it will be."""
        a = assets.save_upload(logo(), "logo.png", store)
        plain = assets.preview_png(a.path, drones=1, threshold=128)
        many = assets.preview_png(a.path, drones=300, threshold=128)
        assert plain != many, "the drone count must change the preview"

    def test_preview_reports_an_impossible_request_instead_of_raising(self, store):
        a = assets.save_upload(logo(), "logo.png", store)
        # Threshold selects nothing: the preview must still render.
        png = assets.preview_png(a.path, drones=100, threshold=256)
        assert Image.open(io.BytesIO(png)).format == "PNG"

    def test_preview_handles_more_drones_than_pixels(self, store):
        a = assets.save_upload(logo(), "logo.png", store)
        png = assets.preview_png(a.path, drones=999999, threshold=128)
        assert Image.open(io.BytesIO(png)).format == "PNG"


# ==========================================================================
# Becoming a formation
# ==========================================================================


class TestArtworkFormations:
    def test_image_sampling_gives_the_exact_drone_count(self, store):
        a = assets.save_upload(logo(), "logo.png", store)
        for n in (30, 120, 400):
            pts = geometry.image(n, str(a.path), width=90.0)
            assert pts.shape == (n, 3)
            assert geometry.min_pair_distance(pts) > 0

    def test_every_drone_lands_on_the_ink(self, store):
        """Regression: k-means centroids fell into the holes in a logo.

        A centroid is the mean of its cluster, so on a non-convex mark -
        a disc beside a bar - clusters that span both put their centre in
        the empty gap. 42% of drones ended up there, and the formation
        read as a rectangle of noise instead of the mark. Every point is
        now snapped onto an actual filled pixel.
        """
        a = assets.save_upload(logo(), "logo.png", store)
        grey = assets.grey_array(a.path)
        mask = grey >= assets.otsu(grey)
        ys, xs = np.nonzero(mask)
        h = mask.shape[0]
        pixels = np.column_stack([xs.astype(float), (h - 1 - ys).astype(float)])

        snapped = geometry._snap_to_mask(
            np.array([[280.0, 150.0], [250.0, 60.0], [10.0, 10.0]]), pixels
        )
        ink = {tuple(p) for p in pixels}
        for p in snapped:
            assert tuple(p) in ink, f"{p} is not on the artwork"

    def test_snapping_keeps_drones_distinct(self, store):
        """Two drones snapped to one pixel is two drones at one position."""
        a = assets.save_upload(logo(), "logo.png", store)
        grey = assets.grey_array(a.path)
        mask = grey >= assets.otsu(grey)
        ys, xs = np.nonzero(mask)
        h = mask.shape[0]
        pixels = np.column_stack([xs.astype(float), (h - 1 - ys).astype(float)])

        # Twenty centroids piled on the same spot: the worst case.
        crowd = np.tile(np.array([[150.0, 150.0]]), (20, 1))
        snapped = geometry._snap_to_mask(crowd, pixels)
        assert len({tuple(p) for p in snapped}) == len(snapped)

    def test_sampled_formations_have_no_coincident_drones(self, store):
        a = assets.save_upload(logo(), "logo.png", store)
        for n in (50, 200, 500):
            pts = geometry.image(n, str(a.path), width=90.0)
            assert geometry.min_pair_distance(pts) > 1e-9, (
                f"{n} drones produced a coincident pair"
            )

    def test_text_formation(self):
        pts = geometry.text(120, "SKY", width=100.0)
        assert pts.shape == (120, 3)
        assert geometry.min_pair_distance(pts) > 0.5

    def test_asking_for_more_drones_than_pixels_is_refused(self, store):
        small = png_bytes(lambda d: d.rectangle([4, 4, 9, 9], fill=(255, 255, 255)),
                          size=(40, 40))
        a = assets.save_upload(small, "dot.png", store)
        with pytest.raises(ValueError, match="higher resolution"):
            geometry.image(5000, str(a.path), width=50.0)


# ==========================================================================
# API
# ==========================================================================


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from skyshine import server

    tmp = pathlib.Path(tempfile.mkdtemp())
    monkeypatch.setattr(server, "SHOW_DIR", tmp / "shows")
    monkeypatch.setattr(server, "LOG_DIR", tmp / "logs")
    monkeypatch.setattr(server, "ASSET_DIR", tmp / "assets")
    monkeypatch.setattr(server, "_fleet", None)
    (tmp / "shows").mkdir()
    return TestClient(server.app)


def upload(client, data=None, name="logo.png"):
    return client.post("/api/design/assets",
                       files={"file": (name, data or logo(), "image/png")})


class TestAssetApi:
    def test_upload_returns_threshold_guidance(self, client):
        r = upload(client)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["width"] == 600 and body["height"] == 300
        assert 0 <= body["suggested_threshold"] <= 255
        assert body["max_drones"] > 100

    def test_upload_rejects_a_non_image_with_400(self, client):
        r = upload(client, b"not an image at all")
        assert r.status_code == 400
        assert "not a readable image" in r.text

    def test_upload_rejects_svg_with_guidance(self, client):
        r = upload(client, b'<svg xmlns="http://www.w3.org/2000/svg"/>', "l.svg")
        assert r.status_code == 400
        assert "PNG" in r.json()["detail"]

    def test_preview_endpoint_returns_a_png(self, client):
        asset_id = upload(client).json()["id"]
        r = client.get(f"/api/design/assets/{asset_id}/preview?drones=80&threshold=128")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert Image.open(io.BytesIO(r.content)).format == "PNG"

    def test_preview_of_an_unknown_asset_is_a_404(self, client):
        assert client.get("/api/design/assets/nope/preview").status_code == 404

    def test_capacity_endpoint(self, client):
        asset_id = upload(client).json()["id"]
        a = client.get(f"/api/design/assets/{asset_id}/capacity?threshold=128").json()
        b = client.get(
            f"/api/design/assets/{asset_id}/capacity?threshold=128&invert=true").json()
        assert a["max_drones"] != b["max_drones"]

    def test_list_and_delete(self, client):
        asset_id = upload(client).json()["id"]
        assert len(client.get("/api/design/assets").json()["assets"]) == 1
        assert client.delete(f"/api/design/assets/{asset_id}").status_code == 200
        assert client.get("/api/design/assets").json()["assets"] == []
        assert client.delete(f"/api/design/assets/{asset_id}").status_code == 404

    def test_a_logo_becomes_a_flyable_formation(self, client):
        asset_id = upload(client).json()["id"]
        client.post("/api/design/shows", json={
            "name": "brand", "drone_count": 150,
            "limits": {"max_altitude": 200.0, "battery_endurance_s": 1800.0},
            "origin": {"latitude": 12.97, "longitude": 77.59, "amsl": 920.0},
        })
        r = client.post("/api/design/shows/brand/artwork", json={
            "name": "our logo", "asset_id": asset_id,
            "threshold": 128, "width": 100.0, "altitude": 60.0,
        })
        assert r.status_code == 200, r.text
        assert r.json()["raw_min_separation_m"] > 0

        prepared = client.post("/api/design/shows/brand/prepare").json()
        report = client.post(
            f"/api/design/shows/{prepared['saved_as']}/validate").json()
        assert report["passed"], [v["message"] for v in report["violations"]]
        assert report["metrics"]["min_separation_m"] >= 3.0

    def test_text_becomes_a_formation(self, client):
        client.post("/api/design/shows", json={
            "name": "words", "drone_count": 120,
            "limits": {"max_altitude": 200.0, "battery_endurance_s": 1800.0},
        })
        r = client.post("/api/design/shows/words/artwork", json={
            "name": "greeting", "text": "SKY", "width": 110.0, "altitude": 55.0,
        })
        assert r.status_code == 200, r.text
        detail = client.get("/api/design/shows/words").json()
        assert detail["formations"][0]["name"] == "greeting"

    def test_both_asset_and_text_is_refused(self, client):
        asset_id = upload(client).json()["id"]
        client.post("/api/design/shows", json={"name": "b", "drone_count": 20})
        r = client.post("/api/design/shows/b/artwork", json={
            "name": "x", "asset_id": asset_id, "text": "HI",
        })
        assert r.status_code == 400
        assert "not both" in r.text

    def test_neither_asset_nor_text_is_refused(self, client):
        client.post("/api/design/shows", json={"name": "b", "drone_count": 20})
        r = client.post("/api/design/shows/b/artwork", json={"name": "x"})
        assert r.status_code == 400

    def test_too_many_drones_for_the_artwork_is_explained(self, client):
        small = png_bytes(lambda d: d.rectangle([4, 4, 9, 9], fill=(255, 255, 255)),
                          size=(40, 40))
        asset_id = upload(client, small, "dot.png").json()["id"]
        client.post("/api/design/shows", json={"name": "big", "drone_count": 500})
        r = client.post("/api/design/shows/big/artwork", json={
            "name": "x", "asset_id": asset_id, "threshold": 128,
        })
        assert r.status_code == 400
        body = r.json()["detail"]
        assert "usable pixels" in body
        assert "higher-resolution" in body or "Lower the threshold" in body


class TestFitWarning:
    def test_adding_a_shape_that_will_bust_the_ceiling_warns_at_once(self, client):
        client.post("/api/design/shows", json={"name": "w", "drone_count": 100})
        outline = client.post("/api/design/shows/w/formations", json={
            "name": "heart", "shape": "heart"}).json()
        assert "ceiling" in outline["warning"]
        # The warning is the same note prepare gives later, not a second opinion.
        assert outline["warning"] in client.post(
            "/api/design/shows/w/prepare").json()["notes"]

        solid = client.post("/api/design/shows/w/formations", json={
            "name": "heart", "shape": "heart", "params": {"solid": True}}).json()
        assert solid["warning"] is None
