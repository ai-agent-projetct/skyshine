"""Uploaded artwork: logos and images that become formations.

Turning a picture into a formation is mostly a thresholding problem, and
the threshold is the thing nobody can guess from the filename. A logo on a
white background and the same logo knocked out of a dark one need opposite
settings, and getting it wrong produces a cloud of drones in the shape of
nothing in particular.

So this module exists to make the choice visible: `preview_png` renders
the mask *and the drone positions that would actually be sampled from it*,
at the real drone count, before anything is added to a show. Otsu gives a
sensible starting threshold so the first preview is usually close.

Uploaded files are untrusted input. They are decoded by Pillow (which
rejects anything that is not an image), stored under a generated name
rather than the one the browser supplied, capped in both bytes and pixels,
and never executed or passed to a shell.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# A 24 megapixel photo is already far more detail than a few hundred
# drones can express, and the cap is mostly here to stop a decompression
# bomb turning an upload into an out-of-memory kill.
MAX_BYTES = 24 * 1024 * 1024
MAX_PIXELS = 40_000_000
PREVIEW_SIZE = 420

# Formats Pillow reads that make sense as show artwork. SVG is absent on
# purpose - Pillow cannot rasterise it, and the honest answer is to say so
# rather than to fail with a confusing decode error.
ACCEPTED = {
    "PNG", "JPEG", "GIF", "BMP", "TIFF", "WEBP", "ICO", "PPM", "TGA", "AVIF",
}


class AssetError(ValueError):
    """Something about the uploaded file means it cannot become a formation."""


@dataclass
class Asset:
    id: str
    name: str
    path: Path
    width: int
    height: int
    format: str
    uploaded: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "uploaded": self.uploaded,
            "path": str(self.path),
        }


def _looks_like_svg(data: bytes) -> bool:
    head = data[:400].lstrip().lower()
    return head.startswith(b"<?xml") and b"<svg" in data[:2000].lower() or head.startswith(b"<svg")


def save_upload(data: bytes, filename: str, directory: Path) -> Asset:
    """Validate an uploaded image and store it under a generated name.

    The caller's filename is used only for display. The stored name comes
    from a hash of the content, so a traversal attempt like
    `../../etc/passwd` cannot reach outside the asset directory, and
    uploading the same logo twice does not make two copies.
    """
    if not data:
        raise AssetError("the uploaded file is empty")
    if len(data) > MAX_BYTES:
        raise AssetError(
            f"file is {len(data) / 1e6:.1f} MB, over the {MAX_BYTES / 1e6:.0f} MB limit"
        )
    if _looks_like_svg(data):
        raise AssetError(
            "that is an SVG, which cannot be rasterised here. Export it as a "
            "PNG at around 1000 px on the long edge and upload that - a logo "
            "only has to carry a few hundred drone positions, so resolution "
            "beyond that buys nothing."
        )

    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()                      # structural check, consumes the file
        img = Image.open(io.BytesIO(data))  # reopen: verify() leaves it unusable
        img.load()
    except Exception as exc:
        raise AssetError(f"that file is not a readable image ({exc})") from None

    fmt = (img.format or "").upper()
    if fmt not in ACCEPTED:
        raise AssetError(
            f"{fmt or 'unknown'} is not a supported image format. "
            f"Use one of: {', '.join(sorted(ACCEPTED))}."
        )
    if img.width * img.height > MAX_PIXELS:
        raise AssetError(
            f"{img.width}x{img.height} is {img.width * img.height / 1e6:.0f} "
            "megapixels, more than this accepts. Downscale it first."
        )
    if min(img.width, img.height) < 8:
        raise AssetError(f"{img.width}x{img.height} is too small to sample")

    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()[:16]
    suffix = {"JPEG": ".jpg"}.get(fmt, f".{fmt.lower()}")
    target = directory / f"{digest}{suffix}"
    if not target.exists():
        target.write_bytes(data)

    asset = Asset(
        id=digest,
        name=Path(filename or "upload").name[:80],
        path=target,
        width=img.width,
        height=img.height,
        format=fmt,
        uploaded=time.time(),
    )
    _write_index(directory, asset)
    return asset


def _index_path(directory: Path) -> Path:
    return directory / "index.json"


def _write_index(directory: Path, asset: Asset) -> None:
    index = _read_index(directory)
    index[asset.id] = {
        "id": asset.id, "name": asset.name, "file": asset.path.name,
        "width": asset.width, "height": asset.height,
        "format": asset.format, "uploaded": asset.uploaded,
    }
    _index_path(directory).write_text(json.dumps(index, indent=1), encoding="utf-8")


def _read_index(directory: Path) -> dict:
    p = _index_path(directory)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def list_assets(directory: Path) -> list[dict]:
    rows = []
    for row in _read_index(directory).values():
        if (directory / row["file"]).exists():
            rows.append(row)
    return sorted(rows, key=lambda r: -r.get("uploaded", 0))


def resolve(directory: Path, asset_id: str) -> Path:
    """Map an asset id to its file, refusing anything outside the directory."""
    row = _read_index(directory).get(asset_id)
    if not row:
        raise AssetError(f"no asset {asset_id!r}")
    target = (directory / row["file"]).resolve()
    base = directory.resolve()
    # Belt and braces: ids come from our own index, but a tampered index
    # must not be able to hand back /etc/shadow.
    if base not in target.parents and target.parent != base:
        raise AssetError("asset path escapes the asset directory")
    if not target.exists():
        raise AssetError(f"asset {asset_id!r} is indexed but its file is gone")
    return target


def delete(directory: Path, asset_id: str) -> None:
    index = _read_index(directory)
    row = index.pop(asset_id, None)
    if row is None:
        raise AssetError(f"no asset {asset_id!r}")
    target = directory / row["file"]
    if target.exists():
        target.unlink()
    _index_path(directory).write_text(json.dumps(index, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------
# Thresholding
# --------------------------------------------------------------------------


def grey_array(path: Path, resolution: int = 512) -> np.ndarray:
    """Load an image as greyscale, flattened onto black, downscaled."""
    img = Image.open(path).convert("RGBA")
    flat = Image.new("RGBA", img.size, (0, 0, 0, 255))
    flat.alpha_composite(img)
    grey = flat.convert("L")
    if max(grey.size) > resolution:
        scale = resolution / max(grey.size)
        grey = grey.resize(
            (max(1, int(grey.width * scale)), max(1, int(grey.height * scale))),
            Image.LANCZOS,
        )
    return np.asarray(grey)


def otsu(grey: np.ndarray) -> int:
    """Otsu's threshold: the split that best separates the histogram.

    Gives the first preview a sensible starting point instead of a fixed
    128, which is wrong for most logos.
    """
    hist = np.bincount(grey.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128
    levels = np.arange(256)
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not np.any(valid):
        return 128
    cum_mean = np.cumsum(hist * levels)
    mean_bg = np.divide(cum_mean, weight_bg, out=np.zeros(256), where=weight_bg > 0)
    mean_total = cum_mean[-1]
    mean_fg = np.divide(
        mean_total - cum_mean, weight_fg, out=np.zeros(256), where=weight_fg > 0
    )
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = -1.0
    return int(np.argmax(between))


def describe(path: Path, resolution: int = 512) -> dict:
    """What the designer needs in order to choose a threshold."""
    grey = grey_array(path, resolution)
    suggested = otsu(grey)
    mask = grey >= suggested
    fill = float(mask.mean())
    # If the suggested threshold selects most of the frame, the subject is
    # almost certainly dark on light and wants inverting.
    invert = fill > 0.5
    if invert:
        mask = ~mask
        fill = float(mask.mean())
    return {
        "suggested_threshold": suggested,
        "suggested_invert": invert,
        "fill_ratio": round(fill, 4),
        "filled_pixels": int(mask.sum()),
        "preview_width": int(grey.shape[1]),
        "preview_height": int(grey.shape[0]),
    }


def max_drones(path: Path, threshold: int, invert: bool, resolution: int = 512) -> int:
    """How many drones this mask can hold at this resolution.

    `from_mask` cannot place more drones than there are filled pixels, so
    this is the hard ceiling and the UI should say so before the attempt
    rather than after the exception.
    """
    grey = grey_array(path, resolution)
    mask = (grey < threshold) if invert else (grey >= threshold)
    return int(mask.sum())


def preview_png(
    path: Path,
    *,
    drones: int,
    threshold: int,
    invert: bool = False,
    resolution: int = 512,
    size: int = PREVIEW_SIZE,
    seed: int = 0,
) -> bytes:
    """Render the mask with the drone positions that would be sampled.

    This is the whole point of the module. The mask alone tells you what
    survived the threshold; the dots tell you what the audience will
    actually see, which at 120 drones is a very different picture.
    """
    from . import geometry

    grey = grey_array(path, resolution)
    mask = (grey < threshold) if invert else (grey >= threshold)
    h, w = mask.shape

    canvas = Image.new("RGB", (w, h), (10, 12, 18))
    canvas.paste(Image.new("RGB", (w, h), (38, 46, 62)), (0, 0),
                 Image.fromarray((mask * 255).astype(np.uint8)))

    note = None
    filled = int(mask.sum())
    if filled == 0:
        note = "threshold selects nothing"
    elif drones > filled:
        note = f"only {filled} pixels for {drones} drones"
    else:
        try:
            pts = geometry.from_mask(drones, mask, width=float(w), seed=seed)
            draw = ImageDraw.Draw(canvas)
            # from_mask centres on the origin and flips y back to image order.
            cx, cy = w / 2.0, h / 2.0
            r = max(1.2, min(w, h) / 220.0)
            for x, _y, z in pts:
                px, py = cx + x, cy - z
                draw.ellipse([px - r, py - r, px + r, py + r], fill=(90, 209, 255))
        except ValueError as exc:
            note = str(exc)[:60]

    if note:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, w, 18], fill=(60, 20, 28))
        draw.text((5, 4), note, fill=(255, 160, 170))

    if max(canvas.size) > size:
        scale = size / max(canvas.size)
        canvas = canvas.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
        )

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()
