"""Validating and storing image bytes.

Uploads are untrusted input, so before anything touches a model:
* size is capped before decoding (``max_bytes``);
* the format is sniffed from the bytes by Pillow, never trusted from the filename
  or Content-Type;
* the pixel count is checked from the header *before* decoding, which stops
  decompression bombs (a tiny file that decodes to gigabytes);
* the full decode happens once, in a guarded block, and yields an RGB image.

Files are content-addressed by SHA-256: identical bytes are stored once, the path is
derived from the hash (never from user input), and re-uploading is a no-op.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

FORMATS = {  # Pillow format -> (mime type, file extension)
    "JPEG": ("image/jpeg", "jpg"),
    "PNG": ("image/png", "png"),
    "WEBP": ("image/webp", "webp"),
    "GIF": ("image/gif", "gif"),
}


class ImageValidationError(ValueError):
    """Raised for anything that is not an acceptable image. Message is user-facing."""


@dataclass
class DecodedImage:
    image: Image.Image          # RGB, fully loaded
    sha256: str
    format: str
    mime_type: str
    extension: str
    width: int
    height: int
    nbytes: int


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_image(data: bytes, *, max_bytes: int, max_pixels: int) -> DecodedImage:
    if not data:
        raise ImageValidationError("empty file")
    if len(data) > max_bytes:
        raise ImageValidationError(f"file is {len(data)} bytes; the limit is {max_bytes}")
    try:
        with Image.open(io.BytesIO(data)) as probe:          # reads the header only
            fmt, (w, h) = probe.format, probe.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError("not a recognised image file") from exc
    if fmt not in FORMATS:
        raise ImageValidationError(f"unsupported format {fmt}; use JPEG, PNG, WEBP or GIF")
    if w * h > max_pixels:
        raise ImageValidationError(f"image is {w}x{h}; the limit is {max_pixels} pixels")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.seek(0)                                      # first frame of a GIF
            rgb = img.convert("RGB")                         # forces a full decode
    except (OSError, Image.DecompressionBombError, ValueError) as exc:
        raise ImageValidationError("image data is corrupt or truncated") from exc
    mime, ext = FORMATS[fmt]
    return DecodedImage(image=rgb, sha256=sha256_bytes(data), format=fmt, mime_type=mime,
                        extension=ext, width=w, height=h, nbytes=len(data))


class MediaStore:
    """Files live at <root>/<sha[:2]>/<sha>.<ext>; the DB stores the relative path."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def relative_path(sha256: str, extension: str) -> str:
        return f"{sha256[:2]}/{sha256}.{extension}"

    def save(self, data: bytes, decoded: DecodedImage) -> str:
        rel = self.relative_path(decoded.sha256, decoded.extension)
        dest = self.root / rel
        if dest.exists():
            return rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file, then rename: a crash never leaves a half-written image.
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".upload-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, dest)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return rel

    def resolve(self, rel: str) -> Path:
        path = (self.root / rel).resolve()
        if self.root not in path.parents:
            raise ValueError("path escapes the media directory")
        return path
