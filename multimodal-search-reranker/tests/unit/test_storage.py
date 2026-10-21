import io

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from mmsearch.media.storage import ImageValidationError, MediaStore, decode_image  # noqa: E402
from tests.unit.fakes import png_bytes  # noqa: E402

LIMITS = dict(max_bytes=1_000_000, max_pixels=1_000_000)


def test_decodes_png_and_jpeg_to_rgb():
    for fmt, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")):
        d = decode_image(png_bytes(fmt=fmt), **LIMITS)
        assert d.mime_type == mime and d.image.mode == "RGB"
        assert (d.width, d.height) == (64, 64)
        assert len(d.sha256) == 64


def test_rgba_and_palette_images_are_converted():
    buf = io.BytesIO()
    Image.new("RGBA", (10, 10), (255, 0, 0, 128)).save(buf, "PNG")
    assert decode_image(buf.getvalue(), **LIMITS).image.mode == "RGB"
    buf = io.BytesIO()
    Image.new("P", (10, 10)).save(buf, "GIF")
    assert decode_image(buf.getvalue(), **LIMITS).mime_type == "image/gif"


@pytest.mark.parametrize("data, msg", [
    (b"", "empty"),
    (b"definitely not an image", "not a recognised image"),
    (b"\x89PNG\r\n\x1a\n" + b"\x00" * 40, "not a recognised image"),
])
def test_rejects_garbage(data, msg):
    with pytest.raises(ImageValidationError, match=msg):
        decode_image(data, **LIMITS)


def test_rejects_oversized_bytes_before_decoding():
    with pytest.raises(ImageValidationError, match="limit"):
        decode_image(png_bytes(), max_bytes=100, max_pixels=10**9)


def test_rejects_too_many_pixels_from_header():
    with pytest.raises(ImageValidationError, match="pixels"):
        decode_image(png_bytes(size=(400, 300)), max_bytes=10**7, max_pixels=100_000)


def test_rejects_truncated_image():
    data = png_bytes(fmt="JPEG", size=(256, 256))
    with pytest.raises(ImageValidationError):
        decode_image(data[: len(data) // 3], **LIMITS)


def test_rejects_unsupported_format():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "BMP")
    with pytest.raises(ImageValidationError, match="unsupported"):
        decode_image(buf.getvalue(), **LIMITS)


def test_media_store_is_content_addressed_and_idempotent(tmp_path):
    store = MediaStore(tmp_path)
    data = png_bytes()
    d = decode_image(data, **LIMITS)
    rel = store.save(data, d)
    assert rel == f"{d.sha256[:2]}/{d.sha256}.png"
    assert store.resolve(rel).read_bytes() == data
    assert store.save(data, d) == rel                       # second save is a no-op
    assert len(list(tmp_path.rglob("*.png"))) == 1
    assert not list(tmp_path.rglob(".upload-*"))            # no temp files left behind


def test_media_store_blocks_path_traversal(tmp_path):
    store = MediaStore(tmp_path / "media")
    with pytest.raises(ValueError):
        store.resolve("../../etc/passwd")
