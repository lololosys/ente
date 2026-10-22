"""Turning a remote file into a decoded image."""

import io

from PIL import Image, ImageOps

from .ente_api import IMAGE, PublicAlbum, RemoteFile

try:  # HEIC originals from phones
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass


def _decode(data: bytes) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        return None


def load_image(f: RemoteFile, album: PublicAlbum) -> Image.Image | None:
    """The original for still images (faces in group shots are small), else
    the thumbnail: videos, live photos, and originals Pillow can't decode."""
    if f.file_type == IMAGE:
        if img := _decode(album.original(f)):
            return img
    return _decode(album.thumbnail(f))
