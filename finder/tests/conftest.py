import io
from dataclasses import dataclass, field

from PIL import Image

from finder.ente_api import IMAGE, RemoteFile


def jpeg(color=(200, 50, 50), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def rf(fid: int, file_type: int = IMAGE, title: str = "x.jpg") -> RemoteFile:
    return RemoteFile(fid, b"k" * 32, file_type, title, "", "")


@dataclass
class FakeAlbum:
    """Duck-types finder.ente_api.PublicAlbum."""

    name: str
    items: list[RemoteFile]
    originals: dict[int, bytes] = field(default_factory=dict)
    thumbs: dict[int, bytes] = field(default_factory=dict)
    original_calls: list[int] = field(default_factory=list)

    def files(self):
        return list(self.items)

    def album_name(self):
        return self.name

    def original(self, f):
        self.original_calls.append(f.id)
        return self.originals.get(f.id, jpeg(size=(400, 300)))

    def thumbnail(self, f):
        return self.thumbs.get(f.id, jpeg(size=(40, 30)))
