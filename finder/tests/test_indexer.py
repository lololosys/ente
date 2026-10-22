from conftest import FakeAlbum, jpeg, rf

from finder.ente_api import VIDEO
from finder.images import load_image
from finder.indexer import index_album, index_folders
from finder.store import Store


def test_folders_written_from_part_links(tmp_path):
    s = Store(tmp_path / "f.db")
    parts = [
        FakeAlbum("Wedding · A", [rf(1), rf(2)]),
        FakeAlbum("Wedding · B", [rf(3)]),
    ]
    index_folders(parts, s)
    assert s.folders() == {"A": [1, 2], "B": [3]}


def test_video_uses_thumbnail():
    album = FakeAlbum("w", [rf(1, VIDEO)])
    img = load_image(album.items[0], album)
    assert img.size == (40, 30)
    assert album.original_calls == []


def test_undecodable_original_falls_back_to_thumbnail():
    album = FakeAlbum("w", [rf(1, title="x.cr3")], originals={1: b"not an image"})
    img = load_image(album.items[0], album)
    assert img.size == (40, 30)


def test_original_used_for_images():
    album = FakeAlbum("w", [rf(1)])
    assert load_image(album.items[0], album).size == (400, 300)


def test_rerun_skips_done_files(tmp_path):
    s = Store(tmp_path / "f.db")
    calls = []
    stages = {"faces": lambda f, img: calls.append(f.id)}
    album = FakeAlbum("w", [rf(1), rf(2)])
    index_album(album, s, stages)
    album.items.append(rf(3))
    index_album(album, s, stages)
    assert calls == [1, 2, 3]


def test_failing_stage_leaves_file_pending_and_continues(tmp_path):
    s = Store(tmp_path / "f.db")

    def boom(f, img):
        if f.id == 1:
            raise RuntimeError("model failed")

    index_album(FakeAlbum("w", [rf(1), rf(2)]), s, {"faces": boom})
    assert s.pending("faces") == [1]
