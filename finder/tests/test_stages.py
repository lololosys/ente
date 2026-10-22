from conftest import FakeAlbum, fixture_image, rf

from finder.indexer import index_album
from finder.stages import clip_stage, face_stage
from finder.store import Store


def test_face_and_clip_stages_store_results(tmp_path, models):
    s = Store(tmp_path / "f.db")
    import io

    buf = io.BytesIO()
    fixture_image("person_a_1.jpg").save(buf, "JPEG")
    album = FakeAlbum("w", [rf(7)], originals={7: buf.getvalue()})
    index_album(album, s, {"faces": face_stage(models, s), "clip": clip_stage(models, s)})
    assert [r.file_id for r in s.face_rows()] == [7]
    assert list(s.clip_rows()) == [7]
    assert s.pending("faces") == [] and s.pending("clip") == []
