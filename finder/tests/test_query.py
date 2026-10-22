import io

import numpy as np
import pytest
from conftest import fixture_image
from PIL import Image

from finder.faces import Face
from finder.query import Finder, ManyFaces, NoFace
from finder.stages import cluster_faces
from finder.store import Store


def unit(*v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


FACE = Face((0, 0, 10, 10), 0.9, ((1, 1),) * 5)


def jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "f.db")
    s.set_faces(1, [(FACE, unit(1, 0.10, 0))])  # 0.995 to the selfie
    s.set_faces(2, [(FACE, unit(1, 0.75, 0))])  # 0.80
    s.set_faces(3, [(FACE, unit(0, 1, 0))])  # 0.0
    s.set_faces(4, [(FACE, unit(1, 1.2, 0))])  # 0.64
    return s


def finder(store, threshold=0.7):
    return Finder(store, models=None, face_threshold=threshold, embed_selfie=lambda b: unit(1, 0, 0))


def test_find_me_orders_by_best_score_and_cuts_at_threshold(store):
    r = finder(store).find_me(b"selfie")
    assert r.file_ids == [1, 2]


def test_find_me_skips_excluded_people(store):
    cluster_faces(store)
    c = {row.file_id: row.cluster for row in store.face_rows()}[1]
    store.exclude_cluster(c)
    assert 1 not in finder(store).find_me(b"selfie").file_ids


def test_search_returns_clip_matches_above_floor_best_first(tmp_path):
    s = Store(tmp_path / "f.db")
    s.set_clip(1, unit(1, 0, 0))
    s.set_clip(2, unit(1, 1, 0))
    s.set_clip(3, unit(0, 0, 1))
    f = Finder(s, models=None, face_threshold=0.6, embed_text=lambda q: unit(1, 0.2, 0))
    assert f.search("cake") == [1, 2]
    assert f.search("   ") == []


def test_folders(tmp_path):
    s = Store(tmp_path / "f.db")
    s.set_folder("Repas", [5, 6])
    (folder,) = Finder(s, models=None, face_threshold=0.6).folders()
    assert (folder.name, folder.cover_file_id, folder.file_ids) == ("Repas", 5, [5, 6])


def test_find_me_no_face_raises_NoFace(tmp_path, models):
    f = Finder(Store(tmp_path / "f.db"), models, face_threshold=0.6)
    with pytest.raises(NoFace):
        f.find_me(jpeg_bytes(Image.new("RGB", (400, 400), (90, 140, 200))))


def test_find_me_not_an_image_raises_NoFace(tmp_path, models):
    with pytest.raises(NoFace):
        Finder(Store(tmp_path / "f.db"), models, face_threshold=0.6).find_me(b"%PDF-1.4 nope")


def test_find_me_two_large_faces_raises_ManyFaces(tmp_path, models):
    a, b = fixture_image("person_a_1.jpg"), fixture_image("person_b.jpg").resize(fixture_image("person_a_1.jpg").size)
    both = Image.new("RGB", (a.width * 2, a.height))
    both.paste(a, (0, 0))
    both.paste(b, (a.width, 0))
    with pytest.raises(ManyFaces):
        Finder(Store(tmp_path / "f.db"), models, face_threshold=0.6).find_me(jpeg_bytes(both))


def test_find_me_real_selfie_finds_same_person(tmp_path, models):
    from finder.faces import detect_faces, embed_face

    s = Store(tmp_path / "f.db")
    for fid, name in [(1, "person_a_2.jpg"), (2, "person_b.jpg")]:
        img = fixture_image(name)
        s.set_faces(fid, [(f, embed_face(models, img, f)) for f in detect_faces(models, img)])
    r = Finder(s, models, face_threshold=0.6).find_me(jpeg_bytes(fixture_image("person_a_1.jpg")))
    assert r.file_ids == [1] and r.faces == 1
