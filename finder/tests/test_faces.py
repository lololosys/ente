import numpy as np
from conftest import fixture_image

from finder.faces import detect_faces, embed_face


def best_embedding(models, name):
    img = fixture_image(name)
    faces = detect_faces(models, img)
    assert faces, f"no face in {name}"
    face = max(faces, key=lambda f: f.area)
    return embed_face(models, img, face)


def test_detect_finds_one_face_in_portrait(models):
    faces = detect_faces(models, fixture_image("person_a_1.jpg"))
    assert len(faces) == 1
    f = faces[0]
    assert f.score >= 0.5 and len(f.landmarks) == 5
    assert 0 <= f.box[0] < f.box[2] <= 600


def test_embedding_is_l2_normalised(models):
    e = best_embedding(models, "person_a_1.jpg")
    assert e.dtype == np.float32 and abs(np.linalg.norm(e) - 1) < 1e-4


def test_same_person_scores_above_0_6_and_other_below(models):
    a1 = best_embedding(models, "person_a_1.jpg")
    a2 = best_embedding(models, "person_a_2.jpg")
    b = best_embedding(models, "person_b.jpg")
    assert float(a1 @ a2) > 0.6
    assert float(a1 @ b) < 0.6


def test_small_face_in_large_image_is_found(models):
    """Group shots: a face ~60 px wide inside a 4000 px photo."""
    from PIL import Image

    portrait = fixture_image("person_a_1.jpg").resize((150, 188))
    canvas = Image.new("RGB", (4000, 2600), (120, 130, 125))
    canvas.paste(portrait, (2900, 1700))
    faces = detect_faces(models, canvas)
    assert any(2900 <= f.box[0] <= 3050 and 1700 <= f.box[1] <= 1890 for f in faces)
