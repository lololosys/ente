from finder.store import Store


def test_folders_round_trip(tmp_path):
    s = Store(tmp_path / "f.db")
    s.set_folder("Repas", [3, 1])
    s.set_folder("Mairie", [2])
    s.set_folder("Repas", [1, 3, 4])  # re-run replaces
    assert s.folders() == {"Mairie": [2], "Repas": [1, 3, 4]}


def test_done_flags(tmp_path):
    s = Store(tmp_path / "f.db")
    s.ensure_files([1, 2])
    s.mark_done(1, "faces")
    assert s.pending("faces") == [2]
    assert s.pending("clip") == [1, 2]


import numpy as np

from finder.faces import Face
from finder.stages import cluster_faces


def unit(*v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def face():
    return Face((0, 0, 10, 10), 0.9, ((1, 1),) * 5)


def test_faces_replace_per_file(tmp_path):
    s = Store(tmp_path / "f.db")
    s.set_faces(1, [(face(), unit(1, 0))])
    s.set_faces(1, [(face(), unit(0, 1)), (face(), unit(1, 1))])
    rows = s.face_rows()
    assert [r.file_id for r in rows] == [1, 1]
    assert rows[0].emb.shape == (2,)


def test_cluster_faces_groups_at_0_76(tmp_path):
    s = Store(tmp_path / "f.db")
    s.set_faces(1, [(face(), unit(1, 0.05))])
    s.set_faces(2, [(face(), unit(1, 0.0))])
    s.set_faces(3, [(face(), unit(0, 1))])
    cluster_faces(s)
    by_file = {r.file_id: r.cluster for r in s.face_rows()}
    assert by_file[1] == by_file[2] != by_file[3]


def test_exclusion_survives_reclustering(tmp_path):
    s = Store(tmp_path / "f.db")
    s.set_faces(1, [(face(), unit(1, 0.05))])
    s.set_faces(2, [(face(), unit(0, 1))])
    cluster_faces(s)
    target = {r.file_id: r.cluster for r in s.face_rows()}[1]
    pid = s.exclude_cluster(target)
    s.set_faces(3, [(face(), unit(0.2, 1))])  # new photos shift cluster numbers
    cluster_faces(s)
    (person,) = s.excluded_people()
    assert person[0] == pid and float((person[1] @ unit(1, 0)).max()) > 0.99
    s.include_person(pid)
    assert s.excluded_people() == []


def test_forget_faces(tmp_path):
    s = Store(tmp_path / "f.db")
    s.ensure_files([1])
    s.set_faces(1, [(face(), unit(1, 0))])
    s.mark_done(1, "faces")
    cluster_faces(s)
    s.exclude_cluster(1)
    s.forget_faces()
    assert s.face_rows() == [] and s.excluded_people() == []
    assert s.pending("faces") == [1]
