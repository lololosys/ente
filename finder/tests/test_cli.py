import numpy as np
import pytest

from finder import cli
from finder.faces import Face
from finder.stages import cluster_faces
from finder.store import Store

FACE = Face((0, 0, 10, 10), 0.9, ((1, 1),) * 5)


def unit(*v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FINDER_MAIN_LINK", "https://x/?t=T#3yZe7d")
    monkeypatch.setenv("FINDER_DATA_DIR", str(tmp_path))
    s = Store(tmp_path / "finder.db")
    s.set_faces(10, [(FACE, unit(1, 0))])
    s.set_faces(11, [(FACE, unit(1, 0.05))])
    s.set_faces(12, [(FACE, unit(0, 1))])
    cluster_faces(s)
    return s


def test_clusters_lists_people_biggest_first_with_photo_ids(env, capsys):
    cli.main(["clusters"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].split()[:3] == ["person", "1:", "2"]
    assert "10" in lines[0] and "11" in lines[0]


def test_exclude_then_people_then_include(env, capsys):
    cli.main(["exclude", "1"])
    cli.main(["excluded"])
    out = capsys.readouterr().out
    assert "excluded 1" in out
    cli.main(["include", "1"])
    assert env.excluded_people() == []


def test_forget_faces_command(env):
    cli.main(["forget-faces", "--yes"])
    assert env.face_rows() == []


def test_forget_faces_needs_yes(env):
    with pytest.raises(SystemExit):
        cli.main(["forget-faces"])
    assert env.face_rows() != []


def test_set_folder_reads_ids_from_stdin(env, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("12\n 10\n\n11\n"))
    cli.main(["set-folder", "Mairie"])
    assert env.folders()["Mairie"] == [10, 11, 12]


def test_remove_folder(env, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("1\n"))
    cli.main(["set-folder", "Old"])
    cli.main(["remove-folder", "Old"])
    assert "Old" not in env.folders()
