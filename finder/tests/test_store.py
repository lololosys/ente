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
