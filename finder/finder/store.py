"""SQLite store for everything the finder knows about the album."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STAGES = ("faces", "clip", "caption")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    done_faces INTEGER NOT NULL DEFAULT 0,
    done_clip INTEGER NOT NULL DEFAULT 0,
    done_caption INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    score REAL,
    emb BLOB NOT NULL,
    cluster INTEGER
);
CREATE INDEX IF NOT EXISTS faces_file ON faces(file_id);
CREATE TABLE IF NOT EXISTS clip (file_id INTEGER PRIMARY KEY, emb BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS captions (file_id INTEGER PRIMARY KEY, en TEXT, fr TEXT, tags TEXT);
CREATE TABLE IF NOT EXISTS folders (name TEXT NOT NULL, file_id INTEGER NOT NULL, pos INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS excluded_people (id INTEGER PRIMARY KEY, emb BLOB NOT NULL);
"""


def to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


@dataclass(frozen=True)
class FaceRow:
    id: int
    file_id: int
    box: tuple[float, float, float, float]
    emb: np.ndarray
    cluster: int | None


class Store:
    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript(_SCHEMA)

    # files and stage progress

    def ensure_files(self, ids: list[int]) -> None:
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO files(id) VALUES (?)", [(i,) for i in ids])

    def pending(self, stage: str) -> list[int]:
        assert stage in STAGES
        rows = self.db.execute(f"SELECT id FROM files WHERE done_{stage} = 0 ORDER BY id")
        return [r[0] for r in rows]

    def mark_done(self, file_id: int, stage: str) -> None:
        assert stage in STAGES
        with self.db:
            self.db.execute(f"UPDATE files SET done_{stage} = 1 WHERE id = ?", (file_id,))

    # folders

    def set_folder(self, name: str, file_ids: list[int]) -> None:
        with self.db:
            self.db.execute("DELETE FROM folders WHERE name = ?", (name,))
            # An empty list removes the folder.
            self.db.executemany(
                "INSERT INTO folders(name, file_id, pos) VALUES (?, ?, ?)",
                [(name, fid, i) for i, fid in enumerate(sorted(file_ids))],
            )

    def folders(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for name, fid in self.db.execute("SELECT name, file_id FROM folders ORDER BY name, pos"):
            out.setdefault(name, []).append(fid)
        return out

    # faces

    def set_faces(self, file_id: int, faces: list) -> None:
        with self.db:
            self.db.execute("DELETE FROM faces WHERE file_id = ?", (file_id,))
            self.db.executemany(
                "INSERT INTO faces(file_id, x1, y1, x2, y2, score, emb) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(file_id, *f.box, f.score, to_blob(e)) for f, e in faces],
            )

    def face_rows(self) -> list[FaceRow]:
        rows = self.db.execute("SELECT id, file_id, x1, y1, x2, y2, emb, cluster FROM faces ORDER BY id")
        return [FaceRow(r[0], r[1], (r[2], r[3], r[4], r[5]), from_blob(r[6]), r[7]) for r in rows]

    def set_clusters(self, assignment: dict[int, int]) -> None:
        with self.db:
            self.db.executemany("UPDATE faces SET cluster = ? WHERE id = ?", [(c, i) for i, c in assignment.items()])

    def exclude_cluster(self, cluster: int) -> int:
        """Remembers the person by the cluster's mean face, so the exclusion
        holds when clusters are renumbered or new photos arrive."""
        embs = [r.emb for r in self.face_rows() if r.cluster == cluster]
        if not embs:
            raise KeyError(f"no faces in cluster {cluster}")
        mean = np.mean(embs, axis=0)
        mean /= np.linalg.norm(mean)
        with self.db:
            cur = self.db.execute("INSERT INTO excluded_people(emb) VALUES (?)", (to_blob(mean),))
        return int(cur.lastrowid)

    def include_person(self, person_id: int) -> None:
        with self.db:
            self.db.execute("DELETE FROM excluded_people WHERE id = ?", (person_id,))

    def excluded_people(self) -> list[tuple[int, np.ndarray]]:
        return [(r[0], from_blob(r[1])) for r in self.db.execute("SELECT id, emb FROM excluded_people ORDER BY id")]

    def forget_faces(self) -> None:
        with self.db:
            self.db.execute("DELETE FROM faces")
            self.db.execute("DELETE FROM excluded_people")
            self.db.execute("UPDATE files SET done_faces = 0")
        self.db.execute("VACUUM")

    # CLIP

    def set_clip(self, file_id: int, emb: np.ndarray) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO clip(file_id, emb) VALUES (?, ?)", (file_id, to_blob(emb)))

    def clip_rows(self) -> dict[int, np.ndarray]:
        return {r[0]: from_blob(r[1]) for r in self.db.execute("SELECT file_id, emb FROM clip ORDER BY file_id")}

    # captions (filled once captioning is enabled)

    def caption_rows(self) -> dict[int, tuple[str, str, str]]:
        return {r[0]: (r[1] or "", r[2] or "", r[3] or "") for r in self.db.execute("SELECT file_id, en, fr, tags FROM captions")}
