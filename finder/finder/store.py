"""SQLite store for everything the finder knows about the album."""

import sqlite3
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
CREATE TABLE IF NOT EXISTS excluded_clusters (cluster INTEGER PRIMARY KEY);
"""


def to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


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
            self.db.executemany(
                "INSERT INTO folders(name, file_id, pos) VALUES (?, ?, ?)",
                [(name, fid, i) for i, fid in enumerate(sorted(file_ids))],
            )

    def folders(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for name, fid in self.db.execute("SELECT name, file_id FROM folders ORDER BY name, pos"):
            out.setdefault(name, []).append(fid)
        return out
