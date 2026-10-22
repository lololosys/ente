"""Answering guests: folders, find-me by selfie, search by words.

The selfie is decoded and embedded in memory and dropped when the call
returns; nothing about it is stored or logged."""

import io
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from . import images  # noqa: F401  (registers the HEIC opener)
from .clip import MATCH_FLOOR, clip_text
from .faces import detect_faces, embed_face
from .models import Models
from .stages import SAME_PERSON
from .store import Store

FIND_ME_CAP = 500
SEARCH_CAP = 300
#: A second face at least this large (relative to the largest) means the
#: selfie shows more than one person.
SECOND_FACE_RATIO = 0.5
SELFIE_SIDE = 1600


class NoFace(Exception):
    pass


class ManyFaces(Exception):
    pass


@dataclass(frozen=True)
class FindResult:
    file_ids: list[int]
    faces: int


@dataclass(frozen=True)
class Folder:
    name: str
    cover_file_id: int
    file_ids: list[int]


class Finder:
    def __init__(
        self,
        store: Store,
        models: Models | None,
        face_threshold: float,
        embed_selfie: Callable[[bytes], np.ndarray] | None = None,
        embed_text: Callable[[str], np.ndarray] | None = None,
    ):
        self.store = store
        self.models = models
        self.face_threshold = face_threshold
        self._embed_selfie = embed_selfie or self._selfie_embedding
        self._embed_text = embed_text or (lambda q: clip_text(models, q))

    def folders(self) -> list[Folder]:
        return [Folder(name, ids[0], ids) for name, ids in self.store.folders().items() if ids]

    def _selfie_embedding(self, data: bytes) -> np.ndarray:
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        except Exception:
            raise NoFace() from None
        img.thumbnail((SELFIE_SIDE, SELFIE_SIDE))
        faces = sorted(detect_faces(self.models, img), key=lambda f: f.area, reverse=True)
        if not faces:
            raise NoFace()
        if len(faces) > 1 and faces[1].area >= SECOND_FACE_RATIO * faces[0].area:
            raise ManyFaces()
        return embed_face(self.models, img, faces[0])

    def find_me(self, selfie: bytes) -> FindResult:
        query = self._embed_selfie(selfie)
        people = [faces for _, faces in self.store.excluded_people()]
        excluded = np.concatenate(people) if people else np.empty((0, query.shape[0]), np.float32)
        # Someone who asked to be excluded can't find themselves either; the
        # stricter same-person rule keeps look-alike relatives searchable.
        if len(excluded) and float((excluded @ query).max()) >= SAME_PERSON:
            return FindResult([], 1)
        best: dict[int, float] = {}
        for row in self.store.face_rows():
            if len(excluded) and float((excluded @ row.emb).max()) >= self.face_threshold:
                continue
            score = float(row.emb @ query)
            if score >= self.face_threshold and score > best.get(row.file_id, -1):
                best[row.file_id] = score
        ranked = sorted(best, key=lambda fid: best[fid], reverse=True)
        return FindResult(ranked[:FIND_ME_CAP], 1)

    def search(self, q: str) -> list[int]:
        q = q.strip()
        if not q:
            return []
        rows = self.store.clip_rows()
        if not rows:
            return []
        ids = list(rows)
        scores = np.stack([rows[i] for i in ids]) @ self._embed_text(q)
        order = np.argsort(-scores)
        return [ids[i] for i in order if scores[i] >= MATCH_FLOOR][:SEARCH_CAP]
