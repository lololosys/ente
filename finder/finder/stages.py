"""Indexing stages that run on the box: faces and CLIP."""

import numpy as np
from PIL import Image

from .clip import clip_image
from .ente_api import RemoteFile
from .faces import detect_faces, embed_face
from .models import Models
from .store import Store

#: Ente's same-person rule (web/packages/new/photos/services/ml/cluster.ts).
SAME_PERSON = 0.76


def face_stage(models: Models, store: Store):
    def run(f: RemoteFile, img: Image.Image) -> None:
        store.set_faces(f.id, [(face, embed_face(models, img, face)) for face in detect_faces(models, img)])

    return run


def clip_stage(models: Models, store: Store):
    def run(f: RemoteFile, img: Image.Image) -> None:
        store.set_clip(f.id, clip_image(models, img))

    return run


def cluster_faces(store: Store) -> None:
    """Greedy clustering: a face joins the first cluster whose centroid it
    matches at SAME_PERSON or better, else starts a new one."""
    rows = store.face_rows()
    centroids: list[np.ndarray] = []
    sums: list[np.ndarray] = []
    assignment: dict[int, int] = {}
    for r in rows:
        best, best_score = -1, SAME_PERSON
        for i, c in enumerate(centroids):
            score = float(r.emb @ c)
            if score >= best_score:
                best, best_score = i, score
        if best == -1:
            centroids.append(r.emb.copy())
            sums.append(r.emb.copy())
            best = len(centroids) - 1
        else:
            sums[best] += r.emb
            centroids[best] = sums[best] / np.linalg.norm(sums[best])
        assignment[r.id] = best + 1
    store.set_clusters(assignment)
