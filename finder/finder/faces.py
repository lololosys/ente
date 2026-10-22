"""Face detection and embedding with Ente's YOLOv5-face and MobileFaceNet,
mirroring rust/crates/ml/src/{preprocess.rs,face/detect.rs,face/align.rs}."""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .models import Models

YOLO_SIZE = 640
PAD = 114
MIN_SCORE = 0.5
IOU = 0.4
FACE_SIZE = 112
IDEAL_LANDMARKS = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)
#: Group shots: faces are small, so large photos are also scanned in
#: 640 px tiles of a copy whose long side is this many pixels.
TILE_LONG_SIDE = 2560
TILE_STRIDE = 480


@dataclass(frozen=True)
class Face:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in image pixels
    score: float
    landmarks: tuple[tuple[float, float], ...]  # eyes, nose, mouth corners

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])


def _detect_window(models: Models, rgb: np.ndarray) -> list[Face]:
    """One letterboxed 640 px pass over an RGB array; coordinates in that array."""
    h, w = rgb.shape[:2]
    scale = min(YOLO_SIZE / w, YOLO_SIZE / h)
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR) if (sw, sh) != (w, h) else rgb
    left, top = (YOLO_SIZE - sw) // 2, (YOLO_SIZE - sh) // 2
    canvas = np.full((YOLO_SIZE, YOLO_SIZE, 3), PAD, dtype=np.uint8)
    canvas[top : top + sh, left : left + sw] = resized
    x = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    sess = models.face_detect
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0].reshape(-1, 16)
    out = out[out[:, 4] >= MIN_SCORE]
    faces = []
    for row in out:
        cx, cy, bw, bh = row[:4]
        box = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        pts = row[5:15].reshape(5, 2)
        box[[0, 2]] = (box[[0, 2]] - left) / scale
        box[[1, 3]] = (box[[1, 3]] - top) / scale
        pts = (pts - [left, top]) / scale
        faces.append(Face(tuple(map(float, box)), float(row[4]), tuple(map(tuple, pts.astype(float)))))
    return faces


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _nms(faces: list[Face]) -> list[Face]:
    kept: list[Face] = []
    for f in sorted(faces, key=lambda f: f.score, reverse=True):
        if all(_iou(f.box, k.box) < IOU for k in kept):
            kept.append(f)
    return kept


def _shift(f: Face, dx: float, dy: float, s: float) -> Face:
    x1, y1, x2, y2 = f.box
    return Face(
        ((x1 + dx) / s, (y1 + dy) / s, (x2 + dx) / s, (y2 + dy) / s),
        f.score,
        tuple(((x + dx) / s, (y + dy) / s) for x, y in f.landmarks),
    )


def _starts(length: int) -> list[int]:
    if length <= YOLO_SIZE:
        return [0]
    starts = list(range(0, length - YOLO_SIZE, TILE_STRIDE))
    return starts + [length - YOLO_SIZE]


def detect_faces(models: Models, img: Image.Image) -> list[Face]:
    rgb = np.asarray(img.convert("RGB"))
    faces = _detect_window(models, rgb)
    h, w = rgb.shape[:2]
    if max(w, h) > 2 * YOLO_SIZE:
        s = min(1.0, TILE_LONG_SIDE / max(w, h))
        big = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
        bh, bw = big.shape[:2]
        for ty in _starts(bh):
            for tx in _starts(bw):
                tile = big[ty : ty + YOLO_SIZE, tx : tx + YOLO_SIZE]
                faces += [_shift(f, tx, ty, s) for f in _detect_window(models, tile)]
    return _nms(faces)


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Umeyama least-squares similarity transform, 2x3."""
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    sc, dc = src - src_mean, dst - dst_mean
    cov = dc.T @ sc / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1
    r = u @ sign @ vt
    scale = np.trace(np.diag(d) @ sign) / sc.var(0).sum()
    t = dst_mean - scale * r @ src_mean
    return np.hstack([scale * r, t[:, None]]).astype(np.float32)


def align_face(img: Image.Image, face: Face) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB"))
    m = _similarity_transform(np.array(face.landmarks, dtype=np.float32), IDEAL_LANDMARKS)
    return cv2.warpAffine(
        rgb, m, (FACE_SIZE, FACE_SIZE), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(PAD, PAD, PAD),
    )


def embed_face(models: Models, img: Image.Image, face: Face) -> np.ndarray:
    crop = align_face(img, face).astype(np.float32) / 127.5 - 1.0
    sess = models.face_embed
    emb = sess.run(None, {sess.get_inputs()[0].name: crop[None]})[0].reshape(-1).astype(np.float32)
    return emb / max(np.linalg.norm(emb), 1e-12)
