"""MobileCLIP-S2 image and text embeddings, mirroring
rust/crates/ml/src/{preprocess.rs::preprocess_clip,clip/}."""

import numpy as np
from PIL import Image

from .models import Models

SIZE = 256
#: Ente's search keeps image/text matches at or above this dot product.
MATCH_FLOOR = 0.175


def _norm(v: np.ndarray) -> np.ndarray:
    v = v.reshape(-1).astype(np.float32)
    return v / max(np.linalg.norm(v), 1e-12)


def clip_image(models: Models, img: Image.Image) -> np.ndarray:
    w, h = img.size
    s = SIZE / min(w, h)
    img = img.convert("RGB").resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    x = np.asarray(img.crop((left, top, left + SIZE, top + SIZE)), dtype=np.float32) / 255.0
    sess = models.clip_image
    return _norm(sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0])


def clip_text(models: Models, text: str) -> np.ndarray:
    sess = models.clip_text
    ids = models.tokenizer.tokenize(text)[None]
    inp = sess.get_inputs()[0]
    if "int64" in inp.type:
        ids = ids.astype(np.int64)
    return _norm(sess.run(None, {inp.name: ids})[0])
