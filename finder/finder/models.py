"""Ente's own ONNX models (rust/crates/ml/src/assets.rs), on CPU."""

import hashlib
from functools import cached_property
from pathlib import Path

import httpx
import onnxruntime as ort

BASE_URL = "https://models.ente.com/"

FILES = {
    "face_detect": ("yolov5s_face_640_640_static_b1.onnx", "e047647409403d52696035ecd445792173e50d7fbdcccac97b958a585db9aa3d"),
    "face_embed": ("mobilefacenet_portable_static_b1.onnx", "0763fc33f54e138476194da95987e133b3e976075a6b1d3e1b2caedb251b1a36"),
    "clip_image": ("mobileclip_s2_image_gelu_opset20.onnx", "205a430af825e501c5138e5bb9abea942482a7a4fd4a680e98e47cf0830dce7e"),
    "clip_text": ("mobileclip_s2_text_opset18_quant.onnx", "d92f33dfcff83077fc2e0d3414250710efbb51795dfd89767bdbefb5fdc47322"),
    "vocab": ("bpe_simple_vocab_16e6.txt", "67603cfda2e032ad77b5f8808af37789d590db664b26df8705d2bf8b3c553fc8"),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_models(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, sha in FILES.values():
        path = models_dir / name
        if path.exists() and _sha256(path) == sha:
            continue
        tmp = path.with_suffix(".part")
        with httpx.stream("GET", BASE_URL + name, follow_redirects=True, timeout=600) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        if _sha256(tmp) != sha:
            tmp.unlink()
            raise RuntimeError(f"checksum mismatch for {name}")
        tmp.rename(path)


class Models:
    def __init__(self, models_dir: Path):
        self.dir = Path(models_dir)

    def _session(self, key: str) -> ort.InferenceSession:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        return ort.InferenceSession(str(self.dir / FILES[key][0]), opts, providers=["CPUExecutionProvider"])

    @cached_property
    def face_detect(self) -> ort.InferenceSession:
        return self._session("face_detect")

    @cached_property
    def face_embed(self) -> ort.InferenceSession:
        return self._session("face_embed")

    @cached_property
    def clip_image(self) -> ort.InferenceSession:
        return self._session("clip_image")

    @cached_property
    def clip_text(self) -> ort.InferenceSession:
        return self._session("clip_text")

    @property
    def vocab_path(self) -> Path:
        return self.dir / FILES["vocab"][0]

    @cached_property
    def tokenizer(self):
        from .clip_tokenizer import Tokenizer

        return Tokenizer(self.vocab_path)
