"""OpenAI CLIP byte-pair tokenizer, as ported in rust/crates/ml/src/clip/tokenizer.rs."""

import gzip
import html
from functools import lru_cache
from pathlib import Path

import ftfy
import numpy as np
import regex

TOKENS = 77
_MERGES_END = 49152 - 256 - 2 + 1
_PATTERN = regex.compile(
    r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
    regex.IGNORECASE,
)


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def _clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text)).strip()
    return regex.sub(r"\s+", " ", text).strip()


class Tokenizer:
    def __init__(self, vocab_path: Path):
        raw = Path(vocab_path).read_bytes()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        merges = [tuple(m.split()) for m in raw.decode().split("\n")[1:_MERGES_END]]
        self.byte_encoder = _bytes_to_unicode()
        vocab = list(self.byte_encoder.values())
        vocab += [v + "</w>" for v in vocab]
        vocab += ["".join(m) for m in merges]
        vocab += ["<|startoftext|>", "<|endoftext|>"]
        self.encoder = {v: i for i, v in enumerate(vocab)}
        self.ranks = {m: i for i, m in enumerate(merges)}
        self.sot, self.eot = self.encoder["<|startoftext|>"], self.encoder["<|endoftext|>"]
        self._bpe = lru_cache(maxsize=10_000)(self._bpe_uncached)

    def _bpe_uncached(self, token: str) -> str:
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            first, second = best
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    out.append(first + second)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = tuple(out)
        return " ".join(word)

    def encode(self, text: str) -> list[int]:
        ids = []
        for token in _PATTERN.findall(_clean(text).lower()):
            token = "".join(self.byte_encoder[b] for b in token.encode())
            ids += [self.encoder[t] for t in self._bpe(token).split(" ")]
        return ids

    def tokenize(self, text: str) -> np.ndarray:
        ids = [self.sot, *self.encode(text)[: TOKENS - 2], self.eot]
        return np.array(ids + [0] * (TOKENS - len(ids)), dtype=np.int32)
