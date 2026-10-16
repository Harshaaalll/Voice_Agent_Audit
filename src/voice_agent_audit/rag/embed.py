"""Embedders: turn text into vectors.

hash      pure python, deterministic, no download. Feature hashing of words + character trigrams.
          Mostly lexical (shares sub-word pieces), so it is a *baseline*, not a semantic model.
fastembed ONNX multilingual MiniLM (about 220 MB, downloaded on first use). Real semantic vectors.
"""
from __future__ import annotations
import math
import re
import zlib

FASTEMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class HashEmbedder:
    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _feats(self, text: str):
        for w in re.findall(r"\w+", text.lower()):
            yield "w:" + w
            padded = f"^{w}$"
            for i in range(len(padded) - 2):
                yield "c:" + padded[i:i + 3]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for f in self._feats(t):
                h = zlib.crc32(f.encode())
                v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
            out.append(_norm(v))
        return out


class FastEmbedder:
    name = "fastembed"

    def __init__(self, model: str = FASTEMBED_MODEL):
        from fastembed import TextEmbedding   # optional dependency
        self._m = TextEmbedding(model_name=model)
        self.dim = len(next(iter(self._m.embed(["x"]))))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_norm([float(x) for x in v]) for v in self._m.embed(texts)]


def get_embedder(name: str):
    if name == "hash":
        return HashEmbedder()
    if name == "fastembed":
        return FastEmbedder()
    raise ValueError(f"unknown embedder {name!r} (hash | fastembed)")
