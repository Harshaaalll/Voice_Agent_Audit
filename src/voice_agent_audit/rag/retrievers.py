"""Retrievers over policy chunks. Every retriever: search(query, k) -> [(chunk_id, score)]."""
from __future__ import annotations
import math
import re
from collections import Counter


def _tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


class BM25:
    name = "bm25"

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.ids = [c["id"] for c in chunks]
        self.docs = [Counter(_tok(c["text"])) for c in chunks]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / len(self.len)
        n = len(self.docs)
        df = Counter(t for d in self.docs for t in d)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 3):
        q = _tok(query)
        scores = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                if t in d:
                    f = d[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            scores.append((self.ids[i], s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


class Dense:
    def __init__(self, chunks: list[dict], embedder):
        self.name = embedder.name
        self.emb = embedder
        self.ids = [c["id"] for c in chunks]
        self.vecs = embedder.embed([c["text"] for c in chunks])

    def search(self, query: str, k: int = 3):
        q = self.emb.embed([query])[0]
        scores = [(i, sum(a * b for a, b in zip(q, v))) for i, v in zip(self.ids, self.vecs)]
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


def rrf(rank_lists: list[list[str]], k: int) -> list[tuple[str, float]]:
    """Reciprocal rank fusion (Cormack et al. 2009, constant 60) over ranked id lists."""
    fused: dict[str, float] = {}
    for ids in rank_lists:
        for rank, cid in enumerate(ids, 1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
    return sorted(fused.items(), key=lambda x: -x[1])[:k]


class Hybrid:
    """Reciprocal rank fusion of two retrievers (Cormack et al. 2009, constant 60)."""

    def __init__(self, a, b, depth: int = 20):
        self.a, self.b, self.depth = a, b, depth
        self.name = f"{a.name}+{b.name}"

    def search(self, query: str, k: int = 3):
        return rrf([[c for c, _ in r.search(query, self.depth)] for r in (self.a, self.b)], k)


class StoreRetriever:
    """Dense search executed by the database (pgvector on PostgreSQL, python cosine on SQLite)."""

    def __init__(self, store, embedder):
        self.store, self.emb, self.name = store, embedder, f"{embedder.name}@{store.kind}"

    def search(self, query: str, k: int = 3):
        return [(h["id"], h["score"]) for h in self.store.search_chunks(self.emb.name, self.emb.embed([query])[0], k)]
