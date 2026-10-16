"""Shared wiring used by both the API process and the worker: store, embedder, retrieval."""
from __future__ import annotations
import os

from ..rag.embed import get_embedder
from ..rag.policy import CHUNK_LIST
from ..rag.retrievers import BM25, rrf
from .store import Store


class Core:
    def __init__(self, url: str):
        self.store = Store(url)
        self.embedder = get_embedder(os.environ.get("VAA_EMBEDDER", "hash"))
        self.mode = os.environ.get("VAA_RETRIEVAL", "dense")            # dense | hybrid
        if self.mode not in ("dense", "hybrid"):
            raise ValueError("VAA_RETRIEVAL must be dense or hybrid")
        self.bm25 = BM25(CHUNK_LIST)
        self.by_id = {c["id"]: c for c in CHUNK_LIST}
        if self.store.count_chunks(self.embedder.name) < len(CHUNK_LIST):   # first start loads the policy
            self.store.upsert_chunks(self.embedder.name, CHUNK_LIST,
                                     self.embedder.embed([c["text"] for c in CHUNK_LIST]))

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        qv = self.embedder.embed([query])[0]
        if self.mode == "dense":
            return self.store.search_chunks(self.embedder.name, qv, k)
        dense = self.store.search_chunks(self.embedder.name, qv, 20)
        fused = rrf([[h["id"] for h in dense], [c for c, _ in self.bm25.search(query, 20)]], k)
        return [{**self.by_id[c], "score": sc} for c, sc in fused]
