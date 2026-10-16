"""Retrieval metrics. `recall@k` is the context-recall idea: how much of what was needed was fetched."""
from __future__ import annotations
import random
import time


def score_query(ranked: list[str], relevant: set[str], k: int) -> dict:
    top = ranked[:k]
    hit = [c for c in top if c in relevant]
    first = next((i for i, c in enumerate(ranked, 1) if c in relevant), None)
    return {"recall": len(set(hit)) / len(relevant), "hit": float(bool(hit)),
            "precision": len(hit) / k, "rr": 1.0 / first if first else 0.0}


def mean_ci(xs: list[float], iters: int = 2000, seed: int = 0):
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return sum(xs) / n, means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def evaluate(retriever, gold: list[tuple], k: int = 3, depth: int = 10) -> dict:
    """gold items: (query, [relevant ids], style)."""
    rows, lat = [], []
    for q, rel, style in gold:
        t0 = time.perf_counter()
        ranked = [c for c, _ in retriever.search(q, depth)]
        lat.append((time.perf_counter() - t0) * 1000)
        rows.append((style, score_query(ranked, set(rel), k), ranked[:k]))
    def agg(sel):
        out = {"n": len(sel)}
        for m in ("recall", "hit", "precision", "rr"):
            mean, lo, hi = mean_ci([r[1][m] for r in sel])
            out[m] = round(mean, 4)
            out[m + "_ci95"] = [round(lo, 3), round(hi, 3)]
        return out
    lat.sort()
    res = {"k": k, "overall": agg(rows),
           "by_style": {s: agg([r for r in rows if r[0] == s]) for s in sorted({r[0] for r in rows})},
           "latency_ms": {"p50": round(lat[len(lat) // 2], 3), "p95": round(lat[int(0.95 * len(lat)) - 1], 3)}}
    res["_rankings"] = [r[2] for r in rows]
    return res
