"""Run an agent variant over scenarios, apply deterministic checks, aggregate."""
from __future__ import annotations
import json
import os
import time
from collections import defaultdict

from .checks import run_all
from .llm import get_backend
from .prompts import build_prompt


def run_variant(rows: list[dict], variant: str, noisy: bool = True, backend=None,
                model: str | None = None, context_fn=None) -> list[dict]:
    backend = backend or get_backend()
    model = model or os.environ.get("VAA_MODEL", "mock")
    results = []
    for row in rows:
        text = row["borrower_noisy" if noisy else "borrower_clean"]
        hits, retrieval_s = None, None
        if context_fn:
            t0 = time.perf_counter()
            hits = context_fn(text)
            retrieval_s = time.perf_counter() - t0
        system = build_prompt(variant, row["persona"]["account"], text,
                              context=[h["text"] for h in hits] if hits is not None else None)
        res = backend.complete(system, [{"role": "user", "content": text}], model)
        checks = run_all(res.text, row)
        results.append({
            "id": row["id"], "variant": variant, "intent": row["persona"]["intent"],
            "style": row["persona"]["style"], "raw": res.text,
            "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
            "token_source": res.token_source, "latency_s": res.latency_s,
            "retrieval_s": retrieval_s, "context_ids": [h["id"] for h in hits] if hits is not None else None,
            "checks": {k: {"pass": v[0], "detail": v[1]} for k, v in checks.items()},
        })
    return results


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], 5)


def summarize(results: list[dict]) -> dict:
    n = len(results)
    per_check = defaultdict(int)
    for r in results:
        for k, v in r["checks"].items():
            per_check[k] += v["pass"]
    return {
        "n": n,
        "pass_rate": {k: round(v / n, 4) for k, v in per_check.items()},
        "mean_input_tokens": round(sum(r["input_tokens"] for r in results) / n, 1),
        "mean_output_tokens": round(sum(r["output_tokens"] for r in results) / n, 1),
        "mean_latency_s": round(sum(r["latency_s"] for r in results) / n, 4),
        "token_source": results[0]["token_source"] if results else None,
        "latency_s": {"p50": _pct([r["latency_s"] for r in results], 0.5),
                      "p95": _pct([r["latency_s"] for r in results], 0.95)},
        **({"retrieval_s": {"p50": _pct([r["retrieval_s"] for r in results], 0.5),
                            "p95": _pct([r["retrieval_s"] for r in results], 0.95)}}
           if results and results[0].get("retrieval_s") is not None else {}),
    }


def decision_confusion(results: list[dict], rows_by_id: dict) -> dict:
    """Per-intent precision/recall/F1 on the policy decision (tool name)."""
    labels = sorted({rows_by_id[r["id"]]["expected_decision"] for r in results})
    pred = {}
    for r in results:
        try:
            tool = (json.loads(r["raw"]).get("tool") or {}).get("name")
        except Exception:  # noqa: BLE001
            tool = None
        pred[r["id"]] = tool
    out = {}
    for lab in labels:
        tp = sum(1 for i, p in pred.items() if p == lab and rows_by_id[i]["expected_decision"] == lab)
        fp = sum(1 for i, p in pred.items() if p == lab and rows_by_id[i]["expected_decision"] != lab)
        fn = sum(1 for i, p in pred.items() if p != lab and rows_by_id[i]["expected_decision"] == lab)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        out[lab] = {"precision": round(pr, 3), "recall": round(rc, 3), "f1": round(f1, 3), "support": tp + fn}
    return out
