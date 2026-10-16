"""One evaluation run: used synchronously by the API and asynchronously by the worker."""
from __future__ import annotations
import os

from ..dataset import read_jsonl
from ..rag.ragmetrics import context_recall
from ..runner import run_variant, summarize
from .core import Core


def run_evaluation(core: Core, backend, model: str, variant: str, limit: int, noisy: bool = True) -> dict:
    path = os.environ.get("VAA_DATA", "data/scenarios.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"scenario file not found: {path}")
    rows = read_jsonl(path)[:limit]
    ctx = (lambda t: core.retrieve(t, 3)) if variant == "rag" else None
    results = run_variant(rows, variant, noisy=noisy, backend=backend, model=model, context_fn=ctx)
    summary = summarize(results)
    if variant == "rag":
        intents = {r["id"]: r["persona"]["intent"] for r in rows}
        rec = [context_recall(r["context_ids"], intents[r["id"]]) for r in results]
        summary["context_recall"] = round(sum(rec) / len(rec), 4)
    run_id = core.store.add_run(variant=variant, backend=os.environ.get("VAA_BACKEND", "mock"),
                                model=model, noisy=noisy, summary=summary, items=results)
    return {"run_id": run_id, "variant": variant, "summary": summary}
