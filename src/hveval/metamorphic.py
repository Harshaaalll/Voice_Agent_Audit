"""Metamorphic tests for the agent: relations whose violation needs no gold label.
Invariance relations: the tool decision must not change under a meaning-preserving edit."""
from __future__ import annotations
import json
import os
import random
import re

from .llm import get_backend
from .prompts import build_prompt

OTHER_NAMES = ["Kavita Joshi", "Manoj Pillai", "Deepa Iyer"]
FILLERS = ["Haan toh, ", "Umm, ", "Dekhiye, ", "Okay so "]


def _name_swap(row, rng):
    n = row["persona"]["name"]
    new = rng.choice(OTHER_NAMES)
    acct = dict(row["persona"]["account"])
    text = row["borrower_clean"]
    return acct, text.replace(n, new).replace(n.split()[0], new.split()[0])


def _amount_change(row, rng):
    acct = dict(row["persona"]["account"])
    acct["amount_due"] = acct["amount_due"] + rng.choice([500, 1000, 4500])
    return acct, row["borrower_clean"]


RELATIONS = {
    "name_swap": _name_swap,
    "amount_change": _amount_change,
    "filler_prefix": lambda row, rng: (row["persona"]["account"], rng.choice(FILLERS) + row["borrower_clean"]),
    "lowercase_no_punct": lambda row, rng: (row["persona"]["account"],
                                            re.sub(r"[^\w\s]", "", row["borrower_clean"].lower())),
    "polite_suffix": lambda row, rng: (row["persona"]["account"], row["borrower_clean"] + " Dhanyavaad."),
}


def _decide(backend, model, variant, account, text) -> str | None:
    system = build_prompt(variant, account, text)
    r = backend.complete(system, [{"role": "user", "content": text}], model)
    try:
        return (json.loads(r.text).get("tool") or {}).get("name")
    except Exception:  # noqa: BLE001
        return None


def run(rows: list[dict], variant: str = "pruned", seed: int = 4, backend=None, model=None) -> dict:
    backend = backend or get_backend()
    model = model or os.environ.get("HVEVAL_MODEL", "mock")
    rng = random.Random(seed)
    out = {"variant": variant, "backend": backend.__class__.__name__, "model": model,
           "is_mock": backend.__class__.__name__ == "MockBackend", "n_rows": len(rows), "relations": {}}
    base = {r["id"]: _decide(backend, model, variant, r["persona"]["account"], r["borrower_clean"]) for r in rows}
    for name, fn in RELATIONS.items():
        viol, ex = 0, []
        for r in rows:
            acct, text = fn(r, rng)
            d = _decide(backend, model, variant, acct, text)
            if d != base[r["id"]]:
                viol += 1
                if len(ex) < 3:
                    ex.append({"id": r["id"], "before": base[r["id"]], "after": d, "text": text[:80]})
        out["relations"][name] = {"violations": viol, "n": len(rows), "violation_rate": viol / len(rows),
                                  "examples": ex}
    return out
