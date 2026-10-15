"""Generate the synthetic scenario set (JSONL). Ground truth is fixed by construction."""
from __future__ import annotations
import json
import random
from pathlib import Path

from .noise import asr_noise
from .personas import INTENTS, make_persona
from .utterances import borrower_turn


def generate(n: int, seed: int = 7, noise_level: float = 0.15) -> list[dict]:
    rng = random.Random(seed)
    intents = list(INTENTS)
    rows = []
    for i in range(n):
        p = make_persona(rng, intent=intents[i % len(intents)])  # balanced intents
        clean = borrower_turn(rng, p.intent, p.style, p.name, p.account.amount_due)
        noisy = asr_noise(clean, rng, level=noise_level)
        rows.append({
            "id": f"sc{i:04d}",
            "persona": p.to_dict(),
            "borrower_clean": clean,
            "borrower_noisy": noisy,
            "expected_decision": INTENTS[p.intent]["decision"],
        })
    return rows


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
