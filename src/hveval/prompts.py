"""Builds the two agent prompt variants.

static : core + EVERY snippet, always (the long baseline).
pruned : core + compliance + top-k snippets retrieved by keyword match on the borrower's
         message. Retrieval never sees the ground-truth intent label (no leakage).
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "prompts"
INTENT_SNIPPETS = ["will_pay", "needs_time", "hardship", "dispute", "wrong_person", "abusive"]
ALWAYS_PRUNED = ["compliance", "language"]
ALL_SNIPPETS = INTENT_SNIPPETS + ["compliance", "language", "faq"]

KEYWORDS = {
    "will_pay": r"pay|payment|bhugtan|paise|upi|kar dunga|bhej",
    "needs_time": r"call back|call me|phone kijiye|busy|baat karte|samay|next week|agle hafte|kal",
    "hardship": r"job|naukri|hospital|medical|ilaaj|kharcha|nahi kar|cannot|can't pay",
    "dispute": r"galat|wrong|already|pehle|never took|liya hi nahi|mistaken",
    "wrong_person": r"wrong number|galat number|nahi rehta|no .* here|galat insaan|wrong person",
    "abusive": r"harass|chup|bakwaas|pareshan|never call|mat karo|shut up|leave me",
}


def _read(name: str) -> str:
    return (ROOT / name).read_text().strip()


def _snippet(name: str) -> str:
    return _read(f"snippets/{name}.txt")


def retrieve(text: str, k: int = 2) -> list[str]:
    scored = []
    for name, pat in KEYWORDS.items():
        n = len(re.findall(pat, text.lower()))
        if n:
            scored.append((n, name))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [n for _, n in scored[:k]]


def account_block(account: dict) -> str:
    return "ACCOUNT:\n" + json.dumps(account, ensure_ascii=False)


def build_prompt(variant: str, account: dict, borrower_text: str, k: int = 2) -> str:
    parts = [_read("core.txt")]
    if variant == "static":
        parts += [_snippet(n) for n in ALL_SNIPPETS]
    elif variant == "pruned":
        names = ALWAYS_PRUNED[:1] + retrieve(borrower_text, k)  # compliance + retrieved
        parts += [_snippet(n) for n in names]
    else:
        raise ValueError(variant)
    parts.append(account_block(account))
    return "\n\n".join(parts)
