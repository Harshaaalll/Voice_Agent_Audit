"""Code-switching and ASR-noise injector.

Deterministic given a seed. Simulates common failure patterns of Indian-English /
Hinglish ASR: homophone swaps, dropped words, vowel confusions, number mangling.
"""
from __future__ import annotations
import random
import re

HOMOPHONES = {
    "payment": ["pement", "paymen"], "kal": ["call", "kaal"], "nahi": ["nahin", "nahi"],
    "paise": ["pase", "pais"], "tomorrow": ["tomorow"], "call": ["kol"],
    "loan": ["lon"], "galat": ["galt"], "haan": ["han", "haa"],
}
FILLERS = ["uh", "matlab", "you know", "acha", "um"]

CODE_SWITCH = {  # English -> Hinglish swaps used by the code-switch injector
    "tomorrow": "kal", "payment": "payment", "not": "nahi", "yes": "haan",
    "wrong": "galat", "money": "paise", "call": "call", "next week": "agle hafte",
}


def asr_noise(text: str, rng: random.Random, level: float = 0.15) -> str:
    """level in [0,1]: per-word probability of corruption."""
    out = []
    for w in text.split():
        core = re.sub(r"\W", "", w).lower()
        r = rng.random()
        if r < level * 0.3:
            continue  # dropped word
        if r < level * 0.7 and core in HOMOPHONES:
            out.append(rng.choice(HOMOPHONES[core]))
        elif r < level:
            out.append(w + " " + rng.choice(FILLERS) if rng.random() < 0.3 else w.lower())
        else:
            out.append(w)
    return " ".join(out)


def code_switch(text: str, rng: random.Random, p: float = 0.7) -> str:
    out = text
    for en, hi in CODE_SWITCH.items():
        if en in out.lower() and rng.random() < p:
            out = re.sub(en, hi, out, flags=re.IGNORECASE)
    return out
