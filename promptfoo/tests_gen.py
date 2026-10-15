"""Turn data/scenarios.jsonl into promptfoo test cases."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "scenarios.jsonl"


def generate_tests(config=None):
    limit = (config or {}).get("limit")
    rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    if limit:
        rows = rows[: int(limit)]
    return [{
        "description": f"{r['id']} {r['persona']['intent']}/{r['persona']['style']}",
        "vars": {
            "borrower_text": r["borrower_noisy"],
            "account": r["persona"]["account"],
            "row": r,
        },
    } for r in rows]
