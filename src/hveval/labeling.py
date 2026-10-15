"""Terminal labeling tool for the human gold labels. Blind: never shows judge or seeded defects."""
from __future__ import annotations
import json
import random
import time
from pathlib import Path

from .judge import RUBRIC

CRITERIA = list(RUBRIC)


def load_labels(path) -> dict[str, dict]:
    p = Path(path)
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r  # last write wins (allows relabel)
    return out


def _show(t: dict) -> str:
    a = t["persona"]["account"]
    s = [f"\n{'=' * 78}\n{t['id']}   ACCOUNT: due Rs {a['amount_due']} on {a['due_date']}, "
         f"{a['days_overdue']} days overdue, loan {a['loan_id']}, lender {a['lender']}", "-" * 78]
    s += [f"  {x['role'].upper():9s}: {x['text']}" for x in t["turns"]]
    s += [f"  TOOL CALL: {json.dumps(t['tool_call'])}", f"  NOTE     : {t['note']}", "-" * 78]
    return "\n".join(s)


def run(transcripts: list[dict], out_path, seed: int = 5, input_fn=input, print_fn=print) -> int:
    done = load_labels(out_path)
    order = transcripts[:]
    random.Random(seed).shuffle(order)
    todo = [t for t in order if t["id"] not in done]
    print_fn(f"{len(done)}/{len(transcripts)} already labeled. Keys: 1=yes/pass, 0=no/fail, "
             "s=skip item, b=redo previous criterion, q=save & quit")
    count = 0
    for t in todo:
        print_fn(_show(t))
        labels, i = {}, 0
        while i < len(CRITERIA):
            c = CRITERIA[i]
            print_fn(f"\n[{c}] {RUBRIC[c]}")
            ans = input_fn(f"  {c} (1/0/s/b/q) > ").strip().lower()
            if ans == "q":
                print_fn(f"saved. {len(load_labels(out_path))}/{len(transcripts)} labeled.")
                return count
            if ans == "s":
                labels = None
                break
            if ans == "b":
                i = max(0, i - 1)
                continue
            if ans in ("0", "1"):
                labels[c] = int(ans)
                i += 1
        if labels is None:
            continue
        with open(out_path, "a") as f:
            f.write(json.dumps({"id": t["id"], "labels": labels, "ts": time.time()}) + "\n")
        count += 1
    print_fn(f"all done. {len(load_labels(out_path))}/{len(transcripts)} labeled.")
    return count
