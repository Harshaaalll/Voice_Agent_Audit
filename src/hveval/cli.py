from __future__ import annotations
import argparse
import json
from pathlib import Path

from .dataset import generate, read_jsonl, write_jsonl
from .runner import decision_confusion, run_variant, summarize


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hveval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="generate synthetic scenarios")
    g.add_argument("--n", type=int, default=120)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--out", default="data/scenarios.jsonl")
    r = sub.add_parser("run", help="run both prompt variants and write results")
    r.add_argument("--data", default="data/scenarios.jsonl")
    r.add_argument("--out", default="results/run.json")
    r.add_argument("--clean", action="store_true", help="use clean text instead of ASR-noisy")
    a = ap.parse_args(argv)

    if a.cmd == "gen":
        rows = generate(a.n, a.seed)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(rows, a.out)
        print(f"wrote {len(rows)} scenarios to {a.out}")
    else:
        rows = read_jsonl(a.data)
        by_id = {r["id"]: r for r in rows}
        report = {}
        for variant in ("static", "pruned"):
            res = run_variant(rows, variant, noisy=not a.clean)
            report[variant] = {"summary": summarize(res), "decision_prf": decision_confusion(res, by_id)}
        s, p = report["static"]["summary"], report["pruned"]["summary"]
        report["token_reduction_pct"] = round(100 * (1 - p["mean_input_tokens"] / s["mean_input_tokens"]), 1)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(json.dumps({k: report[k]["summary"] for k in ("static", "pruned")}, indent=2))
        print("input-token reduction %:", report["token_reduction_pct"])
