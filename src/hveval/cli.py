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
    t = sub.add_parser("transcripts", help="generate synthetic transcripts + notes (Weekend 2)")
    t.add_argument("--n", type=int, default=160)
    t.add_argument("--seed", type=int, default=11)
    sub.add_parser("label", help="terminal tool to create YOUR human gold labels")
    j = sub.add_parser("judge", help="run the LLM judge over transcripts")
    j.add_argument("--limit", type=int)
    sub.add_parser("report", help="judge/deterministic vs human agreement + kappa")
    sub.add_parser("bias", help="position and verbosity bias probes for the judge")
    sub.add_parser("tgen", help="generate translation items (Weekend 3)")
    sub.add_parser("rate", help="terminal tool to rate translations (YOUR human ratings)")
    sub.add_parser("tmetrics", help="BLEU/chrF/fact-check vs human ratings")
    m = sub.add_parser("meta", help="metamorphic invariance tests on the agent")
    m.add_argument("--variant", default="pruned")
    sub.add_parser("dashboard", help="static HTML dashboard in results/")
    a = ap.parse_args(argv)

    if a.cmd in ("transcripts", "label", "judge", "report", "bias"):
        return _weekend2(a)
    if a.cmd in ("tgen", "rate", "tmetrics", "meta", "dashboard"):
        return _weekend3(a)
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


def _weekend2(a):
    from .transcripts import generate_transcripts
    tp, ip, hp, jp = ("data/transcripts.jsonl", "data/injected_defects.jsonl",
                      "data/human_labels.jsonl", "data/judge_labels.jsonl")
    if a.cmd == "transcripts":
        tr, inj = generate_transcripts(a.n, a.seed)
        write_jsonl(tr, tp)
        write_jsonl(inj, ip)
        print(f"wrote {len(tr)} transcripts to {tp} (seeded defects kept separately in {ip})")
        return
    tr = read_jsonl(tp)
    if a.cmd == "label":
        from .labeling import run
        run(tr, hp)
    elif a.cmd == "judge":
        from .judge import Judge
        from .labeling import CRITERIA
        jd = Judge()
        rows = tr[: a.limit] if a.limit else tr
        with open(jp, "w") as f:
            for t in rows:
                f.write(json.dumps({"id": t["id"], "judge": jd.label,
                                    "labels": {c: jd.score(t, c) for c in CRITERIA}}) + "\n")
        print(f"judged {len(rows)} transcripts with {jd.label} -> {jp}")
    elif a.cmd == "report":
        from .report import build
        jl = {r["id"]: r["labels"] for r in read_jsonl(jp)} if Path(jp).exists() else {}
        name = read_jsonl(jp)[0]["judge"] if jl else "none"
        rep = build(tr, hp, jl, name)
        Path("results").mkdir(exist_ok=True)
        Path("results/judge_report.json").write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=2))
    elif a.cmd == "bias":
        from .judge import Judge, position_bias, verbosity_bias
        jd = Judge()
        out = {"judge": jd.label, "is_proxy": jd.is_proxy,
               "position": position_bias(jd, tr), "verbosity": verbosity_bias(jd, tr)}
        Path("results").mkdir(exist_ok=True)
        Path("results/judge_bias.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))


def _weekend3(a):
    from . import translation as T
    ip, hp, rp = "data/translations.jsonl", "data/translation_hidden.jsonl", "data/translation_ratings.jsonl"
    Path("results").mkdir(exist_ok=True)
    if a.cmd == "tgen":
        items, hidden = T.generate_items()
        write_jsonl(items, ip)
        write_jsonl(hidden, hp)
        print(f"wrote {len(items)} items to {ip} (degradation levels hidden in {hp})")
    elif a.cmd == "rate":
        items = read_jsonl(ip)
        done = {}
        if Path(rp).exists():
            for r in read_jsonl(rp):
                done[r["id"]] = r
        print(f"{len(done)}/{len(items)} rated. adequacy/fluency 1-5, critical_error 1/0 (wrong amount/date/"
              "threat/meaning flip), q=quit")
        for it in items:
            if it["id"] in done:
                continue
            print(f"\n{'=' * 70}\n{it['id']}\n SOURCE   : {it['source']}\n CANDIDATE: {it['candidate']}")
            vals = {}
            for k, ok in (("adequacy", "12345"), ("fluency", "12345"), ("critical_error", "01")):
                while True:
                    v = input(f"  {k} ({'/'.join(ok)}, q) > ").strip().lower()
                    if v == "q":
                        print(f"saved. {len(done)}/{len(items)}")
                        return
                    if v in list(ok):
                        vals[k] = int(v)
                        break
            with open(rp, "a") as f:
                f.write(json.dumps({"id": it["id"], **vals}) + "\n")
            done[it["id"]] = vals
        print("all rated.")
    elif a.cmd == "tmetrics":
        items = {i["id"]: i for i in read_jsonl(ip)}
        rated = {r["id"]: r for r in read_jsonl(rp)} if Path(rp).exists() else {}
        rep = {"n_items": len(items), "n_rated": len(rated), "min_n": 60, "metrics": {}}
        if not rated:
            rep["note"] = "UNMEASURED: no human ratings yet"
        else:
            ids = [i for i in rated if i in items]
            h = [rated[i]["adequacy"] for i in ids]
            ce = [rated[i]["critical_error"] for i in ids]
            scores = {"bleu": [T.sentence_bleu(items[i]["candidate"], items[i]["reference"]) for i in ids],
                      "chrf": [T.chrf(items[i]["candidate"], items[i]["reference"]) for i in ids],
                      "fact_check": [float(T.facts_preserved(items[i])) for i in ids]}
            for m, sc in scores.items():
                med = sorted(sc)[len(sc) // 2]
                bad = [s for s, c in zip(sc, ce) if c == 1]
                rho, ci = T.spearman(sc, h), T.boot_spearman(sc, h)
                rep["metrics"][m] = {"n": len(ids), "spearman_adequacy": rho,
                                     "ci95": [round(x, 3) for x in ci] if ci else None,
                                     "missed_critical_rate": (sum(s >= med for s in bad) / len(bad)) if bad else None,
                                     "n_critical": len(bad)}
            rep["note"] = ("underpowered (n<60)" if len(ids) < 60 else "") or "ok"
        Path("results/translation_report.json").write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=2))
    elif a.cmd == "meta":
        from .metamorphic import run as meta_run
        out = meta_run(read_jsonl("data/scenarios.jsonl"), variant=a.variant)
        Path("results/metamorphic.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({k: (v if k != "relations" else {r: x["violation_rate"] for r, x in v.items()})
                          for k, v in out.items()}, indent=2))
    elif a.cmd == "dashboard":
        from .dashboard import build
        print("wrote", build())
