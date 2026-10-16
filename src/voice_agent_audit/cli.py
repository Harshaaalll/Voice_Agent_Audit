from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from .dataset import generate, read_jsonl, write_jsonl
from .runner import decision_confusion, run_variant, summarize


def main(argv=None):
    from .llm import QuotaExhausted, reset_phrase
    try:
        return _main(argv)
    except QuotaExhausted as e:
        phrase = reset_phrase(e.reset_hours)
        msg = f"Daily free-tier quota exhausted for {e.model} ({phrase}). "
        msg += "Completed calls are cached; re-run later to resume."
        print(msg, file=sys.stderr)
        sys.exit(2)


def _print_cache_stats(backend) -> None:
    from .llm import CachedBackend
    if isinstance(backend, CachedBackend):
        print(f"agent cache: {backend.hits} hits / {backend.misses} misses")


def _main(argv=None):
    ap = argparse.ArgumentParser(prog="vaa")
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
    bi = sub.add_parser("bias", help="position and verbosity bias probes for the judge")
    bi.add_argument("--limit", type=int, help="use only the first N transcripts (saves API quota)")
    sub.add_parser("tgen", help="generate translation items (Weekend 3)")
    sub.add_parser("rate", help="terminal tool to rate translations (YOUR human ratings)")
    sub.add_parser("tmetrics", help="BLEU/chrF/fact-check vs human ratings")
    m = sub.add_parser("meta", help="metamorphic invariance tests on the agent")
    m.add_argument("--variant", default="pruned")
    m.add_argument("--limit", type=int, help="use only the first N scenarios (saves API quota)")
    sub.add_parser("dashboard", help="static HTML dashboard in results/")
    i = sub.add_parser("ingest", help="load the policy chunks + embeddings into the database (pgvector)")
    i.add_argument("--embedder", default="hash", choices=["hash", "fastembed"])
    re_ = sub.add_parser("retrieval-eval", help="compare retrievers: recall@k, hit, MRR, latency (Phase 2)")
    re_.add_argument("--k", type=int, default=3)
    re_.add_argument("--db", action="store_true", help="also run the database (pgvector) path and compare")
    sub.add_parser("retrieval-review", help="confirm or fix the draft retrieval gold labels (YOU)")
    w = sub.add_parser("worker", help="run the background evaluation worker (needs REDIS_URL)")
    w.add_argument("--once", action="store_true", help="exit when the queue is empty")
    re2 = sub.add_parser("rag-eval", help="compare static / pruned / rag: pass rates, tokens, p50/p95 latency, context recall")
    re2.add_argument("--limit", type=int, default=60)
    re2.add_argument("--faithfulness", action="store_true", help="also score faithfulness with the judge")
    sub.add_parser("trace-check", help="send one test trace to Langfuse and print the HTTP status")
    a = ap.parse_args(argv)

    if a.cmd in ("transcripts", "label", "judge", "report", "bias"):
        return _weekend2(a)
    if a.cmd in ("tgen", "rate", "tmetrics", "meta", "dashboard"):
        return _weekend3(a)
    if a.cmd in ("ingest", "retrieval-eval", "retrieval-review"):
        return _phase2(a)
    if a.cmd in ("worker", "rag-eval", "trace-check"):
        return _phase3(a)
    if a.cmd == "gen":
        rows = generate(a.n, a.seed)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(rows, a.out)
        print(f"wrote {len(rows)} scenarios to {a.out}")
    else:
        rows = read_jsonl(a.data)
        by_id = {r["id"]: r for r in rows}
        report = {}
        from .llm import get_backend
        backend = get_backend()
        for variant in ("static", "pruned"):
            res = run_variant(rows, variant, noisy=not a.clean, backend=backend)
            report[variant] = {"summary": summarize(res), "decision_prf": decision_confusion(res, by_id)}
        s, p = report["static"]["summary"], report["pruned"]["summary"]
        report["token_reduction_pct"] = round(100 * (1 - p["mean_input_tokens"] / s["mean_input_tokens"]), 1)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(json.dumps({k: report[k]["summary"] for k in ("static", "pruned")}, indent=2))
        print("input-token reduction %:", report["token_reduction_pct"])
        _print_cache_stats(backend)


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
               "n_transcripts": len(tr[: a.limit] if a.limit else tr),
               "position": position_bias(jd, tr[: a.limit] if a.limit else tr),
               "verbosity": verbosity_bias(jd, tr[: a.limit] if a.limit else tr)}
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
        rows = read_jsonl("data/scenarios.jsonl")
        from .llm import get_backend
        backend = get_backend()
        out = meta_run(rows[: a.limit] if a.limit else rows, variant=a.variant, backend=backend)
        Path("results/metamorphic.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({k: (v if k != "relations" else {r: x["violation_rate"] for r, x in v.items()})
                          for k, v in out.items()}, indent=2))
        _print_cache_stats(backend)
    elif a.cmd == "dashboard":
        from .dashboard import build
        print("wrote", build())


def _phase2(a):
    import os
    from .rag.embed import get_embedder
    from .rag.gold import GOLD
    from .rag.policy import CHUNK_LIST
    reviewed_path = Path("data/retrieval_gold_reviewed.jsonl")
    reviewed = {r["query"]: r for r in read_jsonl(reviewed_path)} if reviewed_path.exists() else {}
    if a.cmd == "ingest":
        from .service.store import Store
        url = os.environ.get("DATABASE_URL", "sqlite:///results/vaa.db")
        emb = get_embedder(a.embedder)
        st = Store(url)
        n = st.upsert_chunks(emb.name, CHUNK_LIST, emb.embed([c["text"] for c in CHUNK_LIST]))
        print(f"ingested {n} chunks with embedder={emb.name} (dim={emb.dim}) into {st.kind}")
    elif a.cmd == "retrieval-review":
        print("Chunks:")
        for c in CHUNK_LIST:
            print(f"  {c['id']:16s} {c['text'][:90]}")
        print("\nEnter = accept draft | ids comma-separated = replace | s = skip | q = save & quit")
        for q, rel, style in GOLD:
            if q in reviewed:
                continue
            print(f"\nQUERY ({style}): {q}\n  draft relevant: {', '.join(rel)}")
            for r in rel:
                print("    -", next(c["text"] for c in CHUNK_LIST if c["id"] == r)[:110])
            ans = input("  > ").strip()
            if ans.lower() == "q":
                break
            if ans.lower() == "s":
                continue
            ids = rel if not ans else [x.strip() for x in ans.split(",")]
            valid = {c["id"] for c in CHUNK_LIST}
            if not set(ids) <= valid:
                print("  unknown id(s), skipped")
                continue
            with open(reviewed_path, "a") as f:
                f.write(json.dumps({"query": q, "relevant": ids, "changed": ids != rel}) + "\n")
            reviewed[q] = {"query": q, "relevant": ids}
        print(f"reviewed {len(reviewed)}/{len(GOLD)}")
    else:
        from .rag import metrics as M
        from .rag.retrievers import BM25, Dense, Hybrid, StoreRetriever
        gold = [(q, reviewed[q]["relevant"] if q in reviewed else rel, st) for q, rel, st in GOLD]
        bm25 = BM25(CHUNK_LIST)
        dense = {"hash": Dense(CHUNK_LIST, get_embedder("hash"))}
        skipped = {}
        try:
            dense["fastembed"] = Dense(CHUNK_LIST, get_embedder("fastembed"))
        except Exception as e:  # noqa: BLE001
            skipped["fastembed"] = f"{type(e).__name__}: {str(e)[:80]}"
        rets = {"bm25": bm25}
        for n, d in dense.items():
            rets[n] = d
            rets[f"{n}+bm25"] = Hybrid(d, bm25)
        report = {"k": a.k, "n_queries": len(gold), "n_chunks": len(CHUNK_LIST),
                  "gold": {"n_reviewed_by_human": len(reviewed), "source": "draft authored by the assistant; "
                           "only reviewed items are human-confirmed"},
                  "skipped": skipped, "retrievers": {}}
        rankings = {}
        for name, r in rets.items():
            res = M.evaluate(r, gold, k=a.k)
            rankings[name] = res.pop("_rankings")
            report["retrievers"][name] = res
        if a.db:
            from .service.store import Store
            url = os.environ.get("DATABASE_URL", "sqlite:///results/vaa.db")
            st = Store(url)
            emb = get_embedder("hash")
            st.upsert_chunks(emb.name, CHUNK_LIST, emb.embed([c["text"] for c in CHUNK_LIST]))
            res = M.evaluate(StoreRetriever(st, emb), gold, k=a.k)
            same = sum(x == y for x, y in zip(res.pop("_rankings"), rankings["hash"])) / len(gold)
            report["retrievers"][f"hash@{st.kind}"] = res
            report["db_matches_memory_top_k"] = same
        Path("results").mkdir(exist_ok=True)
        Path("results/retrieval_report.json").write_text(json.dumps(report, indent=2))
        print(f"{'retriever':18s} recall@{a.k}  hit@{a.k}   MRR   p50ms")
        for n, r in report["retrievers"].items():
            o = r["overall"]
            print(f"{n:18s} {o['recall']:.3f}    {o['hit']:.3f}  {o['rr']:.3f}  {r['latency_ms']['p50']}")
        if skipped:
            print("skipped:", skipped)


def _phase3(a):
    import os
    url = os.environ.get("DATABASE_URL", "sqlite:///results/vaa.db")
    if a.cmd == "worker":
        from .llm import get_backend
        from .service.core import Core
        from .service.queue import get_queue
        from .service.worker import loop
        q = get_queue()
        if q is None:
            raise SystemExit("REDIS_URL is not set")
        print("worker started; waiting for jobs", flush=True)
        loop(Core(url), q, get_backend(), os.environ.get("VAA_MODEL", "mock"), once=a.once)
    elif a.cmd == "trace-check":
        from .service.tracing import LangfuseTracer, get_tracer
        t = get_tracer()
        if not isinstance(t, LangfuseTracer):
            raise SystemExit("set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (and LANGFUSE_HOST if not EU cloud)")
        t.background = False
        t.record(name="trace-check", model="none", input="ping", output="pong",
                 usage={"input_tokens": 1, "output_tokens": 1}, latency_s=0.01)
        print("HTTP status:", t.last_status, "(2xx = accepted; 207 means check the response body in Langfuse)")
    else:
        from collections import defaultdict
        from .rag.metrics import mean_ci
        from .rag.ragmetrics import SCENARIO_REQUIRED, context_recall, faithfulness
        from .service.core import Core
        from .llm import get_backend
        core, backend = Core(url), get_backend()
        model = os.environ.get("VAA_MODEL", "mock")
        rows = read_jsonl(os.environ.get("VAA_DATA", "data/scenarios.jsonl"))[: a.limit]
        by_id = {r["id"]: r for r in rows}
        judge = None
        if a.faithfulness:
            from .judge import Judge
            judge = Judge()
        report = {"n": len(rows), "backend": os.environ.get("VAA_BACKEND", "mock"), "model": model,
                  "is_mock": os.environ.get("VAA_BACKEND", "mock") == "mock",
                  "retrieval": {"embedder": core.embedder.name, "mode": core.mode},
                  "required_chunk_map": "draft by the assistant, not human-reviewed", "variants": {}}
        for v in ("static", "pruned", "rag"):
            ctx = (lambda t: core.retrieve(t, 3)) if v == "rag" else None
            res = run_variant(rows, v, noisy=True, backend=backend, model=model, context_fn=ctx)
            out = summarize(res)
            if v == "rag":
                rec = {r["id"]: context_recall(r["context_ids"], by_id[r["id"]]["persona"]["intent"]) for r in res}
                m, lo, hi = mean_ci(list(rec.values()))
                per = defaultdict(list)
                for i, x in rec.items():
                    per[by_id[i]["persona"]["intent"]].append(x)
                out["context_recall"] = {"mean": round(m, 4), "ci95": [round(lo, 3), round(hi, 3)],
                                         "by_intent": {k: round(sum(x) / len(x), 3) for k, x in sorted(per.items())}}
                if judge:
                    fs = []
                    for r in res:
                        row = by_id[r["id"]]
                        texts = [core.by_id[c]["text"] for c in r["context_ids"]]
                        f = faithfulness(judge, r["raw"], texts, row["persona"]["account"], row["borrower_noisy"])
                        if f is not None:
                            fs.append(f)
                    out["faithfulness"] = {"mean": round(sum(fs) / len(fs), 4) if fs else None, "n": len(fs),
                                           "judge": judge.label, "judge_cache": judge.cache.stats()}
            report["variants"][v] = out
        Path("results").mkdir(exist_ok=True)
        Path("results/rag_report.json").write_text(json.dumps(report, indent=2))
        print(f"{'variant':8s} {'in_tok':>7s} {'p50_s':>9s} {'p95_s':>9s} {'decision':>9s} {'ctx_recall':>10s}")
        for v, o in report["variants"].items():
            cr = o.get("context_recall", {}).get("mean", "-")
            print(f"{v:8s} {o['mean_input_tokens']:7.0f} {o['latency_s']['p50']:9.5f} {o['latency_s']['p95']:9.5f} "
                  f"{o['pass_rate']['decision']:9.3f} {cr!s:>10s}")
        if report["is_mock"]:
            print("(mock backend: tokens/latency/pass rates are plumbing, not model quality)")
        _print_cache_stats(backend)
