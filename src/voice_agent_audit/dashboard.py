"""Static HTML dashboard from results/*.json. Anything without data is shown as UNMEASURED."""
from __future__ import annotations
import html
import json
from pathlib import Path


def _load(p):
    p = Path(p)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except (ValueError, OSError):
        return None
    return d if isinstance(d, dict) else None


def _f(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except (ValueError, TypeError):
        return "—"


def _ci(ci):
    """' [lo, hi]' with bad bounds as em dashes, or '' when ci is not a 2-list."""
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return ""
    return f" [{_f(ci[0])}, {_f(ci[1])}]"


def _d(x):
    return x if isinstance(x, dict) else {}


def build(results_dir="results", out="results/dashboard.html") -> str:
    R = Path(results_dir)
    run, jr, jb = _load(R / "run.json"), _load(R / "judge_report.json"), _load(R / "judge_bias.json")
    tr, mm = _load(R / "translation_report.json"), _load(R / "metamorphic.json")
    sec = []

    def table(head, rows):
        h = "".join(f"<th>{html.escape(c)}</th>" for c in head)
        b = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
        return f"<table><tr>{h}</tr>{b}</table>"

    def sect(title, body, note=""):
        sec.append(f"<section><h2>{html.escape(title)}</h2>{('<p class=note>' + html.escape(note) + '</p>') if note else ''}{body}</section>")

    if run:
        rows = [(k, v) for k, v in run.items() if isinstance(v, (int, float, str))][:12]
        sect("Weekend 1 · agent run", table(["field", "value"], rows),
             "Mock backend unless the file says otherwise — plumbing numbers, not model quality.")
    else:
        sect("Weekend 1 · agent run", "<p>UNMEASURED</p>")
    if jr:
        rows = []
        for c, v in jr["criteria"].items():
            if "status" in v:
                rows.append((c, "UNMEASURED", "", "", ""))
                continue
            j = v.get("judge", {})
            rows.append((c, v["n"], _f(j.get("kappa")), str(j.get("kappa_ci95") and [round(x, 2) for x in j["kappa_ci95"]]),
                         "underpowered (<150)" if v["underpowered"] else "ok"))
        sect("Weekend 2 · judge vs human (Cohen's κ)", table(["criterion", "n labeled", "κ", "95% CI", "power"], rows),
             f"Judge: {jr['judge']}")
    else:
        sect("Weekend 2 · judge vs human", "<p>UNMEASURED — run vaa label, judge, report</p>")
    if jb:
        p, v = jb["position"], jb["verbosity"]
        sect("Weekend 2 · judge bias", table(["probe", "value"], [
            ("first-position rate (0.5 = none)", _f(p["first_position_rate"])),
            ("order-inconsistency rate", _f(p["inconsistency_rate"])),
            ("prefers longer note", _f(v["prefers_longer_rate"])),
            ("faithful flip under padding", _f(v["note_faithful_flip_rate"]))]),
             "PROXY judge — meaningless" if jb.get("is_proxy") else f"Judge: {jb['judge']}")
    if tr:
        rows = [(m, v["n"], _f(v["spearman_adequacy"]), str(v["ci95"]), _f(v["missed_critical_rate"]))
                for m, v in tr["metrics"].items()]
        sect("Weekend 3 · translation metrics vs human adequacy", table(
            ["metric", "n rated", "Spearman ρ", "95% CI", "critical errors scored above median"], rows),
             tr.get("note", ""))
    else:
        sect("Weekend 3 · translation metrics vs human", "<p>UNMEASURED — run vaa rate, tmetrics</p>")
    if mm:
        rows = [(k, f"{v['violations']}/{v['n']}", _f(v["violation_rate"])) for k, v in mm["relations"].items()]
        sect("Weekend 3 · metamorphic invariance", table(["relation", "violations", "rate"], rows),
             "MOCK backend — plumbing only" if mm["is_mock"] else f"{mm['backend']}:{mm['model']}")
    rt, rg = _load(R / "retrieval_report.json"), _load(R / "rag_report.json")
    def missing(title, path, cmd):
        why = "UNMEASURED (unreadable report)" if (R / path).exists() else "UNMEASURED"
        sect(title, f"<p>{why} — run {cmd}</p>")

    if rt:
        rows = []
        for name, v in _d(rt.get("retrievers")).items():
            o = _d(_d(v).get("overall"))
            rows.append((name, _f(o.get("recall")) + _ci(o.get("recall_ci95")), _f(o.get("hit")), _f(o.get("rr")),
                         _f(_d(_d(v).get("latency_ms")).get("p50"))))
        note = (f"{_d(rt.get('gold')).get('n_reviewed_by_human', 0)} of {rt.get('n_queries')} gold queries "
                "reviewed by a human (the rest are drafts)")
        sk = rt.get("skipped")
        if isinstance(sk, dict) and sk:
            note += ". Skipped: " + ", ".join(f"{k} ({v})" for k, v in sk.items())
        elif isinstance(sk, list) and sk:
            note += ". Skipped: " + ", ".join(str(k) for k in sk)
        sect("Phase 2 · retrieval", table(
            ["retriever", f"recall@{rt.get('k', 'k')} [95% CI]", "hit", "MRR", "p50 ms"], rows), note)
    else:
        missing("Phase 2 · retrieval", "retrieval_report.json", "vaa retrieval-eval")
    if rg:
        rows = []
        for name, v in _d(rg.get("variants")).items():
            v = _d(v)
            cr, fa = v.get("context_recall"), v.get("faithfulness")
            lat = _d(v.get("latency_s"))
            rows.append((name, _f(v.get("mean_input_tokens"), 1), _f(lat.get("p50"), 4), _f(lat.get("p95"), 4),
                         _f(_d(v.get("pass_rate")).get("decision")),
                         (_f(_d(cr).get("mean")) + _ci(_d(cr).get("ci95"))) if isinstance(cr, dict) else "—",
                         _f(_d(fa).get("mean")) if isinstance(fa, dict) else "—"))
        rv = _d(rg.get("retrieval"))
        note = f"Retrieval setup: embedder {rv.get('embedder')}, mode {rv.get('mode')}."
        if rg.get("is_mock"):
            note = "MOCK backend, plumbing only, not model quality. " + note
        sect("Phase 3 · RAG variants", table(
            ["variant", "mean input tokens", "latency p50 s", "latency p95 s", "decision pass rate",
             "context recall [95% CI]", "faithfulness"], rows), note)
    else:
        missing("Phase 3 · RAG variants", "rag_report.json", "vaa rag-eval")
    css = ("body{font:15px system-ui;margin:2rem auto;max-width:860px;padding:0 1rem;background:#fff;color:#111}"
           "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:.35rem .6rem;text-align:left}"
           "h2{margin-top:2rem}.note{color:#555}@media(prefers-color-scheme:dark){body{background:#111;color:#eee}"
           "td,th{border-color:#444}.note{color:#aaa}}")
    page = f"<!doctype html><meta charset=utf-8><title>Voice Agent Audit</title><style>{css}</style><h1>Voice Agent Audit</h1><p class=note>Evaluation kit for Hinglish payment-reminder voice agents</p>{''.join(sec)}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(page)
    return out
