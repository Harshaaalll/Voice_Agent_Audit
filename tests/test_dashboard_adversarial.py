import json, shutil
from pathlib import Path
import pytest
from voice_agent_audit.dashboard import build

RT = {"k": 3, "n_queries": 49, "gold": {"n_reviewed_by_human": 0}, "skipped": {},
      "retrievers": {"bm25": {"overall": {"recall": 0.449, "recall_ci95": [0.316, 0.592], "hit": 0.47, "rr": 0.45},
                              "latency_ms": {"p50": 0.7}}}}
RG = {"is_mock": True, "retrieval": {"embedder": "fastembed", "mode": "hybrid"},
      "variants": {"static": {"mean_input_tokens": 792.2, "latency_s": {"p50": 0.1, "p95": 0.2},
                              "pass_rate": {"decision": 0.9}}}}


def run(tmp_path, rt=None, rg=None, raw_rt=None, raw_rg=None):
    if rt is not None: (tmp_path / "retrieval_report.json").write_text(json.dumps(rt))
    if raw_rt is not None: (tmp_path / "retrieval_report.json").write_text(raw_rt)
    if rg is not None: (tmp_path / "rag_report.json").write_text(json.dumps(rg))
    if raw_rg is not None: (tmp_path / "rag_report.json").write_text(raw_rg)
    out = tmp_path / "o" / "d.html"
    out.parent.mkdir()
    return Path(build(str(tmp_path), str(out))).read_text()


def mut(base, f):
    d = json.loads(json.dumps(base)); f(d); return d


def test_baseline_ok(tmp_path):
    h = run(tmp_path, RT, RG)
    assert "Phase 2 · retrieval" in h and "0 of 49 gold queries" in h and "MOCK backend, plumbing only, not model quality" in h
    assert "fastembed" in h and "hybrid" in h


def test_missing_both(tmp_path):
    h = run(tmp_path)
    assert h.count("UNMEASURED") >= 2 and "vaa retrieval-eval" in h and "vaa rag-eval" in h


def test_escape(tmp_path):
    rt = mut(RT, lambda d: d["retrievers"].update({"<script>x</script>": d["retrievers"]["bm25"]}))
    rt["skipped"] = {"<b>s</b>": "<i>why</i>"}
    rg = mut(RG, lambda d: d["variants"].update({"<script>y</script>": d["variants"]["static"]}))
    rg["retrieval"]["embedder"] = "<img src=x>"
    h = run(tmp_path, rt, rg)
    for bad in ("<script>x", "<script>y", "<b>s</b>", "<i>why", "<img src=x>"):
        assert bad not in h


def test_skipped_named(tmp_path):
    assert "faiss" in run(tmp_path, mut(RT, lambda d: d.update(skipped={"faiss": "not installed"})))


def test_no_ci(tmp_path):
    run(tmp_path, mut(RT, lambda d: d["retrievers"]["bm25"]["overall"].pop("recall_ci95")))

def test_no_latency(tmp_path):
    run(tmp_path, mut(RT, lambda d: d["retrievers"]["bm25"].pop("latency_ms")))

def test_no_gold(tmp_path):
    run(tmp_path, mut(RT, lambda d: d.pop("gold")))

def test_no_n_queries(tmp_path):
    run(tmp_path, mut(RT, lambda d: d.pop("n_queries")))

def test_empty_retrievers(tmp_path):
    run(tmp_path, mut(RT, lambda d: d.update(retrievers={})))

def test_ctx_recall_no_ci(tmp_path):
    h = run(tmp_path, rg=mut(RG, lambda d: d["variants"]["static"].update(context_recall={"mean": 0.4})))
    assert "0.400" in h

def test_faith_none_mean(tmp_path):
    run(tmp_path, rg=mut(RG, lambda d: d["variants"]["static"].update(faithfulness={"mean": None})))

def test_empty_variants(tmp_path):
    run(tmp_path, rg=mut(RG, lambda d: d.update(variants={})))

def test_no_decision(tmp_path):
    run(tmp_path, rg=mut(RG, lambda d: d["variants"]["static"].update(pass_rate={})))

def test_no_retrieval_key(tmp_path):
    run(tmp_path, rg=mut(RG, lambda d: d.pop("retrieval")))

def test_not_mock_note(tmp_path):
    assert "MOCK backend" not in run(tmp_path, rg=mut(RG, lambda d: d.update(is_mock=False)))

def test_invalid_json_rag(tmp_path):
    run(tmp_path, raw_rg="{not json")

def test_empty_file_rag(tmp_path):
    run(tmp_path, raw_rg="")

def test_invalid_json_retrieval(tmp_path):
    run(tmp_path, raw_rt="{not json")

def test_empty_file_retrieval(tmp_path):
    run(tmp_path, raw_rt="")

def test_unicode_and_long(tmp_path):
    n = "रिट्रीवर-é-" + "x" * 5000
    h = run(tmp_path, mut(RT, lambda d: d["retrievers"].update({n: d["retrievers"]["bm25"]})))
    assert "रिट्रीवर" in h

def test_string_numbers_retrieval(tmp_path):
    run(tmp_path, mut(RT, lambda d: d["retrievers"]["bm25"]["overall"].update(recall="0.5")))

def test_string_numbers_rag(tmp_path):
    run(tmp_path, rg=mut(RG, lambda d: d["variants"]["static"].update(mean_input_tokens="800")))

def test_ci_with_none(tmp_path):
    run(tmp_path, mut(RT, lambda d: d["retrievers"]["bm25"]["overall"].update(recall_ci95=[None, 0.5])))

def test_retriever_null(tmp_path):
    run(tmp_path, mut(RT, lambda d: d["retrievers"].update(bm25=None)))

def test_variant_null(tmp_path):
    run(tmp_path, rg=mut(RG, lambda d: d["variants"].update(static=None)))

def test_json_top_level_list(tmp_path):
    run(tmp_path, raw_rg="[]")

def test_skipped_list(tmp_path):
    run(tmp_path, mut(RT, lambda d: d.update(skipped=["faiss"])))

def test_nested_out_dir(tmp_path):
    out = tmp_path / "a" / "b" / "d.html"
    build(str(tmp_path), str(out))

REAL = Path(__file__).resolve().parents[1] / "results"

@pytest.mark.skipif(not (REAL / "rag_report.json").exists(), reason="no real results")
def test_real_numbers(tmp_path):
    for f in REAL.glob("*.json"): shutil.copy(f, tmp_path / f.name)
    out = tmp_path / "d.html"
    build(str(tmp_path), str(out))
    h = out.read_text()
    rt = json.loads((REAL / "retrieval_report.json").read_text())
    rg = json.loads((REAL / "rag_report.json").read_text())
    for n, v in rt["retrievers"].items():
        assert n in h and f"{v['overall']['recall']:.3f}" in h and f"{v['overall']['rr']:.3f}" in h
    for n, v in rg["variants"].items():
        assert f"{v['mean_input_tokens']:.1f}" in h
        assert f"{v['pass_rate']['decision']:.3f}" in h
