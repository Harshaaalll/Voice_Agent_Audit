import json

from voice_agent_audit.dashboard import build


def _page(tmp_path):
    out = tmp_path / "dash.html"
    build(str(tmp_path), str(out))
    return out.read_text()


def _ret(name="bm25", skipped=None):
    return {"k": 3, "n_queries": 49, "gold": {"n_reviewed_by_human": 0}, "skipped": skipped or {},
            "retrievers": {name: {"overall": {"recall": 0.449, "recall_ci95": [0.316, 0.592], "hit": 0.4694, "rr": 0.453},
                                  "latency_ms": {"p50": 0.705, "p95": 11.6}}}}


def _rag(mock=True):
    pr = {"decision": 0.9333}
    lat = {"p50": 0.0004, "p95": 0.02}
    return {"is_mock": mock, "retrieval": {"embedder": "fastembed", "mode": "hybrid"}, "variants": {
        "static": {"mean_input_tokens": 792.2, "pass_rate": pr, "latency_s": lat},
        "rag": {"mean_input_tokens": 365.4, "pass_rate": pr, "latency_s": lat,
                "context_recall": {"mean": 0.4417, "ci95": [0.333, 0.55]}, "faithfulness": {"mean": 0.81}}}}


def test_missing_reports_unmeasured(tmp_path):
    p = _page(tmp_path)
    assert "Phase 2 · retrieval" in p and "Phase 3 · RAG variants" in p
    assert p.count("UNMEASURED") >= 2
    assert "vaa retrieval-eval" in p and "vaa rag-eval" in p


def test_retrieval_section(tmp_path):
    (tmp_path / "retrieval_report.json").write_text(json.dumps(_ret(skipped={"fastembed": "ImportError"})))
    p = _page(tmp_path)
    assert "0 of 49 gold queries reviewed by a human (the rest are drafts)" in p
    assert "0.449" in p and "0.316" in p and "0.592" in p and "0.453" in p and "0.705" in p
    assert "fastembed" in p


def test_rag_section(tmp_path):
    (tmp_path / "rag_report.json").write_text(json.dumps(_rag()))
    p = _page(tmp_path)
    assert "MOCK backend, plumbing only, not model quality" in p
    assert "fastembed" in p and "hybrid" in p
    assert "792.2" in p and "0.933" in p
    assert "0.442" in p and "0.810" in p
    assert "—" in p  # static variant has no context recall / faithfulness


def test_rag_not_mock(tmp_path):
    (tmp_path / "rag_report.json").write_text(json.dumps(_rag(mock=False)))
    assert "MOCK backend, plumbing only, not model quality" not in _page(tmp_path)


def test_escaping(tmp_path):
    evil = "<script>x</script>"
    r = _rag()
    r["variants"][evil] = r["variants"].pop("static")
    r["retrieval"]["embedder"] = evil
    (tmp_path / "retrieval_report.json").write_text(json.dumps(_ret(name=evil, skipped={evil: evil})))
    (tmp_path / "rag_report.json").write_text(json.dumps(r))
    p = _page(tmp_path)
    assert evil not in p
    assert "&lt;script&gt;x&lt;/script&gt;" in p
