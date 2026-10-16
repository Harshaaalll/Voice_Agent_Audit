import math
import pytest
from fastapi.testclient import TestClient

from voice_agent_audit.rag.embed import HashEmbedder
from voice_agent_audit.rag.gold import GOLD
from voice_agent_audit.rag.metrics import score_query
from voice_agent_audit.rag.policy import CHUNK_LIST
from voice_agent_audit.rag.retrievers import BM25, Dense, Hybrid, StoreRetriever
from voice_agent_audit.service.app import create_app
from voice_agent_audit.service.store import Store

ACCT = {"loan_id": "LN1", "amount_due": 8500, "due_date": "2026-09-09", "days_overdue": 7, "lender": "Sunrise Finance"}


def test_corpus_and_gold_are_consistent():
    ids = [c["id"] for c in CHUNK_LIST]
    assert len(ids) == len(set(ids))
    for q, rel, style in GOLD:
        assert rel and set(rel) <= set(ids), q
        assert style in {"english", "hinglish", "hindi_roman"}


def test_hash_embedder_deterministic_unit_norm_and_similarity():
    e = HashEmbedder()
    a, b = e.embed(["payment link sent by SMS", "payment link sent by SMS"])
    assert a == b and math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)
    close, far = e.embed(["payments links SMS", "grace period three days"])
    assert sum(x * y for x, y in zip(a, close)) > sum(x * y for x, y in zip(a, far))


def test_bm25_finds_exact_wording():
    top = BM25(CHUNK_LIST).search("one time password OTP PIN card number", 1)[0][0]
    assert top == "channel-2"


def test_score_query_hand_computed():
    s = score_query(["x", "a", "y", "b"], {"a", "b"}, k=3)
    assert s["recall"] == 0.5 and s["hit"] == 1.0 and s["rr"] == 0.5 and s["precision"] == pytest.approx(1 / 3)
    z = score_query(["x", "y"], {"a"}, k=2)
    assert z["recall"] == 0.0 and z["rr"] == 0.0


def test_rrf_prefers_items_ranked_well_by_both():
    class R:
        def __init__(s, name, ids): s.name, s.ids = name, ids
        def search(s, q, k): return [(i, 0) for i in s.ids][:k]
    h = Hybrid(R("a", ["x", "shared", "y"]), R("b", ["z", "shared", "w"]))
    assert h.search("q", 1)[0][0] == "shared"


def test_store_search_matches_in_memory_dense(tmp_path):
    emb = HashEmbedder()
    st = Store(f"sqlite:///{tmp_path}/v.db")
    st.upsert_chunks(emb.name, CHUNK_LIST, emb.embed([c["text"] for c in CHUNK_LIST]))
    assert st.count_chunks("hash") == len(CHUNK_LIST)
    dense, db = Dense(CHUNK_LIST, emb), StoreRetriever(st, emb)
    for q, _, _ in GOLD[:10]:
        assert [i for i, _ in dense.search(q, 3)] == [i for i, _ in db.search(q, 3)]
    st.upsert_chunks(emb.name, CHUNK_LIST, emb.embed([c["text"] for c in CHUNK_LIST]))   # idempotent
    assert st.count_chunks("hash") == len(CHUNK_LIST)


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(f"sqlite:///{tmp_path}/t.db")) as c:
        yield c


def test_service_loads_policy_and_retrieves(client):
    h = client.get("/health").json()
    assert h["chunks"] == len(CHUNK_LIST) and h["embedder"] == "hash"
    r = client.post("/retrieve", json={"query": "OTP PIN card number password", "k": 2}).json()
    assert r["hits"][0]["id"] == "channel-2" and len(r["hits"]) == 2


def test_respond_rag_returns_and_stores_context(client):
    r = client.post("/respond", json={"borrower_text": "Meri job chali gayi hai", "account": ACCT, "variant": "rag"})
    assert r.status_code == 200 and len(r.json()["context_ids"]) == 3
    row = client.app.state.core.store._exec("SELECT context_ids FROM responses")[0]
    assert "[" in row["context_ids"]
    plain = client.post("/respond", json={"borrower_text": "hi", "account": ACCT, "variant": "pruned"}).json()
    assert plain["context_ids"] is None


def test_evaluate_rag_variant(client):
    r = client.post("/evaluate", json={"variant": "rag", "limit": 5})
    assert r.status_code == 200 and r.json()["summary"]["n"] == 5


def test_hybrid_mode_in_service(tmp_path, monkeypatch):
    monkeypatch.setenv("VAA_RETRIEVAL", "hybrid")
    with TestClient(create_app(f"sqlite:///{tmp_path}/h.db")) as c:
        assert c.get("/health").json()["retrieval"] == "hybrid"
        hits = c.post("/retrieve", json={"query": "OTP PIN card number password", "k": 3}).json()["hits"]
        assert hits[0]["id"] == "channel-2" and len(hits) == 3
    monkeypatch.setenv("VAA_RETRIEVAL", "nope")
    with pytest.raises(ValueError):
        with TestClient(create_app(f"sqlite:///{tmp_path}/x.db")):
            pass
