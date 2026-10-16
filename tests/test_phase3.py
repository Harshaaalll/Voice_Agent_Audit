import json
import pytest
from fastapi.testclient import TestClient

from voice_agent_audit.judge import Judge
from voice_agent_audit.llm import Backend, LLMResult
from voice_agent_audit.rag.ragmetrics import context_recall, faithfulness, statements
from voice_agent_audit.runner import summarize
from voice_agent_audit.service.app import create_app
from voice_agent_audit.service.cache import MemoryCache, RedisCache, make_key
from voice_agent_audit.service.queue import MemoryQueue, RedisQueue
from voice_agent_audit.service.tracing import LangfuseTracer, NoopTracer
from voice_agent_audit.service.worker import process_one


class FakeRedis:
    def __init__(self): self.d, self.l = {}, []
    def get(self, k): return self.d.get(k)
    def set(self, k, v, ex=None): self.d[k] = v
    def lpush(self, k, v): self.l.insert(0, v)
    def brpop(self, k, timeout=0): return (k, self.l.pop()) if self.l else None


def test_cache_key_changes_with_any_part():
    assert make_key("g", "m", "s", "u") == make_key("g", "m", "s", "u")
    assert make_key("g", "m", "s", "u") != make_key("g", "m2", "s", "u")
    assert make_key("a", "bc") != make_key("ab", "c")


@pytest.mark.parametrize("cache", [MemoryCache(), RedisCache(FakeRedis())])
def test_cache_roundtrip_and_stats(cache):
    assert cache.get("k") is None
    cache.set("k", {"label": 1})
    assert cache.get("k") == {"label": 1}
    s = cache.stats()
    assert (s["hits"], s["misses"], s["hit_rate"]) == (1, 1, 0.5)


class CountingBackend(Backend):
    def __init__(self): self.calls = 0
    def complete(self, system, messages, model):
        self.calls += 1
        return LLMResult(text='{"supported": 1}', input_tokens=1, output_tokens=1, latency_s=0.0, token_source="est")


def _real_judge(cache):
    j = Judge(cache=cache)               # mock => proxy; flip to the "real" path with a fake backend
    j.is_proxy, j.backend, j.backend_name, j.model = False, CountingBackend(), "fake", "m"
    return j


def test_judge_calls_backend_once_per_unique_prompt():
    j = _real_judge(MemoryCache())
    assert j.support("ctx", "stmt") == 1 and j.support("ctx", "stmt") == 1
    assert j.backend.calls == 1                       # second call served from cache
    j.support("ctx", "another")
    assert j.backend.calls == 2
    assert j.cache.stats()["hit_rate"] == pytest.approx(1 / 3, abs=1e-3)


def test_unparseable_judge_output_is_not_cached():
    j = _real_judge(MemoryCache())
    j.backend.complete = lambda *a, **k: LLMResult(text="not json", input_tokens=1, output_tokens=1, latency_s=0, token_source="e")
    assert j.support("c", "s") is None and j.support("c", "s") is None


def test_proxy_support_and_faithfulness():
    j = Judge(cache=MemoryCache())
    assert j.support("Amount 8500 due", "You owe 8500 rupees.") == 1
    assert j.support("Amount 8500 due", "Late fee 750 rupees.") == 0
    assert j.support("anything", "Otherwise legal action will be taken.") == 0
    raw = json.dumps({"reply": "Thank you. Late fee 750 rupees.", "tool": {}})
    assert statements(raw) == ["Thank you.", "Late fee 750 rupees."]
    assert faithfulness(j, raw, ["policy"], {"amount_due": 8500}, "hi") == 0.5
    assert faithfulness(j, "not json", [], {}, "") is None


def test_context_recall():
    assert context_recall(["hardship-1", "x", "y"], "hardship") == 0.5
    assert context_recall(["hardship-1", "hardship-2"], "hardship") == 1.0
    assert context_recall([], "abusive") == 0.0


def test_summarize_percentiles():
    mk = lambda lat, r: {"checks": {"a": {"pass": 1}}, "input_tokens": 1, "output_tokens": 1,
                         "latency_s": lat, "token_source": "e", "retrieval_s": r}
    s = summarize([mk(i / 100, i / 1000) for i in range(1, 101)])
    assert s["latency_s"]["p50"] == pytest.approx(0.51) and s["latency_s"]["p95"] == pytest.approx(0.96)
    assert "retrieval_s" in s and "retrieval_s" not in summarize([{**mk(0.1, None)}])


def test_langfuse_request_shape_and_failure_isolation():
    sent = []
    t = LangfuseTracer("https://lf.example/", "pk", "sk", post=lambda u, j, a: sent.append((u, j, a)) or 207, background=False)
    t.record(name="respond:rag", model="m", input="hi", output="out",
             usage={"input_tokens": 5, "output_tokens": 2}, latency_s=0.5, trace_id="T1", metadata={"x": 1})
    url, payload, auth = sent[0]
    assert url == "https://lf.example/api/public/ingestion" and auth == ("pk", "sk")
    kinds = [e["type"] for e in payload["batch"]]
    assert kinds == ["trace-create", "generation-create"]
    gen = payload["batch"][1]["body"]
    assert gen["traceId"] == "T1" and gen["usage"] == {"input": 5, "output": 2, "unit": "TOKENS"}
    def boom(*a): raise RuntimeError("network down")
    bad = LangfuseTracer("https://x", "a", "b", post=boom, background=False)
    bad.record(name="n", model="m", input="i", output="o", usage={"input_tokens": 1, "output_tokens": 1}, latency_s=0.1)
    assert bad.failures == 1                           # swallowed, counted
    assert NoopTracer().record(anything=1) is None


def test_redis_queue_roundtrip():
    q = RedisQueue(FakeRedis())
    q.enqueue("a"); q.enqueue("b")
    assert (q.pop(), q.pop(), q.pop()) == ("a", "b", None)      # FIFO


@pytest.fixture()
def client(tmp_path):
    q, tr = MemoryQueue(), LangfuseTracer("https://x", "a", "b", post=lambda *a: 200, background=False)
    with TestClient(create_app(f"sqlite:///{tmp_path}/t.db", queue=q, tracer=tr)) as c:
        c.queue = q
        yield c


def test_async_evaluate_through_worker(client):
    r = client.post("/evaluate", json={"variant": "rag", "limit": 6, "run_async": True})
    jid = r.json()["job_id"]
    assert client.get(f"/jobs/{jid}").json()["status"] == "queued"
    core = client.app.state.core
    assert process_one(core, client.queue, client.app.state.backend or __import__("voice_agent_audit.llm", fromlist=["x"]).get_backend(), "mock", 0) == jid
    j = client.get(f"/jobs/{jid}").json()
    assert j["status"] == "done" and j["run_id"]
    m = client.get("/metrics").json()["rag"]
    assert m["run_id"] == j["run_id"] and "context_recall" in m and "retrieval_s" in m


def test_failed_job_is_recorded_not_raised(client, monkeypatch):
    jid = client.post("/evaluate", json={"variant": "pruned", "limit": 3, "run_async": True}).json()["job_id"]
    monkeypatch.setenv("VAA_DATA", "/nonexistent.jsonl")
    from voice_agent_audit.llm import get_backend
    process_one(client.app.state.core, client.queue, get_backend(), "mock", 0)
    j = client.get(f"/jobs/{jid}").json()
    assert j["status"] == "failed" and "not found" in j["error"]


def test_async_without_queue_is_503(tmp_path):
    with TestClient(create_app(f"sqlite:///{tmp_path}/n.db", queue=None, tracer=NoopTracer())) as c:
        r = c.post("/evaluate", json={"variant": "pruned", "run_async": True})
        assert r.status_code == 503 or c.get("/health").json()["queue"] is True   # REDIS_URL unset => no queue


def test_respond_sends_trace(client):
    acct = {"loan_id": "L", "amount_due": 100, "due_date": "2026-01-01", "days_overdue": 1, "lender": "X"}
    r = client.post("/respond", json={"borrower_text": "hello", "account": acct, "variant": "rag"})
    assert r.status_code == 200 and client.app.state.tracer.last_status == 200
    assert client.get("/health").json()["tracing"] is True


def test_worker_loop_survives_transient_queue_errors(client):
    from voice_agent_audit.service.worker import loop
    class Flaky:
        def __init__(self): self.n = 0
        def pop(self, timeout=0):
            self.n += 1
            if self.n == 1:
                raise TimeoutError("Timeout reading from socket")
            return None
    q = Flaky()
    import threading
    t = threading.Thread(target=lambda: loop(client.app.state.core, q, None, "mock", once=False, retry_s=0.01), daemon=True)
    t.start(); t.join(0.3)
    assert q.n >= 3            # raised once, then kept polling
