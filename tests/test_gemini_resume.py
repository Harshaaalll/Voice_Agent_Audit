
import pytest
import requests

from voice_agent_audit import cli
from voice_agent_audit.llm import (CachedBackend, GeminiBackend, LLMResult, MockBackend, QuotaExhausted,
                                   get_backend)
from voice_agent_audit.service.cache import DiskCache, MemoryCache, make_key

MSG = [{"role": "user", "content": "u"}]
OK = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}], "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2}}


class Resp:
    def __init__(self, code, body=None, headers=None):
        self.status_code, self._b, self.headers = code, body or {}, headers or {}
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)
    def json(self): return self._b


def _gemini(monkeypatch, seq):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    calls, it = [], iter(seq)
    def post(url, headers, json, timeout):
        calls.append(1)
        x = next(it)
        if isinstance(x, Exception):
            raise x
        return x
    monkeypatch.setattr(requests, "post", post)
    b = GeminiBackend()
    b.sleep = lambda s: None
    return b, calls


# A1
def test_timeouts_are_retried_then_succeed(monkeypatch):
    b, calls = _gemini(monkeypatch, [requests.exceptions.ReadTimeout(), requests.exceptions.ConnectionError(), Resp(200, OK)])
    assert b.complete("s", MSG, "m").text == "{}" and len(calls) == 3


def test_always_timeout_reraises_original(monkeypatch):
    b, calls = _gemini(monkeypatch, [requests.exceptions.ReadTimeout("x")] * 10)
    b.max_retries = 2
    with pytest.raises(requests.exceptions.ReadTimeout):
        b.complete("s", MSG, "m")
    assert len(calls) == 3


# A2
DAILY = {"error": {"code": 429, "message": "You exceeded your current quota", "details": [
    {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
    {"retryDelay": "49372s"}]}}
PERMIN = {"error": {"code": 429, "details": [
    {"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}, {"retryDelay": "30s"}]}}


def test_daily_429_not_retried(monkeypatch):
    b, calls = _gemini(monkeypatch, [Resp(429, DAILY), Resp(200, OK)])
    with pytest.raises(QuotaExhausted) as e:
        b.complete("s", MSG, "gemma-x")
    assert len(calls) == 1
    assert "gemma-x" in str(e.value) and "daily" in str(e.value).lower() and "14 h" in str(e.value)


def test_per_minute_429_still_retried(monkeypatch):
    b, calls = _gemini(monkeypatch, [Resp(429, PERMIN), Resp(200, OK)])
    assert b.complete("s", MSG, "m").text == "{}" and len(calls) == 2


# A3
def test_thought_parts_ignored_and_last_text_used(monkeypatch):
    body = {"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True}, {"text": "answer"}]}}],
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 4}}
    b, _ = _gemini(monkeypatch, [Resp(200, body)])
    r = b.complete("s", MSG, "m")
    assert r.text == "answer" and (r.input_tokens, r.output_tokens) == (7, 4)


def test_no_text_raises(monkeypatch):
    body = {"candidates": [{"content": {"parts": [{"text": "t", "thought": True}]}}]}
    b, _ = _gemini(monkeypatch, [Resp(200, body)])
    with pytest.raises(ValueError, match="no text in model response"):
        b.complete("s", MSG, "m")


# A4
class Inner:
    name = "fake"
    def __init__(self): self.n = 0; self.fail = False
    def complete(self, system, messages, model):
        self.n += 1
        if self.fail:
            raise RuntimeError("boom")
        return LLMResult(f"t{self.n}", 5, 6, 1.5, "provider")


def test_cache_hit_and_misses(tmp_path):
    inner = Inner()
    c = CachedBackend(inner, DiskCache(str(tmp_path / "c.db")))
    a = c.complete("s", MSG, "m")
    b = c.complete("s", MSG, "m")
    assert inner.n == 1 and not a.cached and b.cached
    assert (b.text, b.input_tokens, b.output_tokens, b.token_source, b.latency_s) == (a.text, 5, 6, "provider", 1.5)
    c.complete("s", MSG, "m2"); c.complete("s2", MSG, "m"); c.complete("s", [{"role": "user", "content": "v"}], "m")
    assert inner.n == 4 and (c.hits, c.misses) == (1, 4)


def test_errors_not_cached():
    inner = Inner(); inner.fail = True
    c = CachedBackend(inner, MemoryCache())
    with pytest.raises(RuntimeError):
        c.complete("s", MSG, "m")
    inner.fail = False
    assert c.complete("s", MSG, "m").text == "t2" and inner.n == 2


def test_key_prefix_default_unchanged():
    assert make_key("a").startswith("vaa:judge:") and make_key("a", prefix="vaa:agent:").startswith("vaa:agent:")


def test_get_backend_wrapping(monkeypatch, tmp_path):
    monkeypatch.delenv("VAA_NO_CACHE", raising=False)
    monkeypatch.setenv("VAA_AGENT_CACHE", str(tmp_path / "a.db"))
    assert isinstance(get_backend("mock"), MockBackend)
    assert isinstance(get_backend("gemini"), CachedBackend)
    assert isinstance(get_backend("gemini", cached=False), GeminiBackend)
    monkeypatch.setenv("VAA_NO_CACHE", "1")
    assert isinstance(get_backend("gemini"), GeminiBackend)


# A5
def test_cli_quota_exit_code(monkeypatch, capsys):
    def boom(a): raise QuotaExhausted("x", "gemma-x", 13.7)
    monkeypatch.setattr(cli, "_weekend3", boom)
    with pytest.raises(SystemExit) as e:
        cli.main(["meta"])
    assert e.value.code == 2
    assert capsys.readouterr().err.strip() == ("Daily free-tier quota exhausted for gemma-x (resets in ~14 h). "
                                              "Completed calls are cached; re-run later to resume.")


# A6
def test_cache_stats_line(capsys):
    cli._print_cache_stats(MockBackend())
    assert capsys.readouterr().out == ""
    c = CachedBackend(Inner(), MemoryCache()); c.hits, c.misses = 3, 2
    cli._print_cache_stats(c)
    assert capsys.readouterr().out.strip() == "agent cache: 3 hits / 2 misses"


# review round 1
@pytest.mark.parametrize("h,expect", [(90 / 3600, "~2 min"), (1.5 / 3600, "~1 min"), (-5.0, "~0 min"),
                                      (1e9, ">7 days"), (None, "unknown"), (13.7, "~14 h"), (1500 / 3600, "~25 min")])
def test_reset_phrase(h, expect):
    from voice_agent_audit.llm import reset_phrase
    assert expect in reset_phrase(h)


def test_non_numeric_retry_after_falls_back(monkeypatch):
    b, calls = _gemini(monkeypatch, [Resp(503, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}), Resp(200, OK)])
    assert b.complete("s", MSG, "m").text == "{}" and len(calls) == 2


def test_perday_in_message_but_perminute_quota_is_retried(monkeypatch):
    body = {"error": {"message": "PerDay mentioned here", "details": [
        {"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}]}}
    b, calls = _gemini(monkeypatch, [Resp(429, body), Resp(200, OK)])
    assert b.complete("s", MSG, "m").text == "{}" and len(calls) == 2
