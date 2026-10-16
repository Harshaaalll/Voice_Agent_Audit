"""Adversarial tests for Gemini retry / quota / cache-resume. Offline only. Tests marked xfail document defects."""
import json
import threading

import pytest
import requests

from voice_agent_audit import cli
from voice_agent_audit.llm import (CachedBackend, GeminiBackend, LLMResult, MockBackend, QuotaExhausted,
                                   get_backend)
from voice_agent_audit.service.cache import DiskCache, make_key

MSG = [{"role": "user", "content": "u"}]


class Resp:
    def __init__(self, code, body=None, headers=None, text_body=None):
        self.status_code, self._b, self.headers, self._t = code, body if body is not None else {}, headers or {}, text_body
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)
    def json(self):
        if self._t is not None:
            raise ValueError("not json")
        return self._b


def _gem(monkeypatch, seq):
    monkeypatch.setenv("GEMINI_API_KEY", "SECRETKEY")
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


def ok(parts, usage=True):
    j = {"candidates": [{"content": {"parts": parts}}]}
    if usage:
        j["usageMetadata"] = {"promptTokenCount": 1, "candidatesTokenCount": 2}
    return Resp(200, j)


def daily(delay="49372s"):
    return Resp(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
        {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
        {"@type": "RetryInfo", "retryDelay": delay}]}})


def minute(extra=""):
    return Resp(429, {"error": {"code": 429, "message": extra, "details": [
        {"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
        {"retryDelay": "23s"}]}})


@pytest.mark.parametrize("r", [Resp(429, text_body="<html>x</html>"), Resp(429, {}), Resp(429, [1, 2]), Resp(429, "str")])
def test_weird_429_bodies_are_retried_not_crash(monkeypatch, r):
    b, calls = _gem(monkeypatch, [r, ok([{"text": "a"}])])
    assert b.complete("s", MSG, "m").text == "a" and len(calls) == 2


def test_perminute_429_with_day_word_is_retried(monkeypatch):
    b, calls = _gem(monkeypatch, [minute("retry in 1 day"), ok([{"text": "a"}])])
    assert b.complete("s", MSG, "m").text == "a"


@pytest.mark.parametrize("delay,expect_h", [("1.5s", 1.5 / 3600), ("3600s", 1.0), (None, None), ("abc", None), ("2h", None)])
def test_retrydelay_formats(monkeypatch, delay, expect_h):
    b, _ = _gem(monkeypatch, [daily(delay)])
    with pytest.raises(QuotaExhausted) as e:
        b.complete("s", MSG, "gm")
    assert e.value.model == "gm"
    assert e.value.reset_hours == (pytest.approx(expect_h) if expect_h is not None else None)
    assert "gm" in str(e.value)


def test_503_then_daily_429_stops(monkeypatch):
    b, calls = _gem(monkeypatch, [Resp(503), daily()])
    with pytest.raises(QuotaExhausted):
        b.complete("s", MSG, "m")
    assert len(calls) == 2


def test_message_has_hours_and_daily(monkeypatch):
    b, _ = _gem(monkeypatch, [daily()])
    with pytest.raises(QuotaExhausted) as e:
        b.complete("s", MSG, "m")
    s = str(e.value).lower()
    assert "daily" in s and "h" in s and "14" in s  # 49372s ~ 13.7h -> "~14 h"


def test_secret_not_in_quota_or_http_errors(monkeypatch):
    b, _ = _gem(monkeypatch, [Resp(400)])
    with pytest.raises(requests.HTTPError) as e:
        b.complete("s", MSG, "m")
    assert "SECRETKEY" not in str(e.value)


# A3 parsing
def test_thought_with_text_ignored_last_text_used(monkeypatch):
    b, _ = _gem(monkeypatch, [ok([{"text": "t", "thought": True}, {"text": "a"}, {"foo": 1}, {"text": "b"}, {"text": "c", "thought": True}])])
    assert b.complete("s", MSG, "m").text == "b"


def test_no_usage_metadata(monkeypatch):
    b, _ = _gem(monkeypatch, [ok([{"text": "a"}], usage=False)])
    r = b.complete("s", MSG, "m")
    assert (r.input_tokens, r.output_tokens, r.token_source) == (0, 0, "provider")


@pytest.mark.parametrize("parts", [[], [{"thought": True, "text": "x"}], [{"inlineData": {}}]])
def test_no_text_raises_valueerror(monkeypatch, parts):
    b, _ = _gem(monkeypatch, [ok(parts)])
    with pytest.raises(ValueError, match="no text in model response"):
        b.complete("s", MSG, "m")


@pytest.mark.parametrize("j", [{"promptFeedback": {"blockReason": "SAFETY"}},
                               {"candidates": []},
                               {"candidates": [{"finishReason": "SAFETY"}]},
                               {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"role": "model"}}]}])
def test_degenerate_200_gives_clean_valueerror(monkeypatch, j):
    # DEFECT: these raise KeyError/IndexError instead of the ValueError("no text in model response")
    b, _ = _gem(monkeypatch, [Resp(200, j)])
    with pytest.raises(ValueError, match="no text"):
        b.complete("s", MSG, "m")


# A4 cache
class Counting(MockBackend):
    name = "cnt"
    def __init__(self):
        self.n = 0
    def complete(self, system, messages, model):
        self.n += 1
        return LLMResult(f"r{self.n}", 5, 6, 1.25, "provider")


def test_key_boundaries_and_roles(tmp_path):
    inner = Counting()
    c = CachedBackend(inner, DiskCache(str(tmp_path / "c.db")))
    u = lambda t, role="user": [{"role": role, "content": t}]
    c.complete("ab", u("c"), "m"); c.complete("a", u("bc"), "m")
    c.complete("a", u("bc", "assistant"), "m"); c.complete("a", u("bc"), "m2")
    c.complete("é" * 5000, u("日本語"), "m")
    assert inner.n == 5
    r = c.complete("é" * 5000, u("日本語"), "m")
    assert r.cached and inner.n == 5 and r.latency_s == 1.25 and r.token_source == "provider" and r.text == "r5"
    assert (c.hits, c.misses) == (1, 5)


def test_exception_not_cached(tmp_path):
    class Boom(Counting):
        def complete(self, *a):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("x")
            return super().complete(*a)
    inner = Boom(); c = CachedBackend(inner, DiskCache(str(tmp_path / "c.db")))
    with pytest.raises(RuntimeError):
        c.complete("s", MSG, "m")
    assert not c.complete("s", MSG, "m").cached
    assert (c.hits, c.misses) == (0, 1)


def test_cache_persists_across_instances(tmp_path):
    p = str(tmp_path / "n" / "e" / "c.db")
    CachedBackend(Counting(), DiskCache(p)).complete("s", MSG, "m")
    i2 = Counting(); r = CachedBackend(i2, DiskCache(p)).complete("s", MSG, "m")
    assert r.cached and i2.n == 0


def test_get_backend_wrapping(monkeypatch, tmp_path):
    monkeypatch.setenv("VAA_AGENT_CACHE", str(tmp_path / "a" / "b" / "x.db"))
    monkeypatch.delenv("VAA_NO_CACHE", raising=False)
    assert isinstance(get_backend("ollama"), CachedBackend)
    assert not isinstance(get_backend("mock"), CachedBackend)
    assert not isinstance(get_backend("ollama", cached=False), CachedBackend)
    monkeypatch.setenv("VAA_NO_CACHE", "1")
    assert not isinstance(get_backend("ollama"), CachedBackend)
    monkeypatch.setenv("VAA_NO_CACHE", "0")
    assert isinstance(get_backend("ollama"), CachedBackend)


def test_judge_bypasses_agent_cache(monkeypatch, tmp_path):
    from voice_agent_audit.judge import Judge
    monkeypatch.setenv("VAA_AGENT_CACHE", str(tmp_path / "a.db"))
    monkeypatch.delenv("VAA_NO_CACHE", raising=False)
    j = Judge(backend_name="ollama", model="jm", agent_backend="mock", agent_model="mock", cache=DiskCache(str(tmp_path / "j.db")))
    assert not isinstance(j.backend, CachedBackend)


def test_thread_safety_shared_diskcache(tmp_path):
    cache = DiskCache(str(tmp_path / "t.db"))
    errs = []
    def w(i):
        try:
            for k in range(50):
                cache.set(f"k{i}-{k}", {"v": k}); cache.get(f"k{i}-{k}")
        except Exception as e:  # noqa
            errs.append(e)
    ts = [threading.Thread(target=w, args=(i,)) for i in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not errs


def test_corrupt_cache_file_degrades(monkeypatch, tmp_path):
    # DEFECT (robustness): get_cache() degrades on bad path, get_backend() does not.
    p = tmp_path / "bad.db"; p.write_bytes(b"not a sqlite file" * 100)
    monkeypatch.setenv("VAA_AGENT_CACHE", str(p)); monkeypatch.delenv("VAA_NO_CACHE", raising=False)
    b = get_backend("ollama")
    assert b is not None


def test_unwritable_cache_dir_degrades(monkeypatch, tmp_path):
    f = tmp_path / "afile"; f.write_text("x")
    monkeypatch.setenv("VAA_AGENT_CACHE", str(f / "sub" / "c.db")); monkeypatch.delenv("VAA_NO_CACHE", raising=False)
    assert get_backend("ollama") is not None


def test_cache_write_failure_does_not_lose_result(tmp_path):
    class Bad(DiskCache):
        def set(self, k, v):
            raise OSError("disk full")
    c = CachedBackend(Counting(), Bad(str(tmp_path / "c.db")))
    assert c.complete("s", MSG, "m").text == "r1"


def test_stale_entry_missing_fields_is_a_miss(tmp_path):
    cache = DiskCache(str(tmp_path / "c.db"))
    inner = Counting(); c = CachedBackend(inner, cache)
    c.complete("s", MSG, "m")
    k = make_key("cnt", "m", "s", json.dumps(MSG, sort_keys=True, ensure_ascii=False), prefix="vaa:agent:")
    cache.set(k, {"text": "old"})
    assert c.complete("s", MSG, "m").text  # should re-call instead of KeyError


def test_make_key_judge_default_prefix():
    assert make_key("a").startswith("vaa:judge:") and make_key("a", prefix="vaa:agent:").startswith("vaa:agent:")


# A5/A6 CLI
def test_cli_quota_exit_2(monkeypatch, capsys):
    def boom(argv=None):
        raise QuotaExhausted("q", "mm", 12.4)
    monkeypatch.setattr(cli, "_main", boom)
    with pytest.raises(SystemExit) as e:
        cli.main([])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "mm" in err and "12" in err


@pytest.mark.parametrize("h", [None, 0.0, 0.2, -5.0, 1e9])
def test_cli_quota_odd_hours_print(monkeypatch, capsys, h):
    monkeypatch.setattr(cli, "_main", lambda argv=None: (_ for _ in ()).throw(QuotaExhausted("q", "mm", h)))
    with pytest.raises(SystemExit) as e:
        cli.main([])
    assert e.value.code == 2 and "mm" in capsys.readouterr().err


@pytest.mark.parametrize("exc", [KeyboardInterrupt, ValueError, requests.ReadTimeout])
def test_cli_other_exceptions_propagate(monkeypatch, exc):
    monkeypatch.setattr(cli, "_main", lambda argv=None: (_ for _ in ()).throw(exc()))
    with pytest.raises(exc):
        cli.main([])


def test_cli_argparse_error_code_2_and_help(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["nonsense"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0


def test_cli_mock_run_end_to_end_no_cache_line(tmp_path, capsys):
    cli.main(["gen", "--n", "6", "--out", str(tmp_path / "g.jsonl")])
    cli.main(["run", "--data", str(tmp_path / "g.jsonl"), "--out", str(tmp_path / "r.json")])
    assert "agent cache" not in capsys.readouterr().out


def test_cli_run_with_cached_backend_prints_stats(monkeypatch, tmp_path, capsys):
    cli.main(["gen", "--n", "6", "--out", str(tmp_path / "g.jsonl")])
    monkeypatch.setattr(cli, "get_backend", lambda *a, **k: CachedBackend(MockBackend(), DiskCache(str(tmp_path / "c.db"))), raising=False)
    import voice_agent_audit.llm as llm
    monkeypatch.setattr(llm, "get_backend", lambda *a, **k: CachedBackend(MockBackend(), DiskCache(str(tmp_path / "c.db"))))
    monkeypatch.setenv("VAA_BACKEND", "mock")
    cli.main(["run", "--data", str(tmp_path / "g.jsonl"), "--out", str(tmp_path / "r.json")])
    assert "agent cache:" in capsys.readouterr().out
