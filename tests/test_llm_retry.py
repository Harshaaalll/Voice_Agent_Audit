import pytest
import requests

from voice_agent_audit.llm import GeminiBackend


class Resp:
    def __init__(self, code, body=None, headers=None):
        self.status_code, self._b, self.headers = code, body or {}, headers or {}
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)
    def json(self): return self._b


OK = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}], "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2}}


def _backend(monkeypatch, codes):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    calls, waits = [], []
    seq = iter(codes)
    def post(url, headers, json, timeout):
        calls.append(1)
        c = next(seq)
        return Resp(c, OK if c == 200 else {}, {"Retry-After": "1"} if c == 429 else {})
    monkeypatch.setattr(requests, "post", post)
    b = GeminiBackend()
    b.sleep = waits.append
    return b, calls, waits


def test_retries_transient_errors_then_succeeds(monkeypatch):
    b, calls, waits = _backend(monkeypatch, [503, 429, 200])
    r = b.complete("s", [{"role": "user", "content": "u"}], "m")
    assert r.input_tokens == 3 and len(calls) == 3 and len(waits) == 2
    assert waits[1] >= 1.0                    # honoured Retry-After


def test_gives_up_after_max_retries(monkeypatch):
    b, calls, _ = _backend(monkeypatch, [503] * 10)
    b.max_retries = 2
    with pytest.raises(requests.HTTPError):
        b.complete("s", [{"role": "user", "content": "u"}], "m")
    assert len(calls) == 3


def test_does_not_retry_client_errors(monkeypatch):
    b, calls, waits = _backend(monkeypatch, [404, 200])
    with pytest.raises(requests.HTTPError):
        b.complete("s", [{"role": "user", "content": "u"}], "m")
    assert len(calls) == 1 and waits == []
