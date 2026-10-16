"""Provider-agnostic LLM client.

Backends: mock (offline, deterministic, for tests/CI plumbing ONLY - its outputs are not
measurements), ollama (local), gemini (free tier, needs GEMINI_API_KEY).
Choose with VAA_BACKEND and VAA_MODEL.
"""
from __future__ import annotations
import json
import math
import os
import random
import re
import sqlite3
import sys
import time
from dataclasses import dataclass

import requests


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    token_source: str  # "provider" or "estimate"
    cached: bool = False


def estimate_tokens(text: str) -> int:
    """Rough estimate (chars/4). Used only when the provider doesn't report usage."""
    return max(1, round(len(text) / 4))


class Backend:
    name = "base"

    def complete(self, system: str, messages: list[dict], model: str) -> LLMResult:
        raise NotImplementedError


class MockBackend(Backend):
    """Keyword rules over the last borrower message. Ignores the system prompt on purpose;
    it exists so the pipeline can be tested without a model."""
    name = "mock"
    RULES = [
        (r"wrong number|galat number|nahi rehta|wrong person|galat insaan|koi .* nahi", "end_call", {"reason": "wrong_person"}),
        (r"harass|chup|bakwaas|pareshan|never call|phone mat|shut up", "end_call", {"reason": "abusive"}),
        (r"job|naukri|hospital|medical|ilaaj|kharcha", "escalate_to_human", {"reason": "hardship"}),
        (r"galat hai|never took|liya hi nahi|already|pehle hi|mistaken", "escalate_to_human", {"reason": "dispute"}),
        (r"call back|call me|phone kijiye|busy|baat karte|samay nahi", "schedule_callback", {"date": "2026-10-16"}),
        (r"pay|payment|bhugtan|paise|upi", "record_promise_to_pay", {"date": "2026-10-16"}),
    ]

    def complete(self, system, messages, model):
        t0 = time.time()
        last = messages[-1]["content"].lower()
        m = re.search(r'"amount_due":\s*(\d+)', system)
        amount = int(m.group(1)) if m else 0
        tool, reply = None, "Theek hai, main samajh gaya."
        for pat, name, args in self.RULES:
            if re.search(pat, last):
                a = dict(args)
                if name == "record_promise_to_pay":
                    a["amount"] = amount
                tool = {"name": name, "args": a}
                break
        text = json.dumps({"reply": reply, "tool": tool})
        return LLMResult(text, estimate_tokens(system) + estimate_tokens(last),
                         estimate_tokens(text), time.time() - t0, "estimate")


class OllamaBackend(Backend):
    name = "ollama"

    def complete(self, system, messages, model):
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
        t0 = time.time()
        r = requests.post(url, json={
            "model": model, "stream": False, "format": "json",
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}] + messages,
        }, timeout=300)
        r.raise_for_status()
        j = r.json()
        return LLMResult(j["message"]["content"], j.get("prompt_eval_count", 0),
                         j.get("eval_count", 0), time.time() - t0, "provider")


RETRYABLE = {429, 500, 502, 503, 504}
NETWORK_ERRORS = (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout,
                  requests.exceptions.ConnectionError)


class QuotaExhausted(RuntimeError):
    """Daily free-tier quota is used up; retrying now is pointless. Re-run after the reset."""

    def __init__(self, message: str, model: str = "", reset_hours: float | None = None):
        super().__init__(message)
        self.model, self.reset_hours = model, reset_hours


def reset_phrase(hours: float | None) -> str:
    """Human wording for the quota reset delay."""
    if hours is None:
        return "reset time unknown"
    hours = max(hours, 0.0)
    if hours > 24 * 7:
        return "resets in >7 days"
    if hours < 1:
        return f"resets in ~{math.ceil(hours * 60)} min"
    return f"resets in ~{round(hours)} h"


def _is_daily(body) -> bool:
    """Decide from the violations' quotaId/quotaMetric when present, else from the whole body."""
    details = (body.get("error") or {}).get("details") if isinstance(body, dict) else None
    violations = []
    for d in details if isinstance(details, list) else []:
        v = d.get("violations") if isinstance(d, dict) else None
        violations += [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []
    if violations:
        return any("PerDay" in str(v.get("quotaId", "")) + str(v.get("quotaMetric", "")) for v in violations)
    return "PerDay" in json.dumps(body)


def _daily_quota_error(r, model: str) -> QuotaExhausted | None:
    """Return QuotaExhausted if a 429 says the PER-DAY quota is gone (per-minute 429s return None)."""
    try:
        body = r.json()
    except Exception:  # noqa: BLE001  - non-JSON body: treat as a normal 429
        return None
    if not _is_daily(body):
        return None
    hours = None
    details = (body.get("error") or {}).get("details") if isinstance(body, dict) else None
    for d in details if isinstance(details, list) else []:
        delay = d.get("retryDelay") if isinstance(d, dict) else None
        m = re.fullmatch(r"(-?\d+(?:\.\d+)?)s", str(delay or ""))
        if m:
            hours = float(m.group(1)) / 3600
            break
    return QuotaExhausted(f"Daily free-tier quota exhausted for {model} ({reset_phrase(hours)}).", model, hours)


def _retry_after(r, attempt: int) -> float:
    """Retry-After seconds if numeric, else exponential backoff (the header may be an HTTP date)."""
    try:
        return float(r.headers.get("Retry-After"))
    except (TypeError, ValueError):
        return min(2 ** attempt * 2, 40)


class GeminiBackend(Backend):
    name = "gemini"
    max_retries = 6
    sleep = staticmethod(time.sleep)

    def complete(self, system, messages, model):
        key = os.environ["GEMINI_API_KEY"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        contents = [{"role": "user" if m["role"] == "user" else "model",
                     "parts": [{"text": m["content"]}]} for m in messages]
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        t0 = time.time()
        r = None
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                r = requests.post(url, headers={"x-goog-api-key": key}, json=body, timeout=120)
            except NETWORK_ERRORS:
                if last:
                    raise
                self.sleep(min(2 ** attempt * 2, 40) + random.random())
                continue
            if r.status_code == 429:
                q = _daily_quota_error(r, model)
                if q:
                    raise q
            if r.status_code not in RETRYABLE or last:
                break
            wait = _retry_after(r, attempt) + random.random()
            self.sleep(wait)                      # latency below includes the waits (wall-clock)
        r.raise_for_status()
        j = r.json()
        u = j.get("usageMetadata", {})
        cands = j.get("candidates") or [{}]
        parts = (cands[0].get("content") or {}).get("parts") or []
        texts = [p["text"] for p in parts if not p.get("thought") and "text" in p]
        if not texts:
            why = (j.get("promptFeedback") or {}).get("blockReason") or cands[0].get("finishReason")
            raise ValueError("no text in model response" + (f" (reason: {why})" if why else ""))
        return LLMResult(texts[-1], u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0),
                         time.time() - t0, "provider")


class CachedBackend(Backend):
    """Wraps a backend with an on-disk cache so an interrupted run resumes where it stopped.
    Errors are never cached. A hit returns the original text, tokens and latency."""

    def __init__(self, inner: Backend, cache):
        self.inner, self.cache = inner, cache
        self.name = inner.name
        self.hits = self.misses = 0

    def _warn(self, what: str, e: Exception) -> None:
        if not getattr(self, "_warned", False):
            self._warned = True
            print(f"warning: agent cache {what} failed ({e.__class__.__name__}); continuing without it", file=sys.stderr)

    def complete(self, system, messages, model):
        from .service.cache import make_key
        key = make_key(self.inner.name, model, system, json.dumps(messages, sort_keys=True, ensure_ascii=False),
                       prefix="vaa:agent:")
        try:
            v = self.cache.get(key)
        except Exception as e:  # noqa: BLE001  - a broken cache is a miss, never a crash
            self._warn("read", e)
            v = None
        if v is not None:
            try:
                r = LLMResult(v["text"], v["input_tokens"], v["output_tokens"], v["latency_s"],
                              v["token_source"], cached=True)
                self.hits += 1
                return r
            except (KeyError, TypeError):
                pass                              # stale/partial entry: treat as a miss and overwrite
        r = self.inner.complete(system, messages, model)
        self.misses += 1
        try:
            self.cache.set(key, {"text": r.text, "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                                 "latency_s": r.latency_s, "token_source": r.token_source})
        except Exception as e:  # noqa: BLE001
            self._warn("write", e)
        return r


def get_backend(name: str | None = None, cached: bool = True) -> Backend:
    name = name or os.environ.get("VAA_BACKEND", "mock")
    b = {"mock": MockBackend, "ollama": OllamaBackend, "gemini": GeminiBackend}[name]()
    if cached and name != "mock" and os.environ.get("VAA_NO_CACHE") != "1":
        from .service.cache import DiskCache
        path = os.environ.get("VAA_AGENT_CACHE", "results/agent_cache.db")
        try:
            b = CachedBackend(b, DiskCache(path))
        except (sqlite3.DatabaseError, OSError) as e:
            print(f"warning: agent cache disabled ({e.__class__.__name__}: {path})", file=sys.stderr)
    return b
