"""Provider-agnostic LLM client.

Backends: mock (offline, deterministic, for tests/CI plumbing ONLY - its outputs are not
measurements), ollama (local), gemini (free tier, needs GEMINI_API_KEY).
Choose with HVEVAL_BACKEND and HVEVAL_MODEL.
"""
from __future__ import annotations
import json
import os
import re
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


class GeminiBackend(Backend):
    name = "gemini"

    def complete(self, system, messages, model):
        key = os.environ["GEMINI_API_KEY"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        contents = [{"role": "user" if m["role"] == "user" else "model",
                     "parts": [{"text": m["content"]}]} for m in messages]
        t0 = time.time()
        r = requests.post(url, headers={"x-goog-api-key": key}, json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }, timeout=120)
        r.raise_for_status()
        j = r.json()
        u = j.get("usageMetadata", {})
        text = j["candidates"][0]["content"]["parts"][0]["text"]
        return LLMResult(text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0),
                         time.time() - t0, "provider")


def get_backend(name: str | None = None) -> Backend:
    name = name or os.environ.get("HVEVAL_BACKEND", "mock")
    return {"mock": MockBackend, "ollama": OllamaBackend, "gemini": GeminiBackend}[name]()
