"""Tracing. Noop by default. LangfuseTracer posts to Langfuse's REST ingestion endpoint.

NOT verified against a live Langfuse project in this repo's tests (no account was available);
the unit test checks the request we send. Run `vaa trace-check` once with real keys to verify.
"""
from __future__ import annotations
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class NoopTracer:
    enabled = False

    def record(self, **_):
        return None


class LangfuseTracer:
    enabled = True

    def __init__(self, host: str, public_key: str, secret_key: str, post=None, background: bool = True):
        self.url = host.rstrip("/") + "/api/public/ingestion"
        self.auth = (public_key, secret_key)
        self._post = post or self._http_post
        self.background = background
        self.failures = 0
        self.last_status: int | None = None

    def _http_post(self, url, json, auth):
        import httpx
        return httpx.post(url, json=json, auth=auth, timeout=5.0).status_code

    def build_batch(self, *, name, model, input, output, usage, latency_s, metadata=None, trace_id=None):
        end = datetime.now(timezone.utc)
        start = end - timedelta(seconds=latency_s)
        tid = trace_id or uuid.uuid4().hex
        return {"batch": [
            {"id": uuid.uuid4().hex, "type": "trace-create", "timestamp": _iso(start),
             "body": {"id": tid, "name": name, "input": input, "output": output, "metadata": metadata or {}}},
            {"id": uuid.uuid4().hex, "type": "generation-create", "timestamp": _iso(end),
             "body": {"id": uuid.uuid4().hex, "traceId": tid, "name": name, "model": model,
                      "startTime": _iso(start), "endTime": _iso(end), "input": input, "output": output,
                      "usage": {"input": usage["input_tokens"], "output": usage["output_tokens"], "unit": "TOKENS"},
                      "metadata": metadata or {}}}]}

    def send(self, payload: dict) -> int | None:
        try:
            self.last_status = self._post(self.url, payload, self.auth)
            if not (200 <= self.last_status < 300):
                self.failures += 1
        except Exception:  # noqa: BLE001  - tracing must never break a request
            self.failures += 1
            self.last_status = None
        return self.last_status

    def record(self, **kw):
        payload = self.build_batch(**kw)
        if self.background:
            threading.Thread(target=self.send, args=(payload,), daemon=True).start()
        else:
            self.send(payload)


def get_tracer():
    pk, sk = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
    if pk and sk:
        host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
        return LangfuseTracer(host, pk, sk)
    return NoopTracer()
