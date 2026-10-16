import pytest

from voice_agent_audit.service.tracing import LangfuseTracer, NoopTracer, get_tracer


def _env(monkeypatch, **kw):
    for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in kw.items():
        monkeypatch.setenv(k, v)


def test_noop_without_keys(monkeypatch):
    _env(monkeypatch)
    assert isinstance(get_tracer(), NoopTracer)
    _env(monkeypatch, LANGFUSE_PUBLIC_KEY="pk")
    assert isinstance(get_tracer(), NoopTracer)          # secret missing


@pytest.mark.parametrize("env,expected", [
    ({}, "https://cloud.langfuse.com/api/public/ingestion"),
    ({"LANGFUSE_BASE_URL": "https://us.cloud.langfuse.com"}, "https://us.cloud.langfuse.com/api/public/ingestion"),
    ({"LANGFUSE_HOST": "https://h.example"}, "https://h.example/api/public/ingestion"),
    ({"LANGFUSE_HOST": "https://h.example", "LANGFUSE_BASE_URL": "https://b.example"}, "https://h.example/api/public/ingestion"),
])
def test_host_from_either_variable_host_wins(monkeypatch, env, expected):
    _env(monkeypatch, LANGFUSE_PUBLIC_KEY="pk", LANGFUSE_SECRET_KEY="sk", **env)
    t = get_tracer()
    assert isinstance(t, LangfuseTracer) and t.url == expected
