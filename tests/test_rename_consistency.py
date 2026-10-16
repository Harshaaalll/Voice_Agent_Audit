"""Durable checks that the hveval -> Voice Agent Audit rename stays consistent. No network."""
import importlib
import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", "node_modules", ".git", "results", "data", "__pycache__", ".pytest_cache"}
VAA = re.compile(r"\bVAA_[A-Z0-9_]+\b")
OLD = "hveval"


def _files(globs):
    out = []
    for g in globs:
        out += [p for p in ROOT.glob(g) if p.is_file()]
    return out


def _vars(paths):
    found = set()
    for p in paths:
        found |= set(VAA.findall(p.read_text(errors="ignore")))
    return found


CODE = _files(["src/**/*.py", "promptfoo/*.py"])
DOCS = _files(["README.md", "deploy/*.md", "Dockerfile", "docker-compose.yml", ".github/workflows/eval.yml"])


def test_documented_env_vars_are_read_by_code():
    unread = _vars(DOCS) - _vars(CODE)
    assert not unread, f"documented/set but never read: {sorted(unread)}"


def test_code_env_vars_are_documented():
    undocumented = _vars(CODE) - _vars(DOCS)
    assert not undocumented, f"read but undocumented: {sorted(undocumented)}"


def test_no_env_var_typos_or_old_prefix():
    # every VAA token in code must be a full VAA_ name; no HVEVAL_ left anywhere
    for p in CODE + DOCS:
        t = p.read_text(errors="ignore")
        assert "HVEVAL_" not in t.upper(), p
        assert not re.search(r"\bVAA(?!_)\b", t), p


def test_old_name_only_in_formerly_line():
    hits = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or set(p.relative_to(ROOT).parts) & SKIP_DIRS:
            continue
        if p.suffix in {".pyc", ".mp3", ".png"} or p.name == Path(__file__).name:
            continue
        if p.parts and any(part.endswith(".egg-info") for part in p.parts):
            continue
        if OLD in p.name.lower():
            hits.append(f"name:{p}")
        try:
            lines = p.read_text().splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(lines, 1):
            if OLD in line.lower():
                hits.append(f"{p.relative_to(ROOT)}:{i}")
    assert hits == ["README.md:3"], hits
    assert (ROOT / "README.md").read_text().count("Formerly named `hveval`") == 1


def test_entry_points_resolve():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert cfg["project"]["name"] == "voice-agent-audit"
    scripts = cfg["project"]["scripts"]
    assert set(scripts) == {"vaa", "voice-agent-audit"}
    for target in scripts.values():
        mod, fn = target.split(":")
        assert callable(getattr(importlib.import_module(mod), fn))
    assert (ROOT / "src" / "voice_agent_audit" / "__init__.py").exists()
    assert not (ROOT / "src" / OLD).exists()


def test_redis_key_prefixes():
    from voice_agent_audit.service.cache import make_key
    from voice_agent_audit.service import queue
    assert make_key("a", "b").startswith("vaa:judge:")
    assert queue.KEY == "vaa:jobs"


def test_default_sqlite_and_titles():
    src = "\n".join(p.read_text() for p in CODE)
    assert "sqlite:///results/vaa.db" in src
    assert 'FastAPI(title="Voice Agent Audit"' in src
    front = (ROOT / "deploy" / "hf-space-README.md").read_text().split("---")[1]
    assert "title: Voice Agent Audit" in front and "sdk: docker" in front


def test_compose_and_dockerfile_consistent():
    import yaml
    c = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
    db, api, worker = c["db"], c["api"], c["worker"]
    assert db["environment"]["POSTGRES_USER"] == "vaa" and db["environment"]["POSTGRES_DB"] == "vaa"
    assert "vaa_local_dev" in db["environment"]["POSTGRES_PASSWORD"]
    assert "-U vaa -d vaa" in db["healthcheck"]["test"][1]
    url = api["environment"]["DATABASE_URL"]
    assert url.startswith("postgresql://vaa:") and url.endswith("@db:5432/vaa") and "vaa_local_dev" in url
    assert worker["command"] == ["vaa", "worker"]
    docker = (ROOT / "Dockerfile").read_text()
    assert "VAA_PROMPTS_DIR=/app/prompts" in docker and "voice_agent_audit.service.app:app" in docker
    for line in docker.splitlines():
        if line.startswith("COPY ") and "--from" not in line:
            for src in line.split()[1:-1]:
                assert (ROOT / src).exists(), src


def test_workflow_commands_exist():
    import yaml
    from voice_agent_audit import cli  # noqa: F401
    wf = yaml.safe_load((ROOT / ".github/workflows/eval.yml").read_text())
    runs = " ".join(s.get("run", "") for s in wf["jobs"]["eval"]["steps"])
    assert "vaa gen" in runs and "vaa run" in runs and ".[dev,service]" in runs
    assert OLD not in runs


def test_promptfoo_paths_import():
    cfg = (ROOT / "promptfooconfig.yaml").read_text()
    for rel in set(re.findall(r"file://([\w/\.]+\.py)", cfg)):
        assert (ROOT / rel).exists(), rel
    spec = importlib.util.spec_from_file_location("pf_provider", ROOT / "promptfoo" / "provider.py")
    importlib.util.module_from_spec(spec)
