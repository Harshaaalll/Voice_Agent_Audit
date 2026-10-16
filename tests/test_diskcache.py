from voice_agent_audit.service.cache import DiskCache, get_cache, make_key


def test_disk_cache_persists_across_instances(tmp_path):
    p = str(tmp_path / "c.db")
    a = DiskCache(p)
    assert a.get("k") is None
    a.set("k", {"supported": 1})
    b = DiskCache(p)                       # a new process would see the same file
    assert b.get("k") == {"supported": 1}
    assert b.stats()["hits"] == 1 and a.stats()["misses"] == 1


def test_get_cache_defaults_to_disk_and_memory_override(tmp_path, monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("VAA_JUDGE_CACHE", str(tmp_path / "x.db"))
    assert get_cache().kind == "disk"
    monkeypatch.setenv("VAA_JUDGE_CACHE", "memory")
    assert get_cache().kind == "memory"
    assert make_key("a") != make_key("b")
