"""Judge-call cache. Redis when REDIS_URL is reachable, otherwise an in-process dict.
Keys are a hash of (backend, model, system prompt, user text), so any change busts the entry."""
from __future__ import annotations
import hashlib
import json
import os
import threading


def make_key(*parts: str, prefix: str = "vaa:judge:") -> str:
    return prefix + hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


class MemoryCache:
    kind = "memory"

    def __init__(self):
        self._d: dict[str, str] = {}
        self.hits = self.misses = 0

    def get(self, key: str):
        v = self._d.get(key)
        self.hits += v is not None
        self.misses += v is None
        return None if v is None else json.loads(v)

    def set(self, key: str, value) -> None:
        self._d[key] = json.dumps(value)

    def stats(self) -> dict:
        n = self.hits + self.misses
        return {"kind": self.kind, "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / n, 4) if n else None}


class RedisCache(MemoryCache):
    kind = "redis"

    def __init__(self, client, ttl_s: int = 7 * 24 * 3600):
        super().__init__()
        self.r, self.ttl = client, ttl_s

    def get(self, key: str):
        v = self.r.get(key)
        self.hits += v is not None
        self.misses += v is None
        return None if v is None else json.loads(v)

    def set(self, key: str, value) -> None:
        self.r.set(key, json.dumps(value), ex=self.ttl)


class DiskCache(MemoryCache):
    """SQLite-backed, survives restarts. Used when Redis is not available so long judge runs can resume."""
    kind = "disk"

    def __init__(self, path: str):
        import sqlite3
        from pathlib import Path
        super().__init__()
        self._lock = threading.Lock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._c = sqlite3.connect(path, check_same_thread=False)
        with self._lock:
            self._c.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")

    def get(self, key: str):
        with self._lock:
            row = self._c.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        self.hits += row is not None
        self.misses += row is None
        return None if row is None else json.loads(row[0])

    def set(self, key: str, value) -> None:
        with self._lock:
            self._c.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, json.dumps(value)))
            self._c.commit()


def get_cache():
    url = os.environ.get("REDIS_URL")
    if url:
        try:
            import redis
            client = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
            client.ping()
            return RedisCache(client)
        except Exception:  # noqa: BLE001  - fall back, caching must never break an eval
            pass
    path = os.environ.get("VAA_JUDGE_CACHE", "results/judge_cache.db")
    try:
        return DiskCache(path) if path != "memory" else MemoryCache()
    except Exception:  # noqa: BLE001  - read-only folder etc.; caching is optional
        return MemoryCache()
