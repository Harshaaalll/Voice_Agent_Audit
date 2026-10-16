"""Tiny job queue on a Redis list (LPUSH / BRPOP). Job state lives in the database, not in Redis."""
from __future__ import annotations
import os
from collections import deque

KEY = "vaa:jobs"


class RedisQueue:
    def __init__(self, client):
        self.r = client

    def enqueue(self, job_id: str) -> None:
        self.r.lpush(KEY, job_id)

    def pop(self, timeout: int = 5) -> str | None:
        item = self.r.brpop(KEY, timeout=timeout)
        return item[1] if item else None


class MemoryQueue:
    """For tests only."""
    def __init__(self):
        self.q: deque[str] = deque()

    def enqueue(self, job_id: str) -> None:
        self.q.appendleft(job_id)

    def pop(self, timeout: int = 0) -> str | None:
        return self.q.pop() if self.q else None


def get_queue():
    url = os.environ.get("REDIS_URL")
    if not url:
        return None
    import redis
    # socket_timeout must exceed the BRPOP wait or an idle queue raises TimeoutError
    client = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=2, socket_timeout=30)
    client.ping()
    return RedisQueue(client)
