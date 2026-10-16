"""Persistence for the service. SQLite for tests/local, PostgreSQL for docker compose.

Both use the same SQL (`?` placeholders, JSON as TEXT, python-generated ids) so one code path
is exercised everywhere. pgvector tables arrive in Phase 2.
"""
from __future__ import annotations
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY, created_at TEXT NOT NULL, variant TEXT NOT NULL, backend TEXT NOT NULL,
        model TEXT NOT NULL, noisy INTEGER NOT NULL, n INTEGER NOT NULL, summary TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS run_items (
        run_id TEXT NOT NULL, item_id TEXT NOT NULL, intent TEXT, style TEXT, checks TEXT NOT NULL,
        input_tokens INTEGER, output_tokens INTEGER, latency_s REAL, PRIMARY KEY (run_id, item_id))""",
    """CREATE TABLE IF NOT EXISTS responses (
        id TEXT PRIMARY KEY, created_at TEXT NOT NULL, variant TEXT NOT NULL, backend TEXT NOT NULL,
        model TEXT NOT NULL, borrower_text TEXT NOT NULL, account TEXT NOT NULL, raw TEXT NOT NULL,
        parsed TEXT, input_tokens INTEGER, output_tokens INTEGER, latency_s REAL)""",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, url: str):
        self.url = url
        self._lock = threading.Lock()
        if url.startswith("sqlite:///"):
            path = url[len("sqlite:///"):]
            if path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self.kind = "sqlite"
        elif url.startswith(("postgresql://", "postgres://")):
            import psycopg
            from psycopg.rows import dict_row
            self._conn = psycopg.connect(url, autocommit=True, row_factory=dict_row)
            self.kind = "postgres"
        else:
            raise ValueError(f"unsupported DATABASE_URL scheme: {url.split(':')[0]}")
        for stmt in SCHEMA:
            self._exec(stmt)
        self._init_chunks()
        self._init_jobs()

    def _exec(self, sql: str, params: tuple = ()):
        if self.kind == "postgres":
            sql = sql.replace("?", "%s")
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall() if cur.description else []
            if self.kind == "sqlite":
                self._conn.commit()
        return [dict(r) for r in rows]

    def _add_column(self, table: str, col: str, typ: str):
        if self.kind == "postgres":
            self._exec(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {typ}")
        elif col not in {r["name"] for r in self._exec(f"PRAGMA table_info({table})")}:
            self._exec(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")

    def _init_jobs(self):
        self._exec("""CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                      status TEXT NOT NULL, params TEXT NOT NULL, run_id TEXT, error TEXT,
                      started_at TEXT, finished_at TEXT)""")
        self._add_column("run_items", "context_ids", "TEXT")
        self._add_column("run_items", "retrieval_s", "REAL")

    def create_job(self, params: dict) -> str:
        jid = uuid.uuid4().hex
        self._exec("INSERT INTO jobs (id,created_at,status,params) VALUES (?,?,?,?)",
                   (jid, _now(), "queued", json.dumps(params)))
        return jid

    def update_job(self, jid: str, **fields) -> None:
        allowed = {"status", "run_id", "error", "started_at", "finished_at"}
        assert set(fields) <= allowed
        sets = ",".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE jobs SET {sets} WHERE id=?", (*fields.values(), jid))

    def get_job(self, jid: str) -> dict | None:
        rows = self._exec("SELECT * FROM jobs WHERE id=?", (jid,))
        if rows:
            rows[0]["params"] = json.loads(rows[0]["params"])
        return rows[0] if rows else None

    def _init_chunks(self):
        if self.kind == "postgres":
            self._exec("CREATE EXTENSION IF NOT EXISTS vector")
            self._exec("""CREATE TABLE IF NOT EXISTS chunks (embedder TEXT NOT NULL, id TEXT NOT NULL,
                          topic TEXT, text TEXT NOT NULL, embedding vector NOT NULL, PRIMARY KEY (embedder, id))""")
            self._exec("ALTER TABLE responses ADD COLUMN IF NOT EXISTS context_ids TEXT")
        else:
            self._exec("""CREATE TABLE IF NOT EXISTS chunks (embedder TEXT NOT NULL, id TEXT NOT NULL,
                          topic TEXT, text TEXT NOT NULL, embedding TEXT NOT NULL, PRIMARY KEY (embedder, id))""")
            cols = {r["name"] for r in self._exec("PRAGMA table_info(responses)")}
            if "context_ids" not in cols:
                self._exec("ALTER TABLE responses ADD COLUMN context_ids TEXT")

    # ---- vector chunks (pgvector on PostgreSQL) ----
    def upsert_chunks(self, embedder: str, rows: list[dict], vectors: list[list[float]]) -> int:
        for r, v in zip(rows, vectors):
            vec = "[" + ",".join(f"{x:.6f}" for x in v) + "]"
            self._exec("DELETE FROM chunks WHERE embedder=? AND id=?", (embedder, r["id"]))
            self._exec("INSERT INTO chunks (embedder,id,topic,text,embedding) VALUES (?,?,?,?,?)"
                       if self.kind == "sqlite" else
                       "INSERT INTO chunks (embedder,id,topic,text,embedding) VALUES (?,?,?,?,?::vector)",
                       (embedder, r["id"], r["topic"], r["text"], vec))
        return len(rows)

    def search_chunks(self, embedder: str, qvec: list[float], k: int = 3) -> list[dict]:
        if self.kind == "postgres":
            vec = "[" + ",".join(f"{x:.6f}" for x in qvec) + "]"
            return self._exec("""SELECT id, topic, text, 1 - (embedding <=> ?::vector) AS score FROM chunks
                                 WHERE embedder=? ORDER BY embedding <=> ?::vector LIMIT ?""", (vec, embedder, vec, k))
        rows = self._exec("SELECT id, topic, text, embedding FROM chunks WHERE embedder=?", (embedder,))
        for r in rows:
            r["score"] = sum(a * b for a, b in zip(qvec, json.loads(r.pop("embedding"))))
        return sorted(rows, key=lambda r: -r["score"])[:k]

    def count_chunks(self, embedder: str) -> int:
        return self._exec("SELECT COUNT(*) AS c FROM chunks WHERE embedder=?", (embedder,))[0]["c"]

    # ---- writes ----
    def add_response(self, *, variant, backend, model, borrower_text, account, raw, parsed, usage, context_ids=None) -> str:
        rid = uuid.uuid4().hex
        self._exec("""INSERT INTO responses (id,created_at,variant,backend,model,borrower_text,account,raw,parsed,
                       input_tokens,output_tokens,latency_s,context_ids) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                   (rid, _now(), variant, backend, model, borrower_text, json.dumps(account), raw,
                    json.dumps(parsed) if parsed is not None else None,
                    usage["input_tokens"], usage["output_tokens"], usage["latency_s"],
                    json.dumps(context_ids) if context_ids is not None else None))
        return rid

    def add_run(self, *, variant, backend, model, noisy, summary, items) -> str:
        rid = uuid.uuid4().hex
        self._exec("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                   (rid, _now(), variant, backend, model, int(noisy), summary["n"], json.dumps(summary)))
        for it in items:
            self._exec("""INSERT INTO run_items (run_id,item_id,intent,style,checks,input_tokens,output_tokens,
                          latency_s,context_ids,retrieval_s) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                       (rid, it["id"], it["intent"], it["style"], json.dumps(it["checks"]),
                        it["input_tokens"], it["output_tokens"], it["latency_s"],
                        json.dumps(it.get("context_ids")) if it.get("context_ids") is not None else None,
                        it.get("retrieval_s")))
        return rid

    # ---- reads ----
    def list_runs(self, limit: int = 20) -> list[dict]:
        rows = self._exec("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        for r in rows:
            r["summary"] = json.loads(r["summary"])
        return rows

    def latest_per_variant(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in self.list_runs(200):          # newest first
            out.setdefault(r["variant"], r)
        return out

    def count(self, table: str) -> int:
        assert table in {"runs", "run_items", "responses", "chunks", "jobs"}
        return self._exec(f"SELECT COUNT(*) AS c FROM {table}")[0]["c"]
