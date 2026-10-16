"""Background worker: pops job ids from the queue and runs evaluations."""
from __future__ import annotations
import os
from datetime import datetime, timezone

from .evaluation import run_evaluation


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def process_one(core, queue, backend, model: str, timeout: int = 5) -> str | None:
    """Handle at most one job. Returns the job id, or None if the queue was empty."""
    jid = queue.pop(timeout)
    if jid is None:
        return None
    job = core.store.get_job(jid)
    if job is None:
        return jid
    core.store.update_job(jid, status="running", started_at=_now())
    try:
        p = job["params"]
        out = run_evaluation(core, backend, model, p["variant"], p["limit"], p.get("noisy", True))
        core.store.update_job(jid, status="done", run_id=out["run_id"], finished_at=_now())
    except Exception as e:  # noqa: BLE001  - record the failure, keep the worker alive
        core.store.update_job(jid, status="failed", error=f"{type(e).__name__}: {e}"[:500], finished_at=_now())
    return jid


def loop(core, queue, backend, model: str, once: bool = False, retry_s: float = 2.0) -> None:
    import time
    while True:
        try:
            jid = process_one(core, queue, backend, model)
        except (ConnectionError, TimeoutError, OSError) as e:       # redis.exceptions subclass these
            print(f"queue error ({type(e).__name__}); retrying in {retry_s}s", flush=True)
            if once:
                raise
            time.sleep(retry_s)
            continue
        if once and jid is None:
            return
