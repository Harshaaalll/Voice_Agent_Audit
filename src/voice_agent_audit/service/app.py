"""FastAPI wrapper around the eval harness.  Run: uvicorn voice_agent_audit.service.app:app"""
from __future__ import annotations
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ..checks import parse_output
from ..llm import get_backend
from ..prompts import build_prompt
from .core import Core
from .evaluation import run_evaluation
from .queue import get_queue
from .tracing import get_tracer

VARIANTS = ("static", "pruned", "rag")


class Account(BaseModel):
    loan_id: str
    amount_due: int = Field(gt=0)
    due_date: str
    days_overdue: int = Field(ge=0)
    lender: str


class RespondIn(BaseModel):
    borrower_text: str = Field(min_length=1, max_length=1000)
    account: Account
    variant: str = "pruned"


class RetrieveIn(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(3, ge=1, le=10)


class EvaluateIn(BaseModel):
    variant: str = "pruned"
    limit: int = Field(30, ge=1, le=200)
    noisy: bool = True
    run_async: bool = False


def create_app(database_url: str | None = None, backend=None, queue=None, tracer=None) -> FastAPI:
    url = database_url or os.environ.get("DATABASE_URL", "sqlite:///results/vaa.db")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.core = Core(url)
        app.state.queue = queue if queue is not None else get_queue()
        app.state.tracer = tracer if tracer is not None else get_tracer()
        yield

    app = FastAPI(title="Voice Agent Audit", version="0.3.0", lifespan=lifespan)
    app.state.backend = backend

    def _backend():
        return app.state.backend or get_backend()

    def _model():
        return os.environ.get("VAA_MODEL", "mock")

    def _bname():
        return os.environ.get("VAA_BACKEND", "mock")

    @app.get("/health")
    def health():
        c: Core = app.state.core
        return {"status": "ok", "db": c.store.kind, "backend": _bname(), "model": _model(),
                "embedder": c.embedder.name, "retrieval": c.mode, "chunks": c.store.count_chunks(c.embedder.name),
                "queue": app.state.queue is not None, "tracing": app.state.tracer.enabled}

    @app.post("/retrieve")
    def retrieve(body: RetrieveIn):
        c: Core = app.state.core
        hits = c.retrieve(body.query, body.k)
        return {"embedder": c.embedder.name, "store": c.store.kind,
                "hits": [{"id": h["id"], "topic": h["topic"], "text": h["text"], "score": round(float(h["score"]), 4)} for h in hits]}

    @app.post("/respond")
    def respond(body: RespondIn):
        if body.variant not in VARIANTS:
            raise HTTPException(422, f"variant must be one of {VARIANTS}")
        c: Core = app.state.core
        acct = body.account.model_dump()
        hits = c.retrieve(body.borrower_text, 3) if body.variant == "rag" else None
        system = build_prompt(body.variant, acct, body.borrower_text,
                              context=[h["text"] for h in hits] if hits is not None else None)
        res = _backend().complete(system, [{"role": "user", "content": body.borrower_text}], _model())
        parsed, err = parse_output(res.text)
        usage = {"input_tokens": res.input_tokens, "output_tokens": res.output_tokens, "latency_s": res.latency_s}
        ids = [h["id"] for h in hits] if hits is not None else None
        rid = c.store.add_response(variant=body.variant, backend=_bname(), model=_model(),
                                   borrower_text=body.borrower_text, account=acct, raw=res.text,
                                   parsed=parsed, usage=usage, context_ids=ids)
        app.state.tracer.record(name=f"respond:{body.variant}", model=_model(), input=body.borrower_text,
                                output=res.text, usage=usage, latency_s=res.latency_s, trace_id=rid,
                                metadata={"variant": body.variant, "backend": _bname(), "context_ids": ids})
        return {"id": rid, "valid_json": parsed is not None, "parse_error": None if parsed else err,
                "reply": (parsed or {}).get("reply"), "tool": (parsed or {}).get("tool"),
                "context_ids": ids, **usage, "token_source": res.token_source}

    @app.post("/evaluate")
    def evaluate(body: EvaluateIn):
        """Synchronous by default. With run_async=true the job goes to the Redis queue and a worker runs it."""
        if body.variant not in VARIANTS:
            raise HTTPException(422, f"variant must be one of {VARIANTS}")
        c: Core = app.state.core
        if body.run_async:
            if app.state.queue is None:
                raise HTTPException(503, "async evaluation needs REDIS_URL (queue not configured)")
            jid = c.store.create_job(body.model_dump())
            app.state.queue.enqueue(jid)
            return {"job_id": jid, "status": "queued"}
        try:
            return run_evaluation(c, _backend(), _model(), body.variant, body.limit, body.noisy)
        except FileNotFoundError as e:
            raise HTTPException(500, str(e))

    @app.get("/jobs/{job_id}")
    def job(job_id: str):
        j = app.state.core.store.get_job(job_id)
        if not j:
            raise HTTPException(404, "unknown job")
        return j

    @app.get("/runs")
    def runs(limit: int = Query(20, ge=1, le=200)):
        return app.state.core.store.list_runs(limit)

    @app.get("/metrics")
    def metrics():
        """Latest run per variant. `is_mock` is true when the backend was the deterministic mock."""
        return {v: {"run_id": r["id"], "created_at": r["created_at"], "backend": r["backend"],
                    "model": r["model"], "is_mock": r["backend"] == "mock", **r["summary"]}
                for v, r in app.state.core.store.latest_per_variant().items()}

    return app


app = create_app()
