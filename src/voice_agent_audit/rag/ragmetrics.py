"""RAG metrics following the Ragas definitions, implemented here (Ragas itself is not a dependency).

context_recall  fraction of the policy chunks REQUIRED for a scenario that were retrieved.
                ID-based, so it needs no LLM. The required-chunk map below is a DRAFT by the assistant.
faithfulness    fraction of statements in the agent's spoken reply that are supported by the retrieved
                policy + account facts + borrower message. Needs a judge (deterministic proxy on mock).
"""
from __future__ import annotations
import json
import re

SCENARIO_REQUIRED = {
    "will_pay": ["promise-1"], "needs_time": ["callback-1"], "hardship": ["hardship-1", "hardship-2"],
    "dispute": ["dispute-1"], "wrong_person": ["wrong-person-1", "wrong-person-2"], "abusive": ["abusive-1"],
}


def context_recall(retrieved_ids: list[str], intent: str) -> float:
    req = set(SCENARIO_REQUIRED[intent])
    return len(req & set(retrieved_ids)) / len(req)


def statements(raw: str) -> list[str]:
    """Spoken-reply sentences. Tool-call arguments are checked by the tool_args rule, not here."""
    try:
        reply = json.loads(raw).get("reply") or ""
    except Exception:  # noqa: BLE001
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", reply) if s.strip()]


def faithfulness(judge, raw: str, context_texts: list[str], account: dict, borrower: str) -> float | None:
    sts = statements(raw)
    if not sts:
        return None
    ctx = "\n".join(context_texts) + "\nACCOUNT: " + json.dumps(account) + "\nBORROWER: " + borrower
    votes = [judge.support(ctx, s) for s in sts]
    votes = [v for v in votes if v is not None]
    return sum(votes) / len(votes) if votes else None
