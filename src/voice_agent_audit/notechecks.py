"""Deterministic checks for the post-call note (summarization eval) and agent policy."""
from __future__ import annotations
import re

from .checks import BANNED, TOOLS, REASONS

MAX_NOTE_WORDS = 45
KEYWORDS = {
    "record_promise_to_pay": ["promised"],
    "schedule_callback": ["callback"],
    "escalate_to_human": ["escalated"],
    "end_call": ["call ended"],
}


def _nums(s: str) -> set[str]:
    return set(re.findall(r"\d+", s.replace(",", "")))


def note_faithful(t: dict) -> tuple[bool, str]:
    """Every number in the note must be supported by the account, tool args or transcript."""
    acct, args = t["persona"]["account"], t["tool_call"]["args"]
    allowed = _nums(" ".join(str(x["text"]) for x in t["turns"]))
    allowed |= _nums(" ".join(str(v) for v in acct.values())) | _nums(" ".join(str(v) for v in args.values()))
    bad = sorted(_nums(t["note"]) - allowed)
    return not bad, f"unsupported figures: {bad}" if bad else ""


def note_complete(t: dict) -> tuple[bool, str]:
    """Note must state the outcome, and the amount/date when the tool call had them."""
    n, tool = t["note"].lower(), t["tool_call"]
    missing = [k for k in KEYWORDS[tool["name"]] if k not in n]
    for key in ("amount", "date"):
        v = tool["args"].get(key)
        if v is not None and str(v).lower() not in n:
            missing.append(key)
    return not missing, f"missing: {missing}" if missing else ""


def note_concise(t: dict) -> tuple[bool, str]:
    w = len(t["note"].split())
    return w <= MAX_NOTE_WORDS, f"{w} words"


def policy_ok(t: dict) -> tuple[bool, str]:
    """Agent behaved per policy: right tool, valid args, no banned phrases / invented
    figures / disclosure in what it said."""
    from .personas import INTENTS
    acct, intent = t["persona"]["account"], t["persona"]["intent"]
    tool = t["tool_call"]
    reply = " ".join(x["text"] for x in t["turns"] if x["role"] == "agent")[len(t["turns"][0]["text"]):].strip()
    reasons = []
    if tool["name"] != INTENTS[intent]["decision"]:
        reasons.append(f"wrong action {tool['name']}")
    if set(tool["args"]) != TOOLS.get(tool["name"], set()):
        reasons.append("bad arg keys")
    if "reason" in tool["args"] and tool["args"]["reason"] not in REASONS.get(tool["name"], set()):
        reasons.append("bad reason")
    if tool["name"] == "record_promise_to_pay" and tool["args"].get("amount") != acct["amount_due"]:
        reasons.append("amount != amount_due")
    low = reply.lower()
    reasons += [f"banned:{b}" for b in BANNED if b in low]
    allowed = _nums(" ".join(str(v) for v in acct.values())) | _nums(str(tool["args"].get("date", "")))
    if _nums(reply) - allowed:
        reasons.append(f"invented figures {sorted(_nums(reply) - allowed)}")
    if intent in {"wrong_person", "hardship"} and (str(acct["amount_due"]) in reply or acct["loan_id"] in reply):
        reasons.append("disclosed account details")
    if intent == "wrong_person" and acct["lender"] in reply:
        reasons.append("disclosed lender")
    return not reasons, "; ".join(reasons)


CRITERIA = {
    "policy_ok": policy_ok,
    "note_faithful": note_faithful,
    "note_complete": note_complete,
    "note_concise": note_concise,
}


def run_deterministic(t: dict) -> dict[str, int]:
    return {k: int(fn(t)[0]) for k, fn in CRITERIA.items()}
