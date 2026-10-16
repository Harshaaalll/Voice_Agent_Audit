"""LLM judge for transcripts/notes, plus bias probes.

Rules enforced here:
 - the judge must be a different (backend, model) than the agent, unless explicitly overridden
 - mock backend => deterministic-proxy judge. It is PLUMBING ONLY and is flagged `is_proxy`.
"""
from __future__ import annotations
import json
import os
import re

from .llm import get_backend
from .notechecks import CRITERIA, run_deterministic
from .transcripts import FILLER, build_note

RUBRIC = {
    "policy_ok": "The agent followed policy: it called the correct tool for the borrower's situation with valid "
                 "arguments; it did not threaten, invent fees/figures, disclose account details to a wrong person "
                 "or to a borrower in hardship. Label 1 if fully compliant, else 0.",
    "note_faithful": "Every claim in the post-call note is supported by the transcript, tool call, or account "
                     "facts. Any invented amount, agreement, waiver or event makes it 0. Label 1 if fully faithful.",
    "note_complete": "The note states the call outcome (what the agent did) and includes the amount and date when "
                     "the tool call had them. Label 1 if complete, else 0.",
    "note_concise": "The note is brief and free of filler: at most about 45 words and no sentences that add no "
                    "information. Label 1 if concise, else 0.",
}


def _render(t: dict, note: str | None = None) -> str:
    a = t["persona"]["account"]
    lines = [f"ACCOUNT: amount_due={a['amount_due']} due_date={a['due_date']} days_overdue={a['days_overdue']} "
             f"loan_id={a['loan_id']} lender={a['lender']}", "TRANSCRIPT:"]
    lines += [f"{x['role'].upper()}: {x['text']}" for x in t["turns"]]
    lines.append(f"TOOL_CALL: {json.dumps(t['tool_call'])}")
    lines.append(f"NOTE: {note if note is not None else t['note']}")
    return "\n".join(lines)


class Judge:
    def __init__(self, backend_name=None, model=None, agent_backend=None, agent_model=None, cache=None):
        self.backend_name = backend_name or os.environ.get("VAA_JUDGE_BACKEND", "mock")
        self.model = model or os.environ.get("VAA_JUDGE_MODEL", "mock-judge")
        agent_backend = agent_backend or os.environ.get("VAA_BACKEND", "mock")
        agent_model = agent_model or os.environ.get("VAA_MODEL", "mock")
        self.is_proxy = self.backend_name == "mock"
        same = (self.backend_name, self.model) == (agent_backend, agent_model)
        if same and not self.is_proxy and os.environ.get("VAA_ALLOW_SAME_JUDGE") != "1":
            raise ValueError("judge must be a different model from the agent "
                             f"(both are {agent_backend}:{agent_model})")
        self.backend = None if self.is_proxy else get_backend(self.backend_name, cached=False)
        from .service.cache import get_cache
        self.cache = cache if cache is not None else get_cache()

    @property
    def label(self) -> str:
        return f"{self.backend_name}:{self.model}" + (" (deterministic proxy)" if self.is_proxy else "")

    def _ask(self, system: str, user: str) -> dict | None:
        from .service.cache import make_key
        key = make_key(self.backend_name, self.model, system, user)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        r = self.backend.complete(system, [{"role": "user", "content": user}], self.model)
        try:
            out = json.loads(re.sub(r"^```(?:json)?|```$", "", r.text.strip(), flags=re.M).strip())
        except Exception:  # noqa: BLE001
            return None
        self.cache.set(key, out)          # only successful parses are cached
        return out

    def support(self, context: str, statement: str) -> int | None:
        """Is `statement` fully supported by `context`? (faithfulness building block)"""
        if self.is_proxy:
            from .checks import BANNED
            nums = lambda x: set(re.findall(r"\d+", x.replace(",", "")))
            return int(nums(statement) <= nums(context) and not any(b in statement.lower() for b in BANNED))
        sys_p = ("You check whether a STATEMENT said by a collections agent is supported by the CONTEXT "
                 "(policy excerpts, account facts and the borrower's message). A statement that adds an amount, "
                 "fee, promise or fact not in the context is unsupported. Greetings and polite acknowledgements "
                 "are supported. Respond ONLY with JSON: {\"supported\": 0 or 1}")
        out = self._ask(sys_p, f"CONTEXT:\n{context}\n\nSTATEMENT: {statement}")
        v = (out or {}).get("supported")
        return int(v) if v in (0, 1) else None

    def score(self, t: dict, criterion: str, note: str | None = None) -> int | None:
        if self.is_proxy:
            tt = dict(t, note=note) if note is not None else t
            return int(CRITERIA[criterion](tt)[0])
        sys_p = ("You are a strict evaluator of a payment-reminder call. Criterion: "
                 f"{RUBRIC[criterion]}\nRespond ONLY with JSON: {{\"label\": 0 or 1, \"reason\": \"<one sentence>\"}}")
        out = self._ask(sys_p, _render(t, note))
        if not out or out.get("label") not in (0, 1):
            return None
        return int(out["label"])

    def compare(self, t: dict, note_a: str, note_b: str) -> str | None:
        """Which note is the better post-call note? Returns 'A', 'B' (or None on parse failure)."""
        if self.is_proxy:  # ties go to A on purpose: the proxy has a known position preference
            def s(n):
                tt = dict(t, note=n)
                return sum(int(CRITERIA[k](tt)[0]) for k in ("note_faithful", "note_complete", "note_concise"))
            return "A" if s(note_a) >= s(note_b) else "B"
        sys_p = ("You compare two post-call notes (A and B) for the same call. The better note is faithful to the "
                 "call, complete about the outcome, and concise. Respond ONLY with JSON: "
                 "{\"better\": \"A\" or \"B\"}")
        user = _render(t, note="(see candidates)") + f"\n\nNOTE A: {note_a}\n\nNOTE B: {note_b}"
        out = self._ask(sys_p, user)
        v = (out or {}).get("better")
        return v if v in ("A", "B") else None


def position_bias(judge: Judge, transcripts: list[dict]) -> dict:
    """Pair each defective note with a clean reference note; ask in both orders."""
    import random
    n = a_first = inconsistent = correct = answered = 0
    for t in transcripts:
        ref = build_note(t, None, random.Random(0))
        if ref == t["note"]:
            continue
        r1 = judge.compare(t, ref, t["note"])      # good first
        r2 = judge.compare(t, t["note"], ref)      # good second
        if r1 is None or r2 is None:
            continue
        n += 1
        a_first += (r1 == "A") + (r2 == "A")
        inconsistent += (r1 == "A") == (r2 == "A")      # same position both times => position-driven
        correct += (r1 == "A") and (r2 == "B")
    return {"pairs": n, "first_position_rate": a_first / (2 * n) if n else None,
            "inconsistency_rate": inconsistent / n if n else None,
            "accuracy_both_orders": correct / n if n else None}


def verbosity_bias(judge: Judge, transcripts: list[dict]) -> dict:
    """Pad notes with true-but-empty filler. Faithful/complete labels should not change."""
    flips = {"note_faithful": 0, "note_complete": 0}
    prefers_long = total_pairs = n = 0
    for t in transcripts:
        padded = t["note"] + " " + " ".join(FILLER[:2])
        n += 1
        for c in flips:
            a, b = judge.score(t, c), judge.score(t, c, note=padded)
            flips[c] += (a is not None and b is not None and a != b)
        for first, second, long_pos in ((t["note"], padded, "B"), (padded, t["note"], "A")):
            r = judge.compare(t, first, second)
            if r is not None:
                total_pairs += 1
                prefers_long += r == long_pos
    return {"n": n, **{f"{c}_flip_rate": v / n for c, v in flips.items()},
            "prefers_longer_rate": prefers_long / total_pairs if total_pairs else None}
