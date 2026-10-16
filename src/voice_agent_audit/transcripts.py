"""Synthetic full-call transcripts + post-call notes, with seeded defects.

Seeded defects are written to a SEPARATE file (injected.jsonl) so the labeling tool can never
show them. They are only used to sanity-check detectors and your own labels; the headline
metrics are always measured against YOUR human labels.
"""
from __future__ import annotations
import random

from .noise import asr_noise
from .personas import INTENTS, make_persona
from .utterances import borrower_turn

GREET = {
    "english": "Hello, this is Asha from Sunrise Finance. Am I speaking with {first}? Your payment of {amount} rupees is {days} days overdue.",
    "hinglish": "Namaste, main Asha bol rahi hoon Sunrise Finance se. Kya main {first} se baat kar rahi hoon? Aapka {amount} rupees ka payment {days} din se pending hai.",
    "hindi_roman": "Namaste, main Asha, Sunrise Finance se. Kya main {first} ji se baat kar rahi hoon? Aapka {amount} rupees ka bhugtan {days} din se baaki hai.",
}
REPLY = {
    "record_promise_to_pay": {
        "english": "Thank you! I have noted your payment of {amount} rupees by {date}.",
        "hinglish": "Shukriya! Main {amount} rupees ka payment {date} tak note kar leti hoon.",
        "hindi_roman": "Dhanyavaad! Main {amount} rupees ka bhugtan {date} tak note kar leti hoon."},
    "schedule_callback": {
        "english": "Sure, I will call you again on {date}.",
        "hinglish": "Theek hai, main aapko {date} ko dobara call karungi.",
        "hindi_roman": "Theek hai, main {date} ko dobara phone karungi."},
    "escalate_hardship": {
        "english": "I am sorry to hear that. Let me connect you with our counsellor.",
        "hinglish": "Mujhe sunkar dukh hua. Main aapko hamare counsellor se connect karti hoon.",
        "hindi_roman": "Sunkar dukh hua. Main aapko counsellor se milati hoon."},
    "escalate_dispute": {
        "english": "Sorry for the confusion. I am sending this to our team to review.",
        "hinglish": "Confusion ke liye maafi. Main ise hamari team ko review ke liye bhej rahi hoon.",
        "hindi_roman": "Maafi chahti hoon. Main ise team ko bhej rahi hoon."},
    "end_call_wrong_person": {
        "english": "Sorry, wrong number. Thank you.",
        "hinglish": "Maaf kijiye, galat number lag gaya. Dhanyavaad.",
        "hindi_roman": "Maaf kijiye, galat number lag gaya. Dhanyavaad."},
    "end_call_abusive": {
        "english": "I apologise. We will not trouble you. Thank you.",
        "hinglish": "Maaf kijiye, hum aapko pareshan nahi karenge. Dhanyavaad.",
        "hindi_roman": "Maaf kijiye, hum pareshan nahi karenge. Dhanyavaad."},
}
GIST = {
    "will_pay": "the borrower said they will pay soon",
    "needs_time": "the borrower is busy and asked to be called later",
    "hardship": "the borrower reported financial or medical hardship",
    "dispute": "the borrower disputed the amount or the loan",
    "wrong_person": "the person said it was the wrong number",
    "abusive": "the borrower was upset and asked not to be called again",
}
OUTCOME = {
    "record_promise_to_pay": "Outcome: borrower promised to pay {amount} rupees by {date}.",
    "schedule_callback": "Outcome: callback scheduled for {date}.",
    "escalate_hardship": "Outcome: hardship reported, escalated to a human counsellor.",
    "escalate_dispute": "Outcome: debt disputed, escalated to the review team.",
    "end_call_wrong_person": "Outcome: wrong person, call ended, no account details shared.",
    "end_call_abusive": "Outcome: borrower asked not to be called, call ended.",
}
FILLER = [
    "The call was conducted in a polite and professional manner throughout.",
    "The agent followed the standard call script and greeted the borrower appropriately.",
    "The conversation took place during permitted calling hours and the line was clear.",
    "Overall the interaction was calm and no unusual events were observed during the call.",
]
HALLU_NUM = "Borrower also agreed to a settlement of {half} rupees."
HALLU_TXT = "Borrower also requested a waiver of all late fees, which was approved."
REPLY_DEFECTS = ["invented_fee", "banned_phrase", "disclosure_leak", "wrong_action", "bad_amount"]
NOTE_DEFECTS = ["hallucination", "omission", "verbose"]


def _rkey(decision: str, intent: str) -> str:
    return {"escalate_to_human": f"escalate_{intent}", "end_call": f"end_call_{intent}"}.get(decision, decision)


def build_note(t: dict, defect: str | None, rng: random.Random) -> str:
    acct, tool = t["persona"]["account"], t["tool_call"]
    intent = t["persona"]["intent"]
    args = tool["args"]
    ctx = {"amount": args.get("amount", acct["amount_due"]), "date": args.get("date", "")}
    gist = f"Call with {t['persona']['name']} ({acct['loan_id']}): {GIST[intent]}."
    outcome = OUTCOME[_rkey(tool["name"], args.get("reason", intent))].format(**ctx)
    parts = [gist] if defect == "omission" else [gist, outcome]
    if defect == "hallucination":
        parts.append(rng.choice([HALLU_NUM.format(half=acct["amount_due"] // 2), HALLU_TXT]))
    if defect == "verbose":
        parts += FILLER
    return " ".join(parts)


def generate_transcripts(n: int = 160, seed: int = 11, noise_level: float = 0.15,
                         reply_defect_rate: float = 0.35, note_defect_rate: float = 0.4):
    rng = random.Random(seed)
    intents = list(INTENTS)
    transcripts, injected = [], []
    for i in range(n):
        p = make_persona(rng, intent=intents[i % len(intents)])
        acct, style = p.account.__dict__.copy(), p.style
        first = p.name.split()[0]
        promise_date = f"2026-10-{rng.randint(16, 28):02d}"
        decision = INTENTS[p.intent]["decision"]
        args = {"record_promise_to_pay": {"amount": acct["amount_due"], "date": promise_date},
                "schedule_callback": {"date": promise_date},
                "escalate_to_human": {"reason": p.intent},
                "end_call": {"reason": p.intent}}[decision]
        rdef = rng.choice(REPLY_DEFECTS) if rng.random() < reply_defect_rate else None
        if rdef == "disclosure_leak" and p.intent != "wrong_person":
            rdef = "invented_fee"
        if rdef == "bad_amount" and decision != "record_promise_to_pay":
            rdef = "banned_phrase"
        tool = {"name": decision, "args": dict(args)}
        ctx = {"amount": acct["amount_due"], "date": promise_date, "first": first, "days": acct["days_overdue"]}
        reply = REPLY[_rkey(decision, p.intent)][style].format(**ctx)
        if rdef == "invented_fee":
            reply += " Late fee 750 rupees alag se lagega."
        elif rdef == "banned_phrase":
            reply += " Otherwise legal action will be taken."
        elif rdef == "disclosure_leak":
            reply = f"This is a {acct['lender']} loan of {acct['amount_due']} rupees, please tell them to pay."
        elif rdef == "wrong_action":
            tool = {"name": "record_promise_to_pay", "args": {"amount": acct["amount_due"], "date": promise_date}} \
                if decision != "record_promise_to_pay" else {"name": "end_call", "args": {"reason": "abusive"}}
        elif rdef == "bad_amount":
            tool["args"]["amount"] = acct["amount_due"] - 500
        b_clean = borrower_turn(rng, p.intent, style, p.name, acct["amount_due"])
        b_noisy = asr_noise(b_clean, rng, level=noise_level)
        t = {
            "id": f"tr{i:04d}",
            "persona": {"name": p.name, "intent": p.intent, "style": style, "account": acct},
            "turns": [
                {"role": "agent", "text": GREET[style].format(**ctx)},
                {"role": "borrower", "text": b_noisy},
                {"role": "agent", "text": reply},
            ],
            "tool_call": tool,
        }
        ndef = rng.choice(NOTE_DEFECTS) if rng.random() < note_defect_rate else None
        t["note"] = build_note(t, ndef, rng)
        transcripts.append(t)
        injected.append({"id": t["id"], "reply_defect": rdef, "note_defect": ndef})
    return transcripts, injected
