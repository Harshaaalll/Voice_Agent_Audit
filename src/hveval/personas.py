"""Fully synthetic borrower personas and account facts. No real data."""
from __future__ import annotations
import random
from dataclasses import dataclass, asdict

FIRST = ["Rahul", "Priya", "Amit", "Sneha", "Vikram", "Anjali", "Rohan", "Kavya", "Imran", "Neha"]
LAST = ["Sharma", "Verma", "Iyer", "Khan", "Patel", "Singh", "Nair", "Das", "Mehta", "Reddy"]

# Intent = what the borrower does. Ground-truth policy decision is fixed by construction.
INTENTS = {
    "will_pay":      {"decision": "record_promise_to_pay"},
    "needs_time":    {"decision": "schedule_callback"},
    "hardship":      {"decision": "escalate_to_human"},
    "dispute":       {"decision": "escalate_to_human"},
    "wrong_person":  {"decision": "end_call"},
    "abusive":       {"decision": "end_call"},
}


@dataclass
class Account:
    loan_id: str
    amount_due: int       # INR
    due_date: str         # ISO date
    days_overdue: int
    lender: str = "Sunrise Finance"


@dataclass
class Persona:
    name: str
    intent: str
    style: str            # "hinglish" | "english" | "hindi_roman"
    account: Account

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def make_persona(rng: random.Random, intent: str | None = None, style: str | None = None) -> Persona:
    intent = intent or rng.choice(list(INTENTS))
    style = style or rng.choice(["hinglish", "english", "hindi_roman"])
    amount = rng.choice(range(1500, 48000, 500))
    overdue = rng.choice([3, 7, 12, 21, 35])
    day = rng.randint(1, 28)
    acct = Account(
        loan_id=f"LN{rng.randint(100000, 999999)}",
        amount_due=amount,
        due_date=f"2026-09-{day:02d}",
        days_overdue=overdue,
    )
    return Persona(name=f"{rng.choice(FIRST)} {rng.choice(LAST)}", intent=intent, style=style, account=acct)
