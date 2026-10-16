import random
from voice_agent_audit.checks import run_all, check_no_invented_figures, check_banned_phrases
from voice_agent_audit.dataset import generate
from voice_agent_audit.noise import asr_noise
from voice_agent_audit.prompts import build_prompt
from voice_agent_audit.runner import run_variant, summarize

ACCT = {"loan_id": "LN123456", "amount_due": 5000, "due_date": "2026-09-10", "days_overdue": 7, "lender": "Sunrise Finance"}


def test_dataset_deterministic_and_balanced():
    a, b = generate(12, seed=1), generate(12, seed=1)
    assert a == b
    assert len({r["expected_decision"] for r in a}) == 4


def test_noise_deterministic():
    assert asr_noise("payment kal karunga", random.Random(3)) == asr_noise("payment kal karunga", random.Random(3))


def test_pruned_shorter_than_static():
    s = build_prompt("static", ACCT, "kal call karo")
    p = build_prompt("pruned", ACCT, "kal call karo")
    assert len(p) < len(s)


def test_invented_figure_detected():
    ok, _ = check_no_invented_figures({"reply": "Aapko 5000 dena hai", "tool": None}, ACCT)
    bad, d = check_no_invented_figures({"reply": "Late fee 750 lagega", "tool": None}, ACCT)
    assert ok and not bad and "750" in d


def test_banned_phrase_detected():
    ok, _ = check_banned_phrases({"reply": "We will take legal action"})
    assert not ok


def test_wrong_arg_amount_fails():
    row = {"persona": {"account": ACCT, "intent": "will_pay"}, "expected_decision": "record_promise_to_pay"}
    raw = '{"reply":"ok","tool":{"name":"record_promise_to_pay","args":{"amount":4000,"date":"2026-10-16"}}}'
    assert not run_all(raw, row)["tool_args"][0]


def test_pipeline_runs_with_mock():
    rows = generate(12, seed=2)
    res = run_variant(rows, "static", backend=None, model="mock")
    assert summarize(res)["n"] == 12
