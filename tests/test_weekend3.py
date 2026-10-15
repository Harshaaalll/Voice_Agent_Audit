import pytest
from hveval import translation as T
from hveval.metamorphic import RELATIONS, run as meta_run
from hveval.llm import Backend, LLMResult

ITEMS, HIDDEN = T.generate_items(per_cell=3, seed=21)
LV = {h["id"]: h["level"] for h in HIDDEN}


def test_bleu_chrf_identity_and_disjoint():
    s = "Aapka 5000 rupees ka payment pending hai"
    assert T.sentence_bleu(s, s) == pytest.approx(1.0)
    assert T.chrf(s, s) == pytest.approx(1.0)
    assert T.sentence_bleu("xyz abc", s) == 0.0
    assert T.chrf("qqq", "zzz") == 0.0


def test_bleu_brevity_penalty():
    ref = "aapka payment kal tak due hai please pay"
    assert T.sentence_bleu("aapka payment kal tak", ref) < T.sentence_bleu(ref, ref)


def test_spearman_known():
    assert T.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert T.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert T.spearman([1, 1, 1], [1, 2, 3]) is None
    # tie handling: ranks of [1,2,2,3] are [1,2.5,2.5,4]
    assert T._ranks([1, 2, 2, 3]) == [1, 2.5, 2.5, 4]


def test_fact_check_catches_bad_fact_and_passes_good():
    for it in ITEMS:
        lv = LV[it["id"]]
        if lv == "bad_fact":
            assert not T.facts_preserved(it)
        if lv in ("good", "medium", "bad_tone"):
            ok = T.facts_preserved(it)
            # medium may drop the sentence holding the date/amount; others must preserve
            assert ok or lv == "medium", (lv, it["candidate"])


def test_bad_fact_always_changes_the_candidate():
    for it in ITEMS:
        if LV[it["id"]] == "bad_fact":
            assert it["candidate"] != it["reference"]


def test_bleu_is_blind_to_wrong_facts():
    """Documented failure mode: a wrong-amount candidate still scores far above word salad
    and close to a good translation, so BLEU alone cannot gate on critical errors."""
    def mean(lv, f):
        xs = [i for i in ITEMS if LV[i["id"]] == lv]
        return sum(f(i["candidate"], i["reference"]) for i in xs) / len(xs)
    assert mean("bad_fact", T.sentence_bleu) > 0.6 > mean("word_salad", T.sentence_bleu)
    assert mean("bad_fact", T.chrf) > 0.9


def test_items_hide_levels():
    assert not any("level" in i for i in ITEMS)


class FlipOnFiller(Backend):
    """Deliberately non-invariant agent: changes decision when text starts with a filler."""
    def complete(self, system, messages, model):
        t = messages[0]["content"]
        name = "end_call" if t.startswith(("Haan toh", "Umm", "Dekhiye", "Okay")) else "record_promise_to_pay"
        return LLMResult(text='{"reply":"x","tool":{"name":"%s","args":{}}}' % name, input_tokens=1,
                         output_tokens=1, token_source="est", latency_s=0.0)


def test_metamorphic_detects_non_invariant_agent():
    rows = [{"id": f"r{i}", "borrower_clean": "Haan main pay karunga",
             "persona": {"name": "Amit Khanna", "account": {"amount_due": 5000, "due_date": "2026-10-01",
                         "days_overdue": 3, "loan_id": "LN1", "lender": "Apex"}}} for i in range(10)]
    out = meta_run(rows, backend=FlipOnFiller(), model="x")
    assert out["relations"]["filler_prefix"]["violation_rate"] == 1.0
    assert out["relations"]["amount_change"]["violation_rate"] == 0.0
