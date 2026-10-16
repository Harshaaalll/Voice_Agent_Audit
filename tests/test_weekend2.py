import json
import pytest

from voice_agent_audit.judge import Judge, position_bias, verbosity_bias
from voice_agent_audit.labeling import load_labels, run as label_run
from voice_agent_audit.notechecks import note_complete, note_concise, note_faithful, policy_ok, run_deterministic
from voice_agent_audit.report import build
from voice_agent_audit.stats import agreement, cohen_kappa, scorecard
from voice_agent_audit.transcripts import generate_transcripts

TR, INJ = generate_transcripts(60, seed=3)


def test_transcripts_deterministic_and_defects_hidden():
    a, _ = generate_transcripts(10, seed=3)
    b, _ = generate_transcripts(10, seed=3)
    assert a == b
    assert not any("defect" in json.dumps(t) for t in TR)   # labeler can never see seeded defects


def test_kappa_hand_computed():
    # po = 0.70; A says 1 for 35/50, B for 30/50 -> pe = .7*.6 + .3*.4 = .54; kappa = .16/.46
    a = [1] * 25 + [1] * 10 + [0] * 5 + [0] * 10
    b = [1] * 25 + [0] * 10 + [1] * 5 + [0] * 10
    assert agreement(a, b) == pytest.approx(0.7)
    assert cohen_kappa(a, b) == pytest.approx(0.16 / 0.46, abs=1e-9)
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)


def test_scorecard_asymmetric_cost():
    gold = [0, 0, 0, 1, 1, 1]      # 0 = violation
    pred = [0, 1, 1, 0, 1, 1]      # catches 1 of 3 violations, 1 false alarm
    s = scorecard(pred, gold, fn_cost=5, fp_cost=1)
    assert (s["tp"], s["fn"], s["fp"]) == (1, 2, 1)
    assert s["cost"] == 5 * 2 + 1
    assert s["recall"] == pytest.approx(1 / 3) and s["precision"] == pytest.approx(0.5)


def test_note_checks_catch_seeded_note_defects():
    caught = {"hallucination": 0, "omission": 0, "verbose": 0}
    seen = dict.fromkeys(caught, 0)
    for t, i in zip(TR, INJ):
        d = i["note_defect"]
        if d == "omission":
            seen[d] += 1; caught[d] += not note_complete(t)[0]
        elif d == "verbose":
            seen[d] += 1; caught[d] += not note_concise(t)[0]
        elif d == "hallucination":
            seen[d] += 1; caught[d] += not note_faithful(t)[0]
    assert seen["omission"] and caught["omission"] == seen["omission"]
    assert seen["verbose"] and caught["verbose"] == seen["verbose"]
    assert caught["hallucination"] >= 1   # text-only hallucinations are intentionally NOT caught


def test_clean_items_are_not_flagged():
    for t, i in zip(TR, INJ):
        if i["reply_defect"] is None:
            assert policy_ok(t)[0], policy_ok(t)[1]
        if i["note_defect"] is None:
            r = run_deterministic(t)
            assert r["note_faithful"] and r["note_complete"] and r["note_concise"]


def test_judge_must_differ_from_agent():
    with pytest.raises(ValueError):
        Judge("gemini", "m", agent_backend="gemini", agent_model="m")
    Judge("gemini", "bigger", agent_backend="gemini", agent_model="m")  # ok


class AlwaysFirst(Judge):
    def __init__(self):
        super().__init__("mock", "x")
    def compare(self, t, a, b):
        return "A"


class LikesLong(Judge):
    def __init__(self):
        super().__init__("mock", "x")
    def compare(self, t, a, b):
        return "A" if len(a) >= len(b) else "B"
    def score(self, t, c, note=None):
        return 1 if (note or t["note"]) and len(note or t["note"]) > 100 else 0


def test_position_bias_probe_detects_biased_judge():
    r = position_bias(AlwaysFirst(), TR)
    assert r["first_position_rate"] == 1.0 and r["inconsistency_rate"] == 1.0


def test_verbosity_probe_detects_length_lover():
    r = verbosity_bias(LikesLong(), TR)
    assert r["prefers_longer_rate"] == 1.0 and r["note_faithful_flip_rate"] > 0


def test_labeling_tool_resume_and_back(tmp_path):
    out = tmp_path / "h.jsonl"
    answers = iter(["1", "0", "b", "1", "1", "1", "q"])   # item 1: back-edit note_faithful
    n = label_run(TR[:3], out, input_fn=lambda _: next(answers), print_fn=lambda *_: None)
    labs = load_labels(out)
    assert n == 1 and len(labs) == 1
    (rec,) = labs.values()
    assert rec["labels"] == {"policy_ok": 1, "note_faithful": 1, "note_complete": 1, "note_concise": 1}
    more = iter(["s"] + ["1"] * 8)                          # resumes; skip one, label next
    label_run(TR[:3], out, input_fn=lambda _: next(more, "q"), print_fn=lambda *_: None)
    assert len(load_labels(out)) >= 2


def test_report_unmeasured_and_underpowered(tmp_path):
    rep = build(TR, tmp_path / "none.jsonl", {}, "none")
    assert all("unmeasured" in v["status"] for v in rep["criteria"].values())
    h = tmp_path / "h.jsonl"
    with open(h, "w") as f:
        for t in TR[:10]:
            f.write(json.dumps({"id": t["id"], "labels": run_deterministic(t), "ts": 0}) + "\n")
    rep = build(TR, h, {t["id"]: run_deterministic(t) for t in TR}, "proxy")
    c = rep["criteria"]["policy_ok"]
    assert c["underpowered"] and c["deterministic"]["kappa"] == 1.0
