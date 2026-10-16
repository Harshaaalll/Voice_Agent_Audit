"""Judge-vs-human agreement report. Reports ONLY what has human labels behind it."""
from __future__ import annotations

from .labeling import CRITERIA, load_labels
from .notechecks import run_deterministic
from .stats import agreement, bootstrap_ci, cohen_kappa, scorecard

MIN_N = 150


def build(transcripts: list[dict], human_path, judge_labels: dict[str, dict], judge_name: str,
          fn_cost: float = 5.0) -> dict:
    human = load_labels(human_path)
    by_id = {t["id"]: t for t in transcripts}
    report = {"judge": judge_name, "n_human_labeled": len(human), "min_n": MIN_N, "criteria": {}}
    for c in CRITERIA:
        ids = [i for i in human if i in by_id and c in human[i]["labels"]]
        if not ids:
            report["criteria"][c] = {"status": "unmeasured: no human labels yet"}
            continue
        h = [human[i]["labels"][c] for i in ids]
        row = {"n": len(ids), "human_pass_rate": sum(h) / len(h),
               "underpowered": len(ids) < MIN_N}
        for name, src in (("deterministic", {i: run_deterministic(by_id[i]) for i in ids}),
                          ("judge", judge_labels)):
            pairs = [(src[i][c], human[i]["labels"][c]) for i in ids if i in src and src[i].get(c) is not None]
            if not pairs:
                row[name] = {"status": "no predictions"}
                continue
            p, g = [x for x, _ in pairs], [y for _, y in pairs]
            lo, hi = bootstrap_ci(p, g)
            row[name] = {"n": len(p), "agreement": agreement(p, g), "kappa": cohen_kappa(p, g),
                         "kappa_ci95": [lo, hi], "violation_detector": scorecard(p, g, fn_cost=fn_cost)}
        report["criteria"][c] = row
    return report
