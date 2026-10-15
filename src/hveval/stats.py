"""Agreement statistics. Pure python, no numpy."""
from __future__ import annotations
import random


def agreement(a: list[int], b: list[int]) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two raters over any hashable labels.
    Degenerate case (both raters constant and equal, expected agreement 1): returns 1.0
    if they agree everywhere, else 0.0. Report prevalence next to kappa when this happens."""
    n = len(a)
    po = agreement(a, b)
    labels = set(a) | set(b)
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def bootstrap_ci(a, b, stat=cohen_kappa, iters: int = 2000, seed: int = 0, alpha: float = 0.05):
    rng = random.Random(seed)
    n = len(a)
    vals = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(stat([a[i] for i in idx], [b[i] for i in idx]))
    vals.sort()
    return vals[int(alpha / 2 * iters)], vals[int((1 - alpha / 2) * iters) - 1]


def confusion(pred: list[int], gold: list[int], positive: int = 0) -> dict:
    """Counts with `positive` as the class being detected (default 0 = 'violation')."""
    tp = sum(p == positive and g == positive for p, g in zip(pred, gold))
    fp = sum(p == positive and g != positive for p, g in zip(pred, gold))
    fn = sum(p != positive and g == positive for p, g in zip(pred, gold))
    tn = len(pred) - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def prf(c: dict, beta: float = 1.0) -> dict:
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0
    return {"precision": p, "recall": r, "f_beta": f, "beta": beta}


def scorecard(pred, gold, fn_cost: float = 5.0, fp_cost: float = 1.0, positive: int = 0) -> dict:
    """Detector scorecard with asymmetric cost: missing a real violation (FN) costs
    `fn_cost`, a false alarm (FP) costs `fp_cost`. Also reports F1 and F2 (recall-weighted)."""
    c = confusion(pred, gold, positive)
    out = dict(c)
    out.update({k: v for k, v in prf(c, 1.0).items() if k != "beta"})
    out["f1"] = prf(c, 1.0)["f_beta"]
    out["f2"] = prf(c, 2.0)["f_beta"]
    out["cost"] = fn_cost * c["fn"] + fp_cost * c["fp"]
    out["cost_per_item"] = out["cost"] / len(pred)
    out["fn_cost"], out["fp_cost"] = fn_cost, fp_cost
    return out
