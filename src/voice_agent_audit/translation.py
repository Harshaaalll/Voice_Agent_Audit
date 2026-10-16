"""Weekend 3: reminder-script translation eval. Pure-python BLEU / chrF + critical-fact check.

Candidates are synthetic degradations of a reference; the degradation level is stored in a
SEPARATE hidden file so the rating tool cannot leak it. Headline numbers use YOUR ratings.
"""
from __future__ import annotations
import math
import random
import re
from collections import Counter

TEMPLATES = [
    ("Hello {name}, this is a reminder from {lender}. Your payment of {amount} rupees was due on {date}. "
     "Please pay at your earliest convenience.",
     "Namaste {name}, ye {lender} ki taraf se ek reminder hai. Aapka {amount} rupees ka payment {date} ko due tha. "
     "Kripya jald se jald payment kar dijiye."),
    ("Hi {name}, we noticed your {amount} rupees payment is still pending. You can pay through UPI or at any branch. "
     "Thank you.",
     "Hi {name}, humne dekha ki aapka {amount} rupees ka payment abhi pending hai. Aap UPI se ya kisi bhi branch par "
     "pay kar sakte hain. Dhanyavaad."),
    ("Dear {name}, your instalment of {amount} rupees is overdue since {date}. If you are facing difficulty, please "
     "let us know so we can help.",
     "Priya {name}, aapki {amount} rupees ki kist {date} se overdue hai. Agar aapko koi dikkat ho rahi hai, to hume "
     "bataiye taaki hum madad kar sakein."),
    ("Hello {name}, your payment of {amount} rupees was received on {date}. Thank you for your cooperation.",
     "Namaste {name}, aapka {amount} rupees ka payment {date} ko mil gaya hai. Aapke sahyog ke liye dhanyavaad."),
]
LENDERS = ["Sunrise Finance", "Apex Credit", "Bharat Lending"]
NAMES = ["Amit Khanna", "Priya Nair", "Imran Verma", "Sunita Rao", "Rahul Das"]
LEVELS = ["good", "medium", "bad_fact", "bad_tone", "word_salad"]
SWAPS = [("Kripya", "Please"), ("jald se jald", "jaldi se jaldi"), ("ye ", "yeh "), ("dhanyavaad", "shukriya"),
         ("Dhanyavaad", "Shukriya"), ("bataiye", "batayein")]


def _degrade(ref: str, level: str, ctx: dict, rng: random.Random) -> str:
    if level == "good":
        out = ref
        for a, b in rng.sample(SWAPS, 2):
            out = out.replace(a, b, 1)
        return out
    if level == "medium":
        sents = re.split(r"(?<=\.)\s+", ref)
        return " ".join(sents[:-1]) if len(sents) > 1 else ref
    if level == "bad_fact":
        if ctx["date"] not in ref or rng.random() < 0.5:
            return ref.replace(str(ctx["amount"]), str(ctx["amount"] + rng.choice([500, 1500, 2500])), 1)
        return ref.replace(ctx["date"], f"2026-11-{rng.randint(1, 28):02d}", 1)
    if level == "bad_tone":
        return ref + " Warna aapke upar legal action liya jayega."
    toks = ref.split()
    rng.shuffle(toks)
    return " ".join(toks)


def generate_items(per_cell: int = 3, seed: int = 21):
    rng = random.Random(seed)
    items, hidden, n = [], [], 0
    for tpl_i, (src, ref) in enumerate(TEMPLATES):
        for level in LEVELS:
            for _ in range(per_cell):
                ctx = {"name": rng.choice(NAMES).split()[0], "lender": rng.choice(LENDERS),
                       "amount": rng.randrange(2000, 30000, 500), "date": f"2026-10-{rng.randint(1, 28):02d}"}
                iid = f"tx{n:03d}"
                n += 1
                s, r = src.format(**ctx), ref.format(**ctx)
                items.append({"id": iid, "source": s, "reference": r,
                              "candidate": _degrade(r, level, ctx, rng), "facts": ctx})
                hidden.append({"id": iid, "level": level, "template": tpl_i})
    order = list(range(len(items)))
    rng.shuffle(order)
    return [items[i] for i in order], [hidden[i] for i in order]


# ---------- metrics ----------
def _tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def _ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def sentence_bleu(hyp: str, ref: str, max_n: int = 4) -> float:
    """Sentence BLEU with add-1 smoothing for n>1 (Lin & Och 2004). Returns 0..1."""
    h, r = _tok(hyp), _tok(ref)
    if not h or not r:
        return 0.0
    logp = 0.0
    for n in range(1, max_n + 1):
        hc, rc = _ngrams(h, n), _ngrams(r, n)
        match = sum(min(c, rc[g]) for g, c in hc.items())
        total = max(sum(hc.values()), 0)
        if n == 1:
            if match == 0 or total == 0:
                return 0.0
            p = match / total
        else:
            p = (match + 1) / (total + 1)
        logp += math.log(p) / max_n
    bp = 1.0 if len(h) > len(r) else math.exp(1 - len(r) / len(h))
    return bp * math.exp(logp)


def chrf(hyp: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    """chrF (Popovic 2015): char n-gram F-beta, whitespace removed. Returns 0..1."""
    h, r = re.sub(r"\s+", "", hyp.lower()), re.sub(r"\s+", "", ref.lower())
    ps, rs = [], []
    for n in range(1, max_n + 1):
        hc, rc = _ngrams(list(h), n), _ngrams(list(r), n)
        if not hc or not rc:
            continue
        m = sum(min(c, rc[g]) for g, c in hc.items())
        ps.append(m / sum(hc.values()))
        rs.append(m / sum(rc.values()))
    if not ps:
        return 0.0
    p, rc_ = sum(ps) / len(ps), sum(rs) / len(rs)
    if p + rc_ == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * rc_ / (b2 * p + rc_)


def facts_preserved(item: dict) -> bool:
    """Deterministic critical-fact check: amount and date from the source appear in the candidate,
    and the candidate introduces no other 3+ digit numbers."""
    c = item["candidate"]
    f = item["facts"]
    nums = set(re.findall(r"\d+", c.replace("-", " ")))
    date_needed = f["date"] in item["source"]
    return str(f["amount"]) in c and (f["date"] in c or not date_needed) and not (nums - set(re.findall(r"\d+", item["source"].replace("-", " "))))


# ---------- correlation ----------
def _ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b) -> float | None:
    ra, rb = _ranks(a), _ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return None if va == 0 or vb == 0 else cov / (va * vb)


def boot_spearman(a, b, iters=2000, seed=0):
    rng = random.Random(seed)
    n, vals = len(a), []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        v = spearman([a[i] for i in idx], [b[i] for i in idx])
        if v is not None:
            vals.append(v)
    vals.sort()
    if not vals:
        return None
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]
