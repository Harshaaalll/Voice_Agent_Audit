# hveval: evaluation kit for a Hinglish payment-reminder voice agent

Fully synthetic, text-mode agent evaluated in layers. **Numbers are reported only when measured with a real model.**

## Status (Weekend 1)
- [x] Synthetic personas, balanced intents, ASR-noise injector, code-switch helper
- [x] Two prompt variants: `static` (everything) vs `pruned` (core + compliance + top-2 retrieved snippets)
- [x] 7 deterministic checks: valid JSON, decision, tool args, invented figures, banned phrases, disclosure, length
- [x] Token and latency measurement, per-decision precision/recall/F1
- [ ] Real-model run (Ollama / Gemini) -- **no measured results yet**
- [x] Weekend 2 code: transcripts + post-call notes with seeded defects, note checks (faithful / complete / concise), labeling tool, LLM judge, kappa + bootstrap CI, asymmetric-cost scorecard, position and verbosity bias probes
- [ ] Weekend 2 measurements: **0 / 150 human labels so far**, no real judge run -> every judge number is "unmeasured"
- [x] Weekend 3 code (BLEU/chrF only, no COMET) — see below
- [ ] Weekend 3 measurements: 0/60 human ratings -> metric-vs-human correlation UNMEASURED
- [ ] (old) Weekend 3: BLEU/COMET/judge vs human, metamorphic tests, promptfoo CI, dashboard

## Run
    python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
    .venv/bin/pytest
    .venv/bin/hveval gen --n 120
    HVEVAL_BACKEND=ollama HVEVAL_MODEL=qwen2.5:7b .venv/bin/hveval run
    # or: HVEVAL_BACKEND=gemini HVEVAL_MODEL=gemini-2.0-flash GEMINI_API_KEY=... hveval run

The default `mock` backend is a keyword rule engine for plumbing tests. It ignores the prompt, so
its quality numbers are meaningless; only prompt-size numbers (token counts) are real, and those are
chars/4 estimates unless the provider reports usage.

## Weekend 2: how to produce real numbers
    hveval transcripts --n 160        # synthetic calls + notes (seeded defects kept in a separate file)
    hveval label                      # YOU label each item on 4 criteria; resumable, blind to judge/defects
    HVEVAL_JUDGE_BACKEND=gemini HVEVAL_JUDGE_MODEL=gemini-2.0-flash hveval judge
    hveval report                     # agreement + Cohen's kappa (95% bootstrap CI) per criterion
    hveval bias                       # position + verbosity bias of the judge

Rules baked in: the judge must differ from the agent model; with the default `mock` backend the judge is a
deterministic proxy and is flagged as such; the report prints "unmeasured" for any criterion without human labels
and marks results `underpowered` below 150 labels.

## Weekend 3: translation metrics, metamorphic tests, dashboard
    hveval tgen          # 60 English->Hinglish reminder items (degradation level hidden in a separate file)
    hveval rate          # YOU rate adequacy, fluency (1-5) and critical_error (0/1); resumable
    hveval tmetrics      # BLEU / chrF / fact-check vs your adequacy: Spearman rho + 95% bootstrap CI
    hveval meta          # metamorphic invariance tests on the agent (name swap, amount change, filler, ...)
    hveval dashboard     # results/dashboard.html; anything without data shows UNMEASURED

COMET is intentionally not used (needs ~3-5 GB of torch + model weights). Known limitation, tested:
BLEU/chrF score a wrong-amount translation almost as high as a correct one, so the deterministic
critical-fact check is what gates on numbers. `hveval meta` on the mock backend is plumbing only.
