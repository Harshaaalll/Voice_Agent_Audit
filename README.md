# hveval: evaluation kit for a Hinglish payment-reminder voice agent

Fully synthetic, text-mode agent evaluated in layers. **Numbers are reported only when measured with a real model.**

## Status (Weekend 1)
- [x] Synthetic personas, balanced intents, ASR-noise injector, code-switch helper
- [x] Two prompt variants: `static` (everything) vs `pruned` (core + compliance + top-2 retrieved snippets)
- [x] 7 deterministic checks: valid JSON, decision, tool args, invented figures, banned phrases, disclosure, length
- [x] Token and latency measurement, per-decision precision/recall/F1
- [ ] Real-model run (Ollama / Gemini) -- **no measured results yet**
- [ ] Weekend 2: summaries eval, labeling tool, calibrated judge (kappa)
- [ ] Weekend 3: BLEU/COMET/judge vs human, metamorphic tests, promptfoo CI, dashboard

## Run
    python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
    .venv/bin/pytest
    .venv/bin/hveval gen --n 120
    HVEVAL_BACKEND=ollama HVEVAL_MODEL=qwen2.5:7b .venv/bin/hveval run
    # or: HVEVAL_BACKEND=gemini HVEVAL_MODEL=gemini-2.0-flash GEMINI_API_KEY=... hveval run

The default `mock` backend is a keyword rule engine for plumbing tests. It ignores the prompt, so
its quality numbers are meaningless; only prompt-size numbers (token counts) are real, and those are
chars/4 estimates unless the provider reports usage.
