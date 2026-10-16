# Project rules for Claude Code

## Default workflow: PM / Builder / Tester loop
Use the `pm-builder-tester` skill (`.claude/skills/pm-builder-tester/SKILL.md`) by default for every change to code, tests,
config, Docker files or CI in this repo. The main session acts as Product Manager: write a small slice spec with testable
acceptance criteria, spawn a separate Builder agent, then a separate Tester agent that has not seen the Builder's reasoning,
re-run the suite yourself, and accept / revise / re-scope. At most 3 revise rounds per slice, then ask the user.
Log each iteration in `docs/loop-log.md`.

**Skip the loop (do it directly) for:** answering questions, reading or explaining code, running existing commands or
evaluations, editing docs or comments only, and trivial one-line fixes that the existing tests already cover.
If unsure whether something is trivial, use the loop.

## Standing rules (also enforced by the skill)
- Report only measured numbers. Label mock-backend results, drafted gold sets and unreviewed labels as such.
- Never write to or read around the human-label files: `data/human_labels.jsonl`, `data/translation_ratings.jsonl`,
  `data/retrieval_gold_reviewed.jsonl`, `data/injected_defects.jsonl`, `data/translation_hidden.jsonl`.
- Never print, log or commit secrets. `.env` is gitignored.
- The user runs all git write commands (`add`, `commit`, `push`). Suggest them; do not run them. Never add Claude as author or co-author.
- Do not start Docker containers unless the user asks. Stop any you started.
- Disk is tight (a few GB free): avoid heavy new dependencies; check `df -h /` before downloading models.
