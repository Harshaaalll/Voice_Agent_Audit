---
name: pm-builder-tester
description: Run a change through a three-agent loop (Product Manager, Builder, Tester) with unit-test gates and a PM check after every build. Use for any feature, fix or phase of work in this project when the user says "use the loop", "PM builder tester", or asks for a build that is verified and reviewed each iteration. The roster can be expanded with extra roles when the work needs them.
---

# PM / Builder / Tester loop

Three separate agents, one loop. The Product Manager (PM) decides what gets built and checks every result. The Builder writes the change. The Tester tries to break it. Nothing is "done" until the PM accepts it with evidence.

## Why separate agents
- The Tester must not share the Builder's context, so it cannot inherit the Builder's blind spots. Spawn each role with the Agent tool as its own agent, and give each only what its brief below allows.
- The PM is the main session (you). It never writes production code during the loop. It writes the spec, reads the evidence, and decides.

## Roles

| Role | Mission | Gets | Returns |
|---|---|---|---|
| **PM** (main session) | Turn the user's goal into small slices with acceptance criteria. Keep checking. Decide accept / revise / re-scope. | User goal, repo, all reports | Slice spec, verdict, loop log |
| **Builder** (agent) | Implement exactly one slice. Run the existing tests. | The slice spec and the relevant file paths | Files changed, commands run, honest test output, open doubts |
| **Tester** (agent) | Prove the slice wrong. Add unit tests, including edge and adversarial cases, and run the full suite. | The acceptance criteria, the diff or file list, NOT the Builder's reasoning | New tests, full test output, a list of defects with reproduction steps |

## The loop

1. **PM: write the slice spec.** Goal in one sentence; acceptance criteria as testable statements ("`POST /x` with y returns 422"); what is out of scope; which project rules apply (see below). Keep slices small enough that one Builder pass can finish them.
2. **Builder: build.** Implement only the slice. Run `pytest -q` and any command the spec names. Report exact output. If a test fails, say so; never edit a test to make it pass unless the spec says the test is wrong.
3. **Tester: attack.** Read the criteria first, then the diff. Write new tests for each criterion plus at least two edge cases the Builder is unlikely to have tried (empty input, wrong type, repeated call, failure of a dependency). Run `pytest -q`. Report defects with the smallest reproduction.
4. **PM: check.** Read both reports and look at the diff yourself. Verify every acceptance criterion has a passing test. Re-run `pytest -q` once yourself. Check the project rules. Then choose:
   - **Accept**: all criteria met, suite green, rules respected. Log it. Move to the next slice.
   - **Revise**: send the Builder a short, specific list ("criterion 3 fails: input X returns Y, expected Z"). Go to step 2.
   - **Re-scope**: the slice was wrong or too big. Rewrite the spec. Go to step 1.
5. **Stop rule.** At most 3 revise rounds per slice. After that, stop and show the user what is blocked and why, instead of looping again.
6. **Log every iteration** in `docs/loop-log.md`: slice, round number, verdict, one line on why, tests added, suite result. The log is how the PM keeps an eye on the whole run.

## What the PM checks every time (project rules)
- **Numbers are measured or labeled.** Anything produced with the mock backend, a drafted gold set, or without human labels is reported as such. Never present a mock result as model quality.
- **Human labels stay human.** Do not write to `data/human_labels.jsonl`, `data/translation_ratings.jsonl` or `data/retrieval_gold_reviewed.jsonl`. Do not read `data/injected_defects.jsonl` or `data/translation_hidden.jsonl` while building anything the human labels.
- **No secrets.** Nothing from `.env` in files, logs or output. `.env` stays gitignored.
- **No git writes.** The user runs `git add`, `commit` and `push`. Suggest commands; do not run them. Never add Claude as author or co-author.
- **Unit tests gate everything.** A slice with no new test for its new behavior is not accepted. Service changes are also run once in Docker (`docker compose up -d --build`, then call the endpoint) because real Redis, Postgres and file paths have broken things the unit tests could not.
- **Disk is tight.** Avoid new heavy dependencies; check `df -h /` before downloading models.

## Expanding the roster
The PM may add a role when the work calls for it. Add a role only if it has a distinct job and a distinct output; otherwise fold it into an existing role. To add one, write a four-line brief (mission, inputs it may see, output it must return, gate it can block on) and spawn it like the others. Examples:

| Role | Add it when | Gate it can block on |
|---|---|---|
| **Reviewer** | The change touches many files or security-sensitive code | Review findings marked blocking |
| **Data steward** | The change touches labels, gold sets, or hidden-defect files | Any write or read that breaks the label rules |
| **SRE / deploy checker** | The change touches Docker, compose, queues, or config | Container starts, health check, one real request |
| **Docs writer** | A user-facing command, endpoint or setting changed | README matches the actual behavior |

Run independent roles in parallel (one message, several Agent calls). Run dependent ones in order. Remove a role from the run once its job is done.

## Brief templates

**Builder brief:** "Slice: <goal>. Acceptance criteria: <list>. Files likely involved: <paths>. Do only this slice. Run `pytest -q` and paste real output. Report files changed, anything you were unsure about, and anything you did not do. Rules: <relevant project rules>."

**Tester brief:** "Verify this slice against these acceptance criteria: <list>. Changed files: <list>. You have not seen the Builder's reasoning and should not ask for it. Write new unit tests in `tests/`, covering each criterion and at least two edge cases. Run the full suite. Return: tests added, full output, defects with minimal reproductions, and any criterion you could not test and why."

## Output to the user
At the end of a run give: slices accepted, rounds each took, tests added, suite result, anything the PM rejected and why, and anything left unmeasured. Do not claim more than the logs show.
