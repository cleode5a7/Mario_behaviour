---
name: verify
description: Check that this project's documentation still describes what its code actually does. Runs the mechanical `invoke verify` checks, then reads every doc and docstring for factual claims about behaviour and tests each one against the code. Use after changing what a step does, before committing, or whenever the docs might have drifted.
---

# Verify

## Overview

An analysis pipeline drifts from its own documentation silently. Nothing breaks
when a README lists a task that was renamed, when a docstring says a step "never
pulls data" after that guarantee was removed, or when a documented default of 30
became 10 two commits ago. No test fails, the figures still render, and the next
reader — human or model — is actively misled.

This skill runs the two layers of checking that catch it:

1. **`invoke verify`** — mechanical, deterministic checks. Fast, and it cannot
   forget.
2. **The judgment pass** — reading prose against code for claims no regex can
   evaluate. This is the part only a model can do, and the part this skill
   exists for.

Report findings. **Do not fix anything unless the user asks.** A drifted claim
has two valid repairs — change the doc, or change the code back — and only the
user knows which was intended.

---

## Step 1 — Run the mechanical checks

```bash
uv run invoke verify        # or: invoke verify
```

If the task does not exist, the project predates it: say so, and offer to wire
it up (`airoh.verify`, see the template's `tasks.py`). Then continue with Step 2
regardless — the judgment pass does not depend on it.

Report each `FAIL` and `WARN`. Do not re-verify them by hand; they are already
verified. Note that a `WARN` about data paths not existing yet is expected in a
fresh clone before `invoke fetch`.

---

## Step 2 — The judgment pass

Read, in full: `README.md`, `CLAUDE.md`, `source_data/CONTENT.md`,
`output_data/CONTENT.md`, every docstring in `tasks.py`, and every module and
function docstring in `analysis/`. Also read the comments in `invoke.yaml`.

For each **factual claim about behaviour**, find the code that implements it and
decide whether the claim is still true. A factual claim is anything a reader
could act on and be wrong: what a step reads or writes, what it retrieves, what
it skips, what raises versus warns, a default value, a threshold, a column name,
an ordering guarantee, a performance property.

Ignore prose that is intent, advice, or rationale ("keep notebooks fast",
"prefer `fetch_data`"). Those cannot be false in the same way.

### Where drift actually happens

Check these first — they are where the claims and the code separate in practice:

- **What is fetched, and what is not.** Docs love absolutes: "only the small
  text files", "no `*.nii.gz` is ever retrieved", "content is never pulled".
  Compare against every glob and every retrieval call. A widened glob almost
  never comes with a doc update.
- **Which task pulls data.** If the project separates "gather assets" from
  "reproduce results", check that no `run-*` step has quietly regained a
  retrieval call, and that the docs still describe the split correctly.
- **Defaults, thresholds and constants.** Any number written in prose:
  thresholds, dpi, cut coordinates, size limits, the smoke-test target. Compare
  against the actual constant, including in notebooks.
- **Tolerant versus strict.** Claims about what raises and what only warns, and
  under which flag. These invert easily during refactoring.
- **Output shape.** Column names and table layouts described in `CONTENT.md`
  against what the writing code actually produces. Read a header line if one
  exists.
- **Task wiring.** Docstrings describing a `pre=` chain, an ordering, or a
  caching rule that the body no longer implements.
- **Skipping and caching.** "Skips if the output exists", "re-runs in strict
  mode" — check the condition is really there.

### For each finding, report

- The claim, quoted, with `file:line`.
- The code that contradicts it, with `file:line`.
- Which one you believe is wrong — the doc or the code — and why. If the code
  looks like the accident, say so plainly; a silently dropped guarantee matters
  more than a stale sentence.

---

## Step 3 — Report

Lead with the count and the severity: how many mechanical failures, how many
drifted claims, and whether any of them means the code lost a guarantee rather
than the doc going stale.

List mechanical findings and judgment findings separately — they need different
kinds of attention. Keep each finding to a couple of lines; the `file:line`
pointers carry the detail.

If nothing has drifted, say that in one sentence. Do not pad a clean result.

Then stop and let the user decide what to repair.
