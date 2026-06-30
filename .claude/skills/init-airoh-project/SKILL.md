---
name: init-airoh-project
description: This skill should be used when initializing a new reproducible analysis project from the airoh-mini template. It guides the user through specifying project metadata, selecting a package manager, implementing fetch/run/clean invoke tasks, updating documentation, and running a smoke test to verify the full pipeline.
---

# Init Airoh Project

## Overview

Wire up the `airoh-mini` template into a working, reproducible analysis pipeline. Walk the user through project metadata, package manager choice, fetch/run/clean task implementation, documentation updates, and a final smoke test. Work incrementally — check for existing scripts and notebooks first, ask only what's needed, and produce placeholder stubs when work is deferred.

Read `references/airoh_api.md` before writing any code.

---

## Workflow

Follow these steps in order. Complete each step before moving to the next.

---

### Step 1 — Project basics

Ask the user for:
1. **Project title** — used in README.md, CLAUDE.md, and `pyproject.toml`
2. **Short overview** (1–3 sentences) — what the analysis does and why
3. **Package manager** — `uv`, `pip`, or `conda`

Do not proceed until all three are confirmed.

---

### Step 2 — Update package manager setup

Based on the chosen package manager, keep or remove the following files:

| File | uv | pip | conda |
|---|---|---|---|
| `pyproject.toml` | keep, update `name` field | delete | delete |
| `uv.lock` | keep | delete | delete |
| `requirements.txt` | keep | keep | keep |
| `environment.yml` | delete | delete | keep |

For **uv**: update `pyproject.toml` — set `name` to a slug of the project title, keep all dependencies.

For **pip**: `requirements.txt` is the only dependency file; no changes needed unless the user wants to add packages.

For **conda**: `environment.yml` is the primary; `requirements.txt` can stay as a pip fallback inside the conda env.

Fix the `code_dir` key in `invoke.yaml` if it still says `code: code` — it should be `code_dir: analysis` to match the actual directory name.

---

### Step 3 — Clean template artifacts

Remove the demo code that ships with the template:

- In `tasks.py`: delete the `run_simulation` task and remove it from all `pre=` chains
- Delete `analysis/simulation.py`; if `analysis/` is now empty (only `__init__.py`), keep it — the user will populate it
- In `invoke.yaml` under `files:`: delete the `papers:` entry (the demo download)
- Overwrite `source_data/CONTENT.md` with a minimal placeholder:
  ```
  # Source Data

  _TODO: document data sources after fetch tasks are set up._
  ```
- Overwrite `output_data/CONTENT.md` with a minimal placeholder:
  ```
  # Output Data

  _TODO: document outputs after run tasks are set up._
  ```

After cleanup, `tasks.py` should contain only stubs: an empty `fetch`, `run_notebooks`, `run`, and `clean`.

---

### Step 4 — Fetch tasks

**Survey first.** Check `source_data/` for any files that are not part of the template (i.e., not `.gitkeep`, `CONTENT.md`). If non-template files are present, ask the user which represent downloadable sources (URL-based) vs. local/manual files.

**Ask the user:**
- "What data sources does this project need?" (name, URL or description, destination path in `source_data/`)
- For each source, whether it is a direct URL download or requires manual steps.

**Implement fetch tasks** using `download_data` from airoh (see `references/airoh_api.md`):
- For each URL-based source, add an entry under `files:` in `invoke.yaml` and call `download_data(c, "name")` from the `fetch` task.
- For manual/non-URL sources, add a `print()` message in `fetch` with instructions for the user.
- If no sources are defined yet, leave `fetch` as a stub with a `# TODO` comment and a `print("TODO: no data sources defined yet")`.

**Update `source_data/CONTENT.md`** to describe each source file.

---

### Step 5 — Run tasks

**Survey first.** Scan `analysis/` for Python files and `notebooks/` for `.ipynb` files that are not part of the template (`simulation.py`, `figure_simulation.ipynb`, `summary.ipynb`). List what is found. If nothing is found, ask the user to describe the planned analysis steps.

**Infer and confirm order.** Based on file names and any imports, propose a linear execution order. Present it to the user and confirm or correct before writing any code.

**Implement one invoke task per step:**

For each step:
- If it is a Python script that processes independent "chunks" (subjects, sessions, runs, files, conditions), use the **chunk-processing pattern** (see `references/airoh_api.md`). Name the chunk concept after the actual unit (subject, file, condition, etc.).
  - Default: process all chunks, skipping those whose output already exists.
  - Accepts a comma-separated parameter to restrict to specific chunks (e.g., `subjects=None`).
  - Accepts a `smoke=False` parameter; when `True`, process only the first chunk.
- If it is a notebook step, use `airoh_run_notebooks` (see API reference); it already skips notebooks whose output folder exists.
- If it is a global aggregation or single-pass script, implement a simple task that checks for output existence and skips if already done.
- If a step has no implementation yet, create a stub task with `print("TODO: <step name>")` and a `# TODO` comment.

**Wire up `run` and `run-smoke`:**

```python
@task(pre=[fetch, step_a, step_b, run_notebooks])
def run(c):
    """Full pipeline."""
    print("Pipeline complete.")

@task
def run_smoke(c):
    """Smoke test: minimal end-to-end pass."""
    fetch(c)
    step_a(c, smoke=True)
    step_b(c, smoke=True)
    run_notebooks(c)
```

Adapt the `pre=` list and the `run_smoke` body to the actual confirmed steps. Stub tasks must not raise errors — they should print a TODO message so the smoke test still passes.

---

### Step 6 — Clean tasks

Create one clean task per output type and per source:

```python
@task
def clean_<step_name>(c):
    """Remove outputs from <step_name>."""
    clean_folder(c, "output_data_dir", "<pattern>")

@task(pre=[clean_step_a, clean_step_b, ...])
def clean(c):
    """Remove all computed outputs."""
    pass

@task
def clean_<source_name>(c):
    """Remove downloaded <source_name> data."""
    clean_folder(c, "source_data_dir", "<pattern>")

@task(pre=[clean_source_a, clean_source_b, ...])
def clean_source(c):
    """Remove all downloaded source data."""
    pass
```

Use glob patterns that match the actual output files. For stub steps with no output yet, add `print("TODO: no outputs to clean for <step>")` as the task body.

Update `output_data/CONTENT.md` to describe each output file or folder that the pipeline will create.

---

### Step 7 — Update README.md and CLAUDE.md

**README.md** is the user-facing project document. Update it to reflect:
- Project title and overview (replace the template description)
- Package manager setup instructions (only the method chosen in Step 2; remove the others)
- A brief description of each invoke task: `fetch`, each run step, `run`, `run-smoke`, `clean`, `clean_source`
- A note pointing to `source_data/CONTENT.md` and `output_data/CONTENT.md` for data asset details — do not duplicate their content inline

**CLAUDE.md** guides future Claude sessions in this project. Update it to reflect:
- Project title and purpose in the Overview section (replace the generic template description)
- Any project-specific conventions: naming scheme, chunk concept, analysis structure
- Keep all generic architecture and workflow guidance intact — do not remove or shorten existing sections

---

### Step 8 — Smoke test

Run the smoke test to verify the full pipeline is wired correctly:

```bash
# uv
uv run invoke run-smoke

# pip or conda (with env activated)
invoke run-smoke
```

If the smoke test fails:
1. Read the error carefully
2. Fix the root cause (missing import, wrong path, stub that raises instead of printing, etc.)
3. Re-run until it passes

Once the smoke test passes, report a summary to the user:
- Which tasks are fully implemented vs. still stubs
- What the user should do next (e.g., implement a specific stub, add real data sources)
