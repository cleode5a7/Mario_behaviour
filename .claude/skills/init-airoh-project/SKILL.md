---
name: init-airoh-project
description: This skill should be used when initializing a new reproducible analysis project from the airoh-mini template. It guides the user through specifying project metadata, selecting a package manager, implementing fetch/run/clean invoke tasks, updating documentation, and running a smoke test to verify the full pipeline.
---

# Init Airoh Project

## Overview

Wire up the `airoh-mini` template into a working, reproducible analysis pipeline. Walk the user through project metadata, package manager choice, fetch/run/clean task implementation, documentation updates, and a final smoke test. Work incrementally — check for existing scripts and notebooks first, ask only what's needed, and produce placeholder stubs when work is deferred.

Read `.claude/skills/init-airoh-project/references/airoh_api.md` before writing any code.

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

Each package manager gets exactly one native dependency file — keeping more than one per manager is how three hand-synced lists drift out of sync:

| File | uv | pip | conda |
|---|---|---|---|
| `pyproject.toml` | keep, update `name` field | delete | delete |
| `uv.lock` | keep | delete | delete |
| `requirements.txt` | delete | keep | delete |
| `environment.yml` | delete | delete | keep, set `name:` to a slug of the project title |

For **uv**: update `pyproject.toml` — set `name` to a slug of the project title, keep all dependencies.

For **pip**: `requirements.txt` is the only dependency file; no changes needed unless the user wants to add packages.

For **conda**: `environment.yml` is the only dependency file — it must list every dependency natively (not defer to `requirements.txt`, which is deleted).

Check that every key in `invoke.yaml` is one the project actually reads — `invoke verify` reports unused keys, and a template vestige nobody reads is exactly the kind of drift this project is trying to avoid.

---

### Step 2b — Tooling choices

Ask the user to choose a **linter** and a **test framework**. Present the options clearly; both are optional.

**Linter options:**
- `ruff` (recommended) — fast, modern; configured in `pyproject.toml` for a **uv** project, or in `ruff.toml` for **pip**/**conda** (which delete `pyproject.toml` in Step 2)
- `flake8` — classic, configured in `setup.cfg`
- none

**Test framework options:**
- `pytest` (recommended)
- none

**If ruff chosen:**
- Add `ruff` to the project dependencies
- **uv**: add to `pyproject.toml`:
  ```toml
  [tool.ruff]
  line-length = 100

  [tool.ruff.lint]
  select = ["E", "F", "W", "I"]
  ```
- **pip/conda**: create `ruff.toml` (same content, without the `[tool.ruff]`/`[tool.ruff.lint]` table nesting — top-level `[lint]` instead):
  ```toml
  line-length = 100

  [lint]
  select = ["E", "F", "W", "I"]
  ```

**If flake8 chosen:**
- Add `flake8` to the project dependencies
- Create `setup.cfg`:
  ```ini
  [flake8]
  max-line-length = 100
  ```

**If pytest chosen:**
- Add `pytest` to the project dependencies
- Create `tests/__init__.py` (empty)
- Create `tests/test_smoke.py` with a stub:
  ```python
  def test_placeholder():
      """Replace with real unit tests for analysis/ functions."""
      pass
  ```

The README and CLAUDE.md updates in Step 7 should include the linter command and `pytest` invocation if applicable.

---

### Step 3 — Clean template artifacts

**Preserve the template's `tasks.py` scaffolding — edit it incrementally, never rewrite it from scratch.** `verify`, `record_sources` (inside `fetch`), and `record_run` (inside `run`) are not demo code; deleting or reimplementing `tasks.py` wholesale routinely drops them by accident, which silently breaks provenance recording and `invoke verify`.

Remove the demo code that ships with the template:

- In `tasks.py`: delete the `run_simulation` task, remove it from `run_notebooks`'s `pre=` list, and remove its call from the bodies of `run` and `run-smoke` (steps are called directly in those bodies, not via `pre=` — see CLAUDE.md's `pre=` warning)
- Delete `analysis/simulation.py`; if `analysis/` is now empty (only `__init__.py`), keep it — the user will populate it
- In `invoke.yaml` under `files:`: delete the `papers:` entry (the demo download)
- The template also ships a worked example of the **Inkscape montage pattern** (`airoh.figures` — see the API reference). Ask the user whether this project assembles a multi-panel figure by hand in Inkscape:
  - **No** (the common case): delete `output_data/figure_montage.svg`, the `figures:` block in `invoke.yaml`, and the `run_figure_layout`/`compose_figure`/`clean_figure` tasks in `tasks.py` — including their calls in the bodies of `run`, `run-smoke` and `clean`, and `run_figure_layout` in `run_notebooks`'s `pre=`. Also drop the `montage_dpi` helper and the `FIGURE_MONTAGE_DPI` export from `run_notebooks`. Keep `figures_dir` and `clean_figures`: those are the plain notebook-output convention, not part of the montage pattern.
  - **Yes**: keep the tasks and the `figures:` block, but replace the demo — the user authors their own montage SVG under `output_data/`, and the entry under `figures:` is renamed to match it.
- If a fresh project still has `.datalad/` (a pre-fix instantiation inherited the template's own datalad dataset identity), remove it: `git rm -r .datalad` and drop any `annex.*`/`datalad` lines from `.gitattributes`
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

After cleanup, `tasks.py` should still contain every permanent task — `fetch`, `run`, `run-smoke`, `verify`, `clean`, `clean-figures` — plus a `fetch-{name}` per remaining asset, and `run-figure-layout`/`compose-figure`/`clean-figure` if the project kept the montage pattern. Only the demo-specific `run_simulation`/`clean_simulation` and the `papers` fetch task are gone (or reduced to stubs) until Steps 4–6 fill them back in.

---

### Step 4 — Fetch tasks

**Survey first.** Check `source_data/` for any files that are not part of the template (i.e., not `.gitkeep`, `CONTENT.md`). If non-template files are present, ask the user which represent downloadable sources (URL-based) vs. local/manual files.

**Ask the user:**
- "What data sources does this project need?" (name, URL or description, destination path in `source_data/`)
- For each source, whether it is a direct URL download, data already present on disk (symlink), or requires manual steps.
- **"Is any of this data sensitive, restricted, or identifiable?"** and **"Does retrieving it need credentials that a new collaborator might not have?"**

Those last two change what you write, not just what you say. If the answer to either is yes:
- Keep `source_data/` gitignored and add a guard line for the file types that would carry the sensitive content, with a comment saying why (see CLAUDE.md, "Sensitive and restricted data").
- Make retrieval tolerant: content behind a credentialed remote fails per-file for anyone without access, and a fetch that aborts on the first failure is useless to them. Warn and continue.
- **Record the access requirements in `source_data/CONTENT.md`**: who can obtain this data, how, and what someone without access will see. A collaborator whose `fetch` came back empty must be able to tell "I lack permission" from "the pipeline is broken".
- Ask whether derived outputs inherit that sensitivity. Aggregate tables are usually safe to track in git; per-participant ones usually are not.

**Implement fetch tasks** using `fetch_data` from `airoh.acquisition` (see `.claude/skills/init-airoh-project/references/airoh_api.md`):
- For each source, add an entry under `files:` in `invoke.yaml`, with a `url` to download and/or a `source` to symlink.
- Give each asset its own `fetch-{name}` task with a plain `--source`/`--copy`, and have the umbrella `fetch` call every `fetch-{name}`, routing a per-asset `--{name}-source` flag to the matching one. `fetch_data`'s `source` is a single path bound to a single asset — never a root directory combined with each asset's filename — which is exactly why each asset needs its own task and its own flag:

  ```python
  @task(help={"source": "Existing 'papers' data to link instead of downloading.",
              "copy": "Copy instead of symlinking."})
  def fetch_papers(c, source=None, copy=False):
      fetch_data(c, "papers", source=source, copy=copy)

  @task(help={"source": "Existing 'atlas' data to link instead of downloading.",
              "copy": "Copy instead of symlinking."})
  def fetch_atlas(c, source=None, copy=False):
      fetch_data(c, "atlas", source=source, copy=copy)

  @task(help={"papers_source": "Source path for the papers asset.",
              "atlas_source":  "Source path for the atlas asset."})
  def fetch(c, papers_source=None, atlas_source=None, copy=False):
      fetch_papers(c, source=papers_source, copy=copy)
      fetch_atlas(c, source=atlas_source, copy=copy)
  ```

  Never forward one shared `source` to several `fetch_data` calls: every asset would link to that same path, and it fails silently — `fetch_data` only checks that the source exists, so it prints a success line per asset and exits 0, leaving several differently-named symlinks pointing at one file. (For a single-asset project, `fetch` and `fetch_{name}` collapse into one task with a plain `--source`, as in the template.)
- **Datalad datasets are not `--source` material.** `--source` symlinks or copies a plain file/folder and does not run `datalad get`, so a symlinked datalad dataset exposes only already-present content (un-fetched files are broken symlinks) and `--copy` raises on those un-fetched files. Use `ensure_submodule` from `airoh.acquisition` for a plain git submodule instead.

  For an actual datalad dataset, add a `datasets:` entry in `invoke.yaml` (`output_dir`, plus `url` and/or `source` — same shape as `files:`) and wire a `fetch-{name}` task around `airoh.datalad`:
  ```python
  @task(help={"source": "Existing checkout to symlink instead of cloning."})
  def fetch_cneuromod(c, source=None):
      from airoh.datalad import install_dataset, get_data
      install_dataset(c, "cneuromod", source=source)
      get_data(c, "cneuromod")  # or path="…" to narrow; add strict=True in run-smoke
  ```
  `install_dataset` makes the checkout available (symlink an existing one, or `datalad clone` from `url`); `get_data` retrieves content and is tolerant of partial failures by default (some content lives on credentialed remotes a given environment may not reach — pass `strict=True` only where that must fail loudly, e.g. `run-smoke`). See `.claude/skills/init-airoh-project/references/airoh_api.md` for the full `airoh.datalad` API, including `prefetch_pattern`/`load_known_failures`/`save_known_failures` for a project that needs to fetch a large set of small files up front rather than the whole dataset.
- For manual/non-URL sources, add a `print()` message in `fetch` with instructions for the user.
- If no sources are defined yet, leave `fetch` as a stub with a `# TODO` comment and a `print("TODO: no data sources defined yet")`.
- **Keep the `record_sources(c)` call at the end of `fetch`.** It writes `source_data/MANIFEST.json` — what each asset actually resolved to, including the commit of a symlinked external checkout. Without datalad this is the only record of which inputs a result came from. It never raises, so it cannot break a fetch.

**Update `source_data/CONTENT.md`** to describe each source file, and the access requirements if there are any.

---

### Step 5 — Run tasks

**Fetch retrieves, run reads.** No `run-{name}` task may pull data — not a download, not a `datalad get`, nothing network-facing. A run step that finds its input missing must error and point at `invoke fetch`, never fetch it silently. This split is what keeps `run` fast, offline-capable, and honest about what it depends on: a `run` that quietly re-fetches on demand is slow in a way nobody can diagnose, and hides that a result was produced from data that arrived halfway through.

**Survey first.** Scan `analysis/` for Python files and `notebooks/` for `.ipynb` files that are not part of the template (`simulation.py`, `figure_simulation.ipynb`, `summary.ipynb`). List what is found. If nothing is found, ask the user to describe the planned analysis steps.

**Infer and confirm order.** Based on file names and any imports, propose a linear execution order. Present it to the user and confirm or correct before writing any code.

**Implement one invoke task per step:**

For each step:
- If it is a Python script that processes independent "chunks" (subjects, sessions, runs, files, conditions), use the **chunk-processing pattern** (see `.claude/skills/init-airoh-project/references/airoh_api.md`). Name the chunk concept after the actual unit (subject, file, condition, etc.).
  - Default: process all chunks, skipping those whose output already exists.
  - Accepts a comma-separated parameter to restrict to specific chunks (e.g., `subjects=None`).
  - Accepts a `smoke=False` parameter; when `True`, process only the first chunk.
- If it is a notebook step, use `airoh_run_notebooks` (see API reference); it already skips notebooks whose output folder exists. **The notebook must write into `{figures_base}/{its own stem}/`** — that folder is the "already ran" marker. A notebook writing anywhere else re-runs on every single `invoke run`.
- If it is a global aggregation or single-pass script, implement a simple task that checks for output existence and skips if already done.
- If a step has no implementation yet, create a stub task with `print("TODO: <step name>")` and a `# TODO` comment.

**Wire up `run` and `run-smoke`:**

```python
@task(help={"force": "Delete every computed output first, then run from scratch."})
def run(c, force=False):
    """Full pipeline."""
    from airoh.provenance import record_run

    if force:
        clean(c)
    step_a(c)
    step_b(c)
    run_notebooks(c)
    record_run(c, tasks="step-a,step-b,run-notebooks")
    print("Pipeline complete.")

@task
def run_smoke(c):
    """Smoke test: minimal end-to-end pass."""
    fetch(c)
    step_a(c, smoke=True)
    step_b(c, smoke=True)
    run_notebooks(c)
```

Adapt both bodies to the actual confirmed steps. Stub tasks must not raise errors — they should print a TODO message so the smoke test still passes.

Three things to get right here:

- **Call the steps in the body, not through `pre=`.** A `pre=` chain runs before the body, so a `--force` (or `--smoke`, or a chunk selector) declared on `run` would arrive too late to reach any of them. `pre=` is still fine on a task that is only ever a command-line entry point.
- **`--force` means clean-then-run.** Caching is by output existence, so an edited script does not invalidate anything; `--force` is the documented way out. Do not build content-hash invalidation or a dependency graph — granular `clean-{name}` tasks plus one `--force` is the whole intended cache story.
- **Keep `record_run(c, ...)` last.** It writes `output_data/PROVENANCE.json` — project commit, environment, inputs consumed, output checksums. It never raises.

---

### Step 6 — Clean tasks

Create one clean task per output type and per source:

```python
@task
def clean_<step_name>(c):
    """Remove outputs from <step_name>."""
    clean_folder(c, "output_data_dir", "<pattern>")

@task
def clean(c):
    """Remove all computed outputs."""
    clean_step_a(c)
    clean_step_b(c)

@task
def clean_<source_name>(c):
    """Remove downloaded <source_name> data."""
    clean_folder(c, "source_data_dir", "<pattern>")

@task
def clean_source(c):
    """Remove all downloaded source data."""
    clean_source_a(c)
    clean_source_b(c)
```

Use glob patterns that match the actual output files. For stub steps with no output yet, add `print("TODO: no outputs to clean for <step>")` as the task body.

**The umbrella clean tasks must do their work in the body, not in `pre=`.** `run --force` calls `clean(c)` as a plain function, and a `pre=` chain does not fire that way — the body would run alone, delete nothing, and print a success message. This failure is completely silent, which is why it is worth getting right the first time.

**Clean the notebook markers too.** If a notebook's output folder survives a clean, the next `run` skips that notebook even though its figures are gone. A `clean-figures` task should remove `{figures_base}/{stem}/` for every notebook, not just the image files.

Update `output_data/CONTENT.md` to describe each output file or folder that the pipeline will create.

---

### Step 7 — Update README.md and CLAUDE.md

**README.md** is the user-facing project document. Update it to reflect:
- Project title and overview (replace the template description)
- Package manager setup instructions (only the method chosen in Step 2; remove the others — the template ships a three-way `uv`/`pip`/`conda` block precisely so it can serve any choice, but a finished project only needs one)
- A brief description of each invoke task: `fetch`, each run step, `run`, `run-smoke`, `verify`, `clean`, `clean_source`. The task list must match `invoke --list` exactly — `invoke verify` checks this, and will fail the project otherwise.
- `invoke run --force` as the way to rebuild after editing something, since caching is by output existence
- Any dependency file mentioned that was deleted in Step 2 (e.g. `requirements.txt` for a uv/conda project) — remove the reference along with the file
- A note pointing to `source_data/CONTENT.md` and `output_data/CONTENT.md` for data asset details — do not duplicate their content inline

**CLAUDE.md** guides future Claude sessions in this project. Update it to reflect:
- Project title and purpose in the Overview section (replace the generic template description)
- The `## Setup` block: prune to the one package manager chosen in Step 2, same as README.md
- Any project-specific conventions: naming scheme, chunk concept, analysis structure
- Any dependency file mentioned that was deleted in Step 2
- Keep all generic architecture and workflow guidance intact — do not remove or shorten existing sections

---

### Step 8 — Verify

Check the project against its own documentation:

```bash
# uv
uv run invoke verify

# pip or conda (with env activated)
invoke verify
```

This compares the task list against the README, the dependency files against each other, the paths named in the docs, each data folder against its `CONTENT.md`, the config keys, what git tracks, and the linter. Resolve **every** finding before moving on:

- A `FAIL` is real drift. Fix the code or fix the doc — do not silence the check.
- A `WARN` about data paths that do not exist yet is expected before the first `fetch`/`run`; anything else deserves a look.
- Only add a name to `verify.skip_checks` or `verify.ignore_paths` in `invoke.yaml` when the check genuinely does not apply to this project, and say why in a comment.

A freshly initialized project should end this step fully green.

---

### Step 9 — Smoke test

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
