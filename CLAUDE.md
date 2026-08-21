# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**Mario Behaviour Analysis** — a reproducible pipeline for studying how players learn to complete scenes in a Mario video game task. The analysis detects learning-phase changepoints from per-scene completion-rate data (Bernoulli) and clusters behavioural traces — both within individual scenes and pooled across scenes sharing a design-pattern category — using Fréchet distance fed into HDBSCAN or k-means. Results are rendered both as matplotlib PNGs and as an interactive browser dashboard served from a static export.

Built on the [`invoke`](https://www.pyinvoke.org/) task runner. The `airoh` pip package provides reusable invoke tasks; this repo customizes them via `tasks.py` and `invoke.yaml`.

## Persona

Respond as Uncle Airoh: patient, warm, and wise. Assume the user may be new to coding. Explain errors gently, encourage before correcting, and frame tradeoffs as learning opportunities. When things get heated, offer a calming cup of jasmine tea.

## Setup

```bash
uv sync
```

## Common Commands

With `uv`:
```bash
uv run invoke fetch           # Download source data
uv run invoke run             # Full pipeline (project-specific pre= chain)
uv run invoke run-notebooks   # Execute notebooks, save figures to output_data/
uv run invoke run-web-export  # Bake trace geometry into output_data/web/ for the dashboard
uv run invoke dashboard       # Serve the interactive clustering dashboard on localhost
uv run invoke clean           # Remove output_data/ contents
uv run invoke --list          # Show all available tasks
```

Without `uv` (activate your environment first):
```bash
invoke fetch              # Download source data (configured in invoke.yaml under files:)
invoke run                # Full pipeline (project-specific pre= chain)
invoke run-notebooks      # Execute notebooks, save figures to output_data/
invoke clean              # Remove output_data/ contents
invoke --list             # Show all available tasks
```

## Architecture

**Always read `tasks.py` first** before proposing or implementing any pipeline change — it is the authoritative source of what tasks exist, how they are wired, and what parameters they accept.

**Execution flow:** `invoke run` triggers the project's analysis pipeline via `pre=` dependencies declared in `tasks.py`. The three permanent tasks — `fetch`, `run`, `clean` — are always present; intermediate steps are project-specific.

- `invoke.yaml` — all path and data config (`output_data_dir`, `source_data_dir`, `notebooks_dir`, `files:` for downloads)
- `tasks.py` — project-specific invoke tasks; imports reusable tasks from `airoh.utils`
- `analysis/` — pure Python analysis logic, called by tasks in `tasks.py`
- `web/` — source of the interactive dashboard (`index.html`, `app.js`, `style.css`); no build step, no dependencies. `invoke dashboard` copies these into `output_data/web/` and serves that directory
- `notebooks/` — Jupyter notebooks executed by `run_notebooks` via `airoh.utils.run_notebooks`; notebooks receive `OUTPUT_DATA_DIR` and `SOURCE_DATA_DIR` as environment variables
- `source_data/CONTENT.md` and `output_data/CONTENT.md` — authoritative docs for what each data folder contains; update these when data assets change, do not duplicate their content elsewhere

**Analysis vs. notebooks:** Heavy computation belongs in `analysis/` Python code, invoked by `run-{name}` tasks, which write results to `output_data/`. Notebooks are for visualization only — they read from `output_data/` and produce figures. This keeps notebooks fast and focused.

**Idempotent tasks:** Each `run-{name}` task must check whether its outputs already exist and skip execution if they do. This means `invoke run` can be called repeatedly during development of a later step — earlier steps are skipped automatically. To force a full rerun, call `invoke clean` first, then `invoke run`.

**Task naming conventions:**
- Analysis tasks are named `run-{name}` (e.g. `run-preprocessing`, `run-model`).
- Cleaning tasks mirror them: `clean-{name}` removes only the outputs of the corresponding step.
- The top-level `clean` task calls all `clean-{name}` tasks in sequence.
- The top-level `run` task wires all steps together via `pre=` chains in `tasks.py`.

**Task parameters:** `run-{name}` tasks should expose chunk or subset parameters (e.g. a subject ID, a chunk index) so that individual pieces can be rerun in isolation. They should also support a `smoke` flag for a fast minimal run useful for testing the pipeline end-to-end without running the full analysis.

**Project-specific conventions:**
- **Changepoints chunk by subject**: `run-changepoints` accepts `--subjects` (comma-separated subject IDs, e.g. `01,02`) and `--smoke` (first subject only). It loads each subject's full chronological attempt sequence across all scenes. Also exposes `--beta` (ruptures penalty scale) and `--n-bkps` (Dynp changepoint count; defaults to the subject's Phase-transition count).
- **Changepoints-by-category also chunk by subject**: `run-changepoints-by-category` accepts `--subjects`, `--smoke`, `--beta`, `--n-bkps`. For each subject it restricts the attempt sequence to scenes sharing each design-pattern category (reusing `load_scene_patterns`/`list_categories` from `analysis/clustering.py`) and runs the ruptures detector on **both** the `Duration` (time-to-complete) and completion signals — the pooled-completion signal is near-ceiling and flat, so `Duration` is the signal that actually reveals learning. Chunked by subject (not by category, unlike clustering) because changepoint cost is cheap and the subject's chronological order is the natural unit; one output file per subject holds all their categories.
- **Clustering chunks by scene**: `run-clustering` accepts `--scenes` (comma-separated scene IDs, e.g. `w1l1s0,w2l3s4`) and `--smoke` (first scene only).
- **Clustering-by-category chunks by category**: `run-clustering-by-category` accepts `--categories` (comma-separated pattern tags, e.g. `"2-Horde,Flagpole"`) and `--smoke` (first category only). It pools traces from every scene sharing a tag, re-based to a common origin (each trace shifted to start at (0,0)) before computing Fréchet distance — otherwise distance would be dominated by each scene's absolute level position rather than trace shape. Cost scales with the *square* of the pooled trace count, which varies enormously by category (e.g. "Enemy" pools ~19,000 traces across 90 scenes vs. ~150 for the smallest category) — see `slurm_clustering_by_category.sh` for a SLURM array template and size resources per-category accordingly.
- Source data lives in `source_data/mario.scenes/` (a datalad dataset, git-ignored). `invoke fetch` clones it, checks out `dev_refactor`, runs `datalad get`, and decompresses archives via `code/archives/decompress.py`.
- `source_data/scenes_mastersheet.csv` (checked into Git, from Zenodo record 15586709) tags each scene with one or more design-pattern categories (28 binary columns, not mutually exclusive) — see `source_data/CONTENT.md`.
- Changepoint inputs: all `*_summary.json` files under `sub-{subject}/`, sorted by (Session, Run, StartFrame). Key field: `Outcome` (`"completed"` or `"death"`) → Bernoulli series.
- Clustering inputs: all `*_variables.json` files for a given scene (or, for clustering-by-category, for every scene tagged with that category). Trajectory built from `player_x_posHi × 256 + player_x_posLo` and `player_y_pos`.
- `analysis/changepoints.py` — changepoint detection on per-subject attempt series. `phase_changepoints()` (primary) reads the experimenters' `Phase` annotations; `bocd_pcp_curve()` is an offline **Bayesian** (BetaBinomial) diagnostic; `ruptures_changepoints()` wraps the generic `_ruptures_detect()`, a **non-Bayesian** detector running `ruptures` PELT/Binseg/Dynp with a **CostL2** mean-shift model on both the raw series and a block-averaged series (PELT/Binseg in penalty mode with an adaptive per-signal penalty `beta*log(n)*var`, `jump=1` for full resolution; Dynp in fixed-`n_bkps` mode). `_ruptures_detect` takes any 1-D signal, so it serves both the per-subject completion diagnostic and the per-category `Duration`+completion analysis. `category_attempt_series()` splits a subject's records by design-pattern category (a scene with multiple tags lands in each). All breakpoints are returned in **attempt-index space**.
- `analysis/clustering.py` — pairwise Fréchet distance matrix → HDBSCAN or k-means clustering, shared by `run_clustering` (per scene) and `run_clustering_by_category` (per scene category) via `cluster_traces()`. See TODOs in file. `load_traces(scene_id, source_dir, y_mode=...)` builds trajectories; `y_mode="raw"` (default) is what clustering distances/labels are computed on — **do not change the default**, existing results depend on it. Traces are ordered by a **unique** key `(Subject, Session, Run, StartFrame, ClipCode)` — `(Subject, Session, Run)` alone is not unique (a scene has many clips per run), so a weaker key would leave ties broken by non-reproducible filesystem order and misalign the positionally-stored labels. `y_mode="screen"` is a plotting-only vertical coordinate (`player_y_pos + 24 - (1 - player_y_screen)*256`, masked where `player_state == 0`) that stacks the two vertical NES screens into one continuous axis and hides dead frames; trace discovery/order is identical across modes so cluster labels still map 1:1.
- `analysis/trace_plots.py` — trajectory-figure rendering (pure visualization), imported by `notebooks/figure_clustering.ipynb`. `load_labeled_traces()` re-joins stored cluster labels to freshly-loaded traces **by clip key** (via `CLIP_KEY_FIELDS`), never by position, so plots stay correct even if the load order differs from when clustering ran (e.g. cluster-computed on SLURM, plotted locally). `plot_scene_clusters()` draws one scene's traces (screen-corrected y), coloring by cluster (`color_by="cluster"`, default) or by outcome (`color_by="outcome"`; green = completed, red = death). If a `<scene_id>.png` background exists under `background_dir` (default `source_data/mario.scenes/sourcedata/scene_backgrounds/`) it is drawn with `imshow`, the y-axis is flipped in image space, and x is offset by the scene's mastersheet `Entry point`; otherwise it falls back to a plain data axis whose vertical range is clipped to the playfield via `playfield_ylim()` (percentile + few-screens cap) — a pit death is a *real* multi-screen downward fall in the screen-corrected coordinate, and clipping lets those honest-but-huge plunges run off the bottom edge instead of dominating the axis. `plot_scene_cluster_panels()` renders the same scene as one-subplot-per-cluster small multiples (shared axes) in a single figure, so individual cluster shapes are easy to compare. Background images are **not** in the dataset and must be regenerated (see `mario.scenes/code/deprecated/make_images/make_scene_img.py` — needs `stable_retro` + the source `mario` `.bk2` dataset); the deferred overlay lights up automatically once the PNGs exist.
- **Dashboard export chunks by scene**: `run-web-export` accepts `--scenes`, `--smoke`, `--step` (frame decimation, default 3) and `--force`. It exports only scenes that already have a `<scene>_clusters.json`, so it depends on `run-clustering` — unlike the other run tasks it is **not** independent. `invoke dashboard` serves the result.
- **Export once, serve statically.** The dashboard must never read `source_data/mario.scenes/` at runtime (it is 23 GB and usually a symlink outside the repo). `analysis/webexport.py` bakes the trace geometry *and* copies the background PNGs into `output_data/web/data/`, which is the only tree the browser sees. Anything the dashboard needs must go through the export.
- **The dashboard is a view, never a re-analysis.** Its subject/phase/outcome filters change only which traces are drawn. Cluster labels come from the single HDBSCAN run over all of a scene's traces, and the colour slot for each cluster is fixed by the **whole-scene** rank-by-size order — never the filtered order — so narrowing to one subject cannot repaint the clusters that survive. Only the `n=` counts and composition percentages recompute; the header then reads "N of M traces shown". If a future change makes a filter alter cluster identity, that is a bug, not a feature.
- **The exported payload is for drawing, not for numbers.** `run-web-export` keeps every `step`-th frame (default 3) and rounds coordinates to integers. Correct at 224 px tall, wrong for any measurement — quantitative work goes back through `load_traces()`.
- `analysis/webexport.py` — bakes the clustering results into the static payload the dashboard reads. `export_scene()` pulls geometry and labels through `load_labeled_traces()` (so the join stays by clip key, and the y coordinate stays screen-corrected), applies the mastersheet `Entry point` offset **to the exported coordinates themselves** when a background exists — so `x`, `xlim` and `bg_size` all live in one space and the browser does no arithmetic — decimates by `step`, and writes masked frames as `null` so the renderer breaks the polyline instead of interpolating across dead frames. `write_index()` rebuilds `index.json` from the exported files on every run so the index cannot drift; `sync_backgrounds()` copies the scene PNGs in.
- `web/app.js` — the dashboard renderer. Palettes and the alpha/linewidth pairs are copied **verbatim** from `analysis/trace_plots.py`; if you change them there, change them here or the dashboard and the PNGs stop being the same figure. `sceneRanking()` (whole-scene, fixed colours) is deliberately separate from `visibleCounts()` (filtered, live labels). Hit-testing is a geometric nearest-point search in screen space rather than a colour picking-buffer, so an antialiased pixel can never decode to the wrong trace.
- `run-changepoints`, `run-changepoints-by-category`, `run-clustering`, and `run-clustering-by-category` are **independent** — they can be run in any order or in isolation. `run-web-export` runs after clustering.

**Adding a new analysis step:** add a function to `analysis/`, add a `run-{name}` task and a matching `clean-{name}` task in `tasks.py`, wire both into the top-level `run` and `clean` tasks via `pre=` chains, and create or extend a notebook in `notebooks/` for visualization.

**Evolving CLAUDE.md:** Keep this file current as the project grows. It should always reflect the actual scope of the project — what it does, what data it uses, and what analysis steps it contains. When adding or removing a task, rename a folder, or change the pipeline structure, update CLAUDE.md in the same commit. Stale guidance here misleads future AI sessions and collaborators alike.

**Keeping README.md current:** README.md is the user-facing documentation for this project. Any structural or workflow change — new tasks, renamed folders, updated commands, new dependencies — must be reflected there in the same commit. The task list in README.md should match `invoke --list` exactly; if a task is added or removed, update README.md accordingly. For data folder contents, point to `source_data/CONTENT.md` and `output_data/CONTENT.md` rather than duplicating their content inline.
