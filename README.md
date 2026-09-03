# Mario Behaviour Analysis

A reproducible pipeline for studying how players learn to complete scenes in a Mario video game task. The analysis detects learning-phase changepoints from completion-rate data (Bernoulli, measured both per scene clip and per level repetition) and clusters behavioural traces — both within individual scenes and pooled across scenes sharing a design-pattern category (e.g. "2-Horde", "Gap stair valley") — using Fréchet distance fed into HDBSCAN or k-means.

Built on [`invoke`](https://www.pyinvoke.org/) and [`airoh`](https://pypi.org/project/airoh/).

---

## Quick Start

```bash
uv sync
uv run invoke fetch
uv run invoke run
```

---

## Setup

```bash
uv sync
```

Creates a `.venv` and installs all dependencies from `pyproject.toml`.

---

## Tasks

| Task                  | Description                                                              |
| --------------------- | ------------------------------------------------------------------------ |
| `fetch`                       | Retrieve all source data assets, then record them in `source_data/MANIFEST.json` |
| `fetch-mario-scenes`          | Make the mario.scenes datalad dataset available: clone it, or `--source` an existing checkout |
| `run-changepoints`            | Detect learning-phase changepoints per subject, on two units: per scene clip and per **level repetition** (Phase annotations + a `ruptures` PELT/Binseg/Dynp diagnostic) |
| `run-changepoints-by-category`| Detect changepoints per subject **within each scene-category**, on both the `Duration` (time-to-complete) and completion signals |
| `run-clustering`              | Cluster behavioural traces per scene via Fréchet distance + **bagged** HDBSCAN (consensus over 200 resamplings) |
| `run-clustering-by-category`  | Cluster behavioural traces pooled across scenes sharing a design-pattern category (from `scenes_mastersheet.csv`) |
| `run-cluster-audit`           | Recover the parameters behind already-stored cluster files, then score how well those partitions resample |
| `run-cluster-sweep`           | Score each `(min_cluster_size, epsilon)` grid point per scene by cluster stability |
| `select-cluster-params`       | Pick one corpus-wide parameter setting from the sweeps (largest k above a stability floor) |
| `run-scheme-comparison`       | Cluster every scene under both resampling schemes (subagging vs Breiman bootstrap) and write the side-by-side |
| `run-coassociation`           | Store each scene's co-association (consensus) matrix; feeds the dashboard's Co-association view |
| `run-web-export`              | Export clustered traces to a static payload for the interactive dashboard |
| `dashboard`                   | Serve the interactive clustering dashboard on `localhost` (needs `run-web-export`) |
| `run-notebooks`               | Execute notebooks and save figures to `output_data/`                     |
| `run`                         | Full pipeline in order: fetch → changepoints → changepoints-by-category → clustering → clustering-by-category → web-export → notebooks |
| `run-smoke`                   | Minimal end-to-end pass (first subject/scene/category only) to verify pipeline wiring |
| `verify`                      | Check that the code, config, data and docs still agree (run before committing) |
| `list-scenes-task`            | Print all scene IDs, one per line (used to build SLURM array inputs)     |
| `list-categories-task`        | Print all scene-category tag names, one per line (SLURM array inputs)    |
| `clean-changepoints`          | Remove `output_data/changepoints/`                                       |
| `clean-changepoints-by-category`| Remove `output_data/changepoints_by_category/`                         |
| `clean-clustering`            | Remove `output_data/clusters/`                                           |
| `clean-clustering-by-category`| Remove `output_data/clusters_by_category/`                              |
| `clean-web-export`            | Remove `output_data/web/`                                                |
| `clean-stability`             | Remove `output_data/stability/` (audit, sweeps, selected parameters)     |
| `clean-coassociation`         | Remove `output_data/coassociation/`                                      |
| `clean-frechet`               | Remove cached Fréchet distance matrices (kept separate so re-clustering reuses them) |
| `clean`                       | Remove all computed outputs                                              |
| `clean-mario-scenes`          | Remove the cloned `source_data/mario.scenes/` dataset                    |
| `clean-source`                | Remove all downloaded source data                                        |

Use `uv run invoke --list` or `uv run invoke --help <task>` for full details.

### Running a subset

```bash
uv run invoke run-changepoints --subjects 01,02   # specific subjects
uv run invoke run-changepoints-by-category --subjects 01,02  # per-category, specific subjects
uv run invoke run-clustering --scenes w1l1s0,w2l3s4  # specific scenes
uv run invoke run-clustering-by-category --categories "2-Horde,Flagpole"  # specific categories
uv run invoke run-changepoints --rep-block-size 20  # coarser level-repetition blocks
uv run invoke run-changepoints --smoke            # first subject only
uv run invoke run-cluster-audit                  # what made the current cluster files, and do they hold up
uv run invoke run-cluster-sweep --scenes w1l1s0  # score the parameter grid for one scene
uv run invoke select-cluster-params              # choose one setting corpus-wide
uv run invoke run-clustering --smoke              # first scene only
uv run invoke run-clustering-by-category --smoke  # first category only
uv run invoke run-web-export --scenes w1l1s0      # export one scene for the dashboard
uv run invoke run-web-export --backfill           # add derived fields to existing payloads (seconds)
```

### Interactive dashboard

The PNGs in `output_data/clusters/` are fixed views. For an interactive one:

```bash
uv run invoke run-web-export     # once — bakes trace geometry out of mario.scenes (~50 MB)
uv run invoke dashboard          # serves http://localhost:8000
```

It renders the same figure as `analysis/trace_plots.py` — scene background, cluster
palette and marker shapes, per-cluster panels, per-subject composition — and adds what a
PNG cannot do: pick any scene, filter to a subject, a phase (discovery / practice), a
stage or an outcome, hover a trajectory for its clip details, and click to isolate it.

**Colour by** offers cluster, outcome, subject, **attempt #** or **stage**. The attempt
ramp is viridis over the order in which a subject played that scene, normalised per
subject — each player's own first attempt is dark and their own last is bright, so the
colour reads as time rather than as how much they played.

**Stage** cuts each phase into thirds (early / middle / late) and crosses with the Phase
filter, so Discovery + Late gives late discovery, Practice + Early gives early practice,
and so on for all six cells. The split is at the 1/3 and 2/3 `ClipCode` quantiles within
each subject and phase, matching `add_stage_column` in the `mario_learning.behavior`
repo. Because a scene lives inside a single level, the level mix is identical across the
three bins — this is a genuine repetition axis, not the difficulty-confounded
chronological binning that whole-session discovery thirds suffer from.

Filters only change which traces are **drawn**. Clustering is never re-run, and cluster
colors are pinned to the whole-scene ordering, so narrowing to one subject cannot
repaint the clusters that remain. Counts and composition percentages do update, and the
header reads "N of M traces shown" whenever a filter is active.

Filter selections carry across scene changes, so you can hold "sub-01, discovery" and
click down a list of scenes to compare the same slice. A subject who did not play the
newly selected scene is dropped from the selection rather than blanking the plot.

The exported polylines are thinned (every 3rd frame) for drawing — use `load_traces()`
for anything quantitative.

### Fetching data — clone or point at an existing copy

`mario.scenes` is ~23 GB. If it already lives somewhere on your machine, symlink it
instead of cloning a second copy:

```bash
uv run invoke fetch-mario-scenes --source /path/to/existing/mario.scenes
uv run invoke fetch --mario-scenes-source /path/to/existing/mario.scenes   # via the umbrella task
```

Each asset gets its own `fetch-{name}` task with its own `--source`; the umbrella
`fetch` routes a per-asset `--{name}-source` flag to the matching one. A `source:` key
under `datasets:` in `invoke.yaml` makes it the default for that machine.

A linked checkout is read exactly as found — no branch checkout, no `datalad get`, no
decompression, since all three write into a dataset this project does not own. Fetch
also never overwrites what already sits at the destination, so re-pointing a stale link
means `invoke clean-mario-scenes` first (that unlinks the symlink and leaves the
external checkout alone).

`fetch` then writes `source_data/MANIFEST.json`, and `run` writes
`output_data/PROVENANCE.json`. Between them they record what each input actually
resolved to — including the commit of a symlinked external checkout — and what produced
the current outputs. `PROVENANCE.json` changes on every run; that is the record working,
not churn to suppress.

### Verification

```bash
uv run invoke verify              # fails on any FAIL
uv run invoke verify --strict     # warnings count as failures too
uv run invoke verify --skip lint
```

Checks that the code, config, data and docs still agree: the task list against this
README and `CLAUDE.md`, `pyproject.toml` against `requirements.txt`, every path named in
the docs, each data folder against its `CONTENT.md`, config keys against `tasks.py`,
tracked file sizes, and provenance freshness. Configure it under `verify:` in
`invoke.yaml`.

Run it before committing. It is deliberately **not** part of `run` — reproducing results
should never depend on documentation hygiene.

### Running on a SLURM cluster

Per-category clustering pools traces across every scene sharing a tag, so cost scales
quadratically with pool size (e.g. "Enemy" pools ~19,000 traces vs. ~150 for the smallest
category) — see `slurm_clustering.sh` (per scene) and `slurm_clustering_by_category.sh`
(per category) for array-job templates, driven by `invoke list-scenes-task` /
`invoke list-categories-task`.

---

## Data

- **Source data:** see [`source_data/CONTENT.md`](source_data/CONTENT.md)
- **Output data:** see [`output_data/CONTENT.md`](output_data/CONTENT.md)

---

## Design principles

- **Analysis in code, visualization in notebooks.** Heavy computation lives in `analysis/` and is run by `invoke` tasks. Notebooks only read results and produce figures. Trace-figure rendering lives in `analysis/trace_plots.py` (imported by `notebooks/figure_clustering.ipynb`): it plots each scene's trajectories with a screen-corrected vertical coordinate and can color traces by cluster (default) or by outcome (green = completed, red = death), set via `COLOR_BY` in the notebook. A `plot_scene_cluster_panels()` view splits a scene into one subplot per cluster (shared axes) in a single figure for reading individual cluster shapes.
- **Aggregate to the unit that carries the signal.** Completion is measured at two units. Per **scene clip** — a few seconds of play — the rate is ~92%: a ceiling, flat, and unsegmentable by any changepoint algorithm. Per **level repetition** (one `.bk2` playthrough) it is 53–81% depending on subject, and `ruptures` recovers the annotated discovery→practice boundary to within 0–14 repetitions for every subject. Two nested criteria are reported: `cleared` (reached the level's end, deaths allowed) and `flawless` (cleared without dying once, 22–54%). A flat signal is usually the wrong unit, not the wrong detector.
- **Keep the time axis honest; adjust for difficulty separately.** During discovery each level is played once, in world order, so "later" and "harder" are the same thing and each third of discovery contains a different set of levels — the raw cleared rate falls (77% → 33%) as a pure difficulty gradient, and **no within-discovery time comparison is valid, adjusted or not** (what looks like a rise after adjustment is a floor effect: the hard late-discovery levels have no headroom). `level_adjusted_series()` expresses each repetition relative to that level's own session-wide average, removing difficulty while preserving true chronological order; subjects sit ~5–12 points below their level average in discovery and ~2–6 above it in practice. **Practice is where the time course is interpretable** — every third contains all 22 levels, and all five subjects rise monotonically. Use the adjusted signal for reporting effect sizes and the raw one for detection.
- **Two baseline checks, and they cover different failures.** `invoke run-smoke` is the behavioural one — does the pipeline run end to end and produce something. `invoke verify` is the structural one — do the code, config, data and docs still describe the same project. Run both before committing; neither substitutes for the other.
- **Gathering assets is a separate job from reproducing results.** `fetch` retrieves; `run` reads what is already on disk and never pulls. A step that finds its input missing says so and points at `invoke fetch` rather than fixing it silently.
- **Idempotent steps.** Each task checks whether its outputs already exist and skips if they do. Call `invoke clean` then `invoke run` to force a full rerun.
- **Chunked for reruns.** `run-changepoints` and `run-changepoints-by-category` process one subject at a time (`--subjects`); `run-clustering` chunks by scene (`--scenes`), `run-clustering-by-category` by category (`--categories`), and `run-web-export` by scene (`--scenes`). Any piece can be restricted or retried in isolation.
- **Clustering is bagged, and it records its own settings.** `run-clustering` defaults to `method=consensus`: HDBSCAN is refit on 200 random 80% subsamples, a co-association matrix records how often each pair of traces landed together, and the final labels come from clustering that matrix. The resampling defaults to **subagging** (subsample without replacement) rather than a Breiman bootstrap. Both are implemented — `--scheme bootstrap` gives classical bagging — and `invoke run-scheme-comparison` runs the corpus under each. Over 308 scenes the two **agree about the clustering** (median ARI 0.97) but not about how much to trust it: a bootstrap replicate holds only ~63% unique points, so its replicates are roughly twice as variable (mean sd of k 1.10 vs 0.61) and it reports a lower stability for the same partition (trace-weighted 73.9% vs 81.6%). That is a property of the estimator, not of the data. A cluster therefore has to survive resampling to exist at all. Each output carries a `params` block and a `stability` report (per-cluster bootstrap Jaccard, per-trace confidence). Earlier files recorded no parameters, and an audit of 308 scenes recovered **14 different settings** in use — `run-cluster-audit` is what reconstructs them for any file predating `params`.
- **Parameters are chosen on evidence, not by hand.** `run-cluster-sweep` scores a 12-point grid per scene by what share of its traces sit in a cluster that reproduces; `select-cluster-params` aggregates trace-weighted and takes the **largest k that still clears an 80% stability floor**. Maximising stability alone would pick the degenerate two-cluster collapse, so resolution is preferred and only traded away when the finer partition genuinely does not survive.
- **Export once, serve statically.** The dashboard reads only `output_data/web/` — trace geometry and background images are baked there by `run-web-export`, so nothing at view time touches the 23 GB `mario.scenes` dataset. Re-export after re-clustering; `--force` overwrites existing payloads, and `--backfill` adds fields derivable from what the payload already holds without reading the source dataset at all.
- **Background-ready trace plots.** Scene trajectory figures overlay a scene background image when one is present at `source_data/mario.scenes/sourcedata/scene_backgrounds/<scene_id>.png` (set `BACKGROUND_DIR` in the notebook, or `None` to disable). These images are **not** part of the dataset and must be regenerated separately (see `source_data/mario.scenes/code/deprecated/make_images/make_scene_img.py`, which needs `stable_retro` + the source `mario` `.bk2` dataset). Until they exist, plots fall back automatically to a plain y-flipped axis.
- **AI-native.** Initialized and extended with Claude Code — `CLAUDE.md` gives future sessions full context on the pipeline.
