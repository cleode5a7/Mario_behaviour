# Mario Behaviour Analysis

A reproducible pipeline for studying how players learn to complete scenes in a Mario video game task. The analysis detects learning-phase changepoints from per-scene completion-rate data (Bernoulli) and clusters behavioural traces — both within individual scenes and pooled across scenes sharing a design-pattern category (e.g. "2-Horde", "Gap stair valley") — using Fréchet distance fed into HDBSCAN or k-means.

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
| `fetch`                       | Clone the mario.scenes datalad dataset and decompress JSON archives      |
| `run-changepoints`            | Detect learning-phase changepoints per subject (Bernoulli completion series; Phase annotations + a `ruptures` PELT/Binseg/Dynp diagnostic) |
| `run-changepoints-by-category`| Detect changepoints per subject **within each scene-category**, on both the `Duration` (time-to-complete) and completion signals |
| `run-clustering`              | Cluster behavioural traces per scene via Fréchet distance + HDBSCAN     |
| `run-clustering-by-category`  | Cluster behavioural traces pooled across scenes sharing a design-pattern category (from `scenes_mastersheet.csv`) |
| `run-notebooks`               | Execute notebooks and save figures to `output_data/`                     |
| `run`                         | Full pipeline in order: fetch → changepoints → changepoints-by-category → clustering → clustering-by-category → notebooks |
| `run-smoke`                   | Minimal end-to-end pass (first subject/scene/category only) to verify pipeline wiring |
| `clean-changepoints`          | Remove `output_data/changepoints/`                                       |
| `clean-changepoints-by-category`| Remove `output_data/changepoints_by_category/`                         |
| `clean-clustering`            | Remove `output_data/clusters/`                                           |
| `clean-clustering-by-category`| Remove `output_data/clusters_by_category/`                              |
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
uv run invoke run-changepoints --smoke            # first subject only
uv run invoke run-clustering --smoke              # first scene only
uv run invoke run-clustering-by-category --smoke  # first category only
```

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
- **Idempotent steps.** Each task checks whether its outputs already exist and skips if they do. Call `invoke clean` then `invoke run` to force a full rerun.
- **Chunked for reruns.** `run-changepoints` and `run-changepoints-by-category` process one subject at a time (`--subjects`); `run-clustering` chunks by scene (`--scenes`) and `run-clustering-by-category` by category (`--categories`). Any piece can be restricted or retried in isolation.
- **Background-ready trace plots.** Scene trajectory figures overlay a scene background image when one is present at `source_data/mario.scenes/sourcedata/scene_backgrounds/<scene_id>.png` (set `BACKGROUND_DIR` in the notebook, or `None` to disable). These images are **not** part of the dataset and must be regenerated separately (see `mario.scenes/code/deprecated/make_images/make_scene_img.py`, which needs `stable_retro` + the source `mario` `.bk2` dataset). Until they exist, plots fall back automatically to a plain y-flipped axis.
- **AI-native.** Initialized and extended with Claude Code — `CLAUDE.md` gives future sessions full context on the pipeline.
