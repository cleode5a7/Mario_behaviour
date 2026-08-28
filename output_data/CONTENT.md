# Output Data

After running the full pipeline (`invoke run`), this folder contains:

## `changepoints/`

One JSON file per subject produced by `invoke run-changepoints`:

- `<subject_id>_changepoints.json` — detected learning-phase changepoints in that subject's full chronological attempt sequence (all scenes combined).
  Fields: `{"subject": str, "n_attempts": int, "changepoints": [<attempt indices>], "phases": [...], "method": "phase_annotation"}`.
  Also carries a `"ruptures"` block — a non-Bayesian diagnostic that runs the `ruptures` PELT/Binseg/Dynp algorithms (CostL2) on the completion signal, on both the raw per-attempt series and a 50-attempt block-averaged series, for comparison against the Phase annotations:
  `{"model": "l2", "block_size": int, "beta": float, "n_bkps": int, "results": {"raw": {"pelt": [...], "binseg": [...], "dynp": [...]}, "blocks": {...}}}` (breakpoints in attempt-index space).

## `changepoints_by_category/`

One JSON file per subject produced by `invoke run-changepoints-by-category`:

- `<subject_id>_changepoints_by_category.json` — changepoints in each subject's attempt series **restricted to scenes sharing a design-pattern category** (from `source_data/scenes_mastersheet.csv`). For every category the subject played, the `ruptures` detector runs on **both** the `Duration` (time-to-complete) signal — continuous, expected to shrink with mastery — and the Bernoulli completion signal, each on raw + block-averaged series.
  Fields: `{"subject": str, "method": "ruptures_by_category", "model": "l2", "block_size": int, "beta": float, "n_bkps": int, "categories": {"<category>": {"n_attempts": int, "duration": {"raw": {...}, "blocks": {...}}, "completion": {"raw": {...}, "blocks": {...}}}, ...}}`.
  `notebooks/figure_changepoints_by_category.ipynb` writes one small-multiples figure per subject here (`figure_changepoints_by_category_<subject>.png`): duration vs. completion curves for the busiest categories, with per-algorithm changepoint lines.

## `clusters/`

One JSON file per scene produced by `invoke run-clustering`:

- `<scene_id>_clusters.json` — cluster label per behavioural trace for that scene.
  Fields: `{"scene": str, "labels": [<per-trace cluster label>]}`

  **No trace geometry is stored here** — the x/y polylines are rebuilt from
  `source_data/mario.scenes/` on demand by `load_traces()`. See `web/` below for the
  exported copy the dashboard reads.

## `clusters_by_category/`

One JSON file per scene category (design-pattern tag from `source_data/scenes_mastersheet.csv`,
e.g. `2-Horde`, `Gap stair valley`) produced by `invoke run-clustering-by-category`:

- `<category-slug>_clusters.json` — cluster label per behavioural trace, pooled across every
  scene tagged with that category. Traces are re-based to a common origin before clustering
  since raw traces from different scenes sit at different absolute level positions.
  Fields: `{"category": str, "method": str, "labels": [<per-trace cluster label>], "metadata": [...]}`
  (each `metadata` entry carries `SceneID` so pooled traces stay attributable to their scene).

## Trace figures

`notebooks/figure_clustering.ipynb` also writes trajectory figures into `clusters/`
(`figure_clustering_<scene_id>.png`, `figure_clustering_category_<slug>.png`, etc.).
Traces use a screen-corrected vertical coordinate and are colored by cluster (default)
or by outcome. `figure_clustering_<scene_id>_panels.png` shows the same scene split into
one subplot per cluster (shared axes) so individual cluster shapes are easy to compare. Each scene figure overlays a scene background image when one exists at
`source_data/mario.scenes/sourcedata/scene_backgrounds/<scene_id>.png`; those images are
not part of the dataset (must be generated separately), so by default the figures render
on a plain y-flipped axis.

## `web/`

The static payload and app behind `invoke dashboard` — an interactive version of the
figures above, with scene / subject / phase / stage / outcome filters. Written by
`invoke run-web-export`; roughly 50 MB in total.

- `data/index.json` — one row per exported scene (`id`, `world`, `level`, `scene`, `n`
  traces, `k` clusters, `noise` fraction, `subjects`, `phases`, `method`, pattern `tags`),
  plus the corpus-wide subject/phase/tag lists. Drives the scene picker without loading
  any scene payload.
- `data/scenes/<scene_id>.json` — the drawable scene: `method`, `entry` (the applied
  x-offset), `bg` + `bg_size`, `xlim`/`ylim` (precomputed from `_shared_xlim` and
  `playfield_ylim`), `step`, and a `traces` array. Each trace carries its cluster label
  `c`, the filterable metadata (`s` subject, `ses`, `run`, `clip`, `sf`, `ph` phase,
  `out` outcome, `dur`), two derived time fields (`a`, `st` — below), tooltip counters
  (`xt`, `ht`, `ek`, `ll`), and the polyline as parallel `x`/`y` integer arrays. A
  `null` in `y` is a masked (dead) frame and must break the line rather than be
  interpolated across.
- `a` — **attempt index**: the nth entry this subject made into this scene, 1-based and
  chronological across the whole experiment. Each clip is one attempt, re-entries after
  a death included. Not the `rep-NNN` from `SourceBk2`, which restarts every session.
- `st` — **phase tertile**: `"e"`, `"m"` or `"l"`, cutting each `(subject, phase)` group
  at the 1/3 and 2/3 `ClipCode` quantiles. The full stage name is `st + "_" + ph`, so
  only the letter is stored. Both fields come from `analysis/stages.py` and can be added
  to an existing payload with `invoke run-web-export --backfill`.
- `data/backgrounds/<scene_id>.png` — copies of the scene backgrounds, so the served
  tree never reaches back into the 23 GB `mario.scenes` dataset at runtime.
- `index.html`, `app.js`, `style.css` — copied from `web/` by `invoke dashboard`.

Two caveats worth carrying:

- **Coordinates are downsampled.** `step` (default 3) means only every 3rd frame was
  kept, plus the first and last, and coordinates are rounded to integers. Fine for
  drawing at 224 px tall; **do not** use this payload for quantitative analysis — go
  back to `load_traces()` for that.
- **Cluster labels are the pooled ones.** HDBSCAN ran once over every trace in the
  scene. The dashboard's filters only change which traces are drawn; they never
  re-cluster, and cluster colors stay fixed to the whole-scene rank-by-size order so a
  filter cannot repaint the traces that survive it.

> Note: output files are ignored by Git. Run `invoke run` to regenerate them.
