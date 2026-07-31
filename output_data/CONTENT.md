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

> Note: output files are ignored by Git. Run `invoke run` to regenerate them.
