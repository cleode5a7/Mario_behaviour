# Output Data

After running the full pipeline (`invoke run`), this folder contains:

## `changepoints/`

One JSON file per subject produced by `invoke run-changepoints`:

- `<subject_id>_changepoints.json` — detected learning-phase changepoints in that subject's full chronological attempt sequence (all scenes combined).
  Fields: `{"subject": str, "n_attempts": int, "changepoints": [<attempt indices>]}`

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

> Note: output files are ignored by Git. Run `invoke run` to regenerate them.
