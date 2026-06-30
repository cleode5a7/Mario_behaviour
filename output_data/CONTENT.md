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

> Note: output files are ignored by Git. Run `invoke run` to regenerate them.
