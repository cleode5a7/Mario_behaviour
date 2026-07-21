# Source Data

## mario.scenes (`source_data/mario.scenes/`)

Datalad dataset cloned from `git@github.com:courtois-neuromod/mario.scenes` on the `dev_refactor` branch.

After `invoke fetch`:
- The full dataset is retrieved via `datalad get`.
- `code/archives/decompress.py` is run to extract JSON files from compressed archives.
- JSON files containing per-scene behavioural traces are then available for globbing within `source_data/mario.scenes/`.

> Note: `source_data/mario.scenes/` is ignored by Git. Run `invoke fetch` to populate it.

## scenes_mastersheet.csv

Scene-level pattern annotations, one row per `(World, Level, Scene)`. Sourced from
Zenodo record [15586709](https://zenodo.org/records/15586709) (`scenes_mastersheet.csv`).

- `World`, `Level`, `Scene` — identify the scene; combine to `SceneID` as `w{World}l{Level}s{Scene}`.
- 27 binary pattern-tag columns (e.g. `2-Horde`, `Gap stair valley`, `Beginning`, `Flagpole`) —
  a scene can carry multiple tags at once (not mutually exclusive). These are the
  "scene categories" used to group clustering/behavioural results across scenes that
  share a design pattern, instead of only looking at one scene at a time.
- Checked into Git (small, static reference table — unlike `mario.scenes/`, not fetched by `invoke fetch`).
