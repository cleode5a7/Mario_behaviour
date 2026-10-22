# Source Data

## mario.scenes (`source_data/mario.scenes/`)

Datalad dataset from `git@github.com:courtois-neuromod/mario.scenes`, on the `dev_refactor` branch. Declared under `datasets:` in `invoke.yaml`, retrieved by `invoke fetch-mario-scenes` (which the umbrella `invoke fetch` calls).

It arrives one of two ways:

**Cloned** (default) — `datalad clone`, then `git checkout dev_refactor`, then:
- The full dataset is retrieved via `datalad get`.
- `source_data/mario.scenes/code/archives/decompress.py` is run to extract JSON files from compressed archives.
- JSON files containing per-scene behavioural traces are then available for globbing within `source_data/mario.scenes/`.

**Symlinked** — the dataset is ~23 GB, so if it already lives elsewhere on this machine, point at it instead of cloning a second copy:

```bash
invoke fetch-mario-scenes --source /path/to/existing/mario.scenes
invoke fetch --mario-scenes-source /path/to/existing/mario.scenes   # same, via the umbrella task
```

A `source:` key under `datasets: mario_scenes:` in `invoke.yaml` makes that the default for a given machine. A linked checkout is read exactly as found: no branch checkout, no `datalad get`, no decompression — those would all write into a dataset this project does not own. So it must already be on the branch you want, with its archives already decompressed; `fetch` says so and stops if the JSON files are missing. Check `MANIFEST.json` to see which commit and branch a link actually resolved to.

Re-pointing a stale link needs `invoke clean-mario-scenes` first — `fetch` never overwrites what already sits at the destination. That clean unlinks the symlink and leaves the external checkout itself untouched.

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
  It is still declared under `files:` in `invoke.yaml`, with no `url` and no fetch task, purely so
  that `MANIFEST.json` records its size and checksum alongside the dataset.

## MANIFEST.json

Written by `invoke fetch` (see `airoh.provenance`). One entry per declared asset: where it landed, whether it is a symlink and what it really resolves to, its size and checksum, and — for an asset in a repository of its own — that repository's commit, branch and dirty flag.

Git-tracked, and the reason a symlinked dataset is not a provenance hole: the record names the commit of the external checkout the results were actually computed from. Its companion is `output_data/PROVENANCE.json`, which records what the run produced.
