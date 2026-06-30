# Source Data

## mario.scenes (`source_data/mario.scenes/`)

Datalad dataset cloned from `git@github.com:courtois-neuromod/mario.scenes` on the `dev_refactor` branch.

After `invoke fetch`:
- The full dataset is retrieved via `datalad get`.
- `code/archives/decompress.py` is run to extract JSON files from compressed archives.
- JSON files containing per-scene behavioural traces are then available for globbing within `source_data/mario.scenes/`.

> Note: `source_data/mario.scenes/` is ignored by Git. Run `invoke fetch` to populate it.
