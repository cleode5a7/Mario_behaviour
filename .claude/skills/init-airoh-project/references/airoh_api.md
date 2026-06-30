# Airoh Utils API Reference

## download_data(c, name)

Downloads a file configured in `invoke.yaml` under `files:`.

**invoke.yaml entry:**
```yaml
files:
  dataset_name:
    url: https://example.com/data.csv
    output_file: source_data/data.csv
```

**tasks.py call:**
```python
from airoh.utils import download_data

@task
def fetch(c):
    download_data(c, "dataset_name")
    download_data(c, "other_file")
```

- Skips if the output file already exists and is non-empty (idempotent)
- Uses a `.part` temp file for atomic writes

## clean_folder(c, name, pattern=None)

Removes files from a directory identified by an `invoke.yaml` key.

- `name`: key in `invoke.yaml` whose value is a directory path (e.g., `"output_data_dir"`)
- `pattern`: glob pattern (e.g., `"*.png"`, `"subjects/*.csv"`); if `None`, removes the entire folder

```python
from airoh.utils import clean_folder

clean_folder(c, "output_data_dir", "*.csv")   # delete all CSVs in output_data/
clean_folder(c, "source_data_dir", "*.tsv")   # delete all TSVs in source_data/
```

## run_notebooks(c, notebooks_path, figures_base, keys)

Executes all `.ipynb` notebooks found in `notebooks_path`. Skips any notebook whose output directory already exists.

```python
from airoh.utils import run_notebooks as airoh_run_notebooks, ensure_dir_exist

@task
def run_notebooks(c):
    notebooks_dir = Path(c.config.get("notebooks_dir"))
    output_dir = Path(c.config.get("output_data_dir")).resolve()
    ensure_dir_exist(c, "output_data_dir")
    airoh_run_notebooks(c, notebooks_dir, output_dir, keys=["source_data_dir", "output_data_dir"])
```

The `keys` list controls which `invoke.yaml` paths are passed to notebooks as environment variables (`SOURCE_DATA_DIR`, `OUTPUT_DATA_DIR`).

## ensure_dir_exist(c, name)

Creates the directory referenced by an `invoke.yaml` key if it does not exist.

```python
ensure_dir_exist(c, "output_data_dir")
```

---

## invoke.yaml structure

```yaml
notebooks_dir: notebooks
source_data_dir: source_data
output_data_dir: output_data

files:
  dataset_name:
    url: https://...
    output_file: source_data/filename.ext
```

---

## Chunk-processing task pattern

Use this when a script processes independent items (subjects, samples, files) and should skip already-completed ones:

```python
@task
def process_subjects(c, subjects=None, smoke=False):
    """Process each subject; skip if output exists."""
    from analysis.process import process_subject, list_subjects
    output_dir = Path(c.config.get("output_data_dir"))
    source_dir = Path(c.config.get("source_data_dir"))

    all_subjects = list_subjects(source_dir)
    if smoke:
        all_subjects = all_subjects[:1]
    if subjects:
        all_subjects = subjects.split(",")

    for subj in all_subjects:
        out = output_dir / f"{subj}.csv"
        if out.exists():
            print(f"Skipping {subj} (output exists)")
            continue
        process_subject(subj, source_dir, output_dir)
```

Adapt the "chunk" concept (subjects, files, conditions, etc.) and the output existence check to match the actual data structure.

## run / run-smoke pattern

```python
@task(pre=[fetch, process_subjects, run_notebooks])
def run(c):
    """Full pipeline."""
    print("Pipeline complete.")

@task
def run_smoke(c):
    """Smoke test: minimal end-to-end run."""
    fetch(c)
    process_subjects(c, smoke=True)
    run_notebooks(c)
```

`run_smoke` calls tasks directly (not via `pre=`) so it can pass `smoke=True`.
