# Airoh API Reference

Acquisition tasks (`fetch_data`, `download_data`, `ensure_submodule`) live in
`airoh.acquisition`. General utilities (`clean_folder`, `run_notebooks`,
`ensure_dir_exist`) live in `airoh.utils`. Consistency checking lives in
`airoh.verify`, and provenance recording in `airoh.provenance`.

## fetch_data(c, name, source=None, copy=False)

The preferred way to make a data asset available. For each entry in `files:` it
either **downloads** from the entry's `url` (default) or **symlinks** to
already-present data when a source path is given — avoiding a re-download of data
that already lives on disk. Prefer this over the lower-level `download_data`.

**invoke.yaml entry:**
```yaml
files:
  dataset_name:
    url: https://example.com/data.csv        # download source (optional if `source` given)
    output_file: source_data/data.csv
    # source: /path/to/existing/data.csv      # symlink instead of downloading (optional)
```

**tasks.py call (single asset):**
```python
from airoh.acquisition import fetch_data

@task(help={
    "source": "Path to already-present data to symlink instead of downloading.",
    "copy": "Copy the source data instead of symlinking it.",
})
def fetch(c, source=None, copy=False):
    fetch_data(c, "dataset_name", source=source, copy=copy)
```

- Source resolution: the `source` argument → the entry's `source:` key → the entry's `url`.
- Symlinks handle both files and whole directories.
- Idempotent: a link already pointing at the source is left untouched; a real
  file at `output_file` is never clobbered.

**Usage:**
```bash
invoke fetch                              # download
invoke fetch --source /data/existing.csv  # symlink to existing data
invoke fetch --source /data/existing.csv --copy   # real copy instead
```

**One asset, one `--source`.** The `source` argument is a single path bound to
the single asset named in the call — it is **not** a root directory that gets
joined with each asset's filename. Passing one shared `--source` to several
`fetch_data` calls links every asset to that same path, and does so silently:
each call prints a success line and the task exits 0, leaving several
differently-named symlinks pointing at one file.

**With more than one asset, give each its own `fetch-{name}` task** and let the
umbrella `fetch` route a per-asset `--{name}-source` flag to the matching one:

```yaml
files:
  papers:
    url: https://example.com/papers.tsv
    output_file: source_data/papers.tsv
  atlas:
    url: https://example.com/atlas.nii.gz
    output_file: source_data/atlas.nii.gz
```

```python
@task(help={"source": "Existing 'papers' data to link instead of downloading."})
def fetch_papers(c, source=None, copy=False):
    fetch_data(c, "papers", source=source, copy=copy)

@task(help={"source": "Existing 'atlas' data to link instead of downloading."})
def fetch_atlas(c, source=None, copy=False):
    fetch_data(c, "atlas", source=source, copy=copy)

@task(help={"papers_source": "Source path for the papers asset.",
            "atlas_source":  "Source path for the atlas asset."})
def fetch(c, papers_source=None, atlas_source=None, copy=False):
    fetch_papers(c, source=papers_source, copy=copy)
    fetch_atlas(c, source=atlas_source, copy=copy)
```

```bash
invoke fetch-papers --source /data/papers.tsv   # one asset
invoke fetch --papers-source /data/papers.tsv --atlas-source /data/atlas.nii.gz
```

A per-asset `source:` key in `invoke.yaml` sets the default source for that
asset without any command-line flag.

**Datalad datasets need `airoh.datalad`, not `fetch_data --source`.**
`--source` symlinks or copies a plain file/folder; it does not run `datalad
get`. A symlinked datalad dataset resolves only content that is already present
(un-fetched files are broken symlinks), and `--copy` raises on those un-fetched
files. Use `install_dataset`/`get_data` from `airoh.datalad` (see below) with a
`datasets:` entry in `invoke.yaml` for a datalad dataset, and `ensure_submodule`
(below) for a plain git submodule.

## download_data(c, name)

Lower-level primitive used by `fetch_data`: downloads a file from the entry's
`url` only (no symlink option). Reach for `fetch_data` unless you specifically
want URL-only behavior.

```python
from airoh.acquisition import download_data

download_data(c, "dataset_name")
```

- Skips if the output file already exists and is non-empty (idempotent)
- Uses a `.part` temp file for atomic writes

## ensure_submodule(c, path, recursive=True)

Initializes or updates a git submodule at `path` (a common way to bring in an
external dataset tracked as a submodule). Also in `airoh.acquisition`.

```python
from airoh.acquisition import ensure_submodule

ensure_submodule(c, "source_data/my-dataset")
```

## airoh.datalad — datalad-backed retrieval

Requires the `datalad` CLI (`pip install airoh[datalad]`). Import always
succeeds without it on PATH — only calling a task raises, with a hint to
install the extra. Every retrieval function is **tolerant by default**: a
file on a credentialed remote a given environment cannot reach warns and is
skipped rather than aborting the whole fetch. Pass `strict=True` only where
that must fail loudly instead — typically `run-smoke`.

**invoke.yaml entry** (either shape is accepted):
```yaml
datasets:
  cneuromod: source_data/cneuromod.all          # plain path string, or:
  cneuromod:
    output_dir: source_data/cneuromod.all
    url: https://github.com/courtois-neuromod/cneuromod.all.git
    # source: /data/cneuromod.all                # symlink an existing checkout
```

**Invoke tasks** (thin wrappers around the plain functions below, looked up by
`name` under `datasets:`):

```python
from airoh.datalad import install_dataset, get_data, update_dataset

@task(help={"source": "Existing checkout to symlink instead of cloning."})
def fetch_cneuromod(c, source=None):
    install_dataset(c, "cneuromod", source=source)   # symlink or `datalad clone` — no content yet
    get_data(c, "cneuromod")                          # retrieve content, tolerant by default

@task
def update_cneuromod(c):
    update_dataset(c, "cneuromod")                    # advance the pin, no content pulled
```

```bash
invoke fetch-cneuromod                         # clone (or symlink via invoke.yaml `source:`) + get
invoke fetch-cneuromod --source /data/cneuromod.all
invoke get-data --name cneuromod --path sub-01 --strict   # narrow + fail loudly (e.g. in run-smoke)
```

**Plain functions**, for a project that composes its own retrieval logic
(e.g. a custom prefetch step) rather than going through the `datasets:`-keyed
tasks above:

- `ensure_dataset(dest, url=None, source=None)` — make a checkout available:
  no-op if `dest` exists, else symlink from `source`, else `datalad clone url
  dest` (tree only, no content).
- `datalad_get(paths, dataset_root, recursive=False, get_content=True, strict=False)`
  — `datalad get` for `paths` (relative to `dataset_root`), retrying once over
  HTTPS on failure.
- `install_subdataset(path, dataset_root, strict=False)` / `update_subdataset(path,
  dataset_root, strict=False)` — install (or refresh the pin of) a subdataset
  nested inside another subdataset. Plain `git submodule update --init` cannot
  reach a nested subdataset; `datalad get -n` can, in one call.
- `load_known_failures(cache_dir)` / `save_known_failures(cache_dir, failures)`
  — persist the set of paths that failed to retrieve last time, so a repeat
  fetch can skip retrying permanently-inaccessible content instead of paying
  its connection-timeout cost again on every run.
- `prefetch_pattern(dataset_root, pattern, subdir="", skip_set=(), match=None)`
  — glob `dataset_root/subdir` for `pattern`, `datalad get` whatever's missing
  and not in `skip_set`, and return `(already_present, newly_fetched, skipped,
  new_failures, resolved)` so the caller can update its failure cache. The
  generic core of "fetch every small file a `run-*` step reads, up front" —
  compose it per marker/glob in a project-specific `fetch` task rather than
  writing the glob-skip-get-reclassify loop by hand:
  ```python
  from airoh.datalad import load_known_failures, prefetch_pattern, save_known_failures

  @task
  def fetch_mriqc_metadata(c):
      source_dir = Path(c.config.get("source_data_dir"))
      cache_dir = source_dir
      failures = load_known_failures(cache_dir)
      present, fetched, skipped, new_failures, resolved = prefetch_pattern(
          source_dir, "*_bold.json", subdir="derivatives/mriqc", skip_set=failures)
      save_known_failures(cache_dir, (failures - resolved) | new_failures)
      print(f"{present} already had, {fetched} newly fetched, {skipped} skipped")
  ```

**Three gotchas:**
- **Nested subdatasets.** A derivative folder is often a subdataset inside
  another subdataset — reach it with `install_subdataset` (`datalad get -n`),
  not plain `git submodule`.
- **Tolerant by default, strict only where it must fail loudly.** Leave
  `strict=False` (the default) in `fetch`; pass `strict=True` only in
  `run-smoke`, where a silently-empty result would otherwise look like success.
- **`git-annex` version.** A dataset in annex v10 format is refused outright by
  an older `git-annex`, and the failure looks like "no content anywhere" rather
  than a clear error. Pin a recent `git-annex` as a declared project dependency
  (the `git-annex` PyPI package bundles a binary) rather than relying on
  whatever is on a collaborator's system.

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

## airoh.figures — the Inkscape montage pattern

Only relevant when a project assembles a **multi-panel figure by hand in
Inkscape** from panels that matplotlib renders. Skip this whole section
otherwise, and delete the `figures:` block from `invoke.yaml` (`invoke verify`
reports unused keys).

The problem it solves: placing a matplotlib PNG into a hand-authored layout
scales it, which stretches its text, so the point sizes the author chose stop
being the point sizes that land on the page. The contract that fixes it — **the
montage SVG is the single source of truth for layout**. It links each panel by
relative path (resolved from its own directory, which is why the SVG lives under
`output_data/` despite being a pipeline *source*), and the box it places a panel
in is that panel's true on-page size.

```python
# tasks.py
from airoh.figures import clean_figure, compose_figure, figure_layout
```

- **`figure_layout(c, name=None)`** — parse every montage in `figures:` and write
  `{panel_path: [width_mm, height_mm]}` to `figures_dir/panel_sizes.json`. Wrap
  it as `run-figure-layout`. It must **always re-run, never skip on existence**:
  it is cheap, and a box resized in Inkscape has to take effect on the very next
  `invoke run`. This is the one deliberate exception to existence-based caching.
- **`panel_size(name, default)`** — called *in the notebook*, returns the panel's
  size in **inches** for `figsize=`. `name` is the panel's path relative to
  `figures_dir` (e.g. `figure_simulation/scatter.png`). A panel the montage does
  not place keeps `default`, so notebooks still run standalone.
- **`compose_figure(c, name=None)`** — render each montage to its `output` via the
  Inkscape CLI (png/pdf/svg/eps by extension). Inkscape is an **optional system
  binary**: a missing one warns and returns, so it never fails `invoke run`.
- **`clean_figure(c, name=None)`** — remove the composed output. Never the SVG —
  that is hand-authored source.

Three rules that must survive into the project, or the 1:1 guarantee quietly breaks:

1. In the notebook, **never** `bbox_inches="tight"` — it resizes the canvas after
   the fact. Use `layout="constrained"` to reclaim margins inside the fixed canvas.
2. Save at the montage's DPI, and **read it from config** rather than hardcoding:
   saved pixels must equal `figsize × dpi`. Export it from the `run-notebooks`
   task (`os.environ["FIGURE_MONTAGE_DPI"] = ...`) and read it in the notebook.
3. Resizing a box only fully lands after the panel **re-renders** — and panels are
   notebook outputs, so they obey existence-based caching. `panel_sizes.json` and
   the montage refresh every run regardless, so a panel whose notebook did not
   re-run gets stretched into the new box. After an Inkscape resize:
   `invoke clean-figures && invoke run`.

`clean-figures` must remove the whole `figures_dir` tree, since it holds both the
notebook "already ran" sentinels and `panel_sizes.json`.

## verify(c, skip=None, strict=False)

Runs a flat list of consistency checks between the project's code, config, data
and documentation, and exits non-zero if any of them fails. Expose it as a task
of its own; never call it from `run`.

```python
@task(help={"skip": "Comma-separated check names to skip.",
            "strict": "Treat warnings as failures."})
def verify(c, skip=None, strict=False):
    """Check that the code, config, data and docs still agree."""
    from airoh.verify import verify as airoh_verify
    airoh_verify(c, skip=skip, strict=strict)
```

The checks: `content_md` (data folders versus their `CONTENT.md`), `task_list`
(tasks defined versus documented), `dependencies` (`pyproject.toml` versus
`requirements.txt` versus `environment.yml`), `doc_paths` (paths named in the
docs exist), `config_keys` (`tasks.py` versus `invoke.yaml`), `tracked_size`
(nothing oversized or binary tracked in git), `provenance` (records present and
current), `lint`.

Configure under `verify:` in `invoke.yaml`:

```yaml
verify:
  skip_checks: []        # e.g. [lint]
  ignore_paths: []       # glob patterns exempt from path/tracked-file checks
  max_tracked_bytes: 10485760
```

## record_sources(c) and record_run(c, tasks=None)

Provenance records, from `airoh.provenance`. Call `record_sources` at the end of
`fetch` and `record_run` at the end of `run`. Neither can raise, so neither can
break a pipeline.

```python
from airoh.provenance import record_run, record_sources

record_sources(c)                              # -> source_data/MANIFEST.json
record_run(c, tasks="step-a,run-notebooks")    # -> output_data/PROVENANCE.json
```

`MANIFEST.json` records what every asset in `files:`/`datasets:` actually
resolved to — the URL or the real path behind a symlink, size, checksum, and the
git commit and datalad id of the repository it belongs to. `PROVENANCE.json`
records the project's own commit and dirty flag, the environment, a checksum of
the manifest it consumed, and the size and checksum of every output.

Paths come from `manifest_file` and `provenance_file` in `invoke.yaml`. Both
records are git-tracked; `PROVENANCE.json` changes on every run by design.

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
figures_dir: output_data/figures   # notebooks write into figures_dir/<stem>/

files:
  dataset_name:
    url: https://...
    output_file: source_data/filename.ext

datasets:               # only for real datalad datasets — see airoh.datalad above
  dataset_name:
    output_dir: source_data/dataset_name
    url: https://...
    # source: /path/to/existing/checkout

figures:                # only for hand-authored Inkscape montages — delete otherwise
  montage_name:
    svg: output_data/montage_name.svg     # hand-authored, a pipeline SOURCE
    output: output_data/montage_name.png
    dpi: 300
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
@task(help={"force": "Delete every computed output first, then run from scratch."})
def run(c, force=False):
    """Full pipeline."""
    if force:
        clean(c)
    process_subjects(c)
    run_notebooks(c)
    print("Pipeline complete.")

@task
def run_smoke(c):
    """Smoke test: minimal end-to-end run."""
    fetch(c)
    process_subjects(c, smoke=True)
    run_notebooks(c)
```

Both call their steps directly in the body, **not** via `pre=`: a `pre=` chain
runs before the body and has already finished by the time `run` sees `--force`
or `run_smoke` would need `smoke=True`, so passing either through would arrive
too late. `pre=` is fine only on a task that is purely a command-line entry
point and never called as a plain function — never on `run`, `clean`, or
anything another task calls programmatically.
