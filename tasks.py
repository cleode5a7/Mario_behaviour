import os
import shutil
from pathlib import Path
from invoke import task


@task
def fetch(c):
    """Clone mario.scenes datalad dataset and decompress JSON archives."""
    source_dir = Path(c.config.get("source_data_dir"))
    scenes_dir = source_dir / "mario.scenes"

    if not (scenes_dir / ".git").exists():
        datalad_check = c.run("datalad --version", warn=True, hide=True)
        if not datalad_check.ok:
            print("datalad is not installed — cannot fetch mario.scenes.")
            print("Install it with:  pip install datalad  (or  uv add datalad)")
            print("See: https://www.datalad.org/get_datalad.html")
            return
        print("Cloning mario.scenes dataset...")
        c.run(f"datalad install git@github.com:courtois-neuromod/mario.scenes {scenes_dir}")
        c.run(f"git -C {scenes_dir} checkout dev_refactor")
        c.run(f"datalad get -d {scenes_dir} .")
    else:
        print(f"mario.scenes already cloned at {scenes_dir}, skipping install.")

    json_files = list(scenes_dir.glob("**/*.json"))
    if not json_files:
        print("Decompressing archives...")
        c.run(f"cd {scenes_dir} && python code/archives/decompress.py")
    else:
        print(f"JSON files already present ({len(json_files)} found), skipping decompress.")


@task
def run_changepoints(c, subjects=None, smoke=False, beta=1.0, n_bkps=None, rep_block_size=10):
    """Detect learning-phase changepoints from each subject's full attempt sequence.

    Detection runs on two units: per scene clip (near-ceiling at ~92%, largely
    flat) and per level repetition (~53-81% cleared, the signal with the dynamic
    range that actually tracks learning).

    Args:
        beta:           ruptures penalty scale for the completion-signal diagnostic
        n_bkps:         fixed changepoint count for the Dynp algorithm
                        (default: the subject's number of Phase transitions)
        rep_block_size: repetitions per block for the level-repetition signal
    """
    from analysis.changepoints import list_subjects, run_changepoint

    source_dir = Path(c.config.get("source_data_dir"))
    output_dir = Path(c.config.get("output_data_dir")) / "changepoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_subjects = list_subjects(source_dir)
    if not all_subjects:
        print("No subjects found in source_data/mario.scenes — run invoke fetch first.")
        return
    if smoke:
        all_subjects = all_subjects[:1]
    if subjects:
        all_subjects = subjects.split(",")

    max_attempts = 500 if smoke else None
    _n_bkps = int(n_bkps) if n_bkps is not None else None
    for subject in all_subjects:
        out = output_dir / f"{subject}_changepoints.json"
        if out.exists():
            print(f"Skipping subject {subject} (output exists)")
            continue
        run_changepoint(subject, source_dir, output_dir, max_attempts=max_attempts,
                        beta=float(beta), n_bkps=_n_bkps,
                        rep_block_size=int(rep_block_size))


@task
def run_changepoints_by_category(c, subjects=None, smoke=False, beta=1.0, n_bkps=1):
    """Detect changepoints per scene-category in each subject's attempt sequence.

    Restricts each subject's chronological attempts to scenes sharing a
    design-pattern tag, then runs the ruptures detector on BOTH the Duration
    (time-to-complete) and completion signals. Chunked by subject: each subject
    loads once, then splits cheaply across the categories they played.

    Args:
        beta:   ruptures penalty scale
        n_bkps: fixed changepoint count for the Dynp algorithm
    """
    from analysis.changepoints import list_subjects, run_changepoint_by_category

    source_dir = Path(c.config.get("source_data_dir"))
    output_dir = Path(c.config.get("output_data_dir")) / "changepoints_by_category"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_subjects = list_subjects(source_dir)
    if not all_subjects:
        print("No subjects found in source_data/mario.scenes — run invoke fetch first.")
        return
    if smoke:
        all_subjects = all_subjects[:1]
    if subjects:
        all_subjects = subjects.split(",")

    max_attempts = 500 if smoke else None
    for subject in all_subjects:
        out = output_dir / f"{subject}_changepoints_by_category.json"
        if out.exists():
            print(f"Skipping subject {subject} (output exists)")
            continue
        run_changepoint_by_category(subject, source_dir, output_dir,
                                    max_attempts=max_attempts,
                                    beta=float(beta), n_bkps=int(n_bkps))


@task
def run_clustering(c, scenes=None, smoke=False, method="hdbscan", n_clusters=5, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, recompute_frechet=False, n_jobs=None):
    """Cluster behavioural traces within each scene using Fréchet distance.

    The Fréchet distance matrix is cached under output_data/frechet/; re-running
    with a different epsilon/k/method reuses it (skipping the expensive O(n^2)
    recompute). Delete only the <scene>_clusters.json to re-cluster cheaply.

    Args:
        method:                     hdbscan (default) or kmedoids
        n_clusters:                 number of clusters for kmedoids
        min_cluster_size:           minimum cluster size for hdbscan
        min_samples:                hdbscan noise aggressiveness (defaults to
                                    min_cluster_size); set lower to keep fewer,
                                    larger clusters without discarding so many
                                    traces to noise
        cluster_selection_epsilon:  hdbscan merge threshold (Fréchet units);
                                    raise it to get fewer, merged clusters
        recompute_frechet:          force recompute of the cached distance matrix
    """
    from analysis.changepoints import list_scenes
    from analysis.clustering import run_clustering as _run_clustering

    source_dir = Path(c.config.get("source_data_dir"))
    output_dir = Path(c.config.get("output_data_dir")) / "clusters"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_scenes = list_scenes(source_dir)
    if not all_scenes:
        print("No scenes found in source_data/mario.scenes — run invoke fetch first.")
        return
    if smoke:
        all_scenes = all_scenes[:1]
    if scenes:
        all_scenes = scenes.split(",")

    for scene in all_scenes:
        out = output_dir / f"{scene}_clusters.json"
        if out.exists():
            print(f"Skipping {scene} (output exists)")
            continue
        _n_jobs = int(n_jobs) if n_jobs is not None else int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        _min_samples = int(min_samples) if min_samples is not None else None
        _run_clustering(scene, source_dir, output_dir, method=method, n_clusters=int(n_clusters), min_cluster_size=int(min_cluster_size), min_samples=_min_samples, cluster_selection_epsilon=float(cluster_selection_epsilon), recompute_frechet=recompute_frechet, n_jobs=_n_jobs)


@task
def run_clustering_by_category(c, categories=None, smoke=False, method="hdbscan", n_clusters=5, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, recompute_frechet=False, n_jobs=None):
    """Cluster behavioural traces pooled across scenes sharing a design-pattern category.

    Traces are pooled across every scene tagged with the category (from
    scenes_mastersheet.csv) and re-based to a common origin before clustering,
    since raw traces from different scenes sit at different absolute positions.

    The Fréchet distance matrix is cached under output_data/frechet/; re-running
    with a different epsilon/k/method reuses it (skipping the expensive O(n^2)
    recompute). Delete only the <category>_clusters.json to re-cluster cheaply.

    Args:
        method:                     hdbscan (default) or kmedoids
        n_clusters:                 number of clusters for kmedoids
        min_cluster_size:           minimum cluster size for hdbscan
        min_samples:                hdbscan noise aggressiveness (defaults to
                                    min_cluster_size); set lower to keep fewer,
                                    larger clusters without discarding so many
                                    traces to noise
        cluster_selection_epsilon:  hdbscan merge threshold (Fréchet units);
                                    raise it to get fewer, merged clusters
        recompute_frechet:          force recompute of the cached distance matrix
    """
    from analysis.clustering import list_categories, category_slug, run_clustering_by_category as _run_clustering_by_category

    source_dir = Path(c.config.get("source_data_dir"))
    output_dir = Path(c.config.get("output_data_dir")) / "clusters_by_category"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_categories = list_categories(source_dir)
    if not all_categories:
        print("No scene categories found — check source_data/scenes_mastersheet.csv.")
        return
    if smoke:
        all_categories = all_categories[:1]
    if categories:
        all_categories = categories.split(",")

    for category in all_categories:
        out = output_dir / f"{category_slug(category)}_clusters.json"
        if out.exists():
            print(f"Skipping {category} (output exists)")
            continue
        _n_jobs = int(n_jobs) if n_jobs is not None else int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        _min_samples = int(min_samples) if min_samples is not None else None
        _run_clustering_by_category(category, source_dir, output_dir, method=method, n_clusters=int(n_clusters), min_cluster_size=int(min_cluster_size), min_samples=_min_samples, cluster_selection_epsilon=float(cluster_selection_epsilon), recompute_frechet=recompute_frechet, n_jobs=_n_jobs)


@task
def run_web_export(c, scenes=None, smoke=False, step=3, force=False, backfill=False):
    """Export clustered traces to a static payload for the browser dashboard.

    The cluster JSONs hold labels + metadata but no geometry, so the dashboard
    needs the x/y polylines baked out of mario.scenes once. Writes one file per
    scene under output_data/web/data/scenes/, plus an index and a copy of the
    scene background PNGs, so the served tree never reaches back into the 23 GB
    source dataset.

    Args:
        step:  keep every step-th frame (first and last always kept). Traces are
               60 fps against a 224 px background, so 3 is visually lossless;
               lower it for more fidelity at the cost of payload size.
        force: re-export scenes whose payload already exists
        backfill: only add the derived attempt/stage fields to payloads that
               already exist, then rewrite the index. Reads nothing but the
               exported JSON, so it takes seconds where --force would re-read
               the source dataset. Use it after a change that adds a field
               derivable from what the payload already carries.
    """
    from analysis.webexport import (backfill_derived, export_scene, list_clustered_scenes,
                                    sync_backgrounds, write_index)
    from analysis.trace_plots import default_background_dir, scene_entry_map

    source_dir = Path(c.config.get("source_data_dir"))
    output_dir = Path(c.config.get("output_data_dir"))
    web_dir = output_dir / "web" / "data"
    web_dir.mkdir(parents=True, exist_ok=True)

    if backfill:
        if not (web_dir / "scenes").exists():
            print("Nothing to backfill — no payload at output_data/web/data/scenes.")
            return
        backfill_derived(web_dir)
        write_index(web_dir, source_dir)
        return

    all_scenes = list_clustered_scenes(output_dir)
    if not all_scenes:
        print("No clustering output found in output_data/clusters — run invoke run-clustering first.")
        return
    if smoke:
        all_scenes = all_scenes[:1]
    if scenes:
        all_scenes = scenes.split(",")

    entry_map = scene_entry_map(source_dir)
    background_dir = default_background_dir(source_dir)

    for scene in all_scenes:
        out = web_dir / "scenes" / f"{scene}.json"
        if out.exists() and not force:
            print(f"Skipping {scene} (payload exists)")
            continue
        export_scene(scene, source_dir, output_dir, web_dir, step=int(step),
                     entry_map=entry_map, background_dir=background_dir)

    # Both are cheap and keep the payload internally consistent, so redo on every run.
    copied = sync_backgrounds(web_dir, source_dir)
    if copied:
        print(f"Copied {copied} scene background(s)")
    write_index(web_dir, source_dir)


@task
def dashboard(c, port=8000, open_browser=True):
    """Serve the interactive clustering dashboard at http://localhost:<port>.

    Copies the app source from web/ into output_data/web/ (fresh each run, so
    editing web/app.js and re-running always picks the change up) and serves
    that directory. Requires `invoke run-web-export` to have run first.
    """
    import subprocess
    import sys
    import webbrowser

    output_dir = Path(c.config.get("output_data_dir"))
    web_root = output_dir / "web"
    if not (web_root / "data" / "index.json").exists():
        print("No dashboard data found. Run:  invoke run-web-export")
        return

    app_dir = Path("web")
    if not app_dir.exists():
        print(f"Dashboard app source not found at {app_dir}/.")
        return
    for asset in sorted(app_dir.iterdir()):
        if asset.is_file():
            shutil.copy2(asset, web_root / asset.name)

    url = f"http://localhost:{port}/"
    print(f"Serving dashboard at {url}  (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        subprocess.run([sys.executable, "-m", "http.server", str(port),
                        "--directory", str(web_root)])
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


@task
def list_scenes_task(c):
    """Print all scene IDs, one per line. Used to build SLURM array inputs."""
    from analysis.changepoints import list_scenes
    for scene in list_scenes(Path(c.config.get("source_data_dir"))):
        print(scene)


@task
def list_categories_task(c):
    """Print all scene-category (pattern tag) names, one per line. Used to build SLURM array inputs."""
    from analysis.clustering import list_categories
    for category in list_categories(Path(c.config.get("source_data_dir"))):
        print(category)


@task
def run_notebooks(c):
    """Generate figures from pipeline output using notebooks."""
    from airoh.utils import run_notebooks as airoh_run_notebooks, ensure_dir_exist

    notebooks_dir = Path(c.config.get("notebooks_dir"))
    output_dir = Path(c.config.get("output_data_dir")).resolve()

    ensure_dir_exist(c, "output_data_dir")
    airoh_run_notebooks(c, notebooks_dir, output_dir, keys=["source_data_dir", "output_data_dir"])


@task(pre=[fetch, run_changepoints, run_changepoints_by_category, run_clustering, run_clustering_by_category, run_web_export, run_notebooks])
def run(c):
    """Full pipeline."""
    print("Pipeline complete.")


@task
def run_smoke(c):
    """Smoke test: minimal end-to-end pass."""
    fetch(c)
    run_changepoints(c, smoke=True)             # first subject, first 500 attempts
    run_changepoints_by_category(c, smoke=True)  # first subject, per category
    run_clustering(c, smoke=True)                # first scene only
    run_clustering_by_category(c, smoke=True)    # first category only
    run_web_export(c, smoke=True)                # dashboard payload, first scene
    run_notebooks(c)


@task
def clean_changepoints(c):
    """Remove changepoint detection outputs."""
    output_dir = Path(c.config.get("output_data_dir")) / "changepoints"
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Removed {output_dir}")
    else:
        print("No changepoints output to clean.")


@task
def clean_changepoints_by_category(c):
    """Remove per-category changepoint outputs."""
    output_dir = Path(c.config.get("output_data_dir")) / "changepoints_by_category"
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Removed {output_dir}")
    else:
        print("No changepoints-by-category output to clean.")


@task
def clean_clustering(c):
    """Remove clustering outputs."""
    output_dir = Path(c.config.get("output_data_dir")) / "clusters"
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Removed {output_dir}")
    else:
        print("No clustering output to clean.")


@task
def clean_clustering_by_category(c):
    """Remove per-category clustering outputs."""
    output_dir = Path(c.config.get("output_data_dir")) / "clusters_by_category"
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Removed {output_dir}")
    else:
        print("No per-category clustering output to clean.")


@task
def clean_web_export(c):
    """Remove the exported dashboard payload and copied app assets."""
    output_dir = Path(c.config.get("output_data_dir")) / "web"
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Removed {output_dir}")
    else:
        print("No dashboard export to clean.")


@task
def clean_frechet(c):
    """Remove cached Fréchet distance matrices.

    Kept separate from clean-clustering so clearing cluster labels (to re-cluster
    with a new epsilon/k) preserves the expensive distance matrices. Only a full
    `invoke clean` or an explicit `invoke clean-frechet` discards them.
    """
    output_dir = Path(c.config.get("output_data_dir")) / "frechet"
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Removed {output_dir}")
    else:
        print("No cached Fréchet matrices to clean.")


@task(pre=[clean_changepoints, clean_changepoints_by_category, clean_clustering, clean_clustering_by_category, clean_web_export, clean_frechet])
def clean(c):
    """Remove all computed outputs."""
    pass


@task
def clean_mario_scenes(c):
    """Remove the cloned mario.scenes datalad dataset from source_data/."""
    source_dir = Path(c.config.get("source_data_dir"))
    scenes_dir = source_dir / "mario.scenes"
    if scenes_dir.exists():
        shutil.rmtree(scenes_dir)
        print(f"Removed {scenes_dir}")
    else:
        print("mario.scenes not present, nothing to remove.")


@task(pre=[clean_mario_scenes])
def clean_source(c):
    """Remove all downloaded source data."""
    pass
