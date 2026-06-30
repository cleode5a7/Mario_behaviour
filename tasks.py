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
def run_changepoints(c, subjects=None, smoke=False):
    """Detect learning-phase changepoints from each subject's full attempt sequence."""
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
    for subject in all_subjects:
        out = output_dir / f"{subject}_changepoints.json"
        if out.exists():
            print(f"Skipping subject {subject} (output exists)")
            continue
        run_changepoint(subject, source_dir, output_dir, max_attempts=max_attempts)


@task
def run_clustering(c, scenes=None, smoke=False, method="hdbscan", n_clusters=5, min_cluster_size=5):
    """Cluster behavioural traces within each scene using Fréchet distance.

    Args:
        method:           hdbscan (default) or kmedoids
        n_clusters:       number of clusters for kmedoids
        min_cluster_size: minimum cluster size for hdbscan
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
        _run_clustering(scene, source_dir, output_dir, method=method, n_clusters=int(n_clusters), min_cluster_size=int(min_cluster_size))


@task
def list_scenes_task(c):
    """Print all scene IDs, one per line. Used to build SLURM array inputs."""
    from analysis.changepoints import list_scenes
    for scene in list_scenes(Path(c.config.get("source_data_dir"))):
        print(scene)


@task
def run_notebooks(c):
    """Generate figures from pipeline output using notebooks."""
    from airoh.utils import run_notebooks as airoh_run_notebooks, ensure_dir_exist

    notebooks_dir = Path(c.config.get("notebooks_dir"))
    output_dir = Path(c.config.get("output_data_dir")).resolve()

    ensure_dir_exist(c, "output_data_dir")
    airoh_run_notebooks(c, notebooks_dir, output_dir, keys=["source_data_dir", "output_data_dir"])


@task(pre=[fetch, run_changepoints, run_clustering, run_notebooks])
def run(c):
    """Full pipeline."""
    print("Pipeline complete.")


@task
def run_smoke(c):
    """Smoke test: minimal end-to-end pass."""
    fetch(c)
    run_changepoints(c, smoke=True)   # first subject, first 500 attempts
    run_clustering(c, smoke=True)     # first scene only
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
def clean_clustering(c):
    """Remove clustering outputs."""
    output_dir = Path(c.config.get("output_data_dir")) / "clusters"
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Removed {output_dir}")
    else:
        print("No clustering output to clean.")


@task(pre=[clean_changepoints, clean_clustering])
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
