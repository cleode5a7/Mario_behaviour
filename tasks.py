import os
import re
import shutil
from pathlib import Path
from invoke import task


@task(help={
    "source": "Path to an existing mario.scenes checkout to symlink instead of cloning.",
})
def fetch_mario_scenes(c, source=None):
    """Make the mario.scenes datalad dataset available, and decompress its archives.

    Two ways in, decided by `airoh.datalad.install_dataset`:

    - **clone** (default) — `datalad clone` the `url` from invoke.yaml, check out
      the configured branch, `datalad get` the content, then run the dataset's
      own `code/archives/decompress.py` to extract the per-scene JSON files.
    - **symlink** — `--source /path/to/checkout` (or the entry's `source:` key)
      points source_data/mario.scenes at a checkout that already exists, so the
      23 GB dataset is not duplicated per project.

    A symlinked checkout is **read as-is**: no branch checkout, no `datalad get`,
    no decompression. Those all write into a dataset this project does not own,
    and silently switching a colleague's branch is worse than telling them the
    checkout is not ready. If it lacks JSON files, this says so and stops.

    Idempotent: an existing checkout (linked or cloned) is left alone.
    """
    from airoh.datalad import get_data, install_dataset

    entry = c.config.get("datasets")["mario_scenes"]
    scenes_dir = Path(entry["output_dir"])
    linking = bool(source or entry.get("source"))

    if not scenes_dir.exists() and not linking and not shutil.which("datalad"):
        print("datalad is not installed — cannot clone mario.scenes.")
        print("Install it with:  pip install datalad  (or  uv add datalad)")
        print("Already have the dataset? Point at it instead:")
        print("  invoke fetch-mario-scenes --source /path/to/mario.scenes")
        print("See: https://www.datalad.org/get_datalad.html")
        return

    fresh_clone = not scenes_dir.exists() and not linking
    install_dataset(c, "mario_scenes", source=source)

    if fresh_clone:
        c.run(f"git -C {scenes_dir} checkout {entry['branch']}")
        get_data(c, "mario_scenes")

    json_files = list(scenes_dir.glob("**/*.json"))
    if json_files:
        print(f"JSON files already present ({len(json_files)} found), skipping decompress.")
    elif scenes_dir.is_symlink():
        print(f"⚠️  {scenes_dir} links to an external checkout with no JSON files.")
        print("    Leaving it untouched — run code/archives/decompress.py there yourself.")
    else:
        print("Decompressing archives...")
        c.run(f"cd {scenes_dir} && python code/archives/decompress.py")


@task(help={
    "mario-scenes-source": "Path to an existing mario.scenes checkout to symlink "
                           "instead of cloning.",
})
def fetch(c, mario_scenes_source=None):
    """Retrieve all source data assets.

    Each asset has its own fetch-{name} task; this umbrella routes a per-asset
    --{name}-source flag to the matching one. (scenes_mastersheet.csv has no
    fetch task — it is checked into Git.)

    Records what each asset actually resolved to in source_data/MANIFEST.json,
    including the commit of a symlinked external checkout, so the inputs a later
    run consumed stay identifiable. See CLAUDE.md, "Recording asset versions".
    """
    from airoh.provenance import record_sources

    fetch_mario_scenes(c, source=mario_scenes_source)
    record_sources(c)


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
def run_clustering(c, scenes=None, smoke=False, method="consensus", n_clusters=5, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=20.0, recompute_frechet=False, n_jobs=None, bootstraps=200, frac=0.8, seed=0, consensus_epsilon=0.4, scheme="subagging", consensus_cluster="ward", consensus_input="features", consensus_linkage="ward", consensus_k_max=15, consensus_tol=0.0, consensus_min_cluster_size=1):
    """Cluster behavioural traces within each scene using Fréchet distance.

    Defaults to `consensus` — bagged HDBSCAN, where the ensemble produces the
    labels rather than a single fit being trusted. The ensemble is then cut by Ward
    linkage at a silhouette-chosen k (`--consensus-cluster ward`), which replaced a
    hand-swept merge threshold; `--consensus-cluster hdbscan` restores the older fit
    on 1 - C that `--consensus-epsilon` belongs to.

    The k rule is the plain one: cut the Ward tree at every k from 2 to
    `--consensus-k-max`, score each on 1 - C, take the highest silhouette. Two guards
    on that rule ship **off** (`--consensus-tol 0.0`, `--consensus-min-cluster-size 1`)
    and are documented on those flags; turn them on to compare. min_cluster_size=5 and
    epsilon=20 are not arbitrary: they are what `select-cluster-params` chose from
    a 12-point sweep over 288 scenes (82% of all traces land in a cluster that
    survives resampling, median k=4). See output_data/stability/selected_params.json
    for the full grid, and run the sweep again if the source data changes.

    The Fréchet distance matrix is cached under output_data/frechet/; re-running
    with a different epsilon/k/method reuses it (skipping the expensive O(n^2)
    recompute). Delete only the <scene>_clusters.json to re-cluster cheaply.

    Args:
        method:                     consensus (default), hdbscan (single fit) or kmedoids
        bootstraps:                 consensus replicates (200 verified converged)
        frac:                       subsample fraction, drawn without replacement
        seed:                       makes the consensus labels reproducible
        consensus_cluster:          how the ensemble is turned into labels: ward
                                    (default) or hdbscan
        consensus_input:            for ward: features (default) treats C as an
                                    observation matrix, so Ward runs on Euclidean
                                    distances between co-occurrence profiles — which is
                                    what makes its criterion well posed; distance passes
                                    1 - C as a precomputed dissimilarity instead
        consensus_linkage:          linkage for ward (ward/average/complete)
        consensus_k_max:            largest k the silhouette selection considers
        consensus_tol:              plateau tie-break, off at 0.0 (default) — among
                                    cuts within this of the best silhouette, the
                                    smallest k wins. The silhouette on 1 - C flattens
                                    rather than falling, so raising this stops the
                                    argmax drifting up that tail
        consensus_min_cluster_size: eligibility floor on a Ward cut, off at 1 (default)
                                    — a cut is a candidate only if its smallest cluster
                                    clears it. NOT the same knob as min_cluster_size
                                    below, which is an HDBSCAN density parameter for the
                                    replicate fits (and is scaled by frac); this one is
                                    a test on the full-data cut. Raise it to keep Ward
                                    from cutting off singletons — it has no noise class,
                                    and the silhouette cannot penalise one
        consensus_epsilon:          merge threshold for the *hdbscan* consensus fit
                                    only, in co-association distance (NOT Fréchet
                                    units); 0.4 = "co-clustered in >60% of shared
                                    replicates". Ignored when consensus_cluster=ward.
        scheme:                     subagging (default; 80% without replacement) or
                                    bootstrap (Breiman: n with replacement). Bootstrap
                                    replicates hold ~63% unique points and overestimate
                                    k by ~29% — see CLAUDE.md before switching.
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
        _run_clustering(scene, source_dir, output_dir, method=method, n_clusters=int(n_clusters), min_cluster_size=int(min_cluster_size), min_samples=_min_samples, cluster_selection_epsilon=float(cluster_selection_epsilon), recompute_frechet=recompute_frechet, n_jobs=_n_jobs, bootstraps=int(bootstraps), frac=float(frac), seed=int(seed), consensus_epsilon=float(consensus_epsilon), scheme=scheme, consensus_cluster=consensus_cluster, consensus_input=consensus_input, consensus_linkage=consensus_linkage, consensus_k_max=int(consensus_k_max), consensus_tol=float(consensus_tol), consensus_min_cluster_size=int(consensus_min_cluster_size))


@task
def run_clustering_by_category(c, categories=None, smoke=False, method="hdbscan", n_clusters=5, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, recompute_frechet=False, n_jobs=None, bootstraps=200, frac=0.8, seed=0, consensus_epsilon=0.4, scheme="subagging", consensus_cluster="ward", consensus_input="features", consensus_linkage="ward", consensus_k_max=15, consensus_tol=0.0, consensus_min_cluster_size=1):
    """Cluster behavioural traces pooled across scenes sharing a design-pattern category.

    Traces are pooled across every scene tagged with the category (from
    scenes_mastersheet.csv) and re-based to a common origin before clustering,
    since raw traces from different scenes sit at different absolute positions.

    Unlike the per-scene task this still defaults to a single `hdbscan` fit, not
    `consensus`. A category pools up to ~19,000 traces, where the distance matrix
    alone is ~2.9 GB and each of the 200 subsample copies ~1.9 GB — bagging needs a
    memory strategy (float32, a lower B, and the Jaccard-only path with no
    co-association matrix) that is not built yet. `--method consensus` works and is
    correct for the small categories; size the job accordingly before using it on a
    large one. The Ward consensus step adds an O(n^2) linkage on top of that, so it
    is the same warning, not a new one.

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
        _run_clustering_by_category(category, source_dir, output_dir, method=method, n_clusters=int(n_clusters), min_cluster_size=int(min_cluster_size), min_samples=_min_samples, cluster_selection_epsilon=float(cluster_selection_epsilon), recompute_frechet=recompute_frechet, n_jobs=_n_jobs, bootstraps=int(bootstraps), frac=float(frac), seed=int(seed), consensus_epsilon=float(consensus_epsilon), scheme=scheme, consensus_cluster=consensus_cluster, consensus_input=consensus_input, consensus_linkage=consensus_linkage, consensus_k_max=int(consensus_k_max), consensus_tol=float(consensus_tol), consensus_min_cluster_size=int(consensus_min_cluster_size))


@task
def run_cluster_audit(c, scenes=None, smoke=False, bootstraps=60, frac=0.8):
    """Audit stored cluster files: recover their parameters, then score them.

    The stored <scene>_clusters.json records no parameters, and the scenes were not
    all clustered alike, so this reconstructs the settings that reproduce each
    partition (by grid search, ARI >= 0.999) and *then* measures how well that
    partition survives resampling. Recovering first matters: replicate fits made
    with the wrong settings depress the Jaccard through a parameter mismatch rather
    than through genuine instability.

    Reads only output_data/ (the stored labels and the cached Fréchet matrices);
    never opens mario.scenes. Results merge into output_data/stability/audit.json,
    so --scenes chunks accumulate and a re-run skips what is already recorded.
    """
    from analysis.stability import run_cluster_audit as _run_cluster_audit

    output_dir = Path(c.config.get("output_data_dir"))

    clusters_dir = output_dir / "clusters"
    if not clusters_dir.exists():
        print("No clustering output to audit — run invoke run-clustering first.")
        return

    all_scenes = sorted(p.name[: -len("_clusters.json")]
                        for p in clusters_dir.glob("*_clusters.json"))
    if not all_scenes:
        print("No clustering output to audit — run invoke run-clustering first.")
        return
    if smoke:
        all_scenes = all_scenes[:1]
    if scenes:
        all_scenes = scenes.split(",")

    _run_cluster_audit(all_scenes, output_dir,
                       B=int(bootstraps), frac=float(frac))


@task
def run_cluster_sweep(c, scenes=None, smoke=False, min_cluster_sizes="5,10,20",
                      epsilons="0,10,20,40", bootstraps=60, frac=0.8):
    """Score each (min_cluster_size, epsilon) by how well its clusters resample.

    For every grid point: fit HDBSCAN on the full cached Fréchet matrix, refit on
    B random 80% subsamples, and record what share of the scene's traces sit in a
    cluster whose mean bootstrap Jaccard clears 0.75. Count parameters are scaled
    by the subsample fraction inside the resampler; epsilon, being a distance
    threshold, is not.

    Reads only the cached matrices under output_data/frechet/ — never opens
    mario.scenes. Writes output_data/stability/<scene>_sweep.json.

    Args:
        min_cluster_sizes:  comma-separated grid axis (default 5,10,20)
        epsilons:           comma-separated grid axis, Fréchet units (default 0,10,20,40)
        bootstraps:         replicates per grid point; 60 is ample to *rank* points
        frac:               subsample fraction, drawn without replacement
    """
    from analysis.stability import run_sweep

    output_dir = Path(c.config.get("output_data_dir"))
    frechet_dir = output_dir / "frechet"
    if not frechet_dir.exists():
        print("No cached Fréchet matrices — run invoke run-clustering first.")
        return

    all_scenes = sorted(p.stem for p in frechet_dir.glob("*.npy")
                        if re.fullmatch(r"w\d+l\d+s\d+", p.stem))
    if not all_scenes:
        print("No cached Fréchet matrices — run invoke run-clustering first.")
        return
    if smoke:
        all_scenes = all_scenes[:1]
    if scenes:
        all_scenes = scenes.split(",")

    grid = [(int(m), float(e))
            for m in min_cluster_sizes.split(",")
            for e in epsilons.split(",")]
    run_sweep(all_scenes, output_dir, grid=grid,
              B=int(bootstraps), frac=float(frac))


@task
def select_cluster_params(c, floor=0.80):
    """Pick one global (min_cluster_size, epsilon) from the per-scene sweeps.

    Aggregates trace-weighted, then takes the largest k whose stable share clears
    `floor`. Maximising stability alone would pick the degenerate k=2 collapse, so
    the rule prefers the finest partition that still reproduces. Writes
    output_data/stability/selected_params.json.
    """
    from analysis.stability import select_global_params

    select_global_params(Path(c.config.get("output_data_dir")), floor=float(floor))


@task
def run_scheme_comparison(c, scenes=None, smoke=False, bootstraps=200, frac=0.8,
                          min_cluster_size=5, cluster_selection_epsilon=20.0,
                          consensus_epsilon=0.4, consensus_cluster="ward",
                          consensus_input="features", n_jobs=None):
    """Cluster every scene under BOTH resampling schemes and write the comparison.

    subagging (frac*n without replacement, the default) vs Breiman bootstrap
    (n with replacement), at identical parameters, B and seed — so the only thing
    that differs is how replicates are drawn. Writes
    output_data/stability/scheme_comparison.json.

    The consensus aggregation is held fixed at whatever `--consensus-cluster` says
    (ward by default, matching run-clustering) so this isolates the scheme. To compare
    the aggregations instead, use run-consensus-comparison.

    Reads only the cached Fréchet matrices; never opens mario.scenes. Parallel over
    scenes, so -1 uses all cores.
    """
    from analysis.stability import run_scheme_comparison as _compare

    output_dir = Path(c.config.get("output_data_dir"))
    frechet_dir = output_dir / "frechet"
    if not frechet_dir.exists():
        print("No cached Fréchet matrices — run invoke run-clustering first.")
        return

    all_scenes = sorted(p.stem for p in frechet_dir.glob("*.npy")
                        if re.fullmatch(r"w\d+l\d+s\d+", p.stem))
    if smoke:
        all_scenes = all_scenes[:1]
    if scenes:
        all_scenes = scenes.split(",")

    _n_jobs = int(n_jobs) if n_jobs is not None else int(
        os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    _compare(all_scenes, output_dir,
             params={"min_cluster_size": int(min_cluster_size),
                     "cluster_selection_epsilon": float(cluster_selection_epsilon)},
             B=int(bootstraps), frac=float(frac),
             consensus_epsilon=float(consensus_epsilon),
             consensus_cluster=consensus_cluster, consensus_input=consensus_input,
             n_jobs=_n_jobs)


@task
def run_consensus_comparison(c, scenes=None, smoke=False, bootstraps=200, frac=0.8,
                             min_cluster_size=5, cluster_selection_epsilon=20.0,
                             consensus_epsilon=0.4, scheme="subagging", n_jobs=None):
    """Score every scene under all three consensus aggregations and write the comparison.

    Ward on the co-association matrix as a feature matrix (the default), Ward on 1 - C
    as a precomputed dissimilarity, and the older HDBSCAN fit on 1 - C at
    --consensus-epsilon. Identical replicate parameters, B and seed, and in fact the
    identical ensemble — it is built once per scene and shared — so the only thing that
    differs is the last step. Writes output_data/stability/consensus_comparison.json.

    This is what makes the choice of aggregation inspectable rather than asserted, the
    same job run-scheme-comparison does for the resampling scheme.

    Reads only the cached Fréchet matrices; never opens mario.scenes. Parallel over
    scenes, so -1 uses all cores.
    """
    from analysis.stability import run_consensus_comparison as _compare

    output_dir = Path(c.config.get("output_data_dir"))
    frechet_dir = output_dir / "frechet"
    if not frechet_dir.exists():
        print("No cached Fréchet matrices — run invoke run-clustering first.")
        return

    all_scenes = sorted(p.stem for p in frechet_dir.glob("*.npy")
                        if re.fullmatch(r"w\d+l\d+s\d+", p.stem))
    if smoke:
        all_scenes = all_scenes[:1]
    if scenes:
        all_scenes = scenes.split(",")

    _n_jobs = int(n_jobs) if n_jobs is not None else int(
        os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    _compare(all_scenes, output_dir,
             params={"min_cluster_size": int(min_cluster_size),
                     "cluster_selection_epsilon": float(cluster_selection_epsilon)},
             B=int(bootstraps), frac=float(frac), scheme=scheme,
             consensus_epsilon=float(consensus_epsilon), n_jobs=_n_jobs)


@task
def run_coassociation(c, scenes=None, smoke=False, force=False):
    """Store each scene's co-association matrix (the consensus matrix) as uint8.

    cluster_traces builds this matrix, derives the labels and stability from it, then
    discards it — so nothing could display it. This regenerates it from the cached
    Fréchet matrix plus the params each cluster file records (the draws are seeded, so
    it reproduces exactly), and writes output_data/coassociation/<scene>.npy.

    Feeds the dashboard's Matrix view via run-web-export. Reads only output_data/;
    never opens mario.scenes. ~13 MB for the whole corpus.
    """
    from analysis.stability import run_coassociation as _run

    output_dir = Path(c.config.get("output_data_dir"))
    clusters_dir = output_dir / "clusters"
    if not clusters_dir.exists():
        print("No clustering output — run invoke run-clustering first.")
        return

    all_scenes = sorted(p.name[: -len("_clusters.json")]
                        for p in clusters_dir.glob("*_clusters.json"))
    if smoke:
        all_scenes = all_scenes[:1]
    if scenes:
        all_scenes = scenes.split(",")

    _run(all_scenes, output_dir, force=force)


@task
def clean_coassociation(c):
    """Remove stored co-association matrices."""
    output_dir = Path(c.config.get("output_data_dir")) / "coassociation"
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Removed {output_dir}")
    else:
        print("No co-association matrices to clean.")


@task
def run_practice_distances(c, scenes=None, smoke=False, force=False):
    """Measure every attempt against its scene's cluster medoids.

    Per scene: the medoid of each cluster, then for each attempt its distance to its
    own cluster's medoid (d_own) and to the nearest medoid (d_nearest, and normalised
    by that cluster's spread), each also leave-subject-out. Writes
    output_data/practice/attempts/<scene>.csv. Reads only output_data/ (cluster files
    and cached Fréchet matrices), so it needs run-clustering and never opens
    mario.scenes.
    """
    from analysis.practice import run_practice_distances as _run

    output_dir = Path(c.config.get("output_data_dir"))
    clusters_dir = output_dir / "clusters"
    all_scenes = sorted(p.name[: -len("_clusters.json")]
                        for p in clusters_dir.glob("*_clusters.json")) if clusters_dir.exists() else []
    if not all_scenes:
        print("No clustering output — run invoke run-clustering first.")
        return
    if smoke:
        all_scenes = all_scenes[:1]
    if scenes:
        all_scenes = scenes.split(",")

    _run(all_scenes, output_dir, force=force)


@task
def run_practice_models(c, measures="d_nearest_norm,d_own", min_attempts=10,
                        smoke=False, force=False):
    """Model the practice effect on the medoid distances.

    Per series (subject x scene x cluster for d_own, subject x scene for
    d_nearest_norm): linear / quadratic / power / exponential curves compared by
    AICc. Per subject: a mixed model per measure (random intercept + slope per scene,
    or per scene x cluster for d_own), raw and leave-subject-out, all attempts and
    completed only, each on log(d) (primary, zeros dropped) and log1p(d) (original,
    kept for comparison); fitted in parallel. Beside them, alternative random-effect
    structures (scenes nested in levels, and its like-for-like scene-only twin) in
    nested.json. Plus the strategy-switch trend. Writes
    output_data/practice/{curves.csv,mixed.json,mixed_summary.csv,switching.json}. --smoke fits the
    first five scenes with attempt files.
    """
    from analysis.practice import practice_dir, run_practice_models as _run

    output_dir = Path(c.config.get("output_data_dir"))
    files = sorted((practice_dir(output_dir) / "attempts").glob("*.csv"))
    if not files:
        print("No attempt distances — run invoke run-practice-distances first.")
        return
    scenes = [f.stem for f in files[:5]] if smoke else None
    _run(output_dir, measures=tuple(measures.split(",")),
         min_attempts=int(min_attempts), scenes=scenes, force=force)


@task
def run_web_export(c, scenes=None, smoke=False, step=3, force=False, backfill=False):
    """Export clustered traces to a static payload for the browser dashboard.

    The cluster JSONs hold labels + metadata but no geometry, so the dashboard
    needs the x/y polylines baked out of mario.scenes once. Writes one file per
    scene under output_data/web/data/scenes/, plus an index and a copy of the
    scene background PNGs, so the served tree never reaches back into the 23 GB
    source dataset. Also copies the per-subject practice models
    (output_data/practice/mixed.json) to data/practice.json for the dashboard's
    stats section — on every run, --backfill included, since it is a small copy.

    Args:
        step:  keep every step-th frame (first and last always kept). Traces are
               60 fps against a 224 px background, so 3 is visually lossless;
               lower it for more fidelity at the cost of payload size.
        force: re-export scenes whose payload already exists
        backfill: update payloads that already exist, then rewrite the index,
               without reading mario.scenes — seconds, where --force re-reads the
               source dataset. Two things are refreshed: the derived attempt/stage
               fields (computed from what the payload already carries) and the `cons`
               selection record (re-read from the cluster file, with the dendrogram
               leaf order rebuilt from the stored co-association matrix). Use it after
               a change to either; a change to trace geometry or labels is a real
               re-export.
    """
    from analysis.webexport import (backfill_derived, backfill_selection, export_practice,
                                    export_scene, list_clustered_scenes, sync_backgrounds,
                                    write_index)
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
        backfill_selection(web_dir, output_dir)
        export_practice(output_dir, web_dir)
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
    export_practice(output_dir, web_dir)
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


@task(pre=[fetch, run_changepoints, run_changepoints_by_category, run_clustering, run_clustering_by_category, run_practice_distances, run_practice_models, run_web_export, run_notebooks])
def run(c):
    """Full pipeline.

    Steps run through the `pre=` chain above; the body only records what the
    run produced into output_data/PROVENANCE.json (project commit, environment,
    the manifest consumed, and a checksum per output file).
    """
    from airoh.provenance import record_run

    record_run(c, tasks="fetch,run-changepoints,run-changepoints-by-category,"
                        "run-clustering,run-clustering-by-category,run-practice-distances,"
                        "run-practice-models,run-web-export,"
                        "run-notebooks")
    print("Pipeline complete.")


@task(help={
    "skip": "Comma-separated check names to skip.",
    "strict": "Treat warnings as failures.",
})
def verify(c, skip=None, strict=False):
    """Check that the code, config, data and docs still agree.

    Run this before committing. It is deliberately NOT part of `run`:
    reproducing results should never depend on documentation hygiene. See
    CLAUDE.md, "Verification", for what each check covers.
    """
    from airoh.verify import verify as airoh_verify

    airoh_verify(c, skip=skip, strict=strict)


@task
def run_smoke(c):
    """Smoke test: minimal end-to-end pass."""
    fetch(c)
    run_changepoints(c, smoke=True)             # first subject, first 500 attempts
    run_changepoints_by_category(c, smoke=True)  # first subject, per category
    run_clustering(c, smoke=True)                # first scene only
    run_clustering_by_category(c, smoke=True)    # first category only
    run_practice_distances(c, smoke=True)        # medoid distances, first scene
    run_practice_models(c, smoke=True)           # practice models on what exists
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
def clean_stability(c):
    """Remove the cluster audit, sweep, selected-parameter and comparison outputs.

    Removes the whole stability/ directory, so the scheme and consensus comparisons
    go with it — they need no clean- task of their own.
    """
    output_dir = Path(c.config.get("output_data_dir")) / "stability"
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Removed {output_dir}")
    else:
        print("No stability output to clean.")


@task
def clean_practice(c):
    """Remove practice-effect outputs (medoid distances, fits and figures)."""
    output_dir = Path(c.config.get("output_data_dir")) / "practice"
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Removed {output_dir}")
    else:
        print("No practice output to clean.")


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


@task(pre=[clean_changepoints, clean_changepoints_by_category, clean_clustering, clean_clustering_by_category, clean_web_export, clean_stability, clean_coassociation, clean_practice, clean_frechet])
def clean(c):
    """Remove all computed outputs."""
    pass


@task
def clean_mario_scenes(c):
    """Remove the mario.scenes dataset from source_data/.

    A symlink is unlinked, never followed: pointing at an external checkout with
    `fetch-mario-scenes --source` must not put someone else's 23 GB dataset one
    `clean-source` away from deletion. Use this before re-pointing a stale link
    — `fetch` never overwrites what already sits at the destination.
    """
    scenes_dir = Path(c.config.get("datasets")["mario_scenes"]["output_dir"])
    if scenes_dir.is_symlink():
        target = scenes_dir.resolve()
        scenes_dir.unlink()
        print(f"Removed the link {scenes_dir} -> {target} (the checkout itself is untouched)")
    elif scenes_dir.exists():
        shutil.rmtree(scenes_dir)
        print(f"Removed {scenes_dir}")
    else:
        print("mario.scenes not present, nothing to remove.")


@task(pre=[clean_mario_scenes])
def clean_source(c):
    """Remove all downloaded source data."""
    pass
