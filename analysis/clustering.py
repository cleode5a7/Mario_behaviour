from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import similaritymeasures
import hdbscan
from joblib import Parallel, delayed

from analysis.changepoints import _SCENE_PATTERN

# Binary scene-design-pattern columns in scenes_mastersheet.csv (Zenodo 15586709).
SCENE_PATTERN_COLS = [
    "Enemy", "2-Horde", "3-Horde", "4-Horde", "Roof", "Gap",
    "Multiple gaps", "Variable gaps", "Gap enemy", "Pillar gap", "Valley",
    "Pipe valley", "Empty valley", "Enemy valley", "Roof valley", "2-Path",
    "3-Path", "Risk/Reward", "Stair up", "Stair down", "Empty stair valley",
    "Enemy stair valley", "Gap stair valley", "Reward", "Moving platform",
    "Flagpole", "Beginning", "Bonus zone",
]


def load_scene_patterns(source_dir: Path) -> pd.DataFrame:
    """Long-form (scene_id, pattern) table from scenes_mastersheet.csv.

    A scene can carry multiple pattern tags at once (not mutually exclusive),
    so each scene contributes one row per tag it has set. Use this to group
    per-scene results by scene category (e.g. "Gap stair valley") instead of
    by individual scene.
    """
    df = pd.read_csv(source_dir / "scenes_mastersheet.csv").dropna(subset=["World", "Level", "Scene"])
    df["scene_id"] = df.apply(
        lambda r: f"w{int(r['World'])}l{int(r['Level'])}s{int(r['Scene'])}", axis=1
    )
    long = df[["scene_id"] + SCENE_PATTERN_COLS].melt(
        id_vars="scene_id", var_name="pattern", value_name="present"
    )
    return long[long["present"] == 1].drop(columns="present").reset_index(drop=True)


def list_categories(source_dir: Path) -> list[str]:
    """All scene-category (pattern tag) names, sorted."""
    return sorted(load_scene_patterns(source_dir)["pattern"].unique().tolist())


def category_slug(category: str) -> str:
    """Filesystem-safe slug for a category name, e.g. 'Gap stair valley' -> 'gap_stair_valley'."""
    return category.lower().replace(" ", "_").replace("/", "-")


def load_traces(scene_id: str, source_dir: Path, y_mode: str = "raw") -> tuple[list[np.ndarray], list[dict]]:
    """Load (x, y) player trajectories for all clips of one scene.

    Args:
        y_mode: how to derive the vertical coordinate.
            "raw"    — raw ``player_y_pos`` (default). This is what clustering
                       distances/labels are computed on; keep it unchanged so
                       existing results stay reproducible.
            "screen" — screen-corrected y for plotting only:
                       ``player_y_pos + 24 - (1 - player_y_screen) * 256`` with
                       frames masked where ``player_state == 0``. Stacks the
                       vertical NES screens into one continuous coordinate (no
                       mid-trace teleport) and hides fully-inactive frames.
                       Note a pit death is a *real* multi-screen downward fall
                       (``player_y_screen`` runs up as the player drops off the
                       bottom); the plotting layer clips the view to the playfield
                       so those honest-but-huge plunges don't dominate the axis.

    Returns:
        traces:   list of (T, 2) float arrays, one per clip. In "screen" mode the
                  y column is a masked array (matplotlib skips masked frames).
        metadata: list of dicts with Subject/Session/Run/Outcome per clip

    Trace discovery/order is identical across y_modes, so cluster labels (stored
    in this order) map 1:1 regardless of the y coordinate chosen for plotting.
    """
    if y_mode not in ("raw", "screen"):
        raise ValueError(f"Unknown y_mode '{y_mode}'. Use 'raw' or 'screen'.")
    scenes_dir = source_dir / "mario.scenes"
    records: list[tuple[dict, Path]] = []

    for f in scenes_dir.glob("**/gamelogs/*_variables.json"):
        m = _SCENE_PATTERN.search(f.name)
        if not m or f"{m.group(1)}s{m.group(2)}" != scene_id:
            continue
        summary_path = Path(str(f).replace("_variables.json", "_summary.json"))
        meta = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        records.append((meta, f))

    # Sort by a UNIQUE, content-based key. (Subject, Session, Run) alone is not
    # unique — a scene can have many clips per run (e.g. 11–15) — so those ties
    # would otherwise fall back to filesystem glob order, which is not
    # reproducible. ClipCode is unique within (Subject, Session, Run), making the
    # order deterministic across machines and runs (labels are stored/rejoined by
    # this order, so it must be stable). StartFrame is included for readability;
    # it is not unique on its own.
    records.sort(key=lambda r: (
        r[0].get("Subject", ""),
        r[0].get("Session", ""),
        r[0].get("Run", ""),
        r[0].get("StartFrame", 0),
        r[0].get("ClipCode", ""),
    ))

    traces: list[np.ndarray] = []
    metadata: list[dict] = []

    for meta, f in records:
        data = json.loads(f.read_text())
        x = np.array(data["player_x_posHi"], dtype=float) * 256 + np.array(data["player_x_posLo"], dtype=float)
        if y_mode == "screen":
            y_pos = np.array(data["player_y_pos"], dtype=float)
            y_screen = np.array(data["player_y_screen"], dtype=float)
            state = np.array(data["player_state"], dtype=float)
            y = y_pos + 24 - (1 - y_screen) * 256
            y = np.ma.masked_where(state == 0, y)
            traces.append(np.ma.column_stack([x, y]))
        else:
            y = np.array(data["player_y_pos"], dtype=float)
            traces.append(np.column_stack([x, y]))
        metadata.append(meta)

    return traces, metadata


def frechet_distance_matrix(traces: list[np.ndarray], n_jobs: int = 1) -> np.ndarray:
    """Compute symmetric pairwise Fréchet distance matrix.

    Args:
        n_jobs: number of parallel workers (passed to joblib; -1 = all CPUs)
    """
    n = len(traces)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    distances = Parallel(n_jobs=n_jobs)(
        delayed(similaritymeasures.frechet_dist)(traces[i], traces[j])
        for i, j in pairs
    )
    dist = np.zeros((n, n))
    for (i, j), d in zip(pairs, distances):
        dist[i, j] = dist[j, i] = d
    return dist


def frechet_matrix_cached(
    traces: list[np.ndarray],
    cache_path: Path,
    n_jobs: int = 1,
    recompute: bool = False,
) -> np.ndarray:
    """Load a cached Fréchet distance matrix, or compute-and-cache it.

    The matrix depends only on the traces (and their deterministic load order),
    never on the clustering method/epsilon/k — so it is the expensive part worth
    caching. Re-clustering with a different epsilon can then reuse this and skip
    the O(n^2) Fréchet recomputation entirely.

    The cache is validated by trace count: if the on-disk matrix does not match
    the current number of traces (e.g. new source data landed), it is recomputed.
    Pass ``recompute=True`` to force a fresh computation regardless.
    """
    if cache_path.exists() and not recompute:
        dist = np.load(cache_path)
        if dist.shape == (len(traces), len(traces)):
            print(f"Loaded cached Fréchet matrix from {cache_path} ({len(traces)} traces).")
            return dist
        print(f"Cached matrix at {cache_path} is stale "
              f"({dist.shape[0]} != {len(traces)} traces), recomputing.")

    dist = frechet_distance_matrix(traces, n_jobs=n_jobs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, dist)
    print(f"Saved Fréchet matrix to {cache_path}.")
    return dist


def kmedoids(dist_matrix: np.ndarray, n_clusters: int, max_iter: int = 100, seed: int = 42) -> np.ndarray:
    """K-medoids clustering on a precomputed distance matrix.

    Medoids are always actual data points, so this works correctly with
    any non-Euclidean distance (unlike k-means which needs centroid recomputation).

    Returns array of cluster labels (length n).
    """
    n = len(dist_matrix)
    n_clusters = min(n_clusters, n)  # can't have more medoids than points
    rng = np.random.default_rng(seed)
    medoid_idx = rng.choice(n, n_clusters, replace=False)

    for _ in range(max_iter):
        # assign each point to nearest medoid
        labels = dist_matrix[:, medoid_idx].argmin(axis=1)

        new_medoids = medoid_idx.copy()
        for k in range(n_clusters):
            members = np.where(labels == k)[0]
            if len(members) == 0:
                # Re-seed an emptied cluster to the point farthest from all current
                # medoids, so clusters don't silently collapse onto each other.
                nearest = dist_matrix[:, medoid_idx].min(axis=1)
                new_medoids[k] = int(nearest.argmax())
                continue
            # medoid = member with smallest total distance to all others in cluster
            within = dist_matrix[np.ix_(members, members)].sum(axis=1)
            new_medoids[k] = members[within.argmin()]

        if np.array_equal(new_medoids, medoid_idx):
            break
        medoid_idx = new_medoids

    return labels


def cluster_traces(
    traces: list[np.ndarray] | None = None,
    method: str = "hdbscan",
    n_clusters: int = 5,
    min_cluster_size: int = 5,
    cluster_selection_epsilon: float = 0.0,
    n_jobs: int = 1,
    dist_matrix: np.ndarray | None = None,
) -> list[int]:
    """Pairwise Fréchet distance + clustering. Shared by per-scene and per-category runs.

    Args:
        method:                     "hdbscan" (default, no k needed) or "kmedoids"
        n_clusters:                 number of clusters for kmedoids
        min_cluster_size:           minimum cluster size for hdbscan
        cluster_selection_epsilon:  hdbscan merge threshold, in Fréchet-distance
                                    units. Clusters whose separation is below this
                                    are fused, so raise it to get fewer clusters
                                    (0.0 = HDBSCAN's default, no forced merging).
        dist_matrix:                a precomputed Fréchet distance matrix. If given,
                                    ``traces`` is not needed and no distances are
                                    computed — this is the cheap path for re-running
                                    clustering (new epsilon/k/method) on cached
                                    distances. If omitted, distances are computed
                                    from ``traces``.
    """
    if dist_matrix is None:
        if traces is None:
            raise ValueError("cluster_traces needs either traces or a precomputed dist_matrix.")
        dist_matrix = frechet_distance_matrix(traces, n_jobs=n_jobs)

    if method == "hdbscan":
        return hdbscan.HDBSCAN(
            metric="precomputed",
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=float(cluster_selection_epsilon),
        ).fit_predict(dist_matrix).tolist()
    elif method == "kmedoids":
        return kmedoids(dist_matrix, n_clusters=n_clusters).tolist()
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'hdbscan' or 'kmedoids'.")


def run_clustering(
    scene_id: str,
    source_dir: Path,
    output_dir: Path,
    method: str = "hdbscan",
    n_clusters: int = 5,
    min_cluster_size: int = 5,
    cluster_selection_epsilon: float = 0.0,
    n_jobs: int = 1,
    frechet_cache_dir: Path | None = None,
    recompute_frechet: bool = False,
) -> None:
    """Cluster behavioural traces for one scene using Fréchet distance.

    Input:   all *_variables.json matching scene_id in mario.scenes/
    Output:  output_dir/<scene_id>_clusters.json
             {"scene": str, "method": str, "labels": [int, ...], "metadata": [...]}

    The Fréchet distance matrix is cached (under ``frechet_cache_dir``, default
    ``output_dir/../frechet``); re-running with a different epsilon/k/method reuses
    it and skips the O(n^2) recomputation. Pass ``recompute_frechet=True`` to force.
    """
    traces, metadata = load_traces(scene_id, source_dir)

    if not traces:
        print(f"No trace data found for scene {scene_id}, skipping.")
        return

    cache_dir = frechet_cache_dir if frechet_cache_dir is not None else output_dir.parent / "frechet"
    dist_matrix = frechet_matrix_cached(traces, cache_dir / f"{scene_id}.npy",
                                        n_jobs=n_jobs, recompute=recompute_frechet)
    print(f"Scene {scene_id}: clustering {len(traces)} traces (method={method})...")
    labels = cluster_traces(dist_matrix=dist_matrix, method=method, n_clusters=n_clusters,
                             min_cluster_size=min_cluster_size,
                             cluster_selection_epsilon=cluster_selection_epsilon)

    n_found = len(set(l for l in labels if l != -1))
    print(f"Scene {scene_id}: {n_found} clusters found ({labels.count(-1)} noise points)")

    out = output_dir / f"{scene_id}_clusters.json"
    out.write_text(json.dumps({
        "scene": scene_id,
        "method": method,
        "labels": labels,
        "metadata": metadata,
    }, indent=2))


def run_clustering_by_category(
    category: str,
    source_dir: Path,
    output_dir: Path,
    method: str = "hdbscan",
    n_clusters: int = 5,
    min_cluster_size: int = 5,
    cluster_selection_epsilon: float = 0.0,
    n_jobs: int = 1,
    frechet_cache_dir: Path | None = None,
    recompute_frechet: bool = False,
) -> None:
    """Cluster behavioural traces pooled across every scene tagged with a category.

    Traces come from different scenes at different absolute level positions, so
    each trace is re-based to start at (0, 0) before computing Fréchet distance —
    otherwise distance would be dominated by scene offset rather than trace shape.

    Input:   scenes_mastersheet.csv (category -> scene_ids) + matching
             *_variables.json under mario.scenes/
    Output:  output_dir/<category-slug>_clusters.json
             {"category": str, "method": str, "labels": [int, ...], "metadata": [...]}
             metadata entries carry "SceneID" so traces stay attributable to their scene.

    The Fréchet distance matrix is cached (under ``frechet_cache_dir``, default
    ``output_dir/../frechet``); re-running with a different epsilon/k/method reuses
    it and skips the O(n^2) recomputation. Pass ``recompute_frechet=True`` to force.
    """
    patterns = load_scene_patterns(source_dir)
    scene_ids = sorted(patterns.loc[patterns["pattern"] == category, "scene_id"].unique())

    if not scene_ids:
        print(f"No scenes tagged '{category}', skipping.")
        return

    traces: list[np.ndarray] = []
    metadata: list[dict] = []
    for scene_id in scene_ids:
        scene_traces, scene_metadata = load_traces(scene_id, source_dir)
        traces.extend(trace - trace[0] for trace in scene_traces)
        metadata.extend(scene_metadata)

    if not traces:
        print(f"No trace data found for category '{category}' ({len(scene_ids)} scenes), skipping.")
        return

    cache_dir = frechet_cache_dir if frechet_cache_dir is not None else output_dir.parent / "frechet"
    dist_matrix = frechet_matrix_cached(traces, cache_dir / f"{category_slug(category)}.npy",
                                        n_jobs=n_jobs, recompute=recompute_frechet)
    print(f"Category '{category}': clustering {len(traces)} traces "
          f"across {len(scene_ids)} scenes (method={method})...")
    labels = cluster_traces(dist_matrix=dist_matrix, method=method, n_clusters=n_clusters,
                             min_cluster_size=min_cluster_size,
                             cluster_selection_epsilon=cluster_selection_epsilon)

    n_found = len(set(l for l in labels if l != -1))
    print(f"Category '{category}': {n_found} clusters found ({labels.count(-1)} noise points)")

    out = output_dir / f"{category_slug(category)}_clusters.json"
    out.write_text(json.dumps({
        "category": category,
        "method": method,
        "labels": labels,
        "metadata": metadata,
    }, indent=2))
