from pathlib import Path
import json
import re

import numpy as np
import similaritymeasures
import hdbscan
from joblib import Parallel, delayed

from analysis.changepoints import _SCENE_PATTERN


def load_traces(scene_id: str, source_dir: Path) -> tuple[list[np.ndarray], list[dict]]:
    """Load (x, y) player trajectories for all clips of one scene.

    Returns:
        traces:   list of (T, 2) float arrays, one per clip
        metadata: list of dicts with Subject/Session/Run/Outcome per clip
    """
    scenes_dir = source_dir / "mario.scenes"
    records: list[tuple[dict, Path]] = []

    for f in scenes_dir.glob("**/gamelogs/*_variables.json"):
        m = _SCENE_PATTERN.search(f.name)
        if not m or f"{m.group(1)}s{m.group(2)}" != scene_id:
            continue
        summary_path = Path(str(f).replace("_variables.json", "_summary.json"))
        meta = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        records.append((meta, f))

    records.sort(key=lambda r: (
        r[0].get("Subject", ""),
        r[0].get("Session", ""),
        r[0].get("Run", ""),
    ))

    traces: list[np.ndarray] = []
    metadata: list[dict] = []

    for meta, f in records:
        data = json.loads(f.read_text())
        x = np.array(data["player_x_posHi"], dtype=float) * 256 + np.array(data["player_x_posLo"], dtype=float)
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


def kmedoids(dist_matrix: np.ndarray, n_clusters: int, max_iter: int = 100, seed: int = 42) -> np.ndarray:
    """K-medoids clustering on a precomputed distance matrix.

    Medoids are always actual data points, so this works correctly with
    any non-Euclidean distance (unlike k-means which needs centroid recomputation).

    Returns array of cluster labels (length n).
    """
    n = len(dist_matrix)
    rng = np.random.default_rng(seed)
    medoid_idx = rng.choice(n, n_clusters, replace=False)

    for _ in range(max_iter):
        # assign each point to nearest medoid
        labels = dist_matrix[:, medoid_idx].argmin(axis=1)

        new_medoids = medoid_idx.copy()
        for k in range(n_clusters):
            members = np.where(labels == k)[0]
            if len(members) == 0:
                continue
            # medoid = member with smallest total distance to all others in cluster
            within = dist_matrix[np.ix_(members, members)].sum(axis=1)
            new_medoids[k] = members[within.argmin()]

        if np.array_equal(new_medoids, medoid_idx):
            break
        medoid_idx = new_medoids

    return labels


def run_clustering(
    scene_id: str,
    source_dir: Path,
    output_dir: Path,
    method: str = "hdbscan",
    n_clusters: int = 5,
    min_cluster_size: int = 5,
    n_jobs: int = 1,
) -> None:
    """Cluster behavioural traces for one scene using Fréchet distance.

    Args:
        method:           "hdbscan" (default, no k needed) or "kmedoids"
        n_clusters:       number of clusters for kmedoids
        min_cluster_size: minimum cluster size for hdbscan

    Input:   all *_variables.json matching scene_id in mario.scenes/
    Output:  output_dir/<scene_id>_clusters.json
             {"scene": str, "method": str, "labels": [int, ...], "metadata": [...]}
    """
    traces, metadata = load_traces(scene_id, source_dir)

    if not traces:
        print(f"No trace data found for scene {scene_id}, skipping.")
        return

    print(f"Scene {scene_id}: computing Fréchet distances for {len(traces)} traces (n_jobs={n_jobs})...")
    dist_matrix = frechet_distance_matrix(traces, n_jobs=n_jobs)

    if method == "hdbscan":
        labels = hdbscan.HDBSCAN(
            metric="precomputed",
            min_cluster_size=min_cluster_size,
        ).fit_predict(dist_matrix).tolist()
    elif method == "kmedoids":
        labels = kmedoids(dist_matrix, n_clusters=n_clusters).tolist()
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'hdbscan' or 'kmedoids'.")

    n_found = len(set(l for l in labels if l != -1))
    print(f"Scene {scene_id}: {n_found} clusters found ({labels.count(-1)} noise points)")

    out = output_dir / f"{scene_id}_clusters.json"
    out.write_text(json.dumps({
        "scene": scene_id,
        "method": method,
        "labels": labels,
        "metadata": metadata,
    }, indent=2))
