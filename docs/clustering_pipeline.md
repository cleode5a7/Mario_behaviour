# Per-scene clustering pipeline: call flow

How `invoke run-clustering` turns one scene's traces into cluster labels, from the
Fréchet distance matrix through to cutting the Ward tree at the elbow k. Written as a
reading guide for reviewing the code: it names every function on the default path, in
the order the data moves through them.

This covers the default path only (`--method consensus`, `--consensus-cluster ward`,
`--consensus-k-rule elbow`). The *why* behind each choice — subagging over bootstrap,
Ward over HDBSCAN on `1 - C`, elbow over the silhouette argmax — is recorded in
`CLAUDE.md` and in the docstrings and module constants of `analysis/stability.py`.
This file is only the map.

## Where the code lives

| File | Role |
|---|---|
| `tasks.py` | The `run-clustering` invoke task: picks scenes, forwards flags, skips scenes already done |
| `analysis/clustering.py` | Loads traces, computes and caches the Fréchet matrix, dispatches to the clustering method |
| `analysis/stability.py` | All of the ensemble maths. Takes a distance matrix, never a trace |

## Call tree

`(lib)` marks a call into an external library. `→` is what the step returns.

```
uv run invoke run-clustering
│
└─ tasks.py            run_clustering (invoke task)      loops over scenes, forwards flags
   │
   └─ clustering.py    run_clustering()                  one scene → <scene_id>_clusters.json
      │
      ├─ clustering.py load_traces()                     → traces, metadata
      │
      ├─ clustering.py frechet_matrix_cached()           → dist_matrix (n × n)
      │   └─ clustering.py frechet_distance_matrix()     (only when no valid cache)
      │       ├─ joblib.Parallel                              (lib)
      │       └─ similaritymeasures.frechet_dist per pair     (lib)
      │
      └─ clustering.py cluster_traces()                  dispatches on --method
         │
         └─ clustering.py _consensus()   ◄── orchestrator: calls ①–⑤ in order
            │
            ├─ ① stability.py resample_labels()          → replicates
            │     └─ stability.py _one_replicate()   × B (default 200)
            │         └─ stability.py _fit()
            │             └─ hdbscan.HDBSCAN(metric="precomputed")   (lib)
            │
            ├─ ② stability.py coassociation()            → C (n × n)
            │
            ├─ ③ stability.py ward_consensus_labels()    → labels, report
            │     ├─ stability.py _consensus_tree()      → linkage tree Z
            │     │     └─ scipy linkage(C, method="ward")           (lib)
            │     ├─ scipy leaves_list(Z)                            (lib)
            │     ├─ stability.py _candidate_cuts()      cuts for k = 2 .. k_max
            │     │     └─ scipy cut_tree(Z, n_clusters=k)           (lib)
            │     ├─ sklearn silhouette_score, once per k            (lib)
            │     │        → the silhouette-vs-k curve
            │     ├─ stability.py elbow_k(curve)         → elbow k   ★
            │     ├─ --consensus-k-rule picks the k (elbow by default)
            │     ├─ stability.py cut_bounds()           candidate cuts for the dashboard
            │     └─ stability.py _relabel_by_size()     cluster 0 = largest
            │
            ├─ ④ stability.py jaccard_stability(labels, replicates)
            │
            └─ ⑤ stability.py stable_share(), trace_confidence()
                     → stability dict, stored in the cluster file
```

## What travels between the steps

```
traces ─► dist_matrix ─► B replicates ─► C ─► Ward tree ─► silhouette curve ─► k ─► labels
(list)      (n × n)     (indices,        (n×n)    (Z)        (one score per k)   ★   (one per trace)
                         labels) × B
                            │                                                          │
                            └──────────────────────────────────────────────────────────┴─► jaccard,
                                                                                            stable_share
```

The replicates are used **twice**: once to build `C`, and again at the end to score
how well the final labels survive resampling. That second use is easy to miss.

## The steps

| Step             | Function                                                | Input                                                                | Output                                                                                   |
| ---------------- | ------------------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Load             | `load_traces()`                                         | scene id, `source_data/mario.scenes/`                                | one (x, y) array per clip, plus its metadata                                             |
| Distance         | `frechet_matrix_cached()` → `frechet_distance_matrix()` | traces                                                               | symmetric n × n Fréchet matrix, cached in `output_data/frechet/`                         |
| ① Resample       | `resample_labels()`                                     | distance matrix                                                      | B pairs of (drawn trace indices, HDBSCAN labels on that subsample)                       |
| ② Co-associate   | `coassociation()`                                       | replicates                                                           | `C[i, j]` = share of replicates holding both i and j that put them in one cluster        |
| ③ Cut            | `ward_consensus_labels()`                               | `C` (plus the Fréchet matrix, for a recorded-only second silhouette) | one label per trace, plus a report holding the whole silhouette-vs-k curve and the elbow |
| ④ Score clusters | `jaccard_stability()`                                   | labels, replicates                                                   | mean Jaccard per cluster over the replicates                                             |
| ⑤ Summarise      | `stable_share()`, `trace_confidence()`                  | labels, Jaccard, `C`                                                 | share of traces in a stable cluster; per-trace confidence                                |

Everything ends up in `output_data/clusters/<scene_id>_clusters.json`: the labels, the
metadata that identifies each trace, the `params` the run used, and the `stability`
block (whose `consensus` entry carries the silhouette curve, the elbow and the chosen k).

The co-association matrix itself is **not** saved by this path; `run-coassociation`
regenerates it from the cached Fréchet matrix and the recorded `params`.

## Where the flags go

Every `--consensus-*` flag on `invoke run-clustering` is passed unchanged from the task
in `tasks.py`, through `run_clustering()` and `cluster_traces()`, to `_consensus()`,
which hands it to the step it belongs to:

| Flag | Used by |
|---|---|
| `--bootstraps`, `--frac`, `--seed`, `--scheme`, `--min-cluster-size`, `--cluster-selection-epsilon` | ① `resample_labels()` (the HDBSCAN replicate fits) |
| `--consensus-input`, `--consensus-linkage` | ③ `_consensus_tree()` |
| `--consensus-k-max`, `--consensus-min-cluster-size` | ③ `_candidate_cuts()` |
| `--consensus-k-rule`, `--consensus-tol` | ③ the k choice inside `ward_consensus_labels()` |
| `--consensus-elbow-frac`, `--consensus-elbow-min-drop` | ③ `elbow_k()` |

`--min-cluster-size` and `--consensus-min-cluster-size` are different parameters: the
first is an HDBSCAN density setting for the replicate fits, the second a minimum
cluster size for a Ward cut to be eligible.

## Suggested reading order

1. `_consensus()` in `analysis/clustering.py` — the whole method in five calls.
2. `resample_labels()` and `coassociation()` in `analysis/stability.py`.
3. `ward_consensus_labels()`, then `_consensus_tree()` and `_candidate_cuts()`.
4. `elbow_k()` — it reads only the silhouette curve, never the tree or the traces.
5. `jaccard_stability()` and `stable_share()`.

For each one, write down what you expect it to return before opening it.




stats modelling measures

Part 1: What d_nearest_norm_loso measures

Take one scene, say w1l1s0. Every time a player goes through it, we record their path, which is one trace. The clustering has already sorted all traces of that scene (from all five players) into groups of similar routes, like "jumps over the pipe early" or "runs along the bottom".

The name has four parts:

1. Medoid. Each cluster gets a representative trace: the real trace that is most "central", meaning its total distance to all the other members is smallest. It works like the average route, except it's an actual attempt someone made. (medoids(), analysis/practice.py:82)

2. d_nearest. For a new attempt, we measure its Fréchet distance to every medoid and keep the smallest. That answers "how close is this attempt to the nearest standard way of playing this scene?" It doesn't matter which strategy the player chose, only how cleanly they carried it out.

3. _norm. Some clusters are naturally tight and others loose. A distance of 30 could be sloppy for one cluster and perfectly normal for another. So we divide by the cluster's typical spread (the median distance of its members to its medoid):
- d_nearest_norm = 1 means "as close as a typical member"
- < 1 means tighter than typical
- > 1 means further out
This puts all scenes and clusters on one comparable scale.

4. _loso (leave-subject-out). When measuring sub-01's attempts, the medoids and spreads are chosen only from the other four players' traces. Without this there's a circularity problem. If sub-01 played a scene 100 times, their own late attempts could become the medoid, and then their late attempts would look close to the prototype because they partly defined it. With LOSO, the reference is set by others, so sub-01 can't pull it toward themselves.

One honest caveat: LOSO removes the player from choosing the medoid, but the clusters themselves were formed with everyone's traces. That's a much weaker influence, but it isn't zero.

Summary: d_nearest_norm_loso is roughly "how far this attempt is from the closest standard route, as defined by other players, in units of how spread-out that route normally is." If players improve with practice, it should go down over attempts.


How the fitting works

Take the simplest version, with only a random intercept per scene (random slopes follow the same logic):

y = a + b·t  +  u_scene  +  ε
              u_scene ~ Normal(0, τ²)     how much scenes differ from each other
              ε       ~ Normal(0, σ²)     attempt-to-attempt noise

The model doesn't fit ~300 separate u_scene values. It fits only 4 numbers: a, b, τ² and σ². The per-scene values are computed from those afterwards.

Step 1: Estimate τ² by comparing variation between scenes with variation within them

The intuition is this. Suppose scenes didn't really differ (τ² = 0). Scene averages would still vary a bit, just from noise, and that noise shrinks as a scene has more attempts (σ²/n). So:

- Within a scene, attempts bounce around its own mean. That bounce tells you σ².
- Between scenes, look at how spread out the scene means are. If they're more spread out than noise alone would produce, the excess is real scene-to-scene difference, which is τ².

spread of scene means  ≈  τ²  +  σ²/n
                          real    noise

The model doesn't literally do this subtraction. It uses maximum likelihood: an optimiser tries values of (a, b, τ², σ²), asks "how probable is the data I actually observed if these were true?", and keeps adjusting until that probability is highest. The balance it finds is the same one described above. If attempts within a scene cluster tightly while scenes sit at clearly different heights, only a large τ² makes that pattern probable.


** for each player, how does the log1p normalised LOSO distance change with the number of times they've played a scene, allowing each scene its own baseline and learning rate?**

also how to interpret log log 

log goes up in regular step every time the attempt doubles so log1 log2 log 4 log8 
all go up by one regular step

to interpret we take that step ammount and multiply by the coefficient to see how much percentage change for every doubling of attempt



**** later attempts are, on average, more typical of an established route". Not "attempts move towards a prototype"

****With repeated use of a route, its execution moves, on average, closer to that route's prototype.

Yes, that's the core interpretation. Because both distance and practice are on the log scale, the model says each doubling of practice multiplies the distance by the same factor:

factor per doubling = 2^b        % closer per doubling = (1 − 2^b) × 100