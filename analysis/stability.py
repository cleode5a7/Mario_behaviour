"""Resampling-based cluster stability for the Fréchet/HDBSCAN trace clustering.

Every function here takes a **precomputed distance matrix**, never traces, so the
same code serves the per-scene runs and (later) the pooled per-category ones. The
matrices are already cached under ``output_data/frechet/`` and ``clean-frechet``
is deliberately separate from ``clean-clustering``, so a subsample is just
``dist[np.ix_(idx, idx)]`` — nothing here ever reads ``mario.scenes``.

Two things this module exists to fix:

1. **Nothing recorded which parameters produced a cluster file.** The stored JSON
   carries ``scene``/``method``/``labels``/``metadata`` and no settings, and the
   scenes were not all clustered alike (three sampled scenes recovered to three
   different settings). ``recover_params`` reconstructs them after the fact.
2. **Nothing checked whether a partition survives resampling.** HDBSCAN's own
   ``cluster_persistence_`` returns all-1.0 on these matrices, so it is no help;
   resampling is the only route to a stability number here.

**This is subagging (Bühlmann & Yu 2002), not bagging in Breiman's sense.** The
aggregation step is the same — resample, refit, average the partitions — but the
resampling draws ``frac * n`` points *without* replacement rather than ``n`` points
*with*. That is a deliberate choice, measured over 35 scenes:

    resampling scheme      mean k / full-data k    mean replicate sd
    bootstrap (n, w/ repl)         1.29                  1.23
    subagging (0.8n, w/o)          1.04                  0.71

A bootstrap sample holds only ~63% unique points, so each replicate is effectively a
much smaller sample and HDBSCAN fragments it — k came out 29% above the full-data fit
in 33 of the 35 scenes. That bias is not harmless here: the Jaccard scores the
consensus partition against the replicate partitions, so replicates that carry 29%
more clusters by construction depress it through the mismatch rather than through
real instability. Subagging is near-unbiased (4%) and has half the replicate variance.

(A secondary effect, often cited, is that duplicate points sit at distance 0 and
inflate local density. Measured here it is minor — bootstrap adds ~0.2 pp of
zero-distance pairs, and the data already contains genuinely identical traces. The
unique-point deficit is what actually drives the bias.)
"""

from __future__ import annotations

import warnings

import numpy as np
import hdbscan
from scipy.cluster.hierarchy import ClusterWarning, cut_tree, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

# A cluster counts as reproduced when its mean bootstrap Jaccard clears this.
# Hennig (2007): <0.6 dissolved, 0.6-0.85 pattern present but not trustworthy,
# >0.85 highly stable. 0.75 sits inside the middle band on purpose — the headline
# metric is meant to be demanding without insisting on near-perfection.
STABLE_JACCARD = 0.75

# Sweep grid. min_samples is deliberately absent: measured on three scenes,
# epsilon dominates, while raising min_cluster_size pushed noise to 34-57% and
# lowered stability everywhere. Adding the axis would triple the cost for little.
SWEEP_MIN_CLUSTER_SIZES = (5, 10, 20)
SWEEP_EPSILONS = (0.0, 10.0, 20.0, 40.0)

# Recovery grid — wider, and it *does* include min_samples, because the stored
# files were hand-tuned and w8l2s0 turned out to be min_samples=5.
RECOVERY_MIN_CLUSTER_SIZES = (5, 10, 15, 20)
RECOVERY_MIN_SAMPLES = (None, 3, 5)
RECOVERY_EPSILONS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)

# ARI at or above this counts as an exact reproduction of a stored partition.
RECOVERY_TOL = 0.999

# Consensus (Ward) k-selection. The silhouette is also capped at n - 1 by definition.
CONSENSUS_K_MAX = 15

# The selection rule is the plain one: score every cut by silhouette, take the best.
# Both constants below are the *off* setting of a guard, kept as parameters rather
# than deleted because each one has a measured failure mode behind it.
#
# CONSENSUS_K_TOL > 0 is a plateau tie-break — among cuts within `tol` of the best
# score, the smallest k wins. It exists because the silhouette on 1 - C does not fall
# once clusters are isolated (a pair that never co-clustered sits at distance exactly
# 1.0), so the curve flattens and the argmax walks up that tail. At 0.0 the rule is
# exactly "highest silhouette, smallest k among exact ties".
CONSENSUS_K_TOL = 0.0

# CONSENSUS_MIN_CLUSTER_SIZE > 1 restricts candidates to cuts whose smallest cluster
# clears it. Ward assigns every point and has no noise class, so nothing else stops a
# cut from being mostly singletons — and the silhouette cannot object, since sklearn
# scores a singleton as 0 and a never-co-clustered point already scores 0 inside its
# parent, making the split strictly non-negative. Measured over 12 scenes with both
# guards off, plain argmax produced a singleton cluster in 5 of them (w1l1s5 paid
# +0.0029 of silhouette to go from k=4 to k=15). At 1 the filter never fires.
#
# This is deliberately NOT the replicate `min_cluster_size`: that one is a density
# parameter for the HDBSCAN fits and is scaled by `frac`; this one is an eligibility
# test on a full-data cut, with nothing to scale.
CONSENSUS_MIN_CLUSTER_SIZE = 1

# Which rule reads the silhouette-vs-k curve. "silhouette" is the plain argmax above.
# The other two read the curve's *shape* instead of its maximum, via `elbow_k`:
#
#   "elbow"         cut at the cliff edge itself — the last k before the curve drops.
#   "elbow_capped"  the argmax, searched only at or below that cliff.
#
# All three are computed on every run; this constant only says which one assigns the
# labels. Measured over the corpus, against the argmax (the `elbow` row is from the
# shipped run itself; `elbow_capped` is derived from the stored curves, which fix k
# and the smallest cluster exactly but say nothing about stability):
#
#   rule           k (median/mean)   changes k   dust (<= 2)   stable_share
#   silhouette         6 / 7.45          —        126 scenes      79.1%
#   elbow              7 / 7.79      154 scenes   152 scenes      76.9%
#   elbow_capped       6 / 6.51       40 scenes    99 scenes      not measured
#
# The partitions themselves barely move: median ARI 1.000 against the argmax corpus,
# 266 of 308 scenes above 0.9. The cost is concentrated — 42 scenes disagree, one as
# low as 0.061 — and the 2.2 pp of stable_share is bought by those plus the extra dust.
#
# The two are not variations on one idea, and the table is why. "elbow" is the literal
# reading and it runs *finer* as often as coarser: the curve usually plateaus at the
# top with wiggles below `elbow_min_drop`, so the first real cliff sits above the
# argmax and the rule walks out to meet it. "elbow_capped" can only ever be coarser,
# because the cliff is used as a ceiling rather than as the answer — it changes only
# the 40 scenes where the argmax had climbed *past* a cliff (w1l1s5: k=15 against a
# cliff at k=4), which is exactly the flat-tail pathology CONSENSUS_K_TOL describes,
# and it costs a median of 0.0000 silhouette to fix them.
#
# **The default is "elbow": the corpus on disk is cut at the cliff edge.** That is a
# deliberate choice and not the one the table above would argue for — `elbow_capped`
# is the rule that reduces dust, and plain `elbow` does not. It is what the analysis
# asks for: the last k before the ensemble's agreement breaks, rather than the k that
# scores best on a metric which cannot see a singleton. Pass
# `--consensus-k-rule silhouette` to get the argmax corpus back; the alternatives are
# one flag and one re-cluster away in either direction.
CONSENSUS_K_RULE = "elbow"

# A drop counts as the cliff when it is at least this share of the steepest drop in
# the curve. 1.0 is literally "the k before the steepest drop"; below that, an earlier
# drop of comparable depth wins, which is what keeps the rule from walking past a real
# cliff to a slightly deeper one further up the tail.
CONSENSUS_ELBOW_FRAC = 0.5

# ...and at least this much in absolute silhouette, so a curve that is flat within
# rounding (values are stored to 4 dp) is reported as having no cliff at all rather
# than having one at its largest piece of noise.
CONSENSUS_ELBOW_MIN_DROP = 0.005


def _fit(dist: np.ndarray, min_cluster_size, min_samples, epsilon) -> np.ndarray:
    """One HDBSCAN fit on a precomputed distance matrix."""
    return hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=max(2, int(min_cluster_size)),
        min_samples=None if min_samples is None else max(1, int(min_samples)),
        cluster_selection_epsilon=float(epsilon),
    ).fit_predict(dist)


def _cluster_ids(labels: np.ndarray) -> set[int]:
    """Non-noise labels present, as plain ints."""
    return {int(l) for l in np.asarray(labels).tolist()} - {-1}


def sweep_grid() -> list[tuple[int, float]]:
    """(min_cluster_size, epsilon) pairs for the stability sweep."""
    return [(m, e) for m in SWEEP_MIN_CLUSTER_SIZES for e in SWEEP_EPSILONS]


def recovery_grid() -> list[tuple[int, int | None, float]]:
    """(min_cluster_size, min_samples, epsilon) triples for parameter recovery."""
    return [(m, s, e)
            for m in RECOVERY_MIN_CLUSTER_SIZES
            for s in RECOVERY_MIN_SAMPLES
            for e in RECOVERY_EPSILONS]


def _one_replicate(dist, draw, min_cluster_size, min_samples, epsilon):
    """Fit one replicate. Returns (distinct point indices, their labels).

    **The return is always deduplicated**, which matters only for the bootstrap:
    a point drawn three times contributes three identical rows to the fit, but the
    replicate's partition is a partition of the *distinct* points it contains. If
    the duplicates were passed downstream, ``coassociation`` would increment a pair
    once per copy and the co-association denominator would no longer be "replicates
    this pair was drawn into together". Deduplicating here keeps every downstream
    function scheme-agnostic.
    """
    labels = _fit(dist[np.ix_(draw, draw)], min_cluster_size, min_samples, epsilon)
    # np.unique returns sorted distinct values and the index of each one's first
    # occurrence; identical rows get identical labels, so the first is representative.
    points, first = np.unique(draw, return_index=True)
    return points, labels[first]


def resample_labels(dist, *, min_cluster_size, min_samples=None, epsilon=0.0,
                     B=200, frac=0.8, seed=0, scheme="subagging",
                     n_jobs=1) -> list[tuple[np.ndarray, np.ndarray]]:
    """Refit HDBSCAN on B random subsamples. Returns (indices, labels) per replicate.

    Subsamples are ``frac * n`` points drawn **without replacement** — subagging, not
    a Breiman bootstrap. See the module docstring for the measurement behind that.

    **Count parameters are scaled by ``frac``; ``epsilon`` is not.** A subsample
    holds fewer points, so a fixed ``min_cluster_size`` silently raises the density
    bar and real clusters drop below it — measured full-data k of 45 / 14 / 7 came
    back as 35.2 / 6.8 / 3.8 with the count held fixed, and 42.6 / 13.7 / 6.5 once
    scaled. That first column is sample size being measured, not stability.
    ``epsilon`` is a threshold in Fréchet-distance units, which the subsample does
    not change, so it stays put.

    Args:
        scheme: "subagging" (default) draws ``frac * n`` points without replacement;
                "bootstrap" is Breiman's — ``n`` points *with* replacement. See the
                module docstring for the measured difference. Under "bootstrap" the
                count parameters are NOT scaled, since n points are drawn.
        frac:   subsample fraction, used by "subagging" only
        seed:   replicate draws are seeded, so a stability number is reproducible
        n_jobs: parallel workers (joblib; -1 = all cores). Each replicate is an
                independent fit, so this scales close to linearly.
    """
    if scheme not in ("subagging", "bootstrap"):
        raise ValueError(f"Unknown scheme '{scheme}'. Use 'subagging' or 'bootstrap'.")

    dist = np.asarray(dist, dtype=float)
    n = len(dist)
    rng = np.random.default_rng(seed)

    if scheme == "bootstrap":
        # n draws with replacement: the sample is the same size as the data, so the
        # density thresholds keep their full-data meaning and are not rescaled.
        draws = [np.sort(rng.choice(n, n, replace=True)) for _ in range(int(B))]
        mcs_b, ms_b = min_cluster_size, min_samples
    else:
        m = max(2, int(round(frac * n)))
        draws = [np.sort(rng.choice(n, m, replace=False)) for _ in range(int(B))]
        mcs_b = max(2, int(round(min_cluster_size * frac)))
        ms_b = None if min_samples is None else max(1, int(round(min_samples * frac)))

    # Draws are generated up front from one seeded rng so the replicate set is
    # identical whatever n_jobs is — parallelism must not change the answer.
    if n_jobs == 1:
        return [_one_replicate(dist, d, mcs_b, ms_b, epsilon) for d in draws]

    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs)(
        delayed(_one_replicate)(dist, d, mcs_b, ms_b, epsilon) for d in draws
    )


def jaccard_stability(base_labels, replicates) -> dict[int, float]:
    """Per-cluster mean bootstrap Jaccard (Hennig 2007).

    For each replicate, a base cluster is compared against the *best-matching*
    replicate cluster, restricted to the base members that were actually drawn into
    that subsample — otherwise the missing 20% would be scored as a failure to
    recover points that were never on offer.

    Returns {cluster label: mean Jaccard over replicates}, noise (-1) excluded.
    """
    base = np.asarray(base_labels)
    clusters = sorted(_cluster_ids(base))
    if not clusters:
        return {}

    members = {k: set(np.flatnonzero(base == k).tolist()) for k in clusters}
    scores: dict[int, list[float]] = {k: [] for k in clusters}

    for idx, labels in replicates:
        in_sample = set(idx.tolist())
        rep_sets = [set(idx[labels == c].tolist()) for c in _cluster_ids(labels)]
        for k in clusters:
            drawn = members[k] & in_sample
            if not drawn:
                continue
            scores[k].append(max((len(drawn & r) / len(drawn | r) for r in rep_sets),
                                 default=0.0))

    return {k: (float(np.mean(v)) if v else 0.0) for k, v in scores.items()}


def stable_share(base_labels, jaccard, threshold=STABLE_JACCARD) -> float:
    """Share of **all** traces sitting in a cluster that survived resampling.

    The headline sweep metric. Per-trace rather than per-cluster so a scene with
    forty tiny unreproducible clusters and one solid one is scored by how many
    traces ended up somewhere trustworthy; noise counts against it for free, since
    a noise trace is in no cluster at all.

    It is not sufficient on its own — collapsing a scene to k=2 scores 93% on
    w8l2s0 — which is why ``select_params`` maximises k subject to a floor on this
    rather than maximising this directly.
    """
    base = np.asarray(base_labels)
    if not len(base):
        return 0.0
    good = sum(int((base == k).sum()) for k, v in jaccard.items() if v > threshold)
    return good / len(base)


def coassociation(n, replicates, return_counts=False):
    """C[i,j] = P(i and j land in the same cluster | both were drawn).

    Conditioning on co-inclusion is what makes this comparable across pairs: with
    frac=0.8 a pair is drawn together only ~64% of the time, and that rate is the
    same for every pair only in expectation. Pairs never drawn together get 0.

    Args:
        return_counts: also return the co-inclusion counts. Their **minimum** is the
                       diagnostic for whether B was large enough — it is the smallest
                       denominator anywhere in the matrix.
    """
    co_sampled = np.zeros((n, n), dtype=np.int32)
    co_clustered = np.zeros((n, n), dtype=np.int32)

    for idx, labels in replicates:
        co_sampled[np.ix_(idx, idx)] += 1
        for c in _cluster_ids(labels):
            m = idx[labels == c]
            co_clustered[np.ix_(m, m)] += 1

    coassoc = np.divide(co_clustered, co_sampled,
                        out=np.zeros((n, n), dtype=float), where=co_sampled > 0)
    return (coassoc, co_sampled) if return_counts else coassoc


def consensus_labels(coassoc, *, min_cluster_size=5, min_samples=None, epsilon=0.0):
    """Cluster the ensemble itself: HDBSCAN on 1 - C as a precomputed distance.

    **No longer the default** — ``ward_consensus_labels`` is. Kept because the corpus
    was built with this until the switch, so reproducing an old cluster file needs it,
    and because ``run-consensus-comparison`` scores it against the Ward aggregations.
    The epsilon note below is the record of why it was demoted: the fit needed a tuned
    constant to stop it out-resolving the ensemble it summarises, and Ward plus a
    silhouette-chosen k does the same job without one.

    **``epsilon`` here is NOT the Fréchet-units epsilon used for the replicate
    fits.** Co-association distance is bounded in [0, 1], so a Fréchet threshold
    like 20 would exceed every distance in the matrix and merge the whole scene
    into one cluster. It defaults to 0 and callers should leave it there unless
    they mean a co-association threshold — 0.5 being "pairs that clustered
    together in at least half the replicates they shared".

    ``min_cluster_size`` is a count, so it carries over from the replicate fits
    unchanged. The diagonal is forced to zero: a trace called noise in every
    replicate has C[i,i] = 0 and would otherwise sit at distance 1.0 from itself.
    """
    d = 1.0 - np.asarray(coassoc, dtype=float)
    np.fill_diagonal(d, 0.0)
    return _fit(d, min_cluster_size, min_samples, epsilon)


def _relabel_by_size(labels) -> np.ndarray:
    """0-based labels, ordered by descending cluster size.

    ``cut_tree`` numbers clusters arbitrarily. Everything downstream — the dashboard's
    colour slots, the plot legends — ranks by size anyway, so doing it here makes the
    stored file readable at a glance and the numbering deterministic.
    """
    labels = np.asarray(labels)
    found, counts = np.unique(labels, return_counts=True)
    order = found[np.argsort(-counts, kind="stable")]
    remap = {int(old): new for new, old in enumerate(order.tolist())}
    return np.array([remap[int(l)] for l in labels], dtype=int)


def _consensus_tree(coassoc, *, input, linkage_method):
    """Build the linkage tree over the ensemble. Returns ``(Z, d)``.

    ``d`` is always the co-association distance ``1 - C``, returned separately because
    it is what the silhouette is scored on regardless of how the tree was built — that
    is what keeps the two ``input`` modes comparable.

    Two ways to hand the ensemble to scipy, and they are **not** the same measurement:

    - ``input="features"`` (the default, and what the BASC reference does) passes ``C``
      itself as an *observation matrix*. Each row is one trace's co-occurrence profile
      against every other trace, and the tree is built on ordinary Euclidean distances
      between those profiles. This is what makes Ward well posed: its criterion is
      defined on squared Euclidean distances, and the rows of ``C`` are vectors in R^n
      no matter what the input space was. Ours is Fréchet — a metric, but not
      Euclidean-embeddable — so this step is precisely what launders it.
    - ``input="distance"`` passes ``1 - C`` as a precomputed dissimilarity. First-order
      ("how often did *these two* cluster together") rather than second-order ("how
      differently do they relate to everything else"). Measured over all 308 scenes it
      is marginally the better of the two — 81.1% trace-weighted stable against 79.7%,
      at the same median k — but it applies the Lance-Williams update to something not
      guaranteed to be Euclidean, and the two partitions agree at median ARI 0.985
      anyway. Kept so the choice stays checkable rather than asserted;
      ``run-consensus-comparison`` is what produced those numbers.

    The diagonal is handled differently per branch, on purpose. Under "features" it is
    forced to 1.0, matching ``part2adj`` in the reference — an object always co-occurs
    with itself. Under "distance" the diagonal of ``1 - C`` is zeroed for the reason
    ``consensus_labels`` documents: a trace called noise in every replicate has
    C[i,i] = 0 and would otherwise sit at distance 1.0 from itself.
    """
    C = np.asarray(coassoc, dtype=float)
    d = 1.0 - C
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2.0          # float drift only; C is symmetric by construction

    if input == "features":
        feats = C.copy()
        np.fill_diagonal(feats, 1.0)
        with warnings.catch_warnings():
            # A co-occurrence matrix is square, symmetric and near-hollow, so scipy
            # warns it "looks suspiciously like an uncondensed distance matrix". Here
            # that shape is the whole point, not a mistake.
            warnings.simplefilter("ignore", ClusterWarning)
            Z = linkage(feats, method=linkage_method, metric="euclidean")
    elif input == "distance":
        Z = linkage(squareform(d, checks=False), method=linkage_method)
    else:
        raise ValueError(f"Unknown consensus input '{input}'. Use 'features' or 'distance'.")

    return Z, d


def elbow_k(rows, *, frac=CONSENSUS_ELBOW_FRAC, min_drop=CONSENSUS_ELBOW_MIN_DROP):
    """The last k before the silhouette curve falls off a cliff. Returns a report.

    A second reading of the same curve the argmax reads, and it asks a different
    question. The argmax asks *which cut scores best*; this asks *where does the curve
    stop being able to add a cluster without damage* — the *shape* of the curve rather
    than its maximum. The two differ because the silhouette on ``1 - C`` is not a
    well-behaved objective here: it plateaus once clusters are isolated, and a
    singleton peeled off a cluster costs it nothing (sklearn scores a singleton 0, and
    a never-co-clustered trace already scored ~0 inside its parent). So the argmax can
    wander up a flat tail of dust, while a cliff in the curve is a real event — the cut
    at which Ward had to break something that the resampling did hold together.

    The rule, in one sentence: **find the steepest drop between consecutive candidates,
    then take the k immediately before the first drop at least ``frac`` as deep.**

    ``frac=1.0`` makes that literally "the k before the steepest drop". The default 0.5
    exists because the steepest drop is not always the first real one: a curve can fall
    hard at k=7, wander, and fall slightly harder at k=13, and reporting 12 there would
    describe the tail rather than the structure. Requiring only half the depth takes
    the first cliff of comparable size instead.

    ``min_drop`` is an absolute floor on what counts as a cliff at all. Without it a
    curve that is flat to within rounding would still hand back its largest wiggle.

    Returns ``{"k", "drop", "steepest", "threshold"}`` with ``k`` None when the curve
    has no cliff — a monotonically rising curve genuinely has no last-k-before-a-drop,
    and callers fall back to the argmax rather than inventing one.
    """
    sil = [r["silhouette"] for r in rows if r.get("silhouette") is not None]
    ks = [r["k"] for r in rows if r.get("silhouette") is not None]
    empty = {"k": None, "drop": None, "steepest": None, "threshold": None}
    if len(sil) < 2:
        return empty

    # Positive = the curve fell going from this candidate to the next one.
    drops = [sil[i] - sil[i + 1] for i in range(len(sil) - 1)]
    steepest = max(drops)
    if steepest < min_drop:
        return {**empty, "steepest": round(float(steepest), 4)}

    threshold = max(min_drop, frac * steepest)
    i = next(i for i, d in enumerate(drops) if d >= threshold)
    return {"k": int(ks[i]), "drop": round(float(drops[i]), 4),
            "steepest": round(float(steepest), 4),
            "threshold": round(float(threshold), 4)}


def _candidate_cuts(Z, n, *, k_max, min_cluster_size):
    """Every eligible cut of the tree, as ``{k: labels}``.

    ``cut_tree`` returns exactly k where ``fcluster(..., "maxclust")`` returns *at
    most* k — that difference bit in testing, where a collapsed cut reached
    ``silhouette_score`` as a single label and raised. Ties can still collapse a cut,
    so candidates are keyed by the k actually returned (duplicates collapse) and
    single-label cuts dropped.

    Shared by ``ward_consensus_labels`` and ``consensus_cut_bounds`` so the dashboard's
    alternative cuts are the same objects the selection chose between, not a second
    implementation that could drift from it.
    """
    cuts: dict[int, np.ndarray] = {}
    for k in range(2, min(int(k_max), n - 1) + 1):
        labels = np.squeeze(np.asarray(cut_tree(Z, n_clusters=k)))
        found, counts = np.unique(labels, return_counts=True)
        if len(found) < 2 or counts.min() < min_cluster_size:
            continue
        cuts.setdefault(len(found), labels)
    return cuts


def cut_bounds(labels, leaves) -> list[int] | None:
    """Where a cut falls in the dendrogram's leaf order, as split positions.

    A cut's clusters are contiguous runs in the leaf order — ``cut_tree`` cannot split
    a leaf run, since the leaf order is built from the same merges the cut walks back.
    So a whole partition is ``k - 1`` positions, which is what lets the dashboard carry
    *every* candidate cut in a few hundred bytes instead of one label array per k.

    Returns None if the runs are not contiguous, rather than an encoding that would
    silently decode into a different partition. Nothing in the corpus trips it; it is
    here because a wrong partition drawn confidently is worse than a missing feature.
    """
    labels = np.asarray(labels)
    seq = [int(labels[i]) for i in leaves]
    bounds, seen = [], {seq[0]}
    for p in range(1, len(seq)):
        if seq[p] == seq[p - 1]:
            continue
        if seq[p] in seen:
            return None          # this cluster already ended — not contiguous
        seen.add(seq[p])
        bounds.append(p)
    return bounds


def consensus_cut_bounds(coassoc, *, input="features", linkage_method="ward",
                         k_max=CONSENSUS_K_MAX,
                         min_cluster_size=CONSENSUS_MIN_CLUSTER_SIZE) -> dict[int, list[int]]:
    """``{k: cut_bounds}`` for every candidate, rebuilt from a stored ensemble.

    The migration path for cluster files written before the bounds were recorded.
    ``_consensus_tree`` and ``_candidate_cuts`` are pure functions of ``C`` and the two
    settings the file records, so this reproduces the same tree and the same candidate
    list — the same reason ``run-coassociation`` can regenerate the matrix exactly.
    """
    C = np.asarray(coassoc, dtype=float)
    Z, _ = _consensus_tree(C, input=input, linkage_method=linkage_method)
    leaves = [int(i) for i in leaves_list(Z)]
    out = {}
    for k, labels in _candidate_cuts(Z, len(C), k_max=k_max,
                                     min_cluster_size=min_cluster_size).items():
        bounds = cut_bounds(labels, leaves)
        if bounds is not None:
            out[int(k)] = bounds
    return out


def ward_consensus_labels(coassoc, *, dist=None, input="features",
                          linkage_method="ward",
                          min_cluster_size=CONSENSUS_MIN_CLUSTER_SIZE,
                          k_max=CONSENSUS_K_MAX, tol=CONSENSUS_K_TOL,
                          rule=CONSENSUS_K_RULE,
                          elbow_frac=CONSENSUS_ELBOW_FRAC,
                          elbow_min_drop=CONSENSUS_ELBOW_MIN_DROP):
    """Cluster the ensemble with Ward linkage, choosing k by silhouette.

    The default consensus aggregation, replacing ``consensus_labels`` and its tuned
    ``CONSENSUS_EPSILON``. Returns ``(labels, report)``; the report carries the whole
    selection curve so the choice is inspectable without a re-run.

    The method in one sentence: **Ward linkage on the co-association matrix, cut at the
    k with the highest silhouette on 1 - C.** Ward itself has no k — ``linkage`` returns
    the full merge tree — so every k from 2 to ``k_max`` is cut, scored, and the best
    taken. Ties go to the smaller k.

    ``rule="elbow"`` reads the same curve by its shape instead — the last k before it
    falls off a cliff (see ``elbow_k``) — and ``rule="elbow_capped"`` uses that cliff as
    a ceiling on the argmax rather than as the answer. **The elbow is computed on every
    run regardless**, the way ``silhouette_frechet`` is recorded without selecting, so
    the report always says where the other rules would have cut; ``rule`` only decides
    which one assigns the labels. The default stays ``"silhouette"``: switching it
    repartitions the corpus, which is a measured change, not a default to drift into
    (the module constants carry the measurement).

    Two guards sit on that rule and both default to **off**; ``tol`` and
    ``min_cluster_size`` turn them on. See the module constants for the measured
    failure mode behind each — briefly, the silhouette on 1 - C plateaus rather than
    falling (so the argmax drifts up a flat tail) and cannot penalise a singleton (so
    unguarded cuts can be mostly dust). Neither is switched on by default: the plain
    rule is what the analysis reports, and the guards are there to be turned on
    deliberately and compared.

    **Ward assigns every point, so there is no noise label.** When ``min_cluster_size``
    is raised and nothing qualifies, the k=2 cut is taken and ``selection`` says
    ``"fallback_k2"`` rather than the scene being silently dropped. At the default of 1
    the filter never fires and that branch is unreachable.

    Args:
        dist: the Fréchet matrix. Optional, and **does not select** — it only adds a
              ``silhouette_frechet`` to each candidate, so a reader can see where the
              ensemble's cut and the raw trajectory geometry disagree. It is not used
              for selection because it systematically prefers coarser cuts, and coarser
              flatters ``stable_share`` for free (the same gameability ``select_params``
              guards against by maximising k above a floor).
        rule: which reading of the curve assigns the labels — ``"silhouette"`` (the
              argmax, default), ``"elbow"`` (the last k before the cliff) or
              ``"elbow_capped"`` (the argmax at or below that cliff). The elbow is
              computed and reported whichever one is in force.
        elbow_frac, elbow_min_drop: what counts as that cliff; see ``elbow_k``. They
              shape the reported elbow whether or not it selects.
    """
    C = np.asarray(coassoc, dtype=float)
    n = len(C)
    settings = {
        "consensus_cluster": "ward",
        "input": input,
        "linkage": linkage_method,
        "min_cluster_size": int(min_cluster_size),
        "k_max": int(k_max),
        "tol": float(tol),
        "rule": rule,
        "elbow_frac": float(elbow_frac),
        "elbow_min_drop": float(elbow_min_drop),
    }
    no_elbow = {"k": None, "drop": None, "steepest": None, "threshold": None}

    if n < 3:
        # Nothing to select between: silhouette needs 2 <= k <= n - 1.
        return (np.zeros(n, dtype=int),
                {**settings, "k": 1 if n else 0, "selection": "too_small",
                 "silhouette": None, "silhouette_frechet": None, "k_candidates": [],
                 "elbow": no_elbow, "leaves": None})

    # An ensemble that never once put any pair together carries no information: every
    # replicate called every point noise. 1 - C is then all ones, the silhouette is
    # exactly 0, and Ward would cut that flat matrix into arbitrary blocks — inventing
    # structure the resampling explicitly failed to find. Report no clusters instead,
    # which is what the HDBSCAN aggregation did for these scenes and is the honest
    # answer. Measured on the corpus: 8 scenes, all small (n = 3 to 36).
    off_diagonal = ~np.eye(n, dtype=bool)
    if C[off_diagonal].max() == 0.0:
        return (np.full(n, -1, dtype=int),
                {**settings, "k": 0, "selection": "no_ensemble_signal",
                 "silhouette": None, "silhouette_frechet": None, "k_candidates": [],
                 "elbow": no_elbow, "leaves": None})

    Z, d = _consensus_tree(C, input=input, linkage_method=linkage_method)
    frechet = None if dist is None else np.asarray(dist, dtype=float)
    leaves = [int(i) for i in leaves_list(Z)]

    def cut(k):
        return np.squeeze(np.asarray(cut_tree(Z, n_clusters=k)))

    candidates = _candidate_cuts(Z, n, k_max=k_max, min_cluster_size=min_cluster_size)

    rows = []
    for k, labels in sorted(candidates.items()):
        row = {
            "k": int(k),
            "silhouette": round(float(silhouette_score(d, labels, metric="precomputed")), 4),
            "min_size": int(np.unique(labels, return_counts=True)[1].min()),
        }
        if frechet is not None:
            row["silhouette_frechet"] = round(
                float(silhouette_score(frechet, labels, metric="precomputed")), 4)
        # Where this cut falls in the leaf order, so every candidate partition — not
        # just the one taken — travels with the curve at k-1 integers apiece. That is
        # what lets the dashboard show an alternative cut without re-deriving it in the
        # browser: it is still drawing a stored result, just not the selected one.
        bounds = cut_bounds(labels, leaves)
        if bounds is not None:
            row["bounds"] = bounds
        rows.append(row)

    elbow = elbow_k(rows, frac=elbow_frac, min_drop=elbow_min_drop)

    if rows:
        best = max(r["silhouette"] for r in rows)
        by_silhouette = min((r for r in rows if r["silhouette"] >= best - tol),
                            key=lambda r: r["k"])
        if rule not in ("silhouette", "elbow", "elbow_capped"):
            raise ValueError(f"Unknown k rule '{rule}'. "
                             f"Use 'silhouette', 'elbow' or 'elbow_capped'.")
        if rule == "silhouette":
            chosen, selection = by_silhouette, "silhouette"
        elif elbow["k"] is None:
            # A curve with no cliff has no last-k-before-one. Fall back to the argmax
            # and say so, rather than quietly handing back the largest candidate.
            chosen, selection = by_silhouette, "elbow_no_drop"
        elif rule == "elbow":
            chosen, selection = next(r for r in rows if r["k"] == elbow["k"]), "elbow"
        else:
            # The cliff as a ceiling, not as the answer: best silhouette among the cuts
            # at or below it. Coarser than the argmax or equal to it, never finer.
            capped = [r for r in rows if r["k"] <= elbow["k"]]
            top = max(r["silhouette"] for r in capped)
            chosen = min((r for r in capped if r["silhouette"] >= top - tol),
                         key=lambda r: r["k"])
            selection = "elbow_capped"
        labels = candidates[chosen["k"]]
    else:
        # No cut clears the size floor. Take the coarsest partition available and say
        # so, rather than dropping a small scene entirely.
        labels = cut(2)
        selection = "fallback_k2"
        chosen = {"silhouette": None, "silhouette_frechet": None}
        if len(np.unique(labels)) > 1:
            chosen["silhouette"] = round(
                float(silhouette_score(d, labels, metric="precomputed")), 4)
            if frechet is not None:
                chosen["silhouette_frechet"] = round(
                    float(silhouette_score(frechet, labels, metric="precomputed")), 4)

    labels = _relabel_by_size(labels)
    report = {
        **settings,
        "k": int(len(np.unique(labels))),
        "selection": selection,
        "silhouette": chosen.get("silhouette"),
        "silhouette_frechet": chosen.get("silhouette_frechet"),
        "k_candidates": rows,
        # Recorded on every run, whichever rule selected: where the *other* reading of
        # this curve would have cut it. Same standing as `silhouette_frechet` — visible
        # so the choice can be argued with, and not consulted unless `rule` says so.
        "elbow": elbow,
        # Dendrogram leaf order, in this matrix's own row order. The tree carries
        # structure the labels throw away — which clusters are adjacent in it, and how
        # each one is built internally — and that structure is only visible if the
        # matrix is permuted into this order. Ordering by cluster label instead (what
        # the dashboard did first) blocks the partition correctly but scrambles
        # everything inside a block, so sub-structure a finer cut would have found
        # cannot be seen. `leaves_list` is `dendrogram(..., no_plot=True)["leaves"]`
        # without building the plot dict.
        "leaves": [int(i) for i in leaves_list(Z)],
    }
    return labels, report


def trace_confidence(coassoc, labels) -> np.ndarray:
    """Per trace, mean co-association with the other members of its own cluster.

    0-1, and the value the stability colour ramps encode. Noise traces get 0.0.
    """
    labels = np.asarray(labels)
    conf = np.zeros(len(labels), dtype=float)
    for k in _cluster_ids(labels):
        m = np.flatnonzero(labels == k)
        if len(m) < 2:
            continue
        sub = coassoc[np.ix_(m, m)]
        conf[m] = (sub.sum(axis=1) - np.diag(sub)) / (len(m) - 1)
    return conf


def sweep(dist, grid=None, *, B=60, frac=0.8, seed=0,
          threshold=STABLE_JACCARD) -> list[dict]:
    """Score every grid point by stability. One row per (min_cluster_size, epsilon).

    ``eps_quantile`` records where that epsilon falls in *this* scene's own
    pairwise-distance distribution. A single global epsilon is in absolute Fréchet
    units, so this is the check on whether it means the same thing scene to scene.
    """
    dist = np.asarray(dist, dtype=float)
    n = len(dist)
    upper = dist[np.triu_indices(n, 1)] if n > 1 else np.zeros(1)

    rows = []
    for mcs, eps in (grid or sweep_grid()):
        base = _fit(dist, mcs, None, eps)
        reps = resample_labels(dist, min_cluster_size=mcs, epsilon=eps,
                                B=B, frac=frac, seed=seed)
        jac = jaccard_stability(base, reps)
        rows.append({
            "min_cluster_size": int(mcs),
            "epsilon": float(eps),
            "k": len(_cluster_ids(base)),
            "n": n,
            "noise_frac": float((base == -1).mean()),
            "stable_share": stable_share(base, jac, threshold),
            "eps_quantile": float((upper < eps).mean()),
            "per_cluster": {str(k): round(v, 4) for k, v in sorted(jac.items())},
        })
    return rows


def select_params(rows, floor=0.80, min_k=2) -> tuple[dict, bool]:
    """Most resolution above a stability floor. Returns (row, cleared_floor).

    Maximising ``stable_share`` directly is gameable — collapsing to k=2 scores 93%
    on w8l2s0 — so the rule prefers the finest partition that still reproduces, and
    only trades resolution away when the finer one genuinely does not survive.
    Ties on k break toward the higher share.

    When nothing clears the floor the best row is still returned, with False, so the
    caller reports a miss rather than silently adopting the least-bad setting.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("select_params got no sweep rows.")
    ok = [r for r in rows if r["stable_share"] >= floor and r["k"] >= min_k]
    if ok:
        return max(ok, key=lambda r: (r["k"], r["stable_share"])), True
    return max(rows, key=lambda r: (r["stable_share"], r["k"])), False


def recover_params(dist, stored_labels, grid=None, tol=RECOVERY_TOL):
    """Which settings reproduce an already-stored partition? Returns (params, ari).

    The stored cluster files record no parameters, and the scenes were not all run
    alike, so this is the only way to say what produced one. ``params`` is None when
    nothing in the grid reproduces the labels (a different hdbscan version, or a
    setting outside the grid) — reported rather than guessed at.
    """
    dist = np.asarray(dist, dtype=float)
    stored = np.asarray(stored_labels)
    best_ari, best_params = -1.0, None

    for mcs, ms, eps in (grid or recovery_grid()):
        ari = adjusted_rand_score(stored, _fit(dist, mcs, ms, eps))
        if ari > best_ari:
            best_ari, best_params = ari, {
                "min_cluster_size": int(mcs),
                "min_samples": None if ms is None else int(ms),
                "cluster_selection_epsilon": float(eps),
            }
        if best_ari >= 1.0:
            break   # exact; nothing further to find

    return (best_params if best_ari >= tol else None), float(best_ari)


# --------------------------------------------------------------------------
# Drivers (file-reading; the functions above stay pure so they serve any matrix)
# --------------------------------------------------------------------------

def _load_scene(scene_id, output_dir):
    """(stored_labels, dist_matrix, payload) for a clustered scene, or None if unusable."""
    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    clusters = output_dir / "clusters" / f"{scene_id}_clusters.json"
    matrix = output_dir / "frechet" / f"{scene_id}.npy"
    if not clusters.exists() or not matrix.exists():
        return None

    payload = json.loads(clusters.read_text())
    stored = np.asarray(payload["labels"])
    dist = np.load(matrix)
    if len(dist) != len(stored):
        # The Fréchet cache is validated by trace count; a mismatch means the
        # cache and the labels came from different source data.
        print(f"  {scene_id}: matrix ({len(dist)}) and labels ({len(stored)}) disagree, skipping.")
        return None
    return stored, dist, payload


def audit_scene(scene_id, output_dir, *, B=60, frac=0.8, seed=0) -> dict | None:
    """Recover one scene's clustering parameters, then score that partition.

    Two steps, in this order for a reason: the replicate fits must use the *same*
    settings the stored partition was made with, or the Jaccard is depressed by a
    parameter mismatch rather than by genuine instability.

    Files that already carry a consensus ``params`` block skip both steps and report
    what they recorded — there is nothing to recover, and recovering anyway produces
    a wrong answer (see the comment in the body). ``params_source`` says which path a
    row came from, so a mixed audit stays readable.
    """
    loaded = _load_scene(scene_id, output_dir)
    if loaded is None:
        return None
    stored, dist, payload = loaded

    # A file that records its own params needs no recovery — and must not get it.
    # recover_params searches single HDBSCAN fits, but a consensus partition has no
    # single-fit equivalent in general, so the grid returns a poor match and the
    # replicate fits are then made with the wrong settings: on w1l1s0 that reported
    # 23% stable for a partition whose own consensus run measured 96%.
    recorded = payload.get("params")
    if recorded and recorded.get("method") == "consensus":
        stab = payload.get("stability") or {}
        return {
            "n": int(len(stored)),
            "k": len(_cluster_ids(stored)),
            "noise_frac": float((stored == -1).mean()),
            "recovered": recorded,
            "recovery_ari": 1.0,          # recorded, not inferred
            "params_source": "recorded",
            "stable_share": stab.get("stable_share", 0.0),
            "per_cluster": stab.get("per_cluster", {}),
        }

    params, ari = recover_params(dist, stored)
    # With no exact match we still score, using the closest settings found, and
    # record the ARI so the number is read with the right amount of trust.
    fit_params = params or {"min_cluster_size": 5, "min_samples": None,
                            "cluster_selection_epsilon": 0.0}

    row = {
        "n": int(len(stored)),
        "k": len(_cluster_ids(stored)),
        "noise_frac": float((stored == -1).mean()),
        "recovered": params,
        "recovery_ari": round(ari, 4),
        "params_source": "recovered",
    }

    if row["k"] == 0:
        # Every trace called noise — no partition to score.
        row.update({"stable_share": 0.0, "per_cluster": {}})
        return row

    reps = resample_labels(
        dist,
        min_cluster_size=fit_params["min_cluster_size"],
        min_samples=fit_params["min_samples"],
        epsilon=fit_params["cluster_selection_epsilon"],
        B=B, frac=frac, seed=seed,
    )
    jac = jaccard_stability(stored, reps)
    row["stable_share"] = round(stable_share(stored, jac), 4)
    row["per_cluster"] = {str(k): round(v, 4) for k, v in sorted(jac.items())}
    return row


def run_cluster_audit(scenes, output_dir, *, B=60, frac=0.8, seed=0) -> dict:
    """Audit stored cluster files; merge into output_dir/stability/audit.json.

    Merges rather than overwrites so ``--scenes`` chunks accumulate and a re-run
    skips what is already recorded (the same idempotency every run-* task has).
    """
    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    out_dir = output_dir / "stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "audit.json"

    audit = json.loads(out_path.read_text()) if out_path.exists() else {"scenes": {}}
    audit["settings"] = {"B": int(B), "frac": float(frac), "seed": int(seed),
                         "stable_jaccard": STABLE_JACCARD,
                         "recovery_tol": RECOVERY_TOL}

    for i, scene in enumerate(scenes, 1):
        if scene in audit["scenes"]:
            continue
        row = audit_scene(scene, output_dir, B=B, frac=frac, seed=seed)
        if row is None:
            continue
        audit["scenes"][scene] = row
        flag = "" if row["recovered"] else f"  [no exact params, ari={row['recovery_ari']}]"
        print(f"  [{i}/{len(scenes)}] {scene}: n={row['n']} k={row['k']} "
              f"noise={row['noise_frac']:.0%} stable={row['stable_share']:.0%}{flag}")
        out_path.write_text(json.dumps(audit, indent=2, sort_keys=True))

    out_path.write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(f"Audited {len(audit['scenes'])} scenes -> {out_path}")
    return audit


def run_sweep(scenes, output_dir, *, grid=None, B=60, frac=0.8, seed=0,
              min_traces=None) -> None:
    """Sweep the parameter grid per scene; write output_dir/stability/<scene>_sweep.json.

    Reads the cached Fréchet matrix only — never mario.scenes. Scenes too small for
    the grid's largest ``min_cluster_size`` to produce anything are recorded as
    skipped rather than silently written as all-noise.
    """
    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    out_dir = output_dir / "stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = grid or sweep_grid()
    if min_traces is None:
        min_traces = 3 * max(m for m, _ in grid)

    for i, scene in enumerate(scenes, 1):
        out_path = out_dir / f"{scene}_sweep.json"
        if out_path.exists():
            print(f"  [{i}/{len(scenes)}] {scene}: skipping (output exists)")
            continue

        matrix = output_dir / "frechet" / f"{scene}.npy"
        if not matrix.exists():
            print(f"  [{i}/{len(scenes)}] {scene}: no cached Fréchet matrix, skipping.")
            continue
        dist = np.load(matrix)

        if len(dist) < min_traces:
            out_path.write_text(json.dumps({
                "scene": scene, "n": int(len(dist)),
                "skipped": f"fewer than {min_traces} traces", "rows": [],
            }, indent=2))
            print(f"  [{i}/{len(scenes)}] {scene}: n={len(dist)} too small, recorded as skipped.")
            continue

        rows = sweep(dist, grid, B=B, frac=frac, seed=seed)
        out_path.write_text(json.dumps({
            "scene": scene, "n": int(len(dist)), "skipped": None,
            "B": int(B), "frac": float(frac), "rows": rows,
        }, indent=2))
        best = max(rows, key=lambda r: r["stable_share"])
        print(f"  [{i}/{len(scenes)}] {scene}: n={len(dist)} best "
              f"mcs={best['min_cluster_size']} eps={best['epsilon']:.0f} "
              f"k={best['k']} stable={best['stable_share']:.0%}")


def select_global_params(output_dir, floor=0.80, min_k=2) -> dict:
    """Aggregate every sweep into one corpus-wide (min_cluster_size, epsilon).

    Aggregation is **trace-weighted**: ``stable_share`` is already a per-trace
    proportion, so weighting by n composes the per-scene numbers into a single
    honest quantity — the share of all traces in the corpus that sit in a cluster
    that survives resampling. An unweighted mean would let a 20-trace scene count
    as much as a 761-trace one.

    ``eps_quantile`` is carried through as a spread, because a global epsilon is in
    absolute Fréchet units and only means the same thing scene to scene if that
    quantile is stable.
    """
    import json
    from pathlib import Path
    from collections import defaultdict

    out_dir = Path(output_dir) / "stability"
    files = sorted(out_dir.glob("*_sweep.json"))
    if not files:
        raise FileNotFoundError(f"No *_sweep.json under {out_dir} — run run-cluster-sweep first.")

    totals = defaultdict(lambda: {"n": 0, "stable_n": 0.0, "k": [], "q": [], "scenes": 0})
    skipped = []
    for f in files:
        payload = json.loads(f.read_text())
        if payload.get("skipped") or not payload["rows"]:
            skipped.append(payload["scene"])
            continue
        for r in payload["rows"]:
            t = totals[(r["min_cluster_size"], r["epsilon"])]
            t["n"] += r["n"]
            t["stable_n"] += r["stable_share"] * r["n"]
            t["k"].append(r["k"])
            t["q"].append(r["eps_quantile"])
            t["scenes"] += 1

    rows = [{
        "min_cluster_size": mcs,
        "epsilon": eps,
        "scenes": t["scenes"],
        "traces": t["n"],
        "stable_share": round(t["stable_n"] / t["n"], 4) if t["n"] else 0.0,
        "k": float(np.median(t["k"])),          # median k across scenes
        "k_mean": round(float(np.mean(t["k"])), 2),
        "eps_quantile_median": round(float(np.median(t["q"])), 4),
        "eps_quantile_iqr": round(float(np.percentile(t["q"], 75)
                                        - np.percentile(t["q"], 25)), 4),
    } for (mcs, eps), t in sorted(totals.items())]

    best, cleared = select_params(rows, floor=floor, min_k=min_k)
    result = {
        "selected": {"min_cluster_size": best["min_cluster_size"],
                     "cluster_selection_epsilon": best["epsilon"]},
        "cleared_floor": cleared,
        "floor": floor,
        "scenes_swept": len(files) - len(skipped),
        "scenes_skipped": skipped,
        "rows": rows,
    }

    out_path = out_dir / "selected_params.json"
    out_path.write_text(json.dumps(result, indent=2))
    if not cleared:
        print(f"WARNING: no grid point reached the {floor:.0%} floor — widen the grid "
              f"before adopting these parameters.")
    print(f"Selected min_cluster_size={best['min_cluster_size']} "
          f"epsilon={best['epsilon']:.0f} "
          f"(median k={best['k']:.0f}, {best['stable_share']:.0%} of traces stable) "
          f"-> {out_path}")
    return result


def _compare_one(scene, output_dir, params, B, frac, seed, consensus_epsilon,
                 consensus_cluster="ward", consensus_input="features"):
    """Cluster one scene under both resampling schemes. Returns a comparison row.

    The consensus settings are forwarded rather than left at their defaults so this
    compares the two *schemes* under whatever aggregation the corpus actually uses —
    otherwise a later change of default would silently make this measure two things
    at once.
    """
    import json
    from pathlib import Path
    from sklearn.metrics import adjusted_rand_score

    from analysis.clustering import cluster_traces

    matrix = Path(output_dir) / "frechet" / f"{scene}.npy"
    if not matrix.exists():
        return None
    dist = np.load(matrix)

    row = {"scene": scene, "n": int(len(dist))}
    labels = {}
    for scheme in ("subagging", "bootstrap"):
        lab, stab = cluster_traces(
            dist_matrix=dist, method="consensus", scheme=scheme,
            min_cluster_size=params["min_cluster_size"],
            cluster_selection_epsilon=params["cluster_selection_epsilon"],
            bootstraps=B, frac=frac, seed=seed, consensus_epsilon=consensus_epsilon,
            consensus_cluster=consensus_cluster, consensus_input=consensus_input,
        )
        labels[scheme] = lab
        row[scheme] = {
            "k": len(_cluster_ids(np.asarray(lab))),
            "stable_share": stab["stable_share"],
            "noise_frac": stab["noise_frac"],
            "k_replicates": stab["k_replicates"],
            "min_co_sampled": stab["min_co_sampled"],
        }
    # How different are the two partitions, beyond their summary statistics?
    row["ari"] = round(float(adjusted_rand_score(labels["subagging"],
                                                 labels["bootstrap"])), 4)
    return row


def run_scheme_comparison(scenes, output_dir, *, params=None, B=200, frac=0.8,
                          seed=0, consensus_epsilon=0.4, consensus_cluster="ward",
                          consensus_input="features", n_jobs=1) -> dict:
    """Cluster every scene under both subagging and Breiman bootstrap; write the result.

    The point is to make the choice inspectable rather than asserted. Both schemes
    run at the *same* selected parameters, same B, same seed, so the only difference
    is how the replicates are drawn.

    Parallelism is over **scenes**, not replicates: one scene is ~4-5 s of work while
    one replicate fit is ~0.02 s, and at that granularity joblib's per-task overhead
    exceeds the fit (measured: replicate-level n_jobs=10 made subagging *slower*,
    6.0 s against 4.0 s single-threaded).
    """
    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    out_dir = output_dir / "stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    params = params or {"min_cluster_size": 5, "cluster_selection_epsilon": 20.0}

    if n_jobs == 1:
        rows = [_compare_one(s, output_dir, params, B, frac, seed, consensus_epsilon,
                             consensus_cluster, consensus_input)
                for s in scenes]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_compare_one)(s, output_dir, params, B, frac, seed, consensus_epsilon,
                                  consensus_cluster, consensus_input)
            for s in scenes
        )
    rows = [r for r in rows if r is not None]

    def summarise(scheme):
        ks = [r[scheme]["k"] for r in rows]
        ss = [r[scheme]["stable_share"] for r in rows]
        nz = [r[scheme]["noise_frac"] for r in rows]
        sd = [r[scheme]["k_replicates"]["sd"] for r in rows]
        traces = sum(r["n"] for r in rows)
        return {
            "median_k": float(np.median(ks)),
            "mean_k": round(float(np.mean(ks)), 2),
            "median_stable_share": round(float(np.median(ss)), 4),
            # Trace-weighted, so it is comparable with selected_params.json.
            "weighted_stable_share": round(
                float(sum(r[scheme]["stable_share"] * r["n"] for r in rows) / traces), 4),
            "median_noise_frac": round(float(np.median(nz)), 4),
            "mean_replicate_k_sd": round(float(np.mean(sd)), 3),
            "scenes_below_60pct": sum(1 for v in ss if v < 0.60),
            "scenes_at_or_above_80pct": sum(1 for v in ss if v >= 0.80),
            "scenes_with_zero_clusters": sum(1 for v in ks if v == 0),
        }

    result = {
        "params": {**params, "bootstraps": int(B), "frac": float(frac),
                   "seed": int(seed), "consensus_epsilon": float(consensus_epsilon),
                   "consensus_cluster": consensus_cluster,
                   "consensus_input": consensus_input},
        "scenes": len(rows),
        "summary": {"subagging": summarise("subagging"),
                    "bootstrap": summarise("bootstrap")},
        "median_ari_between_schemes": round(float(np.median([r["ari"] for r in rows])), 4),
        "rows": rows,
    }
    out_path = out_dir / "scheme_comparison.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Compared {len(rows)} scenes -> {out_path}")
    return result


# The three ways the ensemble can be turned into labels, as
# (name, consensus_cluster, consensus_input).
CONSENSUS_AGGREGATIONS = (
    ("ward_features", "ward", "features"),
    ("ward_distance", "ward", "distance"),
    ("hdbscan", "hdbscan", None),
)


def _silhouette_excluding_noise(d, labels):
    """Silhouette on ``1 - C``, computed over non-noise points only.

    HDBSCAN partitions carry a -1 class that is not a cluster but a leftover pile, and
    scoring it as though it were one would make the comparison against Ward — which has
    no noise class — meaningless. Returns None when fewer than two clusters remain.
    """
    labels = np.asarray(labels)
    keep = np.flatnonzero(labels != -1)
    if len(keep) < 3 or len(set(labels[keep].tolist())) < 2:
        return None
    return round(float(silhouette_score(d[np.ix_(keep, keep)], labels[keep],
                                        metric="precomputed")), 4)


def _compare_consensus_one(scene, output_dir, params, B, frac, seed, scheme,
                           consensus_epsilon):
    """Score one scene under every consensus aggregation. Returns a comparison row.

    **The ensemble is built once and shared.** The replicate fits are the expensive part
    (~4-5 s a scene) and are identical across the three aggregations, which differ only
    in what they do with ``C``. Refitting per aggregation would triple the cost and, worse,
    leave room for the comparison to be confounded by a differing ensemble; this way the
    three are guaranteed to be reading the same matrix.
    """
    from pathlib import Path

    matrix = Path(output_dir) / "frechet" / f"{scene}.npy"
    if not matrix.exists():
        return None
    dist = np.load(matrix)
    n = len(dist)
    if n < 3:
        return None

    replicates = resample_labels(
        dist, min_cluster_size=params["min_cluster_size"],
        min_samples=params.get("min_samples"),
        epsilon=params["cluster_selection_epsilon"],
        B=B, frac=frac, seed=seed, scheme=scheme,
    )
    coassoc = coassociation(n, replicates)
    d = 1.0 - coassoc
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2.0

    row = {"scene": scene, "n": int(n)}
    labels = {}
    for name, cluster_kind, input_kind in CONSENSUS_AGGREGATIONS:
        if cluster_kind == "ward":
            # The Ward floor, not the replicate density parameter — they are separate
            # settings now, and passing the latter here is what made an earlier run of
            # this comparison score Ward under a floor of 5 it does not ship with.
            lab, report = ward_consensus_labels(
                coassoc, dist=dist, input=input_kind,
                min_cluster_size=params.get("consensus_min_cluster_size",
                                            CONSENSUS_MIN_CLUSTER_SIZE),
                # ...and the k rule the run itself used, for the same reason: this
                # compares aggregations, so everything else about the cut has to be
                # what the corpus was actually made with.
                rule=params.get("consensus_k_rule", CONSENSUS_K_RULE),
                elbow_frac=params.get("consensus_elbow_frac", CONSENSUS_ELBOW_FRAC),
                elbow_min_drop=params.get("consensus_elbow_min_drop",
                                          CONSENSUS_ELBOW_MIN_DROP))
            silhouette = report["silhouette"]
            silhouette_frechet = report["silhouette_frechet"]
            selection = report["selection"]
        else:
            lab = consensus_labels(coassoc,
                                   min_cluster_size=params["min_cluster_size"],
                                   min_samples=params.get("min_samples"),
                                   epsilon=consensus_epsilon)
            silhouette = _silhouette_excluding_noise(d, lab)
            silhouette_frechet = _silhouette_excluding_noise(dist, lab)
            selection = "epsilon"

        jac = jaccard_stability(lab, replicates)
        labels[name] = np.asarray(lab)
        row[name] = {
            "k": len(_cluster_ids(np.asarray(lab))),
            "stable_share": round(stable_share(lab, jac), 4),
            "noise_frac": round(float((np.asarray(lab) == -1).mean()), 4),
            "silhouette": silhouette,
            "silhouette_frechet": silhouette_frechet,
            "selection": selection,
        }

    # Do the partitions actually differ, beyond their summary statistics?
    names = [name for name, _, _ in CONSENSUS_AGGREGATIONS]
    row["ari"] = {f"{a}|{b}": round(float(adjusted_rand_score(labels[a], labels[b])), 4)
                  for i, a in enumerate(names) for b in names[i + 1:]}
    return row


def run_consensus_comparison(scenes, output_dir, *, params=None, B=200, frac=0.8,
                             seed=0, scheme="subagging", consensus_epsilon=0.4,
                             n_jobs=1) -> dict:
    """Score every scene under all three consensus aggregations; write the result.

    Ward on the co-association matrix as a feature matrix (the BASC form, and the
    default), Ward on ``1 - C`` as a precomputed dissimilarity, and the older HDBSCAN
    fit on ``1 - C`` at ``consensus_epsilon``. Same replicate parameters, same B, same
    seed, same ensemble — the only difference is the last step.

    The point is the same as ``run_scheme_comparison``'s: make the choice inspectable
    rather than asserted. A 12-scene probe put the three within about a point of each
    other on trace-weighted ``stable_share``; this is what checks that over the corpus.

    Parallelism is over **scenes** for the reason given in ``run_scheme_comparison``:
    one scene is seconds of work, one replicate fit is milliseconds, and at that
    granularity joblib's per-task overhead exceeds the fit.
    """
    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    out_dir = output_dir / "stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    params = params or {"min_cluster_size": 5, "cluster_selection_epsilon": 20.0}

    if n_jobs == 1:
        rows = [_compare_consensus_one(s, output_dir, params, B, frac, seed, scheme,
                                       consensus_epsilon)
                for s in scenes]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_compare_consensus_one)(s, output_dir, params, B, frac, seed, scheme,
                                            consensus_epsilon)
            for s in scenes
        )
    rows = [r for r in rows if r is not None]
    if not rows:
        raise FileNotFoundError("No cached Fréchet matrices found for the given scenes.")

    traces = sum(r["n"] for r in rows)

    def summarise(name):
        ks = [r[name]["k"] for r in rows]
        ss = [r[name]["stable_share"] for r in rows]
        sil = [r[name]["silhouette"] for r in rows if r[name]["silhouette"] is not None]
        return {
            "median_k": float(np.median(ks)),
            "mean_k": round(float(np.mean(ks)), 2),
            "median_stable_share": round(float(np.median(ss)), 4),
            # Trace-weighted, so it is comparable with selected_params.json.
            "weighted_stable_share": round(
                float(sum(r[name]["stable_share"] * r["n"] for r in rows) / traces), 4),
            "median_noise_frac": round(float(np.median([r[name]["noise_frac"] for r in rows])), 4),
            "median_silhouette": round(float(np.median(sil)), 4) if sil else None,
            "scenes_below_60pct": sum(1 for v in ss if v < 0.60),
            "scenes_at_or_above_80pct": sum(1 for v in ss if v >= 0.80),
            "scenes_with_zero_clusters": sum(1 for v in ks if v == 0),
            "scenes_fallback_k2": sum(1 for r in rows if r[name]["selection"] == "fallback_k2"),
        }

    names = [name for name, _, _ in CONSENSUS_AGGREGATIONS]
    result = {
        "params": {**params, "bootstraps": int(B), "frac": float(frac),
                   "seed": int(seed), "scheme": scheme,
                   "consensus_epsilon": float(consensus_epsilon)},
        "scenes": len(rows),
        "summary": {name: summarise(name) for name in names},
        "median_ari": {
            pair: round(float(np.median([r["ari"][pair] for r in rows])), 4)
            for pair in rows[0]["ari"]
        },
        "rows": rows,
    }
    out_path = out_dir / "consensus_comparison.json"
    out_path.write_text(json.dumps(result, indent=2))
    for name in names:
        sm = result["summary"][name]
        print(f"  {name:14} median k={sm['median_k']:.0f}  "
              f"{sm['weighted_stable_share']:.1%} of traces stable")
    print(f"Compared {len(rows)} scenes -> {out_path}")
    return result


def coassociation_path(scene_id, output_dir):
    """Where one scene's stored co-association matrix lives."""
    from pathlib import Path
    return Path(output_dir) / "coassociation" / f"{scene_id}.npy"


def run_coassociation(scenes, output_dir, *, force=False) -> None:
    """Persist each scene's co-association matrix as uint8.

    The matrix is the central artefact of consensus clustering — it is what the
    labels are derived from — but ``cluster_traces`` builds it, reads the labels and
    the stability numbers off it, and drops it. Nothing downstream could show it.

    Regenerated here from the cached Fréchet matrix plus the ``params`` block each
    cluster file now records; the draws are seeded, so this reproduces exactly the
    matrix that produced the stored labels (verified by re-deriving the labels from
    it and checking ARI == 1).

    Stored as ``round(C * 65535)`` uint16, in the **same row order as the scene's
    cluster JSON metadata**. uint16 rather than uint8 because this is the analysis
    copy: at uint8 the ~0.002 rounding perturbs enough borderline mutual-reachability
    distances to shift the partition (ARI 0.998 against the stored labels), while
    uint16 re-derives them exactly. 27 MB for the corpus against 106 MB for float64.
    The web export downsamples this to uint8 — that copy is for drawing, not numbers.
    """
    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    out_dir = output_dir / "coassociation"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, scene in enumerate(scenes, 1):
        out_path = coassociation_path(scene, output_dir)
        if out_path.exists() and not force:
            print(f"  [{i}/{len(scenes)}] {scene}: skipping (output exists)")
            continue

        clusters = output_dir / "clusters" / f"{scene}_clusters.json"
        matrix = output_dir / "frechet" / f"{scene}.npy"
        if not clusters.exists() or not matrix.exists():
            print(f"  [{i}/{len(scenes)}] {scene}: no cluster file or matrix, skipping.")
            continue

        payload = json.loads(clusters.read_text())
        params = payload.get("params") or {}
        if params.get("method") != "consensus":
            print(f"  [{i}/{len(scenes)}] {scene}: method is "
                  f"'{params.get('method', 'unknown')}', no ensemble to store. Skipping.")
            continue

        dist = np.load(matrix)
        reps = resample_labels(
            dist,
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
            epsilon=params["cluster_selection_epsilon"],
            B=params["bootstraps"], frac=params["frac"], seed=params["seed"],
            # Files written before `scheme` was recorded were all subagging.
            scheme=params.get("scheme", "subagging"),
        )
        coassoc = coassociation(len(dist), reps)
        np.save(out_path, np.round(coassoc * 65535).astype(np.uint16))
        print(f"  [{i}/{len(scenes)}] {scene}: {len(dist)}x{len(dist)} "
              f"({out_path.stat().st_size / 1024:.0f} KB)")
