"""Practice effect: how far each attempt sits from its strategy's prototype, over time.

Clustering says *which* strategies exist in a scene; this module asks whether players
get better at them. Every cluster gets a **medoid** — the real attempt with the smallest
summed Fréchet distance to the rest of its cluster — and every attempt is measured
against those medoids. Two measures, kept apart because they answer different
questions:

  * ``d_own`` — distance to the medoid of the attempt's *own* cluster, analysed
    **within cluster** (one series per subject × scene × cluster). Every point in a
    series is measured against the same reference, so no normalisation is needed:
    "when you use this strategy, do you execute it more consistently?"
  * ``d_nearest_norm`` — distance to the *nearest* medoid of any cluster, divided by
    that cluster's spread (median member-to-medoid distance), analysed over **all**
    attempts (one series per subject × scene). The reference can change from point
    to point, and clusters differ in how tight they are, hence the normalisation:
    "is your behaviour becoming prototypical of *some* strategy?"

Which strategy was chosen is kept as its own measure (``switch``), so switching is
not mixed into either distance.

**Circularity guard.** Medoids are chosen from all attempts, late practice included,
so a cluster dominated by late attempts has its centre where the late attempts are
and distance falls partly by construction. Every distance therefore also has a
leave-subject-out twin (``*_loso``): subject *s* is measured against medoids searched
among the *other* subjects' members only. A practice effect that survives there is
not an artefact of how the centre was picked.

The medoid attempts themselves score ``d_own = 0`` by definition and are flagged
(``is_medoid``) and excluded from every raw-medoid fit. They stay in the ``_loso`` fits,
where nobody is measured against their own attempt.

**One mixed model per subject.** Five subjects are too few to estimate a between-subject
variance, and a pooled model with subject as a fixed intercept would force all five to
share one learning rate. So each subject is fitted on their own — the claim is about
these five players, replicated five times, not about a population. Inside a subject's
model the repeated measure is the scene (``d_nearest_norm``) or the scene × cluster
series (``d_own``), which gets a random intercept and a random slope on practice.
``d_nearest_norm_loso`` is the primary outcome (``PRIMARY``): it is comparable across
scenes, it does not condition on a strategy the player kept using, and its reference
is set by the other players.

**Two transforms, both kept.** Every model is fitted on ``log(d)`` — the primary
(``PRIMARY_TRANSFORM``), with the ~0.6% of attempts at exactly ``d = 0`` dropped since
the log is undefined there — and on ``log1p(d)``, the original analysis, kept so the
switch stays traceable. ``log1p`` squeezes the 0-1 range against a long upper tail, and
its residuals came out heavily right-skewed (skew 1.7-2.6, excess kurtosis 10-17);
under ``log`` they are 0.4-1.0 and 4-5, with the same conclusions. ``log`` also makes a
slope a proportional change in the distance itself rather than in ``1 + d``.

Reads only ``output_data/``: the cached Fréchet matrix and the cluster file share
trace order (both come from ``load_traces``), and the cluster file's ``metadata``
carries Subject, Phase, ClipCode and Outcome. Never opens ``mario.scenes``.
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from analysis.stages import attempt_index, stage_tertiles

MEASURES = ("d_nearest_norm", "d_own")
PRIMARY = "d_nearest_norm_loso"
# Outcome transform: (forward, inverse, drop d <= 0). The first is the primary.
TRANSFORMS = {
    "log": (np.log, np.exp, True),
    "log1p": (np.log1p, np.expm1, False),
}
PRIMARY_TRANSFORM = "log"
# Series unit per measure: d_own is within cluster, d_nearest over all attempts.
SERIES_KEYS = {
    "d_own": ["scene", "subject", "cluster"],
    "d_own_loso": ["scene", "subject", "cluster"],
    "d_nearest_norm": ["scene", "subject"],
    "d_nearest_norm_loso": ["scene", "subject"],
}
# Random-effect unit inside one subject's mixed model: the series key minus the subject.
GROUP_KEYS = {m: [k for k in keys if k != "subject"] for m, keys in SERIES_KEYS.items()}
# Recorded in mixed.json so a file fitted under an older design is refitted, not reused.
# v2: every fit is made under each of TRANSFORMS, not log1p alone.
MIXED_DESIGN = "per_subject_v2"
MIN_ATTEMPTS = 10
CURVE_MODELS = ("linear", "quadratic", "power", "exponential")
# Fixed practice term, and the time term each scene gets its own random slope on —
# always the form's own time variable, so ``b`` and the scene's ``v`` multiply the same
# x. ``quadratic`` gets no random t**2: every form then carries the same random part
# (intercept, one slope, their correlation) and AIC compares fixed shapes only.
MIXED_FORMS = {
    "linear": ("t", "t"),
    "quadratic": ("t + I(t ** 2)", "t"),
    "log": ("logt", "logt"),
    "log_phase": ("logt + C(phase)", "logt"),
}
T_SCALE = 10.0   # attempts per unit of `t` in the mixed model, for conditioning
# Smallest cluster that may serve as a d_nearest reference. The plain silhouette
# argmax leaves singleton "dust" clusters in ~a third of scenes; a singleton has no
# spread to normalise by, and a 2-member spread is one distance. d_own is unaffected.
NEAREST_MIN_SIZE = 3


def practice_dir(output_dir: Path) -> Path:
    return Path(output_dir) / "practice"


def attempts_path(scene_id: str, output_dir: Path) -> Path:
    return practice_dir(output_dir) / "attempts" / f"{scene_id}.csv"


# ---------------------------------------------------------------------------
# Medoids and per-attempt distances
# ---------------------------------------------------------------------------

def medoids(dist: np.ndarray, labels: np.ndarray, exclude: np.ndarray | None = None) -> dict:
    """``{cluster: (medoid_index, spread)}`` for every non-noise cluster.

    The medoid is the member with the smallest summed distance to the other
    members; ``spread`` is the median distance from the other members to it (NaN for
    a singleton, or when every other member sits at distance 0). ``exclude`` removes
    traces from the *search* — used for leave-subject-out — and a cluster with no
    member left is simply absent from the result.
    """
    keep = np.ones(len(labels), dtype=bool) if exclude is None else ~exclude
    out = {}
    for k in np.unique(labels):
        if k == -1:
            continue
        members = np.flatnonzero((labels == k) & keep)
        if members.size == 0:
            continue
        sub = dist[np.ix_(members, members)]
        m = int(members[np.argmin(sub.sum(axis=1))])
        others = members[members != m]
        spread = float(np.median(dist[others, m])) if others.size else np.nan
        out[int(k)] = (m, spread if spread > 0 else np.nan)
    return out


def _distances(dist, labels, meds, rows, sizes):
    """d_own, d_nearest, nearest_cluster, d_nearest_norm for ``rows`` against ``meds``.

    ``d_own`` uses every medoid; the nearest search only clusters of at least
    ``NEAREST_MIN_SIZE`` members (``sizes`` is the full-scene count, so the
    reference set is the same with or without leave-subject-out).
    """
    n = len(rows)
    d_own = np.full(n, np.nan)
    d_near = np.full(n, np.nan)
    near_k = np.full(n, -1)
    d_norm = np.full(n, np.nan)
    if not meds:
        return d_own, d_near, near_k, d_norm
    ks = np.array(sorted(meds))
    idx = np.array([meds[k][0] for k in ks])
    spreads = np.array([meds[k][1] for k in ks])
    to_meds = dist[np.ix_(rows, idx)]
    pos = {k: p for p, k in enumerate(ks)}
    for r, i in enumerate(rows):
        p = pos.get(int(labels[i]))
        if p is not None:
            d_own[r] = to_meds[r, p]
    ref = np.array([sizes[k] >= NEAREST_MIN_SIZE and np.isfinite(meds[k][1]) for k in ks])
    if not ref.any():
        return d_own, d_near, near_k, d_norm
    masked = np.where(ref[None, :], to_meds, np.inf)
    j = np.argmin(masked, axis=1)
    d_near = to_meds[np.arange(n), j]
    near_k = ks[j]
    d_norm = d_near / spreads[j]
    return d_own, d_near, near_k, d_norm


def attempt_distances(scene_id: str, output_dir: Path) -> pd.DataFrame | None:
    """One row per attempt in ``scene_id``: time, cluster, and every distance.

    Returns ``None`` (with a message) when the scene has no usable partition — a
    missing cluster file or matrix, a matrix whose size does not match the labels, or
    a scene whose ensemble carried no signal (all ``-1``).
    """
    output_dir = Path(output_dir)
    clusters = output_dir / "clusters" / f"{scene_id}_clusters.json"
    matrix = output_dir / "frechet" / f"{scene_id}.npy"
    if not clusters.exists() or not matrix.exists():
        print(f"{scene_id}: no cluster file or Fréchet matrix, skipping.")
        return None

    payload = json.loads(clusters.read_text())
    labels = np.asarray(payload["labels"])
    metadata = payload["metadata"]
    dist = np.load(matrix)
    if dist.shape != (len(labels), len(labels)):
        print(f"{scene_id}: Fréchet matrix {dist.shape} does not match "
              f"{len(labels)} labels, skipping.")
        return None
    if (labels == -1).all():
        print(f"{scene_id}: no clusters (ensemble carried no signal), skipping.")
        return None

    n = len(labels)
    subjects = np.array([m.get("Subject", "?") for m in metadata])
    all_rows = np.arange(n)

    sizes = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    meds = medoids(dist, labels)
    d_own, d_near, near_k, d_norm = _distances(dist, labels, meds, all_rows, sizes)
    medoid_idx = {m for m, _ in meds.values()}

    loso = {name: np.full(n, np.nan) for name in ("d_own", "d_nearest", "d_nearest_norm")}
    for s in np.unique(subjects):
        rows = np.flatnonzero(subjects == s)
        m_s = medoids(dist, labels, exclude=subjects == s)
        o, nr, _, nn = _distances(dist, labels, m_s, rows, sizes)
        loso["d_own"][rows], loso["d_nearest"][rows], loso["d_nearest_norm"][rows] = o, nr, nn

    df = pd.DataFrame({
        "scene": scene_id,
        "subject": subjects,
        "clip": [m.get("ClipCode", "") for m in metadata],
        "phase": [(m.get("Phase") or "").strip() for m in metadata],
        "stage": stage_tertiles(metadata),
        "outcome": [m.get("Outcome", "") for m in metadata],
        "cluster": labels,
        "attempt": attempt_index(metadata),
        "is_medoid": [i in medoid_idx for i in range(n)],
        "d_own": d_own,
        "d_nearest": d_near,
        "nearest_cluster": near_k,
        "d_nearest_norm": d_norm,
        "d_own_loso": loso["d_own"],
        "d_nearest_loso": loso["d_nearest"],
        "d_nearest_norm_loso": loso["d_nearest_norm"],
    })

    # Per subject, in attempt order: nth use of this cluster, and whether the
    # cluster differs from the previous attempt's (NaN on the first attempt).
    df = df.sort_values(["subject", "attempt"], kind="stable")
    df["cluster_use"] = df.groupby(["subject", "cluster"]).cumcount() + 1
    prev = df.groupby("subject")["cluster"].shift()
    df["switch"] = np.where(prev.isna(), np.nan, (df["cluster"] != prev).astype(float))
    return df.reset_index(drop=True)


def run_practice_distances(scenes, output_dir, *, force=False) -> None:
    """Write ``practice/attempts/<scene>.csv`` for each scene; skip ones on disk."""
    output_dir = Path(output_dir)
    (practice_dir(output_dir) / "attempts").mkdir(parents=True, exist_ok=True)
    for i, scene in enumerate(scenes, 1):
        out = attempts_path(scene, output_dir)
        if out.exists() and not force:
            print(f"  [{i}/{len(scenes)}] {scene}: skipping (output exists)")
            continue
        df = attempt_distances(scene, output_dir)
        if df is None:
            continue
        df.to_csv(out, index=False)
        print(f"  [{i}/{len(scenes)}] {scene}: {len(df)} attempts, "
              f"{df['cluster'].nunique()} clusters")


# ---------------------------------------------------------------------------
# Per-series curve comparison
# ---------------------------------------------------------------------------

def _power(t, a, b, c):
    return a + b * t ** (-c)


def _exponential(t, a, b, c):
    return a + b * np.exp(-c * (t - 1))


def _aicc(rss, n, k):
    """Gaussian AICc with ``k`` mean parameters (+1 for the residual variance)."""
    k = k + 1
    rss = max(rss, 1e-12)
    aic = n * np.log(rss / n) + 2 * k
    return aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf


def fit_curves(t, y, transform: str = PRIMARY_TRANSFORM) -> dict:
    """Fit the four practice curves to the transformed ``y`` vs attempt, compare by AICc.

    Returns per-model AICc, the best model, its margin over the runner-up, the linear
    slope (transformed units per attempt), and ``change`` — the best model's predicted
    transformed distance at the last attempt minus at the first, so a negative value
    means the distance fell with practice whatever the curve's shape. A model that
    fails to converge is recorded as ``None`` rather than dropped silently. ``y`` must
    already exclude values the transform cannot take (``series_curves`` does this).
    """
    t = np.asarray(t, dtype=float)
    y = TRANSFORMS[transform][0](np.asarray(y, dtype=float))
    n = len(t)
    preds, aicc = {}, {}

    for name, deg in (("linear", 1), ("quadratic", 2)):
        coef = np.polyfit(t, y, deg)
        preds[name] = lambda x, coef=coef: np.polyval(coef, x)
        aicc[name] = _aicc(float(np.sum((y - np.polyval(coef, t)) ** 2)), n, deg + 1)
    slope = float(np.polyfit(t, y, 1)[0])

    tail = y[t >= np.quantile(t, 2 / 3)].mean()
    head = y[t <= np.quantile(t, 1 / 3)].mean()
    for name, f, c0, c_hi in (("power", _power, 0.5, 5.0),
                              ("exponential", _exponential, 0.05, 2.0)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, _ = curve_fit(f, t, y, p0=(tail, head - tail, c0),
                                 bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, c_hi]),
                                 maxfev=5000)
            preds[name] = lambda x, p=p, f=f: f(x, *p)
            aicc[name] = _aicc(float(np.sum((y - f(t, *p)) ** 2)), n, 3)
        except (RuntimeError, ValueError):
            aicc[name] = None

    ranked = sorted((v, k) for k, v in aicc.items() if v is not None and np.isfinite(v))
    best = ranked[0][1]
    lo, hi = t.min(), t.max()
    return {
        "n": n,
        "t_first": int(lo),
        "t_last": int(hi),
        **{f"aicc_{k}": aicc[k] for k in CURVE_MODELS},
        "best": best,
        "delta_aicc": float(ranked[1][0] - ranked[0][0]) if len(ranked) > 1 else None,
        "slope": slope,
        "change": float(preds[best](hi) - preds[best](lo)),
    }


def _excluded_medoid(df, measure):
    """Attempts that are their own reference, and so score 0 by construction.

    ``is_medoid`` flags the all-subject medoids. Under leave-subject-out a subject is
    never measured against their own attempt, so none of their rows is a medoid there
    and all of them are kept.
    """
    if measure.endswith("_loso"):
        return pd.Series(False, index=df.index)
    return df["is_medoid"]


def series_curves(df: pd.DataFrame, measure: str, min_attempts: int = MIN_ATTEMPTS,
                  transform: str = PRIMARY_TRANSFORM) -> pd.DataFrame:
    """``fit_curves`` for every series of ``measure`` with enough points."""
    keys = SERIES_KEYS[measure]
    d = df[~_excluded_medoid(df, measure) & df[measure].notna() & np.isfinite(df[measure])]
    if TRANSFORMS[transform][2]:
        d = d[d[measure] > 0]
    rows = []
    for key, g in d.groupby(keys):
        if len(g) < min_attempts or g["attempt"].nunique() < 4:
            continue
        rec = dict(zip(keys, key))
        rec.update(measure=measure, transform=transform,
                   **fit_curves(g["attempt"], g[measure], transform))
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-subject mixed model
# ---------------------------------------------------------------------------

def _mixed_frame(df, measure, subject, completed_only=False, transform=PRIMARY_TRANSFORM):
    """One subject's rows for ``measure``, with ``y`` the transformed outcome.

    Under a transform that cannot take zero (``log``) the ``d = 0`` rows are dropped;
    the number dropped is ``attrs["n_zero_dropped"]``.
    """
    d = df[(df["subject"] == subject) & ~_excluded_medoid(df, measure)
           & df[measure].notna() & np.isfinite(df[measure])].copy()
    if completed_only:
        d = d[d["outcome"] == "completed"]
    d = d[d["phase"] != ""]
    forward, _, drop_zero = TRANSFORMS[transform]
    n_zero = int((d[measure] <= 0).sum()) if drop_zero else 0
    if drop_zero:
        d = d[d[measure] > 0]
    d["y"] = forward(d[measure])
    d["t"] = d["attempt"] / T_SCALE
    d["logt"] = np.log(d["attempt"])
    d["group"] = d[GROUP_KEYS[measure]].astype(str).agg("|".join, axis=1)
    d.attrs["n_zero_dropped"] = n_zero
    return d


# Optimisers tried in turn. statsmodels moves along the first list only when a method
# raises; a fit that "succeeds" with a non-finite likelihood (seen with lbfgs on the
# random-intercept-only model) has to be caught here and retried.
FIT_METHODS = (["lbfgs", "powell"], "nm")


def _fit_lmm(smf, formula, d, re_formula, groups="group", vc_formula=None):
    """Fit with the first optimiser that reaches a finite likelihood; one that raises
    (e.g. a singular matrix along the way) is skipped rather than ending the fit."""
    last = None
    for method in FIT_METHODS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.mixedlm(formula, d, groups=d[groups], re_formula=re_formula,
                                  vc_formula=vc_formula).fit(reml=False, method=method)
        except (np.linalg.LinAlgError, ValueError) as err:
            last = err
            continue
        if np.isfinite(res.llf):
            return res
    raise ValueError(f"no optimiser reached a finite likelihood ({last})")


# Optimisers and starting level variances tried for the nested model, best likelihood kept.
NESTED_METHODS = ("lbfgs", "bfgs", "powell", "nm")
NESTED_START_SCALES = (1e-3, 1e-2)


def _fit_nested(smf, d, base):
    """``level_scene`` warm-started from the ``scene_uncorr`` fit ``base``.

    The nested model contains ``base`` as the special case where every level variance is
    zero, so its maximum likelihood can only be higher. Fitted cold, the optimiser often
    stopped in a worse local optimum (9 of 40 fits, by up to 97 log-likelihood units). So
    it starts from ``base``'s solution with small level variances, every optimiser in
    ``NESTED_METHODS`` is tried, and the best likelihood is kept."""
    from statsmodels.regression.mixed_linear_model import MixedLMParams

    model = smf.mixedlm("y ~ logt", d, groups=d["level"], re_formula="~logt",
                        vc_formula=SCENE_VC)
    best, how = None, None
    for scale in NESTED_START_SCALES:
        start = MixedLMParams.from_components(fe_params=base.fe_params.values,
                                              cov_re=np.eye(2) * scale, vcomp=base.vcomp)
        for method in NESTED_METHODS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = model.fit(reml=False, method=method, start_params=start)
            except (np.linalg.LinAlgError, ValueError):
                continue
            if np.isfinite(res.llf) and (best is None or res.llf > best.llf):
                best, how = res, f"{method}, start level var {scale:g}"
    if best is None:
        raise ValueError("no optimiser reached a finite likelihood")
    return best, how


def _random_effects(res) -> dict:
    """Variances of the random intercept and slope, their correlation, and the residual."""
    cov = np.asarray(res.cov_re)
    out = {"var_intercept": float(cov[0, 0]), "resid_var": float(res.scale)}
    if cov.shape == (2, 2):
        out["var_slope"] = float(cov[1, 1])
        denom = np.sqrt(cov[0, 0] * cov[1, 1])
        out["corr"] = float(cov[0, 1] / denom) if denom > 0 else None
    return out


def fit_mixed(df: pd.DataFrame, measure: str, subject: str,
              completed_only: bool = False, transform: str = PRIMARY_TRANSFORM) -> dict:
    """One subject's MixedLM on the transformed measure, per practice form, by ML AIC.

    Groups are the measure's series unit within the subject (scene for
    ``d_nearest_norm``, scene × cluster for ``d_own``), each with a random intercept
    and a random slope on ``t`` (attempt / 10) in every form, so the forms differ only
    in their fixed practice term. Fitted by ML rather than REML so forms with
    different fixed effects can be compared by AIC. ``log_phase`` adds the
    discovery/practice step, so its ``logt`` coefficient is the within-phase trend —
    phase and attempt are partly collinear by design (discovery comes first).

    Each form records its fixed effects, the random-effect variances (``re``) and,
    except ``log_phase``, ``curve``: the fixed-effect prediction on a grid of attempts,
    in transformed units (``y``; back-transform with ``TRANSFORMS[transform][1]``).
    """
    import statsmodels.formula.api as smf

    d = _mixed_frame(df, measure, subject, completed_only, transform)
    out = {"measure": measure, "subject": subject, "completed_only": completed_only,
           "transform": transform, "n_obs": int(len(d)),
           "n_zero_dropped": d.attrs["n_zero_dropped"],
           "n_groups": int(d["group"].nunique()), "forms": {}}
    if d["group"].nunique() < 2:
        out["best_form"] = None
        return out
    grid = np.unique(np.linspace(1, np.percentile(d["attempt"], 95), 60).round())

    for name, (term, slope) in MIXED_FORMS.items():
        formula = f"y ~ {term}"
        try:
            res = _fit_lmm(smf, formula, d, f"~{slope}")
        except (np.linalg.LinAlgError, ValueError) as err:
            out["forms"][name] = {"formula": formula, "error": str(err)}
            continue
        ci = res.conf_int()
        fe = res.fe_params.index
        rec = {
            "formula": formula,
            "re_formula": f"~{slope}",
            "converged": bool(res.converged),
            "aic": float(res.aic),
            "llf": float(res.llf),
            "coef": {k: float(res.fe_params[k]) for k in fe},
            "ci": {k: [float(ci.loc[k, 0]), float(ci.loc[k, 1])] for k in fe},
            "p": {k: float(res.pvalues[k]) for k in fe},
            "re": _random_effects(res),
        }
        if "phase" not in term:
            grid_df = pd.DataFrame({"attempt": grid})
            grid_df["t"] = grid_df["attempt"] / T_SCALE
            grid_df["logt"] = np.log(grid_df["attempt"])
            rec["curve"] = {"attempt": grid.astype(int).tolist(),
                            "y": np.asarray(res.predict(exog=grid_df)).tolist()}
        out["forms"][name] = rec

    aics = {k: v["aic"] for k, v in out["forms"].items() if "aic" in v}
    comparable = {k: v for k, v in aics.items() if k != "log_phase"}
    out["best_form"] = min(comparable, key=comparable.get) if comparable else None
    return out


# ---------------------------------------------------------------------------
# Random-effect structure: scenes nested in levels
# ---------------------------------------------------------------------------

# Alternative random-effect structures, fitted beside the main model for the primary
# practice form (``log``). The main model (structure ``scene``, in mixed.json) gives each
# scene a correlated random intercept and slope and assumes scenes are independent. But a
# level's scenes are played back to back, and the level bootstrap showed some correlation
# between them. ``level_scene`` adds a random intercept and slope per **level**, with the
# scene effects nested inside — the model-based answer to that. statsmodels can only nest
# the scene effects as independent variance components, so ``level_scene`` has no
# scene intercept-slope correlation; ``scene_uncorr`` is the scene-only model with that same
# restriction, so that ``level_scene`` against ``scene_uncorr`` isolates what levels add.
# The ideal model — levels plus correlated scene effects — needs lme4 and is not fitted.
# d_own is left out: its unit is scene x cluster, which would need a third level.
STRUCTURES = ("scene_uncorr", "level_scene")
NESTED_MEASURES = ("d_nearest_norm", "d_nearest_norm_loso")
NESTED_DESIGN = "v2"   # v2: level_scene warm-started from scene_uncorr, best of several optimisers
SCENE_VC = {"scene": "0 + C(group)", "scene_slope": "0 + C(group):logt"}


def level_of(scene: str) -> str:
    """``w1l1s3`` -> ``w1l1``."""
    return re.match(r"(w\d+l\d+)", scene).group(1)


def fit_structure(df: pd.DataFrame, measure: str, subject: str, structure: str,
                  completed_only: bool = False, transform: str = PRIMARY_TRANSFORM) -> dict:
    """One subject's ``log``-form model under an alternative random-effect structure.

    Grouped by level in both cases, with each scene's intercept and slope as independent
    variance components inside it; ``level_scene`` adds a correlated random intercept and
    slope per level, ``scene_uncorr`` does not. Records the fixed effects, the variances
    at each level (``level``: intercept, slope, correlation; ``scene``: intercept, slope),
    the residual variance and the fitted curve, in the fit's transform units.
    """
    import statsmodels.formula.api as smf

    d = _mixed_frame(df, measure, subject, completed_only, transform)
    d["level"] = d["group"].map(level_of)
    out = {"measure": measure, "subject": subject, "completed_only": completed_only,
           "transform": transform, "structure": structure, "form": "log",
           "n_obs": int(len(d)), "n_zero_dropped": d.attrs["n_zero_dropped"],
           "n_groups": int(d["group"].nunique()), "n_levels": int(d["level"].nunique())}
    try:
        base = _fit_lmm(smf, "y ~ logt", d, "0", groups="level", vc_formula=SCENE_VC)
        res, how = (_fit_nested(smf, d, base) if structure == "level_scene"
                    else (base, "cold start"))
    except (np.linalg.LinAlgError, ValueError) as err:
        out["error"] = str(err)
        return out
    out["fit_note"] = how
    if structure == "level_scene":
        # Should never be negative; kept so a stuck optimiser is visible in the record.
        out["llf_gain_over_scene_uncorr"] = float(res.llf - base.llf)
    ci = res.conf_int()
    fe = res.fe_params.index
    vcomp = dict(zip(res.model.exog_vc.names, (float(v) for v in res.vcomp)))
    variances = {"scene": {"var_intercept": vcomp.get("scene"),
                           "var_slope": vcomp.get("scene_slope")},
                 "resid_var": float(res.scale)}
    if structure == "level_scene":
        variances["level"] = _random_effects(res)
        variances["level"].pop("resid_var", None)
    grid = np.unique(np.linspace(1, np.percentile(d["attempt"], 95), 60).round())
    grid_df = pd.DataFrame({"attempt": grid, "logt": np.log(grid)})
    out.update({
        "formula": "y ~ logt", "converged": bool(res.converged),
        "aic": float(res.aic), "llf": float(res.llf),
        "coef": {k: float(res.fe_params[k]) for k in fe},
        "ci": {k: [float(ci.loc[k, 0]), float(ci.loc[k, 1])] for k in fe},
        "p": {k: float(res.pvalues[k]) for k in fe},
        "re": variances,
        "curve": {"attempt": grid.astype(int).tolist(),
                  "y": (res.fe_params["Intercept"] + res.fe_params["logt"] * grid_df["logt"]).tolist()},
    })
    return out


# ---------------------------------------------------------------------------
# Strategy switching
# ---------------------------------------------------------------------------

def switch_summary(df: pd.DataFrame, bin_width: int = 5) -> dict:
    """Does strategy choice settle with practice?

    ``binned``: switch rate per subject per attempt bin (pooled over scenes).
    ``trend``: logistic GLM ``switch ~ C(subject) + log(attempt)`` with standard errors
    clustered by subject × scene, since attempts within a series are not independent.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    d = df[df["switch"].notna()].copy()
    d["bin"] = ((d["attempt"] - 1) // bin_width) * bin_width + 1
    binned = (d.groupby(["subject", "bin"])["switch"]
                .agg(["mean", "size"]).reset_index()
                .rename(columns={"mean": "rate", "size": "n"}))

    d["logt"] = np.log(d["attempt"])
    groups = pd.factorize(d["subject"] + "|" + d["scene"])[0]
    res = smf.glm("switch ~ C(subject) + logt", d, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    ci = res.conf_int().loc["logt"]
    return {
        "bin_width": bin_width,
        "binned": binned.to_dict(orient="records"),
        "trend": {"coef_logt": float(res.params["logt"]),
                  "ci": [float(ci[0]), float(ci[1])],
                  "p": float(res.pvalues["logt"]),
                  "n_obs": int(len(d))},
    }


# Time term each form's slope is reported on, and the attempt the summary predicts at.
SLOPE_TERM = {"linear": "t", "quadratic": "t", "log": "logt", "log_phase": "logt"}
SUMMARY_ATTEMPT = 30


def _predict(coef, form, attempt):
    """Fixed-effect prediction at ``attempt``, in transformed units (discovery level for
    log_phase)."""
    t = attempt / T_SCALE
    y = coef["Intercept"]
    if form in ("linear", "quadratic"):
        y += coef["t"] * t + coef.get("I(t ** 2)", 0.0) * t ** 2
    else:
        y += coef["logt"] * np.log(attempt)
    return float(y)


def mixed_summary(mixed: dict, nested: dict | None = None) -> pd.DataFrame:
    """Flatten ``mixed.json`` into one row per subject × measure × transform × attempts × form.

    With ``nested`` (``nested.json``), the alternative random-effect structures are added
    as further rows, told apart by ``structure`` (``scene`` for the main model,
    ``scene_uncorr``, ``level_scene``), with their level variances in ``var_level_*``.

    ``slope`` is the form's practice coefficient (``t``, per 10 attempts, for linear
    and quadratic; ``log(attempt)`` for the log forms), with its CI, p-value and
    whether the CI excludes zero. ``ratio_first`` / ``ratio_at_30`` back-transform the
    fixed-effect line to the distance ratio at attempt 1 and 30 — for ``log_phase``
    that is the discovery level, so read its ``phase_step`` alongside. ``primary``
    flags the main result (the primary measure and transform, all attempts, ``log``
    form). Slopes are in the row's ``transform`` units, so compare across transforms
    through the ratio columns, never the slopes.
    """
    rows = []
    for f in mixed["fits"]:
        inverse = TRANSFORMS[f["transform"]][1]
        for form, r in f["forms"].items():
            row = {"subject": f["subject"], "measure": f["measure"],
                   "transform": f["transform"], "structure": "scene",
                   "completed_only": f["completed_only"], "form": form,
                   "primary": (f["measure"] == mixed.get("primary")
                               and f["transform"] == mixed.get("primary_transform")
                               and not f["completed_only"] and form == "log"),
                   "is_best_form": form == f["best_form"],
                   "n_obs": f["n_obs"], "n_zero_dropped": f["n_zero_dropped"],
                   "n_groups": f["n_groups"]}
            if "coef" not in r:
                row["error"] = r.get("error")
                rows.append(row)
                continue
            term = SLOPE_TERM[form]
            lo, hi = r["ci"][term]
            re = r["re"]
            row.update({
                "intercept": r["coef"]["Intercept"],
                "slope": r["coef"][term], "slope_ci_low": lo, "slope_ci_high": hi,
                "slope_p": r["p"][term], "ci_excludes_zero": bool(hi < 0 or lo > 0),
                "t2": r["coef"].get("I(t ** 2)"),
                "phase_step": r["coef"].get("C(phase)[T.practice]"),
                "ratio_first": float(inverse(_predict(r["coef"], form, 1))),
                f"ratio_at_{SUMMARY_ATTEMPT}": float(inverse(
                    _predict(r["coef"], form, SUMMARY_ATTEMPT))),
                "var_intercept": re.get("var_intercept"), "var_slope": re.get("var_slope"),
                "corr": re.get("corr"), "resid_var": re.get("resid_var"),
                "aic": r["aic"], "converged": r["converged"],
            })
            rows.append(row)
    for f in (nested or {}).get("fits", []):
        row = {"subject": f["subject"], "measure": f["measure"], "transform": f["transform"],
               "structure": f["structure"], "completed_only": f["completed_only"],
               "form": f["form"], "primary": False, "n_obs": f["n_obs"],
               "n_zero_dropped": f["n_zero_dropped"], "n_groups": f["n_groups"],
               "n_levels": f["n_levels"]}
        if "coef" not in f:
            row["error"] = f.get("error")
            rows.append(row)
            continue
        inverse = TRANSFORMS[f["transform"]][1]
        lo, hi = f["ci"]["logt"]
        lev = f["re"].get("level", {})
        row.update({
            "intercept": f["coef"]["Intercept"], "slope": f["coef"]["logt"],
            "slope_ci_low": lo, "slope_ci_high": hi, "slope_p": f["p"]["logt"],
            "ci_excludes_zero": bool(hi < 0 or lo > 0),
            "ratio_first": float(inverse(_predict(f["coef"], "log", 1))),
            f"ratio_at_{SUMMARY_ATTEMPT}": float(inverse(_predict(f["coef"], "log", SUMMARY_ATTEMPT))),
            "var_intercept": f["re"]["scene"]["var_intercept"],
            "var_slope": f["re"]["scene"]["var_slope"],
            "var_level_intercept": lev.get("var_intercept"), "var_level_slope": lev.get("var_slope"),
            "corr_level": lev.get("corr"), "resid_var": f["re"]["resid_var"],
            "aic": f["aic"], "converged": f["converged"],
        })
        rows.append(row)
    order = {k: i for i, k in enumerate(MIXED_FORMS)}
    torder = {k: i for i, k in enumerate(TRANSFORMS)}
    sorder = {k: i for i, k in enumerate(("scene",) + STRUCTURES)}
    df = pd.DataFrame(rows)
    df["_form"] = df["form"].map(order)
    df["_tr"] = df["transform"].map(torder)
    df["_st"] = df["structure"].map(sorder)
    return (df.sort_values(["primary", "_st", "_tr", "measure", "completed_only", "_form", "subject"],
                           ascending=[False, True, True, True, True, True, True])
              .drop(columns=["_form", "_tr", "_st"]).reset_index(drop=True))


def write_mixed_summary(pdir: Path) -> None:
    """Write ``mixed_summary.csv`` from ``mixed.json`` (and ``nested.json``) — no refitting."""
    mixed = json.loads((pdir / "mixed.json").read_text())
    if mixed.get("design") != MIXED_DESIGN:
        return
    nested_path = pdir / "nested.json"
    nested = json.loads(nested_path.read_text()) if nested_path.exists() else None
    mixed_summary(mixed, nested).to_csv(pdir / "mixed_summary.csv", index=False)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_attempts(output_dir: Path, scenes=None) -> pd.DataFrame:
    files = sorted((practice_dir(output_dir) / "attempts").glob("*.csv"))
    if scenes is not None:
        files = [f for f in files if f.stem in set(scenes)]
    return pd.concat([pd.read_csv(f, dtype={"subject": str, "clip": str,
                                            "phase": str, "stage": str})
                      .fillna({"phase": "", "stage": ""}) for f in files],
                     ignore_index=True)


def run_practice_models(output_dir, *, measures=MEASURES, min_attempts=MIN_ATTEMPTS,
                        scenes=None, force=False, n_jobs=-1) -> None:
    """Fit curves, per-subject mixed models and the switching trend over the stored attempts.

    Writes ``practice/curves.csv`` and ``practice/mixed.json`` (one fit per subject ×
    measure × raw/``_loso`` × transform × all/completed attempts), ``practice/nested.json``
    (the alternative random-effect structures, ``fit_structure``), the flat
    ``mixed_summary.csv`` over both, and ``practice/switching.json``. The two model files
    are checked separately, so adding a structure refits only ``nested.json``. The mixed fits are
    independent, so they run in parallel (``n_jobs``, joblib); each is deterministic, so
    parallelism never changes a number.
    Skips when ``mixed.json`` already records the same scenes, measures and design, so
    it reruns by itself once more attempt files land, or when the model design changes. A subset run
    (``scenes``, i.e. ``--smoke``) skips whenever any result is on disk, so it can never
    overwrite a full fit; and since its scene list differs, a later full run refits.
    """
    output_dir = Path(output_dir)
    pdir = practice_dir(output_dir)
    df = load_attempts(output_dir, scenes)
    used = sorted(df["scene"].unique())
    mixed_path = pdir / "mixed.json"
    nested_path = pdir / "nested.json"
    if mixed_path.exists() and not force and scenes is not None:
        print("Practice models already on disk; a subset run never replaces them, "
              "skipping (use --force to refit on the subset).")
        return
    mixed_ok = nested_ok = False
    if mixed_path.exists() and not force:
        prev = json.loads(mixed_path.read_text())
        mixed_ok = (prev.get("scenes") == used and prev.get("measures") == list(measures)
                    and prev.get("design") == MIXED_DESIGN)
    if nested_path.exists() and not force:
        prev = json.loads(nested_path.read_text())
        nested_ok = prev.get("scenes") == used and prev.get("design") == NESTED_DESIGN
    if mixed_ok and nested_ok:
        print("Practice models already fitted on these scenes, skipping "
              "(use --force to refit).")
        if not (pdir / "mixed_summary.csv").exists():
            write_mixed_summary(pdir)
            print(f"Wrote {pdir}/mixed_summary.csv from the existing fits.")
        return
    from joblib import Parallel, delayed

    subjects = sorted(df["subject"].unique())
    if not nested_ok:
        jobs = [(col, subject, structure, completed_only, transform)
                for transform in TRANSFORMS for col in NESTED_MEASURES
                for structure in STRUCTURES for completed_only in (False, True)
                for subject in subjects]
        print(f"  {len(jobs)} alternative-structure fits (scene_uncorr, level_scene)...")
        nested = Parallel(n_jobs=n_jobs)(
            delayed(fit_structure)(df, col, subject, structure, completed_only, transform)
            for col, subject, structure, completed_only, transform in jobs)
        nested_path.write_text(json.dumps({"design": NESTED_DESIGN, "scenes": used,
                                           "structures": list(STRUCTURES),
                                           "subjects": subjects, "fits": nested}, indent=2))
        print(f"Wrote {nested_path}")
    if mixed_ok:
        write_mixed_summary(pdir)
        print(f"Main models already fitted; wrote {pdir}/mixed_summary.csv with the new structures.")
        return

    print(f"Fitting practice models on {len(df)} attempts over {len(used)} scenes...")
    curves = []
    for transform in TRANSFORMS:
        for m in measures:
            for col in (m, f"{m}_loso"):
                c = series_curves(df, col, min_attempts, transform)
                print(f"  curves {col} [{transform}]: {len(c)} series")
                curves.append(c)
    pd.concat(curves, ignore_index=True).to_csv(pdir / "curves.csv", index=False)

    jobs = [(col, subject, completed_only, transform)
            for transform in TRANSFORMS
            for m in measures for col in (m, f"{m}_loso")
            for completed_only in (False, True)
            for subject in subjects]
    print(f"  {len(jobs)} mixed fits...")
    mixed = Parallel(n_jobs=n_jobs)(
        delayed(fit_mixed)(df, col, subject, completed_only, transform)
        for col, subject, completed_only, transform in jobs)
    for r in mixed:
        print(f"  mixed {r['measure']} [{r['transform']}] sub-{r['subject']}"
              f"{' (completed)' if r['completed_only'] else ''}: best form {r['best_form']}")
    mixed_path.write_text(json.dumps({"design": MIXED_DESIGN, "primary": PRIMARY,
                                      "primary_transform": PRIMARY_TRANSFORM,
                                      "transforms": list(TRANSFORMS),
                                      "scenes": used, "measures": list(measures),
                                      "subjects": subjects, "t_scale": T_SCALE,
                                      "fits": mixed}, indent=2))

    write_mixed_summary(pdir)

    (pdir / "switching.json").write_text(json.dumps(switch_summary(df), indent=2))
    print(f"Wrote {pdir}/curves.csv, mixed.json, mixed_summary.csv, switching.json")
