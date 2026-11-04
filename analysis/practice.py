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

**The scene-scale model is the main model on the primary measure** (``run_practice_lme4``):
``log(d) ~ log(attempt) + (1 + log(attempt) | level) + (1 + log(attempt) | level:scene)``
— a level's scenes are played back to back, so they are not independent, and only lme4
can nest correlated scene effects inside levels. It is fitted in R by
``analysis/practice_lme4.R`` into ``lme4.json``; ``mixed_summary.csv`` marks its rows as
``primary``. Beside it, the **route-scale model** (``lme4_route.json``,
``analysis/practice_lme4_route.R``) is the other main model: distance to the attempt's own
route prototype against the nth use of that route, routes nested in scenes in levels. The
statsmodels models (``fit_mixed``, ``fit_structure``) stay beside them as the robustness
checks they became, and as the scene-scale stand-in when R is not available.

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

# The scene-scale model, fitted in R (analysis/practice_lme4.R). d_own is not fitted: its unit
# is scene x cluster, which would need a third nested level.
LME4_MEASURES = ("d_nearest_norm", "d_nearest_norm_loso")
LME4_DESIGN = "v1"
LME4_SCRIPT = Path(__file__).with_name("practice_lme4.R")
PRIMARY_STRUCTURE = "lme4_nested"     # which rows mixed_summary marks primary, when fitted
# Route-scale (within-route) model, the second main model: distance (px) to the attempt's own route prototype, against
# the nth use of that route, routes nested in scenes nested in levels.
ROUTE_MEASURES = ("d_own", "d_own_loso")
ROUTE_DESIGN = "v1"
ROUTE_SCRIPT = Path(__file__).with_name("practice_lme4_route.R")


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


def mixed_summary(mixed: dict, nested: dict | None = None, lme4: dict | None = None,
                  lme4_route: dict | None = None) -> pd.DataFrame:
    """Flatten ``mixed.json`` into one row per subject × measure × transform × attempts × form.

    With ``nested`` (``nested.json``), the alternative random-effect structures are added
    as further rows, told apart by ``structure`` (``scene`` for the main model,
    ``scene_uncorr``, ``level_scene``), with their level variances in ``var_level_*``.
    With ``lme4`` (``lme4.json``), the R fits are added as ``lme4_scene`` (the main model,
    refitted as a cross-check) and ``lme4_nested`` (the scene-scale model); their CI is the
    profile-likelihood one, Wald where the profile failed (``ci_method``). When lme4 rows
    exist they are the ones marked ``primary``; otherwise the statsmodels main model is.
    With ``lme4_route`` (``lme4_route.json``), the route-scale model is added as
    ``lme4_route``: its ``slope`` is per log nth use of the route (``clock``), its distance
    is in px, and ``n_groups`` counts routes.

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
                   "primary": (lme4 is None and f["measure"] == mixed.get("primary")
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
    for f in (lme4_route or {}).get("fits", []):
        inverse = TRANSFORMS[f["transform"]][1]
        coef = {"Intercept": f["intercept"], "logt": f["b"]}
        lo, hi = f["ci"]
        rows.append({
            "subject": f["subject"], "measure": f["measure"], "transform": f["transform"],
            "structure": "lme4_route", "clock": "route_use", "completed_only": f["completed_only"],
            "form": "log", "primary": False, "n_obs": f["n_obs"], "n_groups": f["n_routes"],
            "n_levels": f["n_levels"], "intercept": f["intercept"], "slope": f["b"],
            "slope_ci_low": lo, "slope_ci_high": hi, "ci_method": f["ci_method"],
            "slope_p": f["p_wald"], "ci_excludes_zero": bool(hi < 0 or lo > 0),
            "ratio_first": float(inverse(_predict(coef, "log", 1))),
            f"ratio_at_{SUMMARY_ATTEMPT}": float(inverse(_predict(coef, "log", SUMMARY_ATTEMPT))),
            "var_intercept": f["route_var_intercept"], "var_slope": f["route_var_slope"],
            "corr": f["route_corr"], "var_level_intercept": f["level_var_intercept"],
            "var_level_slope": f["level_var_slope"], "corr_level": f["level_corr"],
            "resid_var": f["resid_var"], "aic": f["aic"], "converged": not f["warnings"],
            "singular": f["singular"],
        })
    for f in (lme4 or {}).get("fits", []):
        structure = f"lme4_{f['model']}"
        inverse = TRANSFORMS[f["transform"]][1]
        coef = {"Intercept": f["intercept"], "logt": f["b"]}
        lo, hi = f["ci"]
        rows.append({
            "subject": f["subject"], "measure": f["measure"], "transform": f["transform"],
            "structure": structure, "completed_only": f["completed_only"], "form": "log",
            "primary": (structure == PRIMARY_STRUCTURE and f["measure"] == lme4.get("primary")
                        and f["transform"] == lme4.get("primary_transform")
                        and not f["completed_only"]),
            "n_obs": f["n_obs"], "n_groups": f["n_scenes"], "n_levels": f["n_levels"],
            "intercept": f["intercept"], "slope": f["b"], "slope_ci_low": lo, "slope_ci_high": hi,
            "ci_method": f["ci_method"], "slope_p": f["p_wald"],
            "ci_excludes_zero": bool(hi < 0 or lo > 0),
            "ratio_first": float(inverse(_predict(coef, "log", 1))),
            f"ratio_at_{SUMMARY_ATTEMPT}": float(inverse(_predict(coef, "log", SUMMARY_ATTEMPT))),
            "var_intercept": f["scene_var_intercept"], "var_slope": f["scene_var_slope"],
            "corr": f["scene_corr"], "var_level_intercept": f.get("level_var_intercept"),
            "var_level_slope": f.get("level_var_slope"), "corr_level": f.get("level_corr"),
            "resid_var": f["resid_var"], "aic": f["aic"], "converged": not f["warnings"],
            "singular": f["singular"], "lrt_p_levels": f.get("lrt_p"),
        })
    order = {k: i for i, k in enumerate(MIXED_FORMS)}
    torder = {k: i for i, k in enumerate(TRANSFORMS)}
    sorder = {k: i for i, k in enumerate(("lme4_nested", "lme4_route", "scene") + STRUCTURES
                                         + ("lme4_scene",))}
    df = pd.DataFrame(rows)
    df["slope_p_bh"] = _bh_by_family(df)
    df["_form"] = df["form"].map(order)
    df["_tr"] = df["transform"].map(torder)
    df["_st"] = df["structure"].map(sorder)
    return (df.sort_values(["primary", "_st", "_tr", "measure", "completed_only", "_form", "subject"],
                           ascending=[False, True, True, True, True, True, True])
              .drop(columns=["_form", "_tr", "_st"]).reset_index(drop=True))


# One family of tests = one fit setting run once per subject; the Benjamini-Hochberg
# correction runs across the subjects in it, as the headline does ("tested on five subjects").
TEST_FAMILY = ["structure", "measure", "transform", "completed_only", "form"]


def _bh_by_family(df: pd.DataFrame) -> pd.Series:
    """Benjamini-Hochberg FDR-adjusted ``slope_p`` across the subjects of each ``TEST_FAMILY``.

    Computed by ``statsmodels.stats.multitest.multipletests(method="fdr_bh")``. It controls
    the false discovery rate, the expected share of false positives among the subjects
    called significant. A missing p stays missing and does not count towards m. This column
    is the one source of BH values: the dashboard and the summary figure read it rather than
    adjusting again.
    """
    from statsmodels.stats.multitest import multipletests

    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(TEST_FAMILY, dropna=False):
        p = g["slope_p"].dropna()
        if len(p):
            out[p.index] = multipletests(p.to_numpy(), method="fdr_bh")[1]
    return out


def write_mixed_summary(pdir: Path) -> None:
    """Write ``mixed_summary.csv`` from ``mixed.json`` (with ``nested.json`` and ``lme4.json``
    when present) — no refitting."""
    mixed = json.loads((pdir / "mixed.json").read_text())
    if mixed.get("design") != MIXED_DESIGN:
        return
    read = lambda name: json.loads((pdir / name).read_text()) if (pdir / name).exists() else None
    mixed_summary(mixed, read("nested.json"), read("lme4.json"), read("lme4_route.json")).to_csv(
        pdir / "mixed_summary.csv", index=False)


def _nan_to_none(v):
    return None if isinstance(v, float) and not np.isfinite(v) else v


def _run_lme4_script(script: Path, frames: pd.DataFrame, n_cores: int, extra=()):
    """Run one of the lme4 R scripts on ``frames``; returns (fits, curves) DataFrames."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frames.to_csv(tmp / "frames.csv", index=False)
        subprocess.run(["Rscript", str(script), str(tmp / "frames.csv"), str(tmp / "fits.csv"),
                        str(tmp / "curves.csv"), str(n_cores), *extra], check=True)
        return (pd.read_csv(tmp / "fits.csv", dtype={"subject": str}),
                pd.read_csv(tmp / "curves.csv", dtype={"subject": str}))


def _lme4_records(fits, curves, zeros, clock):
    """lme4 output rows as JSON records: CI chosen, zeros counted, curve attached."""
    keys = ["measure", "subject", "transform", "completed_only", "model"]
    records = []
    for _, r in fits.iterrows():
        rec = {k: _nan_to_none(v.item() if hasattr(v, "item") else v) for k, v in r.items()}
        rec["warnings"] = rec.get("warnings") or ""
        profile_ok = rec["profile_low"] is not None and rec["profile_high"] is not None
        rec["ci_method"] = "profile" if profile_ok else "wald"
        rec["ci"] = ([rec["profile_low"], rec["profile_high"]] if profile_ok
                     else [rec["wald_low"], rec["wald_high"]])
        rec["n_zero_dropped"] = int(zeros[(rec["measure"], rec["subject"], rec["transform"],
                                           rec["completed_only"])])
        c = curves[(curves[keys] == pd.Series({k: rec[k] for k in keys})).all(axis=1)]
        rec["curve"] = {clock: c[clock].astype(int).tolist(), "y": c["y"].tolist()}
        if clock != "attempt":
            rec["curve"]["attempt"] = rec["curve"][clock]   # drawn on the same x-axis code
        records.append(rec)
    return records


def route_frame(df: pd.DataFrame, measure: str, subject: str, completed_only: bool = False,
                transform: str = PRIMARY_TRANSFORM) -> pd.DataFrame:
    """The route-scale model's rows for one fit: one per attempt on a route (noise dropped).

    ``y`` is the transformed distance to the attempt's own route prototype, ``use`` the nth
    use of that route at the scene (``cluster_use``). Shared by ``run_practice_lme4`` and
    the route-scale diagnostics and bootstrap in ``docs/``, so all three fit the same rows.
    """
    d = _mixed_frame(df, measure, subject, completed_only, transform)
    d = d[d["cluster"] >= 0]
    return pd.DataFrame({
        "measure": measure, "subject": subject, "transform": transform,
        "completed_only": completed_only, "y": d["y"].values,
        "use": d["cluster_use"].values, "log_use": np.log(d["cluster_use"].values),
        "scene": d["scene"].values, "route": d["cluster"].values,
        "level": d["scene"].map(level_of).values,
        "n_zero_dropped": d.attrs["n_zero_dropped"]})


FIT_CHECK_MODELS = ("nested", "route")    # lme4.json's main model, lme4_route.json's
FIT_CHECK_SINGLE = 5                      # attempts 1..5 get a bin each; log-spaced after
FIT_CHECK_BINS = 12


def fit_check_data(df: pd.DataFrame, model: str, measure: str, subject: str,
                   transform: str = PRIMARY_TRANSFORM, completed_only: bool = False, *,
                   effects: pd.DataFrame | None = None, n_points: int = 1500, seed: int = 0) -> dict:
    """The data an lme4 fit was made on, summarised for drawing under its curve.

    Rows come from the frame the fit itself used — ``_mixed_frame`` for the scene-scale
    model (``model="nested"``, x = attempt), ``route_frame`` for the route scale
    (``model="route"``, x = nth use of the route) — so ``n_obs`` matches the fit's.
    Bins are the mean of the *transformed* outcome, back-transformed: under ``log`` a
    geometric mean, the same quantity the fixed-effect curve is, so the two compare like
    with like (an arithmetic mean of ``d`` would sit above the curve from skew alone).
    Attempts 1..``FIT_CHECK_SINGLE`` get a bin each, then log-spaced bins up to the
    95th-percentile x, where the stored curve stops; bins under 2 rows are skipped.

    ``bins`` pool the raw outcome, and mislead for the route model: the fixed slope is a
    *within*-unit slope, and routes reused 20+ times are already ~30% tighter on their
    first use, so pooled bins fall far below the line even when the model is right.
    ``partial_bins`` are the check that matches the model: the same bins over the
    ggeffects partial residuals (fixed prediction + conditional residual, i.e. each
    attempt with its level's, scene's and route's own intercept and slope removed), from
    ``effects`` — this fit's rows of ``load_effects()[0]``, in frame order (checked).
    Without ``effects`` there are no ``partial_bins``. ``points`` is a seeded thinning,
    ``[x, d, partial d]``, for drawing only. Error bars treat attempts as independent,
    which they are not within a scene, so they are narrower than an honest CI.
    """
    if model == "route":
        d = route_frame(df, measure, subject, completed_only, transform)
        x = d["use"].to_numpy(float)
    else:
        d = _mixed_frame(df, measure, subject, completed_only, transform)
        x = d["attempt"].to_numpy(float)
    y = d["y"].to_numpy(float)
    scene = d["scene"].to_numpy()
    inverse = TRANSFORMS[transform][1]
    out = {"n_obs": int(len(d)), "bins": [], "points": []}
    if not len(d):
        return out
    partial = None
    if effects is not None:
        if len(effects) != len(d) or not np.allclose(effects["y"].to_numpy(float), y):
            raise ValueError(f"effects rows do not line up with the {model} frame for sub-{subject}; "
                             "rerun invoke run-practice-lme4-effects --force")
        partial = effects["partial"].to_numpy(float)
        out["partial_bins"] = []
    top = float(np.quantile(x, 0.95))
    edges = np.arange(1, FIT_CHECK_SINGLE + 2, dtype=float) - 0.5
    if top > edges[-1]:
        edges = np.unique(np.concatenate(
            [edges, np.geomspace(edges[-1], top + 0.5, FIT_CHECK_BINS - FIT_CHECK_SINGLE + 1)]))
    for key, v in (("bins", y), ("partial_bins", partial)):
        if v is None:
            continue
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi)
            n = int(m.sum())
            if n < 2:
                continue
            mu, se = v[m].mean(), v[m].std(ddof=1) / np.sqrt(n)
            out[key].append({"x": float(np.exp(np.log(x[m]).mean())), "mean": float(inverse(mu)),
                             "lo": float(inverse(mu - 1.96 * se)), "hi": float(inverse(mu + 1.96 * se)),
                             "n_obs": n, "n_scenes": int(len(np.unique(scene[m])))})
    keep = np.random.default_rng(seed).permutation(len(d))[:n_points]
    pd_ = inverse(partial[keep]) if partial is not None else [None] * len(keep)
    out["points"] = [[float(a), float(b), None if c is None else float(c)]
                     for a, b, c in zip(x[keep], inverse(y[keep]), pd_)]
    return out


EFFECTS_SCRIPT = Path(__file__).with_name("practice_lme4_effects.R")
EFFECTS_DESIGN = "v1"
EFFECTS_TOLERANCE = 1e-4     # refit fixed effects must match the stored fit to this


def effects_dir(output_dir) -> Path:
    return practice_dir(output_dir) / "lme4_effects"


def fit_id(model: str, measure: str, transform: str, completed_only: bool, subject: str) -> str:
    """One lme4 fit's key, shared by the effects files and the dashboard payload."""
    return f"{model}|{measure}|{transform}|{int(bool(completed_only))}|{subject}"


def _stored_lme4_fits(output_dir):
    """``(model, fit record)`` for every fit in lme4.json (nested) and lme4_route.json."""
    pdir = practice_dir(output_dir)
    out = []
    for model, name in (("nested", "lme4.json"), ("route", "lme4_route.json")):
        if (pdir / name).exists():
            out += [(model, f) for f in json.loads((pdir / name).read_text())["fits"]
                    if f["model"] == model]
    return out


def run_practice_lme4_effects(output_dir, *, smoke=False, force=False, n_cores=None) -> None:
    """Partial residuals and per-unit random effects of every stored lme4 fit.

    Refits each fit in ``lme4.json`` (nested model) and ``lme4_route.json`` in R
    (``analysis/practice_lme4_effects.R``: same frames, formula and optimiser, no CIs) and
    writes ``practice/lme4_effects/``: ``partial.csv.gz`` (per fitted row: x, y, and the
    ggeffects partial residual — the fixed prediction plus the conditional residual, so the
    outcome with its level's, scene's and route's own intercept and slope taken out),
    ``ranef.csv.gz`` (every level's, scene's and route's BLUP intercept ``u`` and slope
    ``v``) and ``fixef.json``. Each refit's fixed effects are checked against the stored fit
    (``EFFECTS_TOLERANCE``) and the run stops on a mismatch, so these are the stored models'
    effects, not a near relative's. ``smoke`` refits the first subject's primary fits only,
    into a separate ``smoke/`` folder, never replacing a full run. Needs R with lme4 and
    ggeffects; without them it says so and returns.
    """
    import os
    import shutil
    import subprocess
    import tempfile

    stored = _stored_lme4_fits(output_dir)
    if not stored:
        print("No lme4 fits on disk — run invoke run-practice-lme4 first.")
        return
    if shutil.which("Rscript") is None or subprocess.run(
            ["Rscript", "-e", "quit(status = !requireNamespace('ggeffects', quietly = TRUE))"],
            capture_output=True).returncode != 0:
        print("Rscript with ggeffects not found — skipping the lme4 effects. Install it "
              "(sudo apt install r-cran-ggeffects, or install.packages('ggeffects') in R) and "
              "rerun invoke run-practice-lme4-effects.")
        return
    edir = effects_dir(output_dir) / ("smoke" if smoke else "")
    if smoke:
        first = sorted({f["subject"] for _, f in stored})[0]
        stored = [(m, f) for m, f in stored if f["subject"] == first and f["transform"] == PRIMARY_TRANSFORM
                  and not f["completed_only"] and f["measure"] in (PRIMARY, "d_own_loso")]
    wanted = sorted(fit_id(m, f["measure"], f["transform"], f["completed_only"], f["subject"])
                    for m, f in stored)
    meta_path = edir / "fixef.json"
    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        if meta.get("design") == EFFECTS_DESIGN and sorted(meta["fits"]) == wanted:
            print(f"lme4 effects already computed for these {len(wanted)} fits, skipping "
                  "(use --force to refit).")
            return

    df = load_attempts(output_dir)
    frames = []
    for model, f in stored:
        args = (df, f["measure"], f["subject"], f["completed_only"], f["transform"])
        if model == "route":
            d = route_frame(*args)
            x, route = d["log_use"].values, d["route"].values
        else:
            d = _mixed_frame(*args)
            x, route = d["logt"].values, -1
        frames.append(pd.DataFrame({
            "fid": fit_id(model, f["measure"], f["transform"], f["completed_only"], f["subject"]),
            "model": model, "subject": f["subject"], "y": d["y"].values, "x": x,
            "scene": d["scene"].values, "level": d["scene"].map(level_of).values, "route": route}))
    frames = pd.concat(frames, ignore_index=True)
    frames["rid"] = np.arange(len(frames))
    n_cores = n_cores or max(1, (os.cpu_count() or 2) - 1)
    print(f"Refitting {len(wanted)} lme4 fits for partial residuals and random effects "
          f"on {n_cores} cores...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frames.to_csv(tmp / "frames.csv", index=False)
        out = [tmp / n for n in ("partial.csv", "ranef.csv", "fixef.csv")]
        subprocess.run(["Rscript", str(EFFECTS_SCRIPT), str(tmp / "frames.csv"), *map(str, out),
                        str(n_cores)], check=True)
        partial, ranef, fixef = (pd.read_csv(p, keep_default_na=False, na_values=["NA"],
                                             dtype={"fid": str, "id": str}) for p in out)

    by_id = {fit_id(m, f["measure"], f["transform"], f["completed_only"], f["subject"]): f
             for m, f in stored}
    fixef["stored_b"] = fixef["fid"].map(lambda k: by_id[k]["b"])
    fixef["stored_intercept"] = fixef["fid"].map(lambda k: by_id[k]["intercept"])
    fixef["stored_n_obs"] = fixef["fid"].map(lambda k: by_id[k]["n_obs"])
    gap = np.maximum(abs(fixef["b"] - fixef["stored_b"]), abs(fixef["intercept"] - fixef["stored_intercept"]))
    bad = fixef[(gap > EFFECTS_TOLERANCE) | (fixef["n_obs"] != fixef["stored_n_obs"])]
    if len(bad):
        raise RuntimeError("lme4 refits do not reproduce the stored fits — rerun "
                           "invoke run-practice-lme4 --force first:\n" + bad.to_string())

    rows = frames.drop(columns=["model", "level"]).merge(partial.drop(columns="fid"), on="rid")
    edir.mkdir(parents=True, exist_ok=True)
    rows.drop(columns="rid").to_csv(edir / "partial.csv.gz", index=False)
    ranef.to_csv(edir / "ranef.csv.gz", index=False)
    meta_path.write_text(json.dumps({
        "design": EFFECTS_DESIGN, "fits": wanted,
        "max_fixef_gap": float(gap.max()), "max_ggeffects_gap": float(fixef["max_gap"].max()),
        "fixef": {r.fid: {"intercept": r.intercept, "b": r.b, "singular": bool(r.singular)}
                  for r in fixef.itertuples()}}, indent=2))
    print(f"Wrote {edir}: {len(rows)} partial residuals, {len(ranef)} random effects; refits match "
          f"the stored fits to {gap.max():.1e}, ggeffects matches lme4 to {fixef['max_gap'].max():.1e}.")


def load_effects(output_dir, smoke=False):
    """``(partial, ranef, fixef)`` from ``run_practice_lme4_effects``, or ``None`` if not run.

    ``partial`` has one row per fitted attempt (``fid``, subject, scene, route, ``x`` on
    the model's log scale, ``y`` and ``partial`` on the transform's scale); ``ranef`` one
    row per level / scene / route and fit (``kind``, ``id`` as lme4 names it — ``w1l1``,
    ``w1l1:w1l1s0``, ``w1l1:w1l1s0:3`` — ``u`` intercept, ``v`` slope).
    """
    edir = effects_dir(output_dir) / ("smoke" if smoke else "")
    if not (edir / "fixef.json").exists():
        return None
    partial = pd.read_csv(edir / "partial.csv.gz", dtype={"subject": str, "fid": str, "scene": str})
    ranef = pd.read_csv(edir / "ranef.csv.gz", dtype={"fid": str, "id": str})
    return partial, ranef, json.loads((edir / "fixef.json").read_text())["fixef"]


def unit_lines(ranef: pd.DataFrame, fixef: dict, fid: str) -> pd.DataFrame:
    """Every scene's (and, for the route model, every route's) own fitted line for one fit.

    A unit's line is ``(b0 + u_level + u_scene [+ u_route]) + (b1 + v_level + v_scene
    [+ v_route]) * log(x)`` on the transform's scale. Columns: ``scene``, ``route`` (-1 for
    the scene model), ``a`` intercept, ``b`` slope.
    """
    r = ranef[ranef["fid"] == fid]
    fe = fixef[fid]
    lv = r[r["kind"] == "level"].set_index("id")[["u", "v"]]
    sc = r[r["kind"] == "scene"].set_index("id")[["u", "v"]]
    out = pd.DataFrame({"key": sc.index, "scene": [k.split(":")[1] for k in sc.index]})
    out["level"] = [k.split(":")[0] for k in sc.index]
    out["a"] = fe["intercept"] + sc["u"].values + lv.loc[out["level"], "u"].values
    out["b"] = fe["b"] + sc["v"].values + lv.loc[out["level"], "v"].values
    routes = r[r["kind"] == "route"]
    if not len(routes):
        out["route"] = -1
        return out[["scene", "route", "a", "b"]]
    parts = routes["id"].str.split(":", expand=True)
    rt = pd.DataFrame({"scene_key": parts[0] + ":" + parts[1], "scene": parts[1],
                       "route": parts[2].astype(int), "u": routes["u"].values, "v": routes["v"].values})
    base = out.set_index("key")
    rt["a"] = base.loc[rt["scene_key"], "a"].values + rt["u"]
    rt["b"] = base.loc[rt["scene_key"], "b"].values + rt["v"]
    return rt[["scene", "route", "a", "b"]].reset_index(drop=True)


SPAGHETTI_TOP = 6       # scenes per subject in the notebook figure: the most-played ones


def spaghetti_data(partial: pd.DataFrame, ranef: pd.DataFrame, fixef: dict, fid: str,
                   scenes=None, top: int = SPAGHETTI_TOP) -> dict:
    """Per scene, the attempts and each unit's own fitted line, for one lme4 fit.

    ``scenes`` defaults to the ``top`` scenes with the most attempts in this fit. Each
    scene holds ``points`` (``[x, d, route]``, every attempt, x as attempt or nth use)
    and ``lines``: one per scene for the scene-scale model, one per route for the route
    model (``route``, intercept ``a`` and slope ``b`` on the transform's scale, drawn from
    x = 1 to that unit's last attempt ``x1``), plus ``fixed`` — the average-scene line,
    for reference. The lines are the fixed effects plus that unit's BLUPs (``unit_lines``).
    """
    transform = fid.split("|")[2]
    inverse = TRANSFORMS[transform][1]
    rows = partial[partial["fid"] == fid]
    if scenes is None:
        scenes = rows["scene"].value_counts().index[:top].tolist()
    lines = unit_lines(ranef, fixef, fid)
    fe = fixef[fid]
    out = {"transform": transform, "fixed": {"a": fe["intercept"], "b": fe["b"]}, "scenes": {}}
    for sc in scenes:
        r = rows[rows["scene"] == sc]
        if not len(r):
            continue
        x = np.rint(np.exp(r["x"].to_numpy(float))).astype(int)
        route = r["route"].to_numpy(int)
        last = pd.Series(x).groupby(route).max()
        ls = lines[lines["scene"] == sc]
        out["scenes"][sc] = {
            "n": int(len(r)),
            "points": [[int(a), float(b), int(c)] for a, b, c in zip(x, inverse(r["y"].to_numpy(float)), route)],
            "lines": [{"route": int(l.route), "a": float(l.a), "b": float(l.b),
                       "x1": int(last.get(l.route, x.max()) if l.route >= 0 else x.max())}
                      for l in ls.itertuples() if l.route < 0 or l.route in last.index]}
    return out


def run_practice_lme4(output_dir, *, scenes=None, force=False, n_cores=None) -> None:
    """Fit the two main lme4 practice models: the scene-scale and the route-scale model.

    **Scene scale** (``lme4.json``): exports the exact model frames the statsmodels fits use
    (``_mixed_frame``) for ``LME4_MEASURES`` x ``TRANSFORMS`` x all/completed attempts x
    subject, runs ``analysis/practice_lme4.R`` on them (parallel over fits), and stores, per
    fit, both the nested model and its scene-only twin: intercept, slope, Wald and profile
    CIs, the CI used (``ci``: profile, Wald where the profile failed), Wald p, variances at
    every level, AIC, singularity, the level likelihood-ratio test and the fitted curve.

    **Route scale** (``lme4_route.json``): for ``ROUTE_MEASURES`` (``d_own``,
    distance in px to the attempt's OWN route prototype, so the origin never changes along
    a series), practice counted as the nth use of that route (``cluster_use``), routes
    nested in scenes nested in levels (``analysis/practice_lme4_route.R``); Wald CIs.

    Each file is checked separately and skipped when it already covers these scenes, so
    adding one does not refit the other; a subset run (``scenes``) never replaces a full
    fit. Then rewrites ``mixed_summary.csv`` (its ``primary`` rows are the nested model's).
    Needs R with lme4; without ``Rscript`` it says so and returns, so the rest of the
    pipeline still runs.
    """
    import os
    import shutil

    output_dir = Path(output_dir)
    pdir = practice_dir(output_dir)
    out_path = pdir / "lme4.json"
    route_path = pdir / "lme4_route.json"
    if shutil.which("Rscript") is None:
        print("Rscript not found — skipping the lme4 models (the two main practice models). "
              "Install R and lme4 (e.g. sudo apt install r-base r-cran-lme4) and rerun "
              "invoke run-practice-lme4.")
        return
    df = load_attempts(output_dir, scenes)
    used = sorted(df["scene"].unique())

    def up_to_date(path, design):
        if not path.exists() or force:
            return False
        if scenes is not None:
            return True            # a subset run never replaces a fit on disk
        prev = json.loads(path.read_text())
        return prev.get("scenes") == used and prev.get("design") == design

    main_ok, route_ok = up_to_date(out_path, LME4_DESIGN), up_to_date(route_path, ROUTE_DESIGN)
    if main_ok and route_ok:
        print("lme4 models already fitted on these scenes, skipping (use --force to refit).")
        return
    subjects = sorted(df["subject"].unique())
    n_cores = n_cores or max(1, (os.cpu_count() or 2) - 1)
    combos = [(t, c, s) for t in TRANSFORMS for c in (False, True) for s in subjects]

    if not main_ok:
        frames = []
        for measure in LME4_MEASURES:
            for transform, completed_only, s in combos:
                d = _mixed_frame(df, measure, s, completed_only, transform)
                frames.append(pd.DataFrame({
                    "measure": measure, "subject": s, "transform": transform,
                    "completed_only": completed_only, "y": d["y"].values,
                    "logt": d["logt"].values, "attempt": d["attempt"].values,
                    "scene": d["group"].values, "level": d["group"].map(level_of).values,
                    "n_zero_dropped": d.attrs["n_zero_dropped"]}))
        frames = pd.concat(frames, ignore_index=True)
        zeros = (frames.groupby(["measure", "subject", "transform", "completed_only"])
                       ["n_zero_dropped"].first())
        print(f"Fitting the primary lme4 model: {zeros.size} fits x 2 structures on {n_cores} cores...")
        fits, curves = _run_lme4_script(LME4_SCRIPT, frames.drop(columns="n_zero_dropped"), n_cores)
        out_path.write_text(json.dumps({
            "design": LME4_DESIGN, "primary": PRIMARY, "primary_transform": PRIMARY_TRANSFORM,
            "formula": "y ~ logt + (1 + logt | level) + (1 + logt | level:scene)",
            "scenes": used, "subjects": subjects, "measures": list(LME4_MEASURES),
            "transforms": list(TRANSFORMS),
            "fits": _lme4_records(fits, curves, zeros, "attempt")}, indent=2))
        print(f"Wrote {out_path}")

    if not route_ok:
        frames = pd.concat([route_frame(df, measure, s, completed_only, transform)
                            for measure in ROUTE_MEASURES
                            for transform, completed_only, s in combos], ignore_index=True)
        zeros = (frames.groupby(["measure", "subject", "transform", "completed_only"])
                       ["n_zero_dropped"].first())
        print(f"Fitting the route-scale lme4 model: {zeros.size} fits on {n_cores} cores...")
        fits, curves = _run_lme4_script(ROUTE_SCRIPT, frames.drop(columns="n_zero_dropped"), n_cores)
        route_path.write_text(json.dumps({
            "design": ROUTE_DESIGN, "primary": "d_own_loso", "primary_transform": PRIMARY_TRANSFORM,
            "clock": "nth use of the route (cluster_use)",
            "formula": ("y ~ log_use + (1 + log_use | level) + (1 + log_use | level:scene)"
                        " + (1 + log_use | level:scene:route)"),
            "scenes": used, "subjects": subjects, "measures": list(ROUTE_MEASURES),
            "transforms": list(TRANSFORMS),
            "fits": _lme4_records(fits, curves, zeros, "use")}, indent=2))
        print(f"Wrote {route_path}")

    if (pdir / "mixed.json").exists():
        write_mixed_summary(pdir)
        print(f"Rewrote {pdir}/mixed_summary.csv with the lme4 rows.")


# Checks on the route-scale effect (analysis/practice_route_checks.R): the route plot's data
# under three corrections, each linear on log-log axes and with a quadratic in log(use).
ROUTE_CHECKS_SCRIPT = Path(__file__).with_name("practice_route_checks.R")
ROUTE_CHECKS_DESIGN = "v3"   # v2: adds route_int; v3: whole-polynomial change and joint test
ROUTE_CHECK_CORRECTIONS = ("pooled", "scene", "route_int", "route")
ROUTE_CHECK_FAMILY = ["measure", "transform", "completed_only", "correction", "shape"]
ROUTE_CHECK_TOLERANCE = 1e-4   # route/linear must reproduce lme4_route.json's slope


def run_practice_route_checks(output_dir, *, scenes=None, force=False, n_cores=None) -> None:
    """Is the route-scale decrease a scene effect, a route effect, and is it curved?

    Fits ``route_frame()``'s rows (the route-scale model's own data: distance in px to the
    attempt's own route prototype against the nth use of that route) per subject x
    ``ROUTE_MEASURES`` x ``TRANSFORMS`` x all/completed attempts, at three corrections:
    ``pooled`` (plain least squares, no random effects: the line through the pooled plot),
    ``scene`` (level and scene random intercepts and slopes, no route effect), ``route_int``
    (plus a random intercept per route: routes differ only in how tight they are) and
    ``route`` (plus a random slope per route too: ``lme4_route.json``'s model). Each is fitted linear in
    ``log(use)`` and with a ``log(use)^2`` term, compared by AIC and a 1-df likelihood-ratio
    test. Writes ``practice/route_checks.json``: every fit's coefficients, Wald CIs, the
    local slope at uses 1/5/20/50, the whole polynomial's change from use 1 to the end of
    the range (``change_end`` +/- ``change_end_se``, at ``u_end``) and a joint Wald test of
    all its practice terms (``joint_p``), its curve, and Benjamini-Hochberg p across the five
    subjects (``p_bh`` for the slope, ``lrt_p_bh`` for the curvature, ``joint_p_bh``), per
    ``ROUTE_CHECK_FAMILY``. Stops if the ``route`` linear fit does not reproduce
    ``lme4_route.json`` to ``ROUTE_CHECK_TOLERANCE``. Skips when already fitted on these
    scenes; a subset run (``scenes``) writes ``route_checks_smoke.json`` instead and never
    replaces the full file. Needs R with lme4; without ``Rscript`` it says so and returns.
    """
    import os
    import shutil

    from statsmodels.stats.multitest import multipletests

    output_dir = Path(output_dir)
    pdir = practice_dir(output_dir)
    out_path = pdir / ("route_checks.json" if scenes is None else "route_checks_smoke.json")
    if shutil.which("Rscript") is None:
        print("Rscript not found — skipping the route checks. Install R and lme4 "
              "(e.g. sudo apt install r-base r-cran-lme4) and rerun invoke run-practice-route-checks.")
        return
    df = load_attempts(output_dir, scenes)
    used = sorted(df["scene"].unique())
    if out_path.exists() and not force:
        prev = json.loads(out_path.read_text())
        if prev.get("scenes") == used and prev.get("design") == ROUTE_CHECKS_DESIGN:
            print("Route checks already fitted on these scenes, skipping (use --force to refit).")
            return
    subjects = sorted(df["subject"].unique())
    n_cores = n_cores or max(1, (os.cpu_count() or 2) - 1)
    frames = pd.concat([route_frame(df, measure, s, completed_only, transform)
                        for measure in ROUTE_MEASURES for transform in TRANSFORMS
                        for completed_only in (False, True) for s in subjects], ignore_index=True)
    n_groups = frames.groupby(["measure", "subject", "transform", "completed_only"]).ngroups
    print(f"Fitting the route checks: {n_groups} fits x {2 * len(ROUTE_CHECK_CORRECTIONS)} models "
          f"on {n_cores} cores...")
    fits, curves = _run_lme4_script(ROUTE_CHECKS_SCRIPT, frames.drop(columns="n_zero_dropped"),
                                    n_cores)

    fits["p_bh"] = np.nan
    fits["lrt_p_bh"] = np.nan
    fits["joint_p_bh"] = np.nan
    for _, g in fits.groupby(ROUTE_CHECK_FAMILY):
        fits.loc[g.index, "p_bh"] = multipletests(g["p_wald"].to_numpy(), method="fdr_bh")[1]
        fits.loc[g.index, "joint_p_bh"] = multipletests(g["joint_p"].to_numpy(), method="fdr_bh")[1]
        if g["lrt_p"].notna().all():
            fits.loc[g.index, "lrt_p_bh"] = multipletests(g["lrt_p"].to_numpy(), method="fdr_bh")[1]

    route_path = pdir / "lme4_route.json"
    max_gap = None
    if scenes is None and route_path.exists():
        stored = {(f["measure"], f["subject"], f["transform"], f["completed_only"]): f["b"]
                  for f in json.loads(route_path.read_text())["fits"]}
        mine = fits[(fits["correction"] == "route") & (fits["shape"] == "linear")]
        gaps = [abs(r.b - stored[(r.measure, r.subject, r.transform, bool(r.completed_only))])
                for r in mine.itertuples()
                if (r.measure, r.subject, r.transform, bool(r.completed_only)) in stored]
        max_gap = max(gaps) if gaps else None
        if max_gap is not None and max_gap > ROUTE_CHECK_TOLERANCE:
            raise RuntimeError(f"route/linear differs from lme4_route.json by {max_gap:.1e} — "
                               "rerun invoke run-practice-lme4 --force first")

    keys = ["measure", "subject", "transform", "completed_only", "correction", "shape"]
    records = []
    for _, r in fits.iterrows():
        rec = {k: _nan_to_none(v.item() if hasattr(v, "item") else v) for k, v in r.items()}
        rec["warnings"] = rec.get("warnings") or ""
        c = curves[(curves[keys] == pd.Series({k: rec[k] for k in keys})).all(axis=1)]
        rec["curve"] = {"use": c["use"].astype(int).tolist(), "y": c["y"].tolist()}
        records.append(rec)
    out_path.write_text(json.dumps({
        "design": ROUTE_CHECKS_DESIGN, "primary": "d_own_loso", "primary_transform": PRIMARY_TRANSFORM,
        "clock": "nth use of the route (cluster_use)",
        "formulas": {
            "pooled": "y ~ log_use (least squares, no random effects)",
            "scene": "y ~ log_use + (1 + log_use | level) + (1 + log_use | level:scene)",
            "route_int": ("y ~ log_use + (1 + log_use | level) + (1 + log_use | level:scene)"
                          " + (1 | level:scene:route)"),
            "route": ("y ~ log_use + (1 + log_use | level) + (1 + log_use | level:scene)"
                      " + (1 + log_use | level:scene:route)"),
            "quadratic": "the same, + log_use^2 (fixed effect only)"},
        "max_gap_vs_lme4_route": max_gap,
        "scenes": used, "subjects": subjects, "measures": list(ROUTE_MEASURES),
        "transforms": list(TRANSFORMS), "fits": records}, indent=2))
    print(f"Wrote {out_path}" + (f" (route/linear matches lme4_route.json to {max_gap:.1e})"
                                 if max_gap is not None else ""))


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
