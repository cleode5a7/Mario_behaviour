"""Route usage: does route choice concentrate on fewer routes across the stages of play?

The practice-effect analysis (``analysis/practice.py``) asks whether attempts get closer
to a route's prototype. This module asks the other half of the question: *which* routes
get used — does a player settle on a few routes, or keep spreading attempts over many?

**Unit: one cell = subject × scene × stage.** Each phase is cut in half by attempt
order within ``(subject, scene, phase)``, giving four stages (``STAGES``): early/late
discovery and early/late practice. An odd count puts the extra attempt in the early
half. Within one scene this is a genuine repetition axis, for the same reason the stage
tertiles are (``analysis/stages.py``): a scene sits inside one level, so the level mix
cannot change between stages. Routes are the scene's consensus cluster labels; ``-1``
(noise, single-fit runs only) is not a route and is dropped.

**Main measure: the Simpson index, estimated without bias.** ``D`` is the probability
that two *different* attempts in a cell used the same route, ``sum p_i^2``; it is high
when route use is concentrated and low when it is spread. ``1 / D`` is the effective
number of routes and ``-log D`` the Rényi entropy of order 2. The estimator
``sum n_i (n_i - 1) / (N (N - 1))`` counts matching pairs of different attempts, so it
is unbiased at any ``N >= 2``. That matters here because discovery cells hold a median of
4 attempts against 13 in practice: the plug-in ``sum (n_i / N)^2`` counts every attempt
paired with itself, which inflates ``D`` most where ``N`` is small and would make
discovery look concentrated for a player who never changed. Plain Shannon entropy has
the mirror problem (it creeps up with ``N``).

**Check measure: rarefied Shannon entropy.** The expected Shannon entropy (bits) of
``RAREFY_N`` attempts drawn without replacement from the cell, computed exactly
(``rarefied_shannon``) rather than by simulation. Every cell then answers "how varied
are 4 attempts from this stage?". Cells with fewer than ``RAREFY_N`` attempts cannot take
part. Its direction is the opposite of ``D``: higher means more varied.

**Model: one per subject, fitted in R's lme4** (``analysis/route_usage_lme4.R``):
``y ~ stage + (1 | level) + (1 | level:scene)``, with early discovery as the reference.
The scene intercept absorbs "this scene has 2 routes, that one 12", so stages are
compared within scenes; there are no random slopes, because each scene has at most four
cells per subject. Fitted weighted by the cell's attempt count (primary: bigger cells
are less noisy) and unweighted (the check that the weights do not drive the result).
``CONTRASTS`` are computed from the fixed effects and their covariance: late − early
discovery, late − early practice, and practice − discovery (the mean of the two practice
stages minus the mean of the two discovery stages). Wald CIs.

Two resampling checks, both stored next to each fit:

  * **Chance level** (``null``): shuffle each subject's route labels across stages
    within a scene (cell sizes kept) and refit, ``null_reps`` times. Every contrast's
    null mean should sit at 0 — which is what shows the estimator carries no
    sample-size bias — and ``p_perm`` is the permutation p-value.
  * **Level bootstrap** (``boot``): resample whole levels with replacement and refit,
    the same independence check the practice models have, since a level's scenes are
    played back to back.

One model per subject, never pooled, as in ``analysis/practice.py``: five subjects are
too few to estimate a between-subject variance. Reads only
``output_data/practice/attempts/``, so it needs ``run-practice-distances`` and never
opens ``mario.scenes``.
"""

from __future__ import annotations

import json
import os
import shutil
from functools import lru_cache
from math import comb, log2
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.practice import level_of, load_attempts

STAGES = ["early_discovery", "late_discovery", "early_practice", "late_practice"]
MEASURES = ("simpson", "shannon_rarefied")
PRIMARY = {"measure": "simpson", "weighting": "n"}
WEIGHTINGS = ("n", "none")
CONTRASTS = {
    "late_vs_early_discovery": "late discovery − early discovery",
    "late_vs_early_practice": "late practice − early practice",
    "practice_vs_discovery": "mean of practice stages − mean of discovery stages",
}
RAREFY_N = 4
MIN_CELL = 2          # the unbiased Simpson estimator needs two different attempts
DESIGN = "v1"
SCRIPT = Path(__file__).with_name("route_usage_lme4.R")


def route_usage_dir(output_dir: Path) -> Path:
    return Path(output_dir) / "route_usage"


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

def simpson_unbiased(counts) -> float:
    """``sum n_i (n_i - 1) / (N (N - 1))``: chance two different attempts share a route.

    NaN below two attempts, where no pair exists.
    """
    c = np.asarray(counts, dtype=float)
    n = c.sum()
    return float((c * (c - 1)).sum() / (n * (n - 1))) if n >= 2 else float("nan")


def _entropy_bits(pattern) -> float:
    m = sum(pattern)
    return -sum(k / m * log2(k / m) for k in pattern)


@lru_cache(maxsize=None)
def _rarefied(counts: tuple, m: int) -> float:
    # Multivariate hypergeometric over the routes: the entropy of a draw depends only on
    # the multiset of its non-zero counts, so accumulate probability mass per pattern.
    states = {(m, ()): 1}
    for n_i in counts:
        nxt = {}
        for (rem, pattern), ways in states.items():
            for k in range(0, min(n_i, rem) + 1):
                key = (rem - k, tuple(sorted(pattern + (k,))) if k else pattern)
                nxt[key] = nxt.get(key, 0) + ways * comb(n_i, k)
        states = nxt
    total = comb(sum(counts), m)
    return sum(ways * _entropy_bits(pattern)
               for (rem, pattern), ways in states.items() if rem == 0) / total


def rarefied_shannon(counts, m: int = RAREFY_N) -> float:
    """Expected Shannon entropy (bits) of ``m`` attempts drawn without replacement.

    Exact, not simulated. NaN when the cell holds fewer than ``m`` attempts.
    """
    c = tuple(sorted(int(x) for x in counts if x > 0))
    return _rarefied(c, m) if sum(c) >= m else float("nan")


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def assign_stages(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stage``-half labels (``STAGES``) by attempt order within subject × scene × phase."""
    df = df[df["phase"].isin(["discovery", "practice"]) & (df["cluster"] >= 0)].copy()
    df = df.sort_values(["subject", "scene", "attempt"], kind="stable")
    keys = ["subject", "scene", "phase"]
    rank = df.groupby(keys).cumcount()
    size = df.groupby(keys)["attempt"].transform("size")
    half = np.where(rank < np.ceil(size / 2), "early", "late")
    df["stage4"] = pd.Series(half, index=df.index) + "_" + df["phase"]
    return df.reset_index(drop=True)


def cell_table(df: pd.DataFrame, route_col: str = "cluster") -> pd.DataFrame:
    """One row per subject × scene × stage with ``n``, routes used and both measures."""
    counts = (df.groupby(["subject", "scene", "stage4", route_col]).size()
                .rename("c").reset_index())
    rows = []
    for (s, scene, stage), g in counts.groupby(["subject", "scene", "stage4"], sort=False):
        c = g["c"].to_numpy()
        rows.append((s, scene, stage, int(c.sum()), len(c),
                     simpson_unbiased(c), rarefied_shannon(c)))
    cells = pd.DataFrame(rows, columns=["subject", "scene", "stage", "n", "routes_used",
                                        "simpson", "shannon_rarefied"])
    cells = cells[cells["n"] >= MIN_CELL]
    cells["level"] = cells["scene"].map(level_of)
    cells["stage"] = pd.Categorical(cells["stage"], STAGES)
    return cells.sort_values(["subject", "scene", "stage"]).reset_index(drop=True)


def shuffled_cells(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Cells after permuting each subject's route labels within a scene (sizes kept)."""
    d = df[["subject", "scene", "stage4", "cluster"]].copy()
    d["cluster"] = d.groupby(["subject", "scene"])["cluster"].transform(
        lambda x: rng.permutation(x.to_numpy()))
    return cell_table(d)


# ---------------------------------------------------------------------------
# Models (R)
# ---------------------------------------------------------------------------

def _run_script(frames: pd.DataFrame, subjects, n_boot: int, seed: int, n_cores: int) -> pd.DataFrame:
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frames.to_csv(tmp / "frames.csv", index=False)
        subprocess.run(["Rscript", str(SCRIPT), str(tmp / "frames.csv"), str(tmp / "out.csv"),
                        str(n_cores), str(n_boot), str(seed), ",".join(subjects)], check=True)
        # Only R's NA is missing: pandas would otherwise read kind == "null" as NaN.
        return pd.read_csv(tmp / "out.csv", dtype={"subject": str},
                           keep_default_na=False, na_values=["NA"])


def _summarise(out: pd.DataFrame, cells: pd.DataFrame) -> list[dict]:
    """Long R output -> one record per subject × measure × weighting."""
    fits = []
    obs = out[out["kind"] == "fit"]
    for (s, measure, weighting), g in obs.groupby(["subject", "measure", "weighting"]):
        sel = (out["subject"] == s) & (out["measure"] == measure) & (out["weighting"] == weighting)
        null, boot = out[sel & (out["kind"] == "null")], out[sel & (out["kind"] == "boot")]
        c = cells[(cells["subject"] == s) & cells[measure].notna()]
        meta = g.iloc[0]
        rec = {"subject": s, "measure": measure, "weighting": weighting,
               "n_cells": int(len(c)), "n_attempts": int(c["n"].sum()),
               "n_scenes": int(c["scene"].nunique()), "n_levels": int(c["level"].nunique()),
               "cells_per_stage": {st: int((c["stage"] == st).sum()) for st in STAGES},
               "singular": bool(meta["singular"]) if pd.notna(meta["singular"]) else None,
               "warnings": meta["warnings"] if isinstance(meta["warnings"], str) else "",
               "stage_means": {}, "contrasts": {}}
        for _, r in g.iterrows():
            est, se = float(r["est"]), float(r["se"])
            block = {"est": est, "se": se, "ci": [est - 1.96 * se, est + 1.96 * se]}
            if r["quantity"] in STAGES:
                if measure == "simpson":
                    # 1/D is monotone, so the CI's ends convert directly (while D > 0).
                    lo, hi = block["ci"]
                    block["effective_routes"] = 1 / est if est > 0 else None
                    block["effective_routes_ci"] = [1 / hi if hi > 0 else None,
                                                    1 / lo if lo > 0 else None]
                rec["stage_means"][r["quantity"]] = block
                continue
            block["p"] = float(r["p"])
            nv = null.loc[null["quantity"] == r["quantity"], "est"].to_numpy()
            if len(nv):
                block["null"] = {"reps": int(len(nv)), "mean": float(nv.mean()), "sd": float(nv.std()),
                                 "p_perm": float((1 + (np.abs(nv) >= abs(est)).sum()) / (1 + len(nv)))}
            bv = boot.loc[boot["quantity"] == r["quantity"], "est"].dropna().to_numpy()
            if len(bv):
                lo, hi = np.percentile(bv, [2.5, 97.5])
                block["boot"] = {"reps": int(len(bv)), "ci": [float(lo), float(hi)],
                                 "width_ratio": float((hi - lo) / (3.92 * se))}
            rec["contrasts"][r["quantity"]] = block
        fits.append(rec)
    return fits


def _finite(obj):
    """NaN/inf -> None throughout, so a failed fit is ``null`` in valid JSON."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


def smoke_scenes(scenes, n_levels: int = 3, per_level: int = 2) -> list[str]:
    """A few scenes from each of the first levels: the model needs more than one level."""
    by_level: dict[str, list[str]] = {}
    for sc in sorted(scenes):
        by_level.setdefault(level_of(sc), []).append(sc)
    return [sc for lv in sorted(by_level)[:n_levels] for sc in by_level[lv][:per_level]]


def run_route_usage(output_dir, *, subjects=None, scenes=None, null_reps=200, bootstraps=200,
                    seed=0, force=False, n_cores=None) -> None:
    """Write ``route_usage/cells.csv`` and fit ``route_usage/models.json``.

    ``cells.csv`` is written alongside the fits, so it always matches ``models.json``
    (and is written alone when R is missing). The fits are skipped when
    ``models.json`` already holds every requested subject at the same design, scenes and
    resampling settings. ``subjects`` refits just those subjects and merges them into an
    existing compatible file; ``scenes`` (``--smoke``) restricts the data and never
    replaces a ``models.json`` on disk (as in the practice tasks), while a smoke file is
    replaced by the next full run because its scene list differs. Fits that fail are
    stored as ``null`` estimates with the reason in ``warnings``.
    Needs R with lme4; without ``Rscript`` it says so and returns.
    """
    output_dir = Path(output_dir)
    out_dir = route_usage_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "models.json"

    df = assign_stages(load_attempts(output_dir, scenes))
    cells = cell_table(df)

    def write_cells():
        cells.to_csv(out_dir / "cells.csv", index=False)
        print(f"Wrote {out_dir / 'cells.csv'}: {len(cells)} cells over "
              f"{cells['scene'].nunique()} scenes")

    if shutil.which("Rscript") is None:
        write_cells()
        print("Rscript not found — skipping the route-usage models. Install R and lme4 "
              "(e.g. sudo apt install r-base r-cran-lme4) and rerun invoke run-route-usage.")
        return

    all_subjects = sorted(cells["subject"].unique())
    wanted = sorted(subjects) if subjects else all_subjects
    missing = set(wanted) - set(all_subjects)
    if missing:
        raise ValueError(f"no attempts for subject(s) {sorted(missing)}; have {all_subjects}")
    settings = {"design": DESIGN, "scenes": sorted(cells["scene"].unique()),
                "null_reps": int(null_reps), "bootstraps": int(bootstraps), "seed": int(seed),
                "rarefy_n": RAREFY_N, "min_cell": MIN_CELL}
    previous = []
    if model_path.exists() and not force and scenes is not None:
        print("models.json exists; a scene-subset run (--smoke) never replaces it, skipping.")
        return
    if model_path.exists() and not force:
        prev = json.loads(model_path.read_text())
        if all(prev.get(k) == v for k, v in settings.items()):
            have = {f["subject"] for f in prev["fits"]}
            if set(wanted) <= have and not subjects:
                print("Route-usage models already fitted on these scenes, skipping "
                      "(use --force to refit).")
                return
            previous = [f for f in prev["fits"] if f["subject"] not in set(wanted)]
        elif subjects:
            changed = [k for k, v in settings.items() if prev.get(k) != v]
            print(f"models.json was fitted with different {', '.join(changed)}; merging "
                  "--subjects into it would mix settings. Refit every subject (omit "
                  "--subjects), or pass --force to replace it with just these subjects.")
            return

    write_cells()
    # Observed cells are rep 0; the chance-level reps reshuffle every subject from one
    # seeded generator, so a subject's null never depends on which others are fitted.
    rng = np.random.default_rng(seed)
    frames = [cells.assign(rep=0)]
    for rep in range(1, null_reps + 1):
        frames.append(shuffled_cells(df, rng).assign(rep=rep))
    frames = pd.concat(frames, ignore_index=True)
    frames = frames[frames["subject"].isin(wanted)]
    frames["stage"] = frames["stage"].astype(str)

    n_cores = n_cores or max(1, (os.cpu_count() or 2) - 1)
    print(f"Fitting route-usage models for {len(wanted)} subject(s): {null_reps} null reps, "
          f"{bootstraps} level bootstraps, on {n_cores} cores...")
    out = _run_script(frames[["rep", "subject", "scene", "level", "stage", "n",
                              *MEASURES]], wanted, bootstraps, seed, n_cores)
    fits = previous + _summarise(out, cells)
    fits.sort(key=lambda f: (f["subject"], f["measure"], f["weighting"]))
    model_path.write_text(json.dumps(_finite({
        **settings, "primary": PRIMARY, "stages": STAGES, "contrasts_defined": CONTRASTS,
        "formula": "y ~ stage + (1 | level) + (1 | level:scene)",
        "weightings": {"n": "weights = attempts in the cell (normalised to mean 1)",
                       "none": "unweighted"},
        "subjects": sorted({f["subject"] for f in fits}), "fits": fits}), indent=2))
    print(f"Wrote {model_path}")
