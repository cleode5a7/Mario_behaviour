"""Exploratory: level bootstrap of the scene-scale practice model, the main lme4 model.

The scene-scale model (``output_data/practice/lme4.json``, ``model == "nested"``) is
``log(d_nearest_norm_loso) ~ log(attempt) + (1 + log(attempt) | level)
+ (1 + log(attempt) | level:scene)``. The earlier scene and level bootstraps
(``docs/practice_scene_bootstrap.py``) refit the scene-only statsmodels model, which was the
main model when they were made; this one refits the current model itself. It resamples whole
*levels* per subject (``docs/practice_lme4_scene_boot.R``, in parallel) and compares the
percentile CI of the replicate slopes with the model's own CI (profile, Wald where the
profile failed). All attempts, both transforms. The twin of ``docs/practice_route_bootstrap.py``.

Writes ``docs/practice_lme4_level_bootstrap.csv`` (one row per subject x transform, in the
same columns as ``docs/practice_route_level_bootstrap.csv``) and
``docs/practice_lme4_level_bootstrap_draws.csv`` (every replicate slope). Reads only
``output_data/practice/`` (the attempt files and ``lme4.json``). B=200 took 28 min on
11 cores: a resampled fit is ~2.8 s for the largest subject run alone, slower with every
core busy.

    uv run python docs/practice_lme4_scene_bootstrap.py              # B=200, both transforms
    uv run python docs/practice_lme4_scene_bootstrap.py 50 log       # quicker, one transform
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from analysis.practice import PRIMARY, TRANSFORMS, _mixed_frame, level_of, load_attempts

HERE = Path(__file__).parent
MEASURE = PRIMARY                     # d_nearest_norm_loso
SEED = 0
SUMMARY = HERE / "practice_lme4_level_bootstrap.csv"
DRAWS = HERE / "practice_lme4_level_bootstrap_draws.csv"
LME4_FITS = HERE.parent / "output_data" / "practice" / "lme4.json"


def frame(df, subject, transform):
    """The scene-scale model's rows, exactly as run_practice_lme4 exports them."""
    d = _mixed_frame(df, MEASURE, subject, False, transform)
    return pd.DataFrame({"subject": subject, "transform": transform, "y": d["y"].values,
                         "logt": d["logt"].values, "scene": d["scene"].values,
                         "level": d["scene"].map(level_of).values})


def main():
    args = sys.argv[1:]
    n_boot = next((int(a) for a in args if a.isdigit()), 200)
    transforms = [a for a in args if a in TRANSFORMS] or list(TRANSFORMS)
    n_cores = max(1, (os.cpu_count() or 2) - 1)

    df = load_attempts(Path("output_data"))
    subjects = sorted(df["subject"].unique())
    frames = pd.concat([frame(df, s, t) for t in transforms for s in subjects], ignore_index=True)
    fits = [f for f in json.loads(LME4_FITS.read_text())["fits"]
            if f["model"] == "nested" and f["measure"] == MEASURE and not f["completed_only"]]
    for (t, s), g in frames.groupby(["transform", "subject"]):
        f = next(x for x in fits if x["subject"] == s and x["transform"] == t)
        if f["n_obs"] != len(g):
            raise SystemExit(f"sub-{s} {t}: {len(g)} rows but lme4.json was fitted on {f['n_obs']} "
                             "— rerun invoke run-practice-lme4 --force first")
    print(f"Scene-scale lme4 level bootstrap: B={n_boot}, {len(subjects)} subjects x "
          f"{len(transforms)} transform(s), {n_cores} cores")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frames.to_csv(tmp / "frames.csv", index=False)
        subprocess.run(["Rscript", str(HERE / "practice_lme4_scene_boot.R"), str(tmp / "frames.csv"),
                        str(tmp / "draws.csv"), str(n_boot), str(SEED), str(n_cores)], check=True)
        draws = pd.read_csv(tmp / "draws.csv", dtype={"subject": str})

    rows = []
    for (t, s), g in draws.groupby(["transform", "subject"]):
        f = next(x for x in fits if x["subject"] == s and x["transform"] == t)
        b = g["b"].dropna().to_numpy()
        lo, hi = np.percentile(b, [2.5, 97.5])
        mlo, mhi = f["ci"]
        sub = frames[(frames["subject"] == s) & (frames["transform"] == t)]
        rows.append({
            "transform": t, "subject": s, "b": f["b"], "model_ci_low": mlo, "model_ci_high": mhi,
            "model_ci_method": f["ci_method"],
            "boot_ci_low": lo, "boot_ci_high": hi, "boot_sd": b.std(ddof=1),
            "width_ratio": (hi - lo) / (mhi - mlo),
            "model_excludes_zero": bool(mhi < 0 or mlo > 0),
            "boot_excludes_zero": bool(hi < 0 or lo > 0),
            "share_boot_below_zero": float((b < 0).mean()), "n_boot": len(b),
            "n_failed": int(g["b"].isna().sum()), "share_singular": float(g["singular"].mean()),
            "n_scenes": int(sub["scene"].nunique()), "unit": "level",
            "n_units": int(sub["level"].nunique()),
        })
    summary = pd.DataFrame(rows)
    summary.round(5).to_csv(SUMMARY, index=False)
    draws.to_csv(DRAWS, index=False)
    print(summary[["transform", "subject", "b", "model_ci_low", "model_ci_high", "boot_ci_low",
                   "boot_ci_high", "width_ratio", "model_excludes_zero", "boot_excludes_zero"]]
          .round(4).to_string(index=False))
    print(f"Wrote {SUMMARY.name} and {DRAWS.name}")


if __name__ == "__main__":
    main()
