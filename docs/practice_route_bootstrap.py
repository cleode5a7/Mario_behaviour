"""Exploratory: level bootstrap of the route-scale practice model.

The route-scale model (``output_data/practice/lme4_route.json``) is
``log(d_own) ~ log(use) + (1 + log(use) | level) + (1 + log(use) | level:scene)
+ (1 + log(use) | level:scene:route)``, with Wald CIs. This resamples whole *levels* per
subject, refits that model on each draw (``docs/practice_lme4_route_boot.R``, in parallel)
and compares the percentile CI of the replicate slopes with the model's own CI — row 9 of
the route-scale diagnostics ("does it matter?"). Run on the main measure,
``d_own_loso``, all attempts, both transforms.

Writes ``docs/practice_route_level_bootstrap.csv`` (one row per subject x transform, in the
same columns as ``docs/practice_level_bootstrap.csv``) and
``docs/practice_route_level_bootstrap_draws.csv`` (every replicate slope). Reads only
``output_data/practice/`` (the attempt files and ``lme4_route.json``).

    uv run python docs/practice_route_bootstrap.py              # B=200, both transforms
    uv run python docs/practice_route_bootstrap.py 50 log       # quicker, one transform
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

from analysis.practice import TRANSFORMS, load_attempts, route_frame

HERE = Path(__file__).parent
MEASURE = "d_own_loso"
SEED = 0
SUMMARY = HERE / "practice_route_level_bootstrap.csv"
DRAWS = HERE / "practice_route_level_bootstrap_draws.csv"
ROUTE_FITS = HERE.parent / "output_data" / "practice" / "lme4_route.json"


def main():
    args = sys.argv[1:]
    n_boot = next((int(a) for a in args if a.isdigit()), 200)
    transforms = [a for a in args if a in TRANSFORMS] or list(TRANSFORMS)
    n_cores = max(1, (os.cpu_count() or 2) - 1)

    df = load_attempts(Path("output_data"))
    subjects = sorted(df["subject"].unique())
    frames = pd.concat([route_frame(df, MEASURE, s, False, t)
                        for t in transforms for s in subjects], ignore_index=True)
    print(f"Route-scale level bootstrap: B={n_boot}, {len(subjects)} subjects x "
          f"{len(transforms)} transform(s), {n_cores} cores")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frames.drop(columns="n_zero_dropped").to_csv(tmp / "frames.csv", index=False)
        subprocess.run(["Rscript", str(HERE / "practice_lme4_route_boot.R"), str(tmp / "frames.csv"),
                        str(tmp / "draws.csv"), str(n_boot), str(SEED), str(n_cores)], check=True)
        draws = pd.read_csv(tmp / "draws.csv", dtype={"subject": str})

    fits = json.loads(ROUTE_FITS.read_text())["fits"]
    rows = []
    for (t, s), g in draws.groupby(["transform", "subject"]):
        f = next(x for x in fits if x["measure"] == MEASURE and x["subject"] == s
                 and x["transform"] == t and not x["completed_only"])
        b = g["b"].dropna().to_numpy()
        lo, hi = np.percentile(b, [2.5, 97.5])
        mlo, mhi = f["ci"]
        rows.append({
            "transform": t, "subject": s, "b": f["b"], "model_ci_low": mlo, "model_ci_high": mhi,
            "boot_ci_low": lo, "boot_ci_high": hi, "boot_sd": b.std(ddof=1),
            "width_ratio": (hi - lo) / (mhi - mlo),
            "model_excludes_zero": bool(mhi < 0 or mlo > 0),
            "boot_excludes_zero": bool(hi < 0 or lo > 0),
            "share_boot_below_zero": float((b < 0).mean()), "n_boot": len(b),
            "n_failed": int(g["b"].isna().sum()), "share_singular": float(g["singular"].mean()),
            "n_scenes": int(frames.loc[(frames["subject"] == s) & (frames["transform"] == t),
                                       "scene"].nunique()),
            "unit": "level",
            "n_units": int(frames.loc[(frames["subject"] == s) & (frames["transform"] == t),
                                      "level"].nunique()),
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
