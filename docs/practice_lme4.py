"""Exploratory: fit the full nested practice model in R's lme4 and save the results.

Exports, for the primary measure (``d_nearest_norm_loso``), the exact model frames the
statsmodels fits use — ``analysis.practice._mixed_frame``, so the same transforms, the
same dropped zeros and the same leave-subject-out distances — for every subject, both
transforms, all and completed attempts; runs ``docs/practice_lme4.R`` on them; and writes
``docs/practice_lme4.csv``, then prints the lme4 scene-only model against the statsmodels
main model as a cross-check between the two programs. Needs R with lme4
(``sudo apt install r-base r-cran-lme4``). Reads only ``output_data/practice/``.

    uv run python docs/practice_lme4.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from analysis.practice import PRIMARY, TRANSFORMS, _mixed_frame, level_of, load_attempts

HERE = Path(__file__).parent
OUT = HERE / "practice_lme4.csv"


def main():
    if shutil.which("Rscript") is None:
        sys.exit("Rscript not found - install R and lme4: sudo apt install r-base r-cran-lme4")
    df = load_attempts(Path("output_data"))
    frames = []
    for transform in TRANSFORMS:
        for completed_only in (False, True):
            for s in sorted(df["subject"].unique()):
                d = _mixed_frame(df, PRIMARY, s, completed_only, transform)
                frames.append(pd.DataFrame({
                    "subject": s, "transform": transform, "completed_only": completed_only,
                    "y": d["y"].values, "logt": d["logt"].values,
                    "scene": d["group"].values, "level": d["group"].map(level_of).values,
                }))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "frames.csv"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        subprocess.run(["Rscript", str(HERE / "practice_lme4.R"), str(path), str(OUT)], check=True)

    # Cross-check: lme4's scene-only model is the statsmodels main model.
    r = pd.read_csv(OUT, dtype={"subject": str})
    sm = pd.read_csv(HERE.parent / "output_data/practice/mixed_summary.csv", dtype={"subject": str})
    sm = sm[(sm["measure"] == PRIMARY) & (sm["structure"] == "scene") & (sm["form"] == "log")]
    m = r[r["model"] == "scene"].merge(sm, on=["subject", "transform", "completed_only"])
    m["b_diff"] = (m["b"] - m["slope"]).abs()
    m["aic_diff"] = (m["aic_x"] - m["aic_y"]).abs()
    print("\nlme4 scene-only vs statsmodels main model: largest |b difference| "
          f"{m['b_diff'].max():.2g}, largest |AIC difference| {m['aic_diff'].max():.2g}")


if __name__ == "__main__":
    main()
