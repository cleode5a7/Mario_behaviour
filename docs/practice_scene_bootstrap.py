"""Exploratory: scene-level cluster bootstrap of the main practice slope.

For each subject, draws that subject's scenes **with replacement** (whole scenes, every
attempt in a drawn scene kept; a scene drawn twice enters as two separate groups), refits
the primary model — ``<transform>(d_nearest_norm_loso) ~ log(attempt)`` with a random
intercept and slope on log(attempt) per scene, the ``log`` form of
``analysis.practice.fit_mixed`` — and keeps the slope ``b``. The 2.5th-97.5th percentile of
those slopes is a confidence interval that does not assume normal residuals, equal noise
across scenes or constant spread, which the residual diagnostics showed do not hold.

Writes ``docs/practice_scene_bootstrap.csv`` (one row per subject × transform: model CI,
bootstrap CI, their width ratio), ``docs/practice_scene_bootstrap_draws.csv`` (every
replicate slope) and ``docs/practice_scene_bootstrap.png`` (model CI against bootstrap CI,
and each subject's distribution of replicate slopes with the percentile cut points that
make the bootstrap CI). Reads only ``output_data/practice/attempts/``.

**Resampling unit.** By default whole *scenes* are resampled, which assumes scenes are
independent of each other. Scenes of the same level are played back to back in the same
run, so they may not be; ``level`` resamples whole *levels* instead (every scene of a drawn
level comes along, together), assuming only levels are independent. The model is the same
either way — random effects stay per scene; only the resampling unit changes. If the level
CI is clearly wider than the scene CI, scenes within a level are correlated and both the
model's CI and the scene bootstrap's were too narrow. Results go to
``docs/practice_<unit>_bootstrap.csv`` / ``_draws.csv``; the figure shows every unit run.

    uv run python docs/practice_scene_bootstrap.py              # B=200, scenes, both transforms
    uv run python docs/practice_scene_bootstrap.py level        # B=200, levels
    uv run python docs/practice_scene_bootstrap.py 50 log       # quicker, one transform
    uv run python docs/practice_scene_bootstrap.py plot         # redraw from the CSVs only
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from joblib import Parallel, delayed

from analysis.practice import PRIMARY, PRIMARY_TRANSFORM, TRANSFORMS, _fit_lmm, _mixed_frame, load_attempts
from analysis.trace_plots import CLUSTER_COLORS, SUBJECT_COLORS

HERE = Path(__file__).parent
SEED = 0


def slope(d):
    res = _fit_lmm(smf, "y ~ logt", d, "~logt")
    return float(res.fe_params["logt"]), res


UNITS = ("scene", "level")
UNIT_COLORS = {"scene": 1, "level": 2}      # CLUSTER_COLORS slots; slot 0 is the model CI


def unit_of(scene: str, unit: str) -> str:
    """The resampling unit a scene belongs to: itself, or its level (``w1l1s3`` -> ``w1l1``)."""
    return scene if unit == "scene" else re.match(r"(w\d+l\d+)", scene).group(1)


def summary_path(unit):
    return HERE / f"practice_{unit}_bootstrap.csv"


def draws_path(unit):
    return HERE / f"practice_{unit}_bootstrap_draws.csv"


def replicate(d, units, members, idx):
    """Refit on units ``units[idx]``, each bringing all its scenes; a unit drawn twice
    enters as distinct groups."""
    by_scene = dict(tuple(d.groupby("group")))
    parts = [by_scene[sc].assign(group=f"{sc}#{k}")
             for k, i in enumerate(idx) for sc in members[units[i]]]
    try:
        return slope(pd.concat(parts, ignore_index=True))[0]
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def plot():
    """Draw the bootstrap figure from the two CSVs — no refitting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    units = [u for u in UNITS if summary_path(u).exists()]
    summs = {u: pd.read_csv(summary_path(u), dtype={"subject": str}) for u in units}
    summ = summs["scene"]
    draws = pd.read_csv(draws_path("scene"), dtype={"subject": str})
    transforms = [t for t in TRANSFORMS if t in set(summ["transform"])]
    subjects = sorted(summ["subject"].unique())
    MODEL, BOOT = CLUSTER_COLORS[0], CLUSTER_COLORS[1]

    fig = plt.figure(figsize=(3.6 * len(subjects), 4.2 + 3.1 * len(transforms)))
    gs = fig.add_gridspec(1 + len(transforms), len(subjects),
                          height_ratios=[1.35] + [1] * len(transforms),
                          top=0.87, bottom=0.06, left=0.06, right=0.99, hspace=0.55, wspace=0.32)
    fig.suptitle("Bootstrap: is the model's confidence interval for the practice slope right?",
                 fontsize=15, fontweight="bold", y=0.985)
    fig.text(0.5, 0.952, "Each subject's scenes were redrawn with replacement 200 times and the model "
             "refitted; the spread of those 200 slopes is the uncertainty measured directly, without the "
             "normal / equal-noise assumptions the model's CI formula relies on.\nThe bootstrap CI is the "
             "2.5th to 97.5th percentile of the 200 slopes (the 5 most extreme cut off at each end). It "
             "checks the CI only - the slope b itself comes from the model fitted once on the real data."
             + ("\nThe level bootstrap resamples whole levels (all their scenes together), so it also "
                "holds if scenes of one level are correlated; the scene bootstrap assumes they are not."
                if "level" in units else ""),
             ha="center", va="top", fontsize=10.5, color="#444")

    # Top: model CI against bootstrap CI, one panel per transform.
    span = max(1, len(subjects) // len(transforms))
    for k, t in enumerate(transforms):
        ax = fig.add_subplot(gs[0, k * span:(k + 1) * span if k < len(transforms) - 1 else len(subjects)])
        m = summ[summ["transform"] == t].set_index("subject").loc[subjects]
        ms = {u: summs[u][summs[u]["transform"] == t].set_index("subject") for u in units}
        offs = np.linspace(-0.2, 0.2, 1 + len(units))
        for i, s in enumerate(subjects):
            r = m.loc[s]
            bars = [(r.model_ci_low, r.model_ci_high, MODEL, "model CI (formula)")]
            bars += [(ms[u].loc[s].boot_ci_low, ms[u].loc[s].boot_ci_high,
                      CLUSTER_COLORS[UNIT_COLORS[u]], f"{u} bootstrap CI")
                     for u in units if s in ms[u].index]
            # Segments, not errorbars: a bootstrap CI need not be centred on b.
            for off, (lo, hi, col, lab) in zip(offs, bars):
                y = i + off
                ax.plot([lo, hi], [y, y], color=col, lw=1.6, label=lab if i == 0 else None)
                ax.plot([lo, lo], [y - 0.06, y + 0.06], color=col, lw=1.6)
                ax.plot([hi, hi], [y - 0.06, y + 0.06], color=col, lw=1.6)
                ax.plot(r.b, y, "o", color=col, ms=5)
        ax.axvline(0, color="#898781", lw=1, ls=":")
        ax.set_yticks(range(len(subjects)),
                      [f"sub-{s}\n" + " ".join(f"{u[0]} x{ms[u].loc[s].width_ratio:.2f}"
                                                for u in units if s in ms[u].index)
                       for s in subjects], fontsize=8.5)
        ax.invert_yaxis()
        role = "main analysis" if t == PRIMARY_TRANSFORM else "original analysis"
        ax.set_title(f"{t}(d) - {role}", loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"practice slope b, log(attempt) coefficient ({t} units)", fontsize=9)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left",
                  title="width = bootstrap / model (s = scene, l = level)", title_fontsize=8)
        ax.grid(alpha=0.25)

    # Below: each subject's 200 bootstrap slopes, one row per transform.
    for k, t in enumerate(transforms):
        for j, s in enumerate(subjects):
            ax = fig.add_subplot(gs[1 + k, j])
            x = np.sort(draws[(draws["transform"] == t) & (draws["subject"] == s)]["b"].dropna().values)
            r = summ[(summ["transform"] == t) & (summ["subject"] == s)].iloc[0]
            inside = (x >= r.boot_ci_low) & (x <= r.boot_ci_high)
            bins = np.linspace(x.min(), x.max(), 30)
            ax.hist(x[inside], bins=bins, color=SUBJECT_COLORS[j], alpha=0.75, label="middle 95% (scene)")
            ax.hist(x[~inside], bins=bins, color="#c3c2b7", alpha=0.9, label="cut off (2.5% each end)")
            ax.axvline(r.boot_ci_low, color=BOOT, lw=1.6)
            ax.axvline(r.boot_ci_high, color=BOOT, lw=1.6, label="scene bootstrap CI")
            ax.axvline(r.b, color="k", lw=1.6, label="b (model, real data)")
            ax.axvspan(r.model_ci_low, r.model_ci_high, color=MODEL, alpha=0.12, label="model CI")
            if "level" in units:
                lv = summs["level"]
                lv = lv[(lv["transform"] == t) & (lv["subject"] == s)]
                if len(lv):
                    for v in (lv.iloc[0].boot_ci_low, lv.iloc[0].boot_ci_high):
                        ax.axvline(v, color=CLUSTER_COLORS[2], lw=1.4, ls="--",
                                   label="level bootstrap CI" if v == lv.iloc[0].boot_ci_high else None)
            ax.axvline(0, color="#898781", lw=1, ls=":")
            ax.set_title(f"sub-{s}, {t}(d): {(x < 0).mean():.0%} of slopes < 0", fontsize=10,
                         loc="left", color=SUBJECT_COLORS[j], fontweight="bold")
            ax.set_xlabel("scene-bootstrap slope b", fontsize=8.5)
            ax.tick_params(labelsize=7.5)
            ax.grid(alpha=0.25)
            if j == 0:
                ax.set_ylabel("replicates", fontsize=9)
            if k == 0 and j == len(subjects) - 1:
                ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    out = HERE / "practice_scene_bootstrap.png"
    fig.savefig(out, dpi=110)
    print(f"Wrote {out}")


def main():
    if sys.argv[1:2] == ["plot"]:
        plot()
        return
    args = sys.argv[1:]
    unit = "level" if "level" in args else "scene"
    args = [a for a in args if a not in UNITS]
    B = int(args[0]) if args and args[0].isdigit() else 200
    transforms = [a for a in args if a in TRANSFORMS] or list(TRANSFORMS)
    df = load_attempts(Path("output_data"))
    rng = np.random.default_rng(SEED)
    rows, draws = [], []
    for transform in transforms:
        for s in sorted(df["subject"].unique()):
            d = _mixed_frame(df, PRIMARY, s, transform=transform)
            b, res = slope(d)
            ci = res.conf_int().loc["logt"].values
            scenes = np.array(sorted(d["group"].unique()))
            members = {}
            for sc in scenes:
                members.setdefault(unit_of(sc, unit), []).append(sc)
            units = np.array(sorted(members))
            idxs = [rng.integers(0, len(units), len(units)) for _ in range(B)]
            boot = np.array(Parallel(n_jobs=-1)(
                delayed(replicate)(d, units, members, i) for i in idxs))
            ok = boot[np.isfinite(boot)]
            lo, hi = np.percentile(ok, [2.5, 97.5])
            rows.append({
                "transform": transform, "subject": s, "b": b,
                "model_ci_low": ci[0], "model_ci_high": ci[1],
                "boot_ci_low": lo, "boot_ci_high": hi, "boot_sd": ok.std(ddof=1),
                "width_ratio": (hi - lo) / (ci[1] - ci[0]),
                "model_excludes_zero": bool(ci[1] < 0 or ci[0] > 0),
                "boot_excludes_zero": bool(hi < 0 or lo > 0),
                "share_boot_below_zero": float((ok < 0).mean()),
                "n_boot": len(ok), "n_failed": int(B - len(ok)), "n_scenes": len(scenes),
                "unit": unit, "n_units": len(units),
            })
            draws += [{"transform": transform, "subject": s, "b": v} for v in boot]
            r = rows[-1]
            print(f"{unit} {transform} sub-{s} ({len(units)} {unit}s): b {b:+.4f}  "
                  f"model [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
                  f"boot [{lo:+.4f}, {hi:+.4f}]  width x{r['width_ratio']:.2f}", flush=True)
    pd.DataFrame(rows).round(5).to_csv(summary_path(unit), index=False)
    pd.DataFrame(draws).round(6).to_csv(draws_path(unit), index=False)
    print(f"Wrote {summary_path(unit)}")
    plot()


if __name__ == "__main__":
    main()
