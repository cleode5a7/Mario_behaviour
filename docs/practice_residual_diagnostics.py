"""Exploratory: residual and random-effect diagnostics for the practice models.

Refits the primary model per subject (``<transform>(d_nearest_norm_loso) ~ log(attempt)``,
random intercept + slope on log(attempt) per scene — the ``log`` form of
``analysis.practice.fit_mixed``) under each outcome transform, and draws one row per
assumption, one column per subject, into ``docs/practice_residual_diagnostics_<transform>.png``.
The numbers behind each verdict go to ``docs/practice_residual_diagnostics_<transform>.csv``.
Row 9 does not refit: it reads the scene bootstrap's saved results
(``docs/practice_scene_bootstrap*.csv``, from ``docs/practice_scene_bootstrap.py``) and says
"not run" when they are missing — it is the check that the "CI" rows above actually matter.
Verdicts are computed from those numbers (thresholds in ``verdicts()``), never written in
by hand, so the figure for each transform says what that transform's residuals show.
Reads only ``output_data/practice/attempts/``; writes nothing under ``output_data/``.

**lme4 mode.** With ``lme4`` the same nine checks run on the full nested model instead
(``(1 + log(attempt) | level) + (1 + log(attempt) | level:scene)``, fitted in R by
``docs/practice_lme4_resid.R`` on the same frames), into
``docs/practice_residual_diagnostics_lme4_<transform>.png`` / ``.csv``. Residuals then net
out the level's and the scene's effects; row 6 adds the level effects to the scene ones;
and row 9 compares that model's own CI (the ``lme4_nested`` rows of
``output_data/practice/mixed_summary.csv``, from ``invoke run-practice-lme4``: profile, Wald
where the profile failed) with the level bootstrap — the bootstrap that, like this model, does not
assume a level's scenes are independent. A bootstrap of the nested model itself (hours of
refits) is not run.

    uv run python docs/practice_residual_diagnostics.py            # both transforms
    uv run python docs/practice_residual_diagnostics.py log        # one
    uv run python docs/practice_residual_diagnostics.py lme4       # nested lme4 model, both
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from analysis.practice import (PRIMARY, PRIMARY_TRANSFORM, TRANSFORMS, _fit_lmm, _mixed_frame,
                               level_of, load_attempts)
from analysis.trace_plots import CLUSTER_COLORS, SUBJECT_COLORS

HERE = Path(__file__).parent
BINS = [0, 2, 5, 10, 20, 40, 80, 400]
MIN_BIN = 200          # attempt bins smaller than this are drawn but not judged
OK, BAD, WARN = "#1b7a3e", "#b3261e", "#8a5a00"

# (title, question, what to look for, what a failure affects)
# "Slope" = the estimate b itself is wrong; "CI" = b stays right on average but its
# confidence interval / p-value can be too narrow or too wide.
ROWS = [
    ("0. Classic residual plot",
     "Centred on 0 and equally wide everywhere?",
     "black line flat at 0, dashed lines parallel",
     "BOTH - the centre (row 1) protects the slope, the width (rows 5, 7) the CI"),
    ("1. Shape of the trend",
     "Centre half of row 0, against attempt number: is the average residual 0 at every stage?",
     "dots inside the grey band (±0.05) with no up-down pattern",
     "SLOPE - a leftover trend means the curve shape is wrong, so b itself is wrong; no CI fix helps"),
    ("2. Independence",
     "Is an attempt's residual unrelated to the previous attempt's at that scene?",
     "a round, untilted cloud; r near 0",
     "CI (mainly) - correlated attempts carry less information than counted, so the CI comes out too narrow"),
    ("3. Normality (Q-Q plot)",
     "Do residual quantiles follow a normal distribution's?",
     "points on the red line",
     "CI (mainly) - the CI formula assumes normal noise; a heavy tail can also let extreme attempts pull the slope"),
    ("4. Normality (histogram)",
     "Is the residual distribution bell-shaped?",
     "bars following the black normal curve; skew ~0, excess kurtosis ~0",
     "CI (mainly) - same assumption as row 3, seen as a histogram"),
    ("5. Constant spread",
     "Width half of row 0: is the residual SD (= constant variance) the same wherever the model predicts?",
     "a flat line at the dashed overall SD",
     "CI - b stays right on average (slightly less precise); one shared noise level misjudges its uncertainty"),
    ("6. Scene random effects",
     "Are the scene intercepts u and slopes v bell-shaped?",
     "both sets of points on the line (standardised)",
     "SCENE VARIANCES, and the slope a little - extreme scenes can pull b and distort shrinkage"),
    ("7. Equal variance across scenes",
     "Is attempt noise the same in every scene?",
     "outside share no higher than with residuals shuffled across scenes",
     "CI - a residual check grouped by scene (not a random-effect check); noisy and quiet scenes get equal weight"),
    ("8. Equal random-effect variance",
     "Do u and v spread equally in every world?",
     "boxes equally tall (u filled, v hollow); ratio within shuffled-world chance",
     "SCENE VARIANCES and CI - the random-effect variance check: one var(u), one var(v) for all scenes"),
    ("9. Does it matter? (bootstrap)",
     "Is the model's CI right without its assumptions?",
     "both line pairs at the blue band's edges",
     "CI - the answer for every 'CI' row above: if the intervals match, those violations do not change the conclusions"),
]
BOOT_SUMMARY = Path(__file__).parent / "practice_scene_bootstrap.csv"
BOOT_DRAWS = Path(__file__).parent / "practice_scene_bootstrap_draws.csv"
LEVEL_SUMMARY = Path(__file__).parent / "practice_level_bootstrap.csv"
LEVEL_DRAWS = Path(__file__).parent / "practice_level_bootstrap_draws.csv"
MIXED_SUMMARY = Path(__file__).resolve().parent.parent / "output_data" / "practice" / "mixed_summary.csv"
LME4_ROW9 = ("9. Does it matter? (level bootstrap)",
             "Is this model's CI right without its assumptions?",
             "green lines at the pink band's edges",
             "CI - the level bootstrap does not assume normal noise, nor that a level's scenes are independent")
N_SHUFFLE = 200        # permutations behind the chance level in rows 7 and 8
RNG_SEED = 0
MIN_SCENE = 5          # scenes with fewer attempts have no usable residual SD
MIN_WORLD = 10         # worlds with fewer scenes are drawn but not judged


def fit_lme4_all(df, transform):
    """{subject: (d, scene effects, level effects)} for the nested lme4 model, via R."""
    if shutil.which("Rscript") is None:
        sys.exit("Rscript not found - install R and lme4: sudo apt install r-base r-cran-lme4")
    frames = {}
    for s in sorted(df["subject"].unique()):
        d = _mixed_frame(df, PRIMARY, s, transform=transform).reset_index(drop=True)
        d["rid"] = np.arange(len(d))
        d["level"] = d["group"].map(level_of)
        frames[s] = d
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pd.concat([d.assign(subject=s)[["subject", "rid", "y", "logt", "group", "level"]]
                   .rename(columns={"group": "scene"}) for s, d in frames.items()]
                  ).to_csv(tmp / "frames.csv", index=False)
        subprocess.run(["Rscript", str(HERE / "practice_lme4_resid.R"), str(tmp / "frames.csv"),
                        str(tmp / "resid.csv"), str(tmp / "ranef.csv")], check=True)
        resid = pd.read_csv(tmp / "resid.csv", dtype={"subject": str})
        ranef = pd.read_csv(tmp / "ranef.csv", dtype={"subject": str})
    out = {}
    for s, d in frames.items():
        r = resid[resid["subject"] == s].set_index("rid").loc[d["rid"]]
        attrs = d.attrs
        d = d.assign(fit=r["fit"].values, res=r["res"].values).sort_values(["group", "attempt"])
        d.attrs.update(attrs)
        d["prev"] = d.groupby("group")["res"].shift()
        re = ranef[(ranef["subject"] == s) & (ranef["kind"] == "scene")].set_index("id")[["u", "v"]]
        lev = ranef[(ranef["subject"] == s) & (ranef["kind"] == "level")].set_index("id")[["u", "v"]]
        out[s] = (d, re, lev)
    return out


def fit(df, subject, transform):
    d = _mixed_frame(df, PRIMARY, subject, transform=transform)
    res = _fit_lmm(smf, "y ~ logt", d, "~logt")
    d = d.assign(fit=res.fittedvalues, res=res.resid).sort_values(["group", "attempt"])
    d["prev"] = d.groupby("group")["res"].shift()
    re = pd.DataFrame(res.random_effects).T
    re.columns = ["u", "v"]
    return d, re


def verdicts(m: pd.DataFrame, boot: pd.DataFrame | None) -> list[tuple[str, str]]:
    """One (text, colour) per row, from the per-subject numbers in ``m`` (and ``boot``)."""
    def grade(value, ok, mild):
        return ("OK", OK) if value <= ok else ("MILD", WARN) if value <= mild else ("VIOLATED", BAD)

    bias = m["max_bin_bias"].max()
    shape = grade(bias, 0.05, 0.10)
    r = m["lag1_r"].abs().max()
    indep = grade(r, 0.10, 0.20)
    sk, ku = m["skew"].abs().max(), m["kurtosis"].max()
    normal = (("OK", OK) if sk <= 0.5 and ku <= 1 else
              ("MILD", WARN) if sk <= 1.0 and ku <= 5 else ("VIOLATED", BAD))
    ratio = m["sd_ratio"].max()
    spread = grade(ratio, 1.5, 2.0)
    z = m["max_abs_z_re"].max()
    reff = ("OK", OK) if z <= 4 else ("CONCERN", WARN)
    classic = shape if spread[0] == "OK" else (("MIXED", WARN) if shape[0] == "OK" else spread)
    # Rows 7 and 8 are judged against their shuffled chance level, per subject.
    excess = (m["share_scenes_outside_funnel"] / m["outside_shuffled_p95"]).max()
    scene_var = grade(excess, 1.0, 1.5)
    n_world_out = int((m["world_sd_ratio"] > m["world_sd_ratio_shuffled_p95"]).sum())
    world_var = (("OK", OK) if n_world_out == 0 else
                 ("MILD", WARN) if n_world_out <= 1 else ("CONCERN", WARN))

    norm_txt = (f"skew {m['skew'].min():.1f} to {m['skew'].max():.1f}, "
                f"kurtosis {m['kurtosis'].min():.0f} to {m['kurtosis'].max():.0f}")
    return [
        (f"{classic[0]} - centre {'flat' if shape[0] == 'OK' else 'drifts'}, "
         f"width {'even' if spread[0] == 'OK' else 'fans out'} (rows 1 and 5)", classic[1]),
        (f"{shape[0]} - largest bin bias {bias:.2f} (bins of >= {MIN_BIN})", shape[1]),
        (f"{indep[0]} - largest |r| {r:.2f}", indep[1]),
        (f"{normal[0]} - {norm_txt}", normal[1]),
        (f"{normal[0]} - {norm_txt}", normal[1]),
        (f"{spread[0]} - SD grows up to x{ratio:.1f} low to high", spread[1]),
        (f"{reff[0]} - most extreme scene {z:.1f} SD out", reff[1]),
        (f"{scene_var[0]} - outside share up to x{excess:.1f} its shuffled 95th pct", scene_var[1]),
        (f"{world_var[0]} - {n_world_out} of {len(m)} subjects beyond shuffled chance", world_var[1]),
        boot_verdict(boot),
    ]


def boot_verdict(boot):
    """OK when every clear/not-clear call agrees and widths are within x0.8-1.25.

    ``boot`` holds the scene and, when run, the level bootstrap rows together."""
    if boot is None or boot.empty:
        return ("NOT RUN - run docs/practice_scene_bootstrap.py", WARN)
    flips = int((boot["model_excludes_zero"] != boot["boot_excludes_zero"]).sum())
    lo, hi = boot["width_ratio"].min(), boot["width_ratio"].max()
    found = set(boot.get("unit", []))
    units = ("level boot vs lme4" if found == {"level"} else
             "scene + level" if "level" in found else "scene")
    if flips:
        return (f"CHANGES - {flips} call(s) flip ({units})", BAD)
    if lo >= 0.8 and hi <= 1.25:
        return (f"OK - {units}: widths x{lo:.2f} to x{hi:.2f}, every call unchanged", OK)
    return (f"MILD - {units}: widths x{lo:.2f} to x{hi:.2f}, every call unchanged", WARN)


def load_boot(transform):
    if not (BOOT_SUMMARY.exists() and BOOT_DRAWS.exists()):
        return None, None
    summ = pd.read_csv(BOOT_SUMMARY, dtype={"subject": str})
    draws = pd.read_csv(BOOT_DRAWS, dtype={"subject": str})
    summ = summ.assign(unit=summ.get("unit", "scene"))
    if LEVEL_SUMMARY.exists():
        summ = pd.concat([summ, pd.read_csv(LEVEL_SUMMARY, dtype={"subject": str})],
                         ignore_index=True)
    summ, draws = summ[summ["transform"] == transform], draws[draws["transform"] == transform]
    return (summ, draws) if len(summ) else (None, None)


def load_lme4_vs_level(transform):
    """Row 9 in lme4 mode: the nested model's CI against the level bootstrap.

    Returns rows shaped like the bootstrap summary (model_* = the lme4 CI), plus the
    level-bootstrap draws, so the same verdict and drawing code serve both modes."""
    if not (MIXED_SUMMARY.exists() and LEVEL_SUMMARY.exists() and LEVEL_DRAWS.exists()):
        return None, None
    m = pd.read_csv(MIXED_SUMMARY, dtype={"subject": str})
    m = m[(m["structure"] == "lme4_nested") & (m["measure"] == PRIMARY)
          & (m["transform"] == transform) & ~m["completed_only"].astype(bool)]
    lv = pd.read_csv(LEVEL_SUMMARY, dtype={"subject": str})
    lv = lv[lv["transform"] == transform]
    rows = []
    for _, r in m.iterrows():
        b = lv[lv["subject"] == r["subject"]]
        if b.empty:
            continue
        b = b.iloc[0]
        lo, hi = r["slope_ci_low"], r["slope_ci_high"]
        rows.append({"subject": r["subject"], "unit": "level", "b": r["slope"],
                     "model_ci_low": lo, "model_ci_high": hi,
                     "boot_ci_low": b["boot_ci_low"], "boot_ci_high": b["boot_ci_high"],
                     "width_ratio": (b["boot_ci_high"] - b["boot_ci_low"]) / (hi - lo),
                     "model_excludes_zero": bool(hi < 0 or lo > 0),
                     "boot_excludes_zero": bool(b["boot_excludes_zero"])})
    draws = pd.read_csv(LEVEL_DRAWS, dtype={"subject": str})
    return (pd.DataFrame(rows), draws[draws["transform"] == transform]) if rows else (None, None)


def run(df, transform, model="scene"):
    lme4 = model == "lme4"
    boot, boot_draws = load_lme4_vs_level(transform) if lme4 else load_boot(transform)
    fits = fit_lme4_all(df, transform) if lme4 else None
    rows = ROWS[:-1] + [LME4_ROW9] if lme4 else ROWS
    subjects = sorted(df["subject"].unique())
    n = len(subjects)
    fig = plt.figure(figsize=(3.7 * n, 3.55 * len(rows) + 1.4))
    gs = fig.add_gridspec(len(rows), n, top=1 - 2.6 / (3.55 * len(rows) + 1.4),
                          bottom=0.02, left=0.05, right=0.99, hspace=0.95, wspace=0.32)
    role = "main analysis" if transform == PRIMARY_TRANSFORM else "original analysis, kept for comparison"
    if lme4:
        fig.suptitle(f"Residual checks: FULL NESTED lme4 model, {transform}({PRIMARY}) ~ log(attempt) "
                     "+ (1 + log(attempt) | level) + (1 + log(attempt) | level:scene)  "
                     f"({role})", fontsize=14, fontweight="bold", y=0.997)
    else:
        fig.suptitle(f"Residual checks: {transform}({PRIMARY}) ~ log(attempt), one LMM per subject "
                     f"({role})", fontsize=15, fontweight="bold", y=0.997)
    effects = "the level's and the scene's u and v" if lme4 else "the scene's own u and v"
    fig.text(0.5, 0.984, f"A residual is observed - predicted ({effects} "
             "included). If the model captured the structure, residuals look like "
             "patternless noise: centred on 0 everywhere, the same spread everywhere, "
             "independent, bell-shaped.", ha="center", fontsize=10.5, color="#444")
    fig.text(0.5, 0.976, "Each row says what a failure would affect: the SLOPE b itself, or only "
             "its CI (b right on average, uncertainty misjudged). Row 9 checks whether the CI "
             "problems matter, with " + ("the level bootstrap against this model's profile CI."
                                         if lme4 else
                                         "the scene bootstrap (docs/practice_scene_bootstrap.py)."),
             ha="center", fontsize=10.5, color="#1f3f73", style="italic")

    metrics = []
    for j, s in enumerate(subjects):
        if lme4:
            d, re, lev = fits[s]
        else:
            (d, re), lev = fit(df, s, transform), None
        c = SUBJECT_COLORS[j]
        lo, hi = np.percentile(d["res"], [0.1, 99.9])
        lo, hi = min(lo, -1.0), max(hi, 1.5)
        ax = [fig.add_subplot(gs[i, j]) for i in range(len(rows))]
        classic, ax = ax[0], ax[1:]

        # 0. classic residuals-vs-fitted, with the decile mean and +-SD on top
        classic.scatter(d["fit"], d["res"], s=1, alpha=0.06, color=c, rasterized=True)
        classic.axhline(0, color="red", lw=0.8)
        q = pd.qcut(d["fit"], 10, labels=False)
        m = d.groupby(q).agg(f=("fit", "mean"), r=("res", "mean"), sd=("res", "std"))
        classic.plot(m["f"], m["r"], "k-o", ms=3, lw=1.5, label="mean")
        classic.plot(m["f"], m["r"] + m["sd"], "k--", lw=1, label="mean ± SD")
        classic.plot(m["f"], m["r"] - m["sd"], "k--", lw=1)
        classic.set_ylim(lo, hi)
        classic.set_xlabel(f"fitted value ({transform})", fontsize=8)
        classic.set_ylabel("residual", fontsize=8)
        classic.legend(fontsize=7, frameon=False, loc="upper left")
        classic.tick_params(labelsize=7)
        classic.grid(alpha=0.25)
        classic.set_title(f"sub-{s}", fontsize=12, fontweight="bold", color=c, pad=8)

        # 1. mean residual by attempt bin
        d["ab"] = pd.cut(d["attempt"], BINS)
        b = d.groupby("ab", observed=True)["res"].agg(["mean", "sem", "size"])
        x = np.arange(len(b))
        lim = max(0.16, float((b["mean"].abs() + 1.96 * b["sem"]).max()) * 1.1)
        ax[0].axhspan(-0.05, 0.05, color="#e6e6e6", zorder=0)
        ax[0].axhline(0, color="k", lw=0.8)
        ax[0].errorbar(x, b["mean"], yerr=1.96 * b["sem"], fmt="o", color=c, capsize=2)
        ax[0].set_xticks(x, [f"{iv.left + 1}-{iv.right}" for iv in b.index], rotation=45, fontsize=7)
        ax[0].set_xlabel("attempt at scene", fontsize=8)
        ax[0].set_ylabel("mean residual", fontsize=8)
        ax[0].set_ylim(-lim, lim)

        # 2. residual vs previous residual
        p = d[["res", "prev"]].dropna()
        r = p.corr().iloc[0, 1]
        ax[1].hexbin(p["prev"], p["res"], gridsize=40, bins="log", cmap="Greys",
                     extent=(lo, hi, lo, hi))
        ax[1].set_xlabel("previous attempt's residual", fontsize=8)
        ax[1].set_ylabel("this attempt's residual", fontsize=8)
        ax[1].text(0.04, 0.92, f"r = {r:.2f}", transform=ax[1].transAxes, fontsize=10,
                   fontweight="bold")

        # 3. Q-Q
        (osm, osr), (slope, icpt, _) = stats.probplot(d["res"], dist="norm")
        ax[2].plot(osm, osr, ".", ms=1.5, color=c, alpha=0.4)
        ax[2].plot(osm, slope * osm + icpt, color="red", lw=1)
        ax[2].set_xlabel("normal quantile", fontsize=8)
        ax[2].set_ylabel("residual quantile", fontsize=8)

        # 4. histogram
        sk, ku = stats.skew(d["res"]), stats.kurtosis(d["res"])
        ax[3].hist(d["res"], bins=90, range=(lo, hi), density=True, color=c, alpha=0.7)
        xx = np.linspace(lo, hi, 200)
        ax[3].plot(xx, stats.norm.pdf(xx, d["res"].mean(), d["res"].std()), "k", lw=1.2)
        ax[3].set_xlabel("residual", fontsize=8)
        ax[3].text(0.97, 0.9, f"skew {sk:.1f}\nkurtosis {ku:.0f}", transform=ax[3].transAxes,
                   ha="right", va="top", fontsize=9, fontweight="bold")

        # 5. residual SD by fitted decile
        g = d.groupby(q).agg(f=("fit", "mean"), sd=("res", "std"))
        ratio = g["sd"].max() / g["sd"].min()
        ax[4].plot(g["f"], g["sd"], "-o", color=c, ms=4)
        ax[4].axhline(d["res"].std(), color="k", ls="--", lw=1)
        ax[4].set_ylim(0, None)
        ax[4].set_xlabel(f"model prediction ({transform}, decile means)", fontsize=8)
        ax[4].set_ylabel("residual SD", fontsize=8)
        ax[4].text(0.04, 0.9, f"x{ratio:.1f} lowest to highest", transform=ax[4].transAxes,
                   fontsize=9, fontweight="bold")

        # 6. random effects Q-Q, standardised
        zmax = 0.0
        for col, mk, lab in (("u", "o", "u (intercept)"), ("v", "^", "v (slope)")):
            z = (re[col] - re[col].mean()) / re[col].std()
            zmax = max(zmax, float(z.abs().max()))
            (qm, qr), _ = stats.probplot(z, dist="norm")
            ax[5].plot(qm, qr, mk, ms=3, color=c, alpha=0.6 if col == "u" else 0.35,
                       mfc=c if col == "u" else "none", label=("scene " + lab) if lme4 else lab)
        if lev is not None and len(lev) > 2:
            for col, mk, lab in (("u", "s", "level u"), ("v", "D", "level v")):
                sd = lev[col].std()
                if not sd > 0:
                    continue        # a level variance estimated at zero: nothing to plot
                z = (lev[col] - lev[col].mean()) / sd
                zmax = max(zmax, float(z.abs().max()))
                (qm, qr), _ = stats.probplot(z, dist="norm")
                ax[5].plot(qm, qr, mk, ms=5, color="k", alpha=0.75,
                           mfc="k" if col == "u" else "none", label=lab)
        line = np.array([-3.2, 3.2])
        ax[5].plot(line, line, color="red", lw=1)
        ax[5].set_xlabel("normal quantile", fontsize=8)
        ax[5].set_ylabel("standardised scene effect", fontsize=8)
        ax[5].legend(fontsize=7, frameon=False, loc="upper left")

        # 7. per-scene residual SD against attempts, in the chance funnel for equal variance
        sc = d.groupby("group")["res"].agg(["std", "size"])
        sc = sc[sc["size"] >= MIN_SCENE]
        sigma = float(d["res"].std())
        nn = np.unique(np.geomspace(MIN_SCENE, sc["size"].max(), 60).round())
        band = [sigma * np.sqrt(stats.chi2.ppf(q, nn - 1) / (nn - 1)) for q in (0.05, 0.95)]
        lo_b = sigma * np.sqrt(stats.chi2.ppf(0.05, sc["size"] - 1) / (sc["size"] - 1))
        hi_b = sigma * np.sqrt(stats.chi2.ppf(0.95, sc["size"] - 1) / (sc["size"] - 1))
        outside = float(((sc["std"] < lo_b) | (sc["std"] > hi_b)).mean())
        # Chance level: the same residuals shuffled across scenes keep their real (heavy-
        # tailed) distribution but have equal variance by construction. Conservative,
        # since the pooled tails include whatever scene-to-scene variance exists.
        rng = np.random.default_rng(RNG_SEED)
        keep = d[d["group"].isin(sc.index)]
        sizes_sc = keep.groupby("group").size().loc[sc.index]
        lo_s = sigma * np.sqrt(stats.chi2.ppf(0.05, sizes_sc - 1) / (sizes_sc - 1))
        hi_s = sigma * np.sqrt(stats.chi2.ppf(0.95, sizes_sc - 1) / (sizes_sc - 1))
        chance = []
        for _ in range(N_SHUFFLE):
            sh = keep.assign(res=rng.permutation(keep["res"].values))
            sd = sh.groupby("group")["res"].std().loc[sc.index]
            chance.append(float(((sd < lo_s) | (sd > hi_s)).mean()))
        chance_mean, chance_p95 = float(np.mean(chance)), float(np.quantile(chance, 0.95))
        ax[6].fill_between(nn, band[0], band[1], color="#e6e6e6", zorder=0)
        ax[6].axhline(sigma, color="k", lw=0.8, ls="--")
        ax[6].scatter(sc["size"], sc["std"], s=8, color=c, alpha=0.6)
        ax[6].set_xscale("log")
        ax[6].set_xlabel("attempts in the scene (log scale)", fontsize=8)
        ax[6].set_ylabel("residual SD in the scene", fontsize=8)
        ax[6].text(0.97, 0.92, f"{outside:.0%} outside\nshuffled: {chance_mean:.0%} "
                   f"(95th pct {chance_p95:.0%})", transform=ax[6].transAxes,
                   ha="right", va="top", fontsize=8.5, fontweight="bold")

        # 8. spread of standardised u and v per world
        re = re.assign(world=[int(str(g).split("|")[0][1]) for g in re.index])
        zre = re.assign(u=(re["u"] - re["u"].mean()) / re["u"].std(),
                        v=(re["v"] - re["v"].mean()) / re["v"].std())
        worlds = sorted(zre["world"].unique())
        for k, (col, off, fill) in enumerate((("u", -0.18, True), ("v", 0.18, False))):
            data = [zre.loc[zre["world"] == w, col].values for w in worlds]
            bp = ax[7].boxplot(data, positions=np.arange(len(worlds)) + off, widths=0.32,
                               patch_artist=True, showfliers=True, manage_ticks=False,
                               flierprops=dict(marker=".", ms=3, mec=c, alpha=0.5))
            for box in bp["boxes"]:
                box.set(facecolor=c if fill else "white", edgecolor=c, alpha=0.6 if fill else 1)
            for part in ("whiskers", "caps", "medians"):
                for line in bp[part]:
                    line.set(color=c if part != "medians" else "k", lw=1)
        ax[7].set_xticks(np.arange(len(worlds)), [f"w{w}" for w in worlds])
        ax[7].set_ylim(-5, 5)
        ax[7].axhline(0, color="k", lw=0.6)
        ax[7].set_xlabel("world", fontsize=8)
        ax[7].set_ylabel("standardised scene effect", fontsize=8)
        sizes = zre.groupby("world").size()
        big = sizes[sizes >= MIN_WORLD].index
        sds = zre[zre["world"].isin(big)].groupby("world")[["u", "v"]].std()
        world_ratio = float((sds.max() / sds.min()).max())
        zbig = zre[zre["world"].isin(big)]
        wchance = []
        for _ in range(N_SHUFFLE):
            sh = zbig.assign(world=rng.permutation(zbig["world"].values))
            ssd = sh.groupby("world")[["u", "v"]].std()
            wchance.append(float((ssd.max() / ssd.min()).max()))
        world_p95 = float(np.quantile(wchance, 0.95))
        ax[7].text(0.97, 0.92, f"SD ratio x{world_ratio:.1f}\nshuffled 95th pct x{world_p95:.1f}",
                   transform=ax[7].transAxes, ha="right", va="top", fontsize=8.5, fontweight="bold")

        # 9. scene bootstrap: model CI (band) against bootstrap CI (lines), from saved draws
        row = None if boot is None else boot[(boot["subject"] == s)
                                             & (boot["unit"] == ("level" if lme4 else "scene"))]
        if row is None or row.empty:
            ax[8].text(0.5, 0.5, "bootstrap not run\nfor this subject", ha="center", va="center",
                       transform=ax[8].transAxes, fontsize=9, color="#888")
            ax[8].set_xticks([]); ax[8].set_yticks([])
        else:
            r9 = row.iloc[0]
            x = boot_draws.loc[boot_draws["subject"] == s, "b"].dropna().values
            ax[8].hist(x, bins=30, color=c, alpha=0.7)
            if lme4:
                ax[8].axvspan(r9.model_ci_low, r9.model_ci_high, color=CLUSTER_COLORS[4], alpha=0.2,
                              label="lme4 nested CI")
                ax[8].axvline(r9.boot_ci_low, color=CLUSTER_COLORS[2], lw=1.6, ls="--",
                              label="level bootstrap CI")
                ax[8].axvline(r9.boot_ci_high, color=CLUSTER_COLORS[2], lw=1.6, ls="--")
            else:
                ax[8].axvspan(r9.model_ci_low, r9.model_ci_high, color=CLUSTER_COLORS[0], alpha=0.15,
                              label="model CI")
                ax[8].axvline(r9.boot_ci_low, color=CLUSTER_COLORS[1], lw=1.6, label="scene bootstrap CI")
                ax[8].axvline(r9.boot_ci_high, color=CLUSTER_COLORS[1], lw=1.6)
            ax[8].axvline(r9.b, color="k", lw=1.4, label="b")
            lv = boot[(boot["subject"] == s) & (boot["unit"] == "level")] if not lme4 else boot.iloc[0:0]
            if len(lv):
                ax[8].axvline(lv.iloc[0].boot_ci_low, color=CLUSTER_COLORS[2], lw=1.4, ls="--",
                              label="level bootstrap CI")
                ax[8].axvline(lv.iloc[0].boot_ci_high, color=CLUSTER_COLORS[2], lw=1.4, ls="--")
            ax[8].axvline(0, color="#898781", lw=1, ls=":")
            ax[8].set_xlabel("bootstrap slope b (200 " + ("level" if lme4 else "scene")
                             + " resamples)", fontsize=8)
            ax[8].set_ylabel("replicates", fontsize=8)
            txt = (f"level boot / lme4 x{r9.width_ratio:.2f}" if lme4
                   else f"scene x{r9.width_ratio:.2f}")
            if len(lv):
                txt += f"\nlevel x{lv.iloc[0].width_ratio:.2f}"
            ax[8].text(0.03, 0.92, txt, transform=ax[8].transAxes, fontsize=8.5,
                       fontweight="bold", va="top")
            if j == 0:
                ax[8].legend(fontsize=7, frameon=False, loc="upper right")

        for a in ax:
            a.tick_params(labelsize=7)
            a.grid(alpha=0.25)

        judged = b[b["size"] >= MIN_BIN]
        metrics.append({
            "transform": transform, "subject": s, "n_obs": len(d),
            "n_zero_dropped": d.attrs.get("n_zero_dropped", 0),
            "max_bin_bias": float(judged["mean"].abs().max()), "lag1_r": float(r),
            "skew": float(sk), "kurtosis": float(ku), "sd_ratio": float(ratio),
            "max_abs_z_re": zmax, "share_scenes_outside_funnel": outside,
            "outside_shuffled_mean": chance_mean, "outside_shuffled_p95": chance_p95,
            "world_sd_ratio": world_ratio, "world_sd_ratio_shuffled_p95": world_p95,
        })

    metrics = pd.DataFrame(metrics)
    for i, ((title, q, look, affects), (verdict, vc)) in enumerate(zip(rows, verdicts(metrics, boot))):
        bbox = gs[i, 0].get_position(fig)
        y = bbox.y1 + (0.036 if i == 0 else 0.03)
        fig.text(0.012, y, title, fontsize=12.5, fontweight="bold", va="bottom")
        fig.text(0.012 + 0.0075 * len(title), y, f"   {q}  |  look for: {look}",
                 fontsize=9.5, color="#444", va="bottom")
        fig.text(0.99, y, verdict, fontsize=10.5, fontweight="bold", color=vc,
                 ha="right", va="bottom")
        fig.text(0.012, y - 0.0045, f"If violated, affects: {affects}", fontsize=9.5,
                 color="#1f3f73", style="italic", va="top")

    out = HERE / f"practice_residual_diagnostics_{'lme4_' if lme4 else ''}{transform}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    metrics.round(4).to_csv(out.with_suffix(".csv"), index=False)
    print(f"Wrote {out} and {out.with_suffix('.csv').name}")


def main():
    args = sys.argv[1:]
    model = "lme4" if "lme4" in args else "scene"
    df = load_attempts(Path("output_data"))
    for transform in [a for a in args if a in TRANSFORMS] or list(TRANSFORMS):
        run(df, transform, model)


if __name__ == "__main__":
    main()
