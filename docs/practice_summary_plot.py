"""Summary: every confidence interval for the practice slope, one figure.

For the primary measure (``d_nearest_norm_loso``, ``log`` practice form, all attempts),
per subject: the original ``log1p(d)`` model, the scene-scale lme4 model and the
bootstraps (left), and the ``log(d)`` model, the statsmodels nested model (no scene
correlation), the scene-scale lme4 model (profile CI, Wald where the profile failed) and
the scene and level bootstraps (right). The two
transforms' slopes are in different units, so they get separate panels, never one axis.
A filled marker means the CI clears zero, hollow means it does not. Below the panels, a
table of the p-value each method gives for the same slope, with how each is computed:
Wald z tests for the fitted models (Holm-adjusted across the five subjects for the main
model) and percentile-bootstrap p-values for the two bootstraps.

Reads ``output_data/practice/mixed_summary.csv`` (the statsmodels fits and the primary lme4
model, ``run-practice-lme4``) and the bootstrap CSVs under ``docs/``; refits nothing. Writes ``docs/practice_summary.png``.

    uv run python docs/practice_summary_plot.py
"""

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.practice import PRIMARY
from analysis.trace_plots import CLUSTER_COLORS

HERE = Path(__file__).parent
SUMMARY = HERE.parent / "output_data" / "practice" / "mixed_summary.csv"

# (key, label, colour) per interval, top to bottom within a subject's row.
METHODS = {
    "log1p": [("model", "model CI (formula)", CLUSTER_COLORS[0]),
              ("lme4", "scene scale, lme4", CLUSTER_COLORS[4]),
              ("scene_boot", "scene bootstrap", CLUSTER_COLORS[1]),
              ("level_boot", "level bootstrap", CLUSTER_COLORS[2])],
    "log": [("model", "model CI (formula)", CLUSTER_COLORS[0]),
            ("nested", "nested, no scene corr. (statsmodels)", CLUSTER_COLORS[6]),
            ("lme4", "scene scale, lme4", CLUSTER_COLORS[4]),
            ("scene_boot", "scene bootstrap", CLUSTER_COLORS[1]),
            ("level_boot", "level bootstrap", CLUSTER_COLORS[2])],
}
TITLES = {"log1p": "Original analysis: log(1 + d)", "log": "Main analysis: log(d)"}


def intervals():
    """{(transform, method, subject): (b, lo, hi)}."""
    out = {}
    s = pd.read_csv(SUMMARY, dtype={"subject": str})
    s = s[(s["measure"] == PRIMARY) & ~s["completed_only"].astype(bool) & (s["form"] == "log")]
    for _, r in s.iterrows():
        # lme4_nested: the scene-scale model; its CI is profile likelihood, Wald where that failed.
        key = {"scene": "model", "level_scene": "nested", "lme4_nested": "lme4"}.get(r["structure"])
        if key:
            out[(r["transform"], key, r["subject"])] = (r["slope"], r["slope_ci_low"], r["slope_ci_high"])
    for unit in ("scene", "level"):
        path = HERE / f"practice_{unit}_bootstrap.csv"
        if not path.exists():
            continue
        for _, r in pd.read_csv(path, dtype={"subject": str}).iterrows():
            out[(r["transform"], f"{unit}_boot", r["subject"])] = (r["b"], r["boot_ci_low"], r["boot_ci_high"])
    return out


def holm(p):
    """Holm step-down adjustment of a Series of p-values."""
    order = p.sort_values().index
    adj, running = {}, 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[k]))
        adj[k] = running
    return pd.Series(adj)[p.index]


def pvalues():
    """{(transform, method, subject): p} for every interval drawn, plus ("log", "holm", s)."""
    out = {}
    s = pd.read_csv(SUMMARY, dtype={"subject": str})
    s = s[(s["measure"] == PRIMARY) & ~s["completed_only"].astype(bool) & (s["form"] == "log")]
    for _, r in s.iterrows():
        key = {"scene": "model", "level_scene": "nested", "lme4_nested": "lme4"}.get(r["structure"])
        if key:
            out[(r["transform"], key, r["subject"])] = r["slope_p"]
    main_p = pd.Series({k[2]: v for k, v in out.items() if k[:2] == ("log", "lme4")})
    for subj, v in holm(main_p).items():
        out[("log", "holm", subj)] = v
    for unit in ("scene", "level"):
        path = HERE / f"practice_{unit}_bootstrap_draws.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path, dtype={"subject": str}).dropna(subset=["b"])
        for (t, subj), g in d.groupby(["transform", "subject"]):
            b = g["b"].to_numpy()
            out[(t, f"{unit}_boot", subj)] = (min(1.0, 2 * min((b >= 0).mean(), (b <= 0).mean())),
                                              len(b))
    return out


def fmt_p(v):
    """A p-value as text; bootstrap p-values arrive as (p, n_replicates)."""
    if isinstance(v, tuple):
        p, n = v
        return f"< {2 / n:.2f}" if p == 0 else f"{p:.2f}", p < 0.05
    if pd.isna(v):
        return "-", False
    return ("< 0.0001" if v < 1e-4 else f"{v:.4f}" if v < 0.01 else f"{v:.3f}"), v < 0.05


TABLE_COLS = [("log1p", "model", "model\n(formula)"), ("log1p", "lme4", "scene scale\n(lme4)"),
              ("log1p", "scene_boot", "scene\nbootstrap"), ("log1p", "level_boot", "level\nbootstrap"),
              ("log", "lme4", "scene scale\n(lme4), main"), ("log", "holm", "scene scale,\nHolm-adjusted"),
              ("log", "model", "model\n(formula)"), ("log", "nested", "nested, no scene\ncorr. (statsmodels)"),
              ("log", "scene_boot", "scene\nbootstrap"), ("log", "level_boot", "level\nbootstrap")]
P_NOTE = (
    "How each p-value is computed - every one is a two-sided test of 'no practice effect' (b = 0) for that subject.\n"
    "  -  Model (formula), statsmodels nested, scene scale (lme4): Wald z test. The fitted model gives the slope b and its standard error SE "
    "(which assumes normal, equal-variance, independent noise); z = b / SE, and p = 2 x P(|Z| > |z|) on a normal curve.\n"
    "      lme4's intervals drawn above are profile-likelihood CIs; its p-value here is the Wald one, which gives the same call in all 18 fits with a profile CI (the 2 without one fall back to Wald CIs above too).\n"
    "  -  Holm-adjusted: the main analysis's (scene-scale lme4, log(d)) p-values corrected for testing five subjects (smallest p x 5, next x 4, ...; never decreasing).\n"
    "  -  Scene / level bootstrap: the share of the 200 resampled slopes on the far side of zero, doubled - p = 2 x min(share >= 0, share <= 0). "
    "No normality is assumed.\n"
    "      With 200 replicates the smallest p it can show is 2/200 = 0.01, so '< 0.01' means none of the 200 slopes crossed zero.\n"
    "  Bold = p < 0.05 - the same call as an interval clearing zero in the panels above. The model and bootstrap columns are separate methods "
    "for the same question, not repeated tests.")


def main():
    iv = intervals()
    pv = pvalues()
    subjects = sorted({k[2] for k in iv})
    fig = plt.figure(figsize=(16, 1.4 * len(subjects) + 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4 * len(subjects), 3.0],
                          width_ratios=[1, 1.15], wspace=0.08, hspace=0.42,
                          top=0.875, bottom=0.165, left=0.07, right=0.99)
    axes = [fig.add_subplot(gs[0, 0])]
    axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0]))
    plt.setp(axes[1].get_yticklabels(), visible=False)
    fig.suptitle("Practice slope by method: does the conclusion depend on how the CI is computed?",
                 fontsize=14, fontweight="bold", x=0.5, y=0.985)
    fig.text(0.5, 0.965, f"{PRIMARY}, log(attempt) practice form, all attempts. Each bar is a 95% "
             "confidence interval for one subject's practice slope b; left of 0 = attempts move closer "
             "to the cluster prototypes with practice.\nFilled marker = the interval clears zero (a clear "
             "effect); hollow = it does not. The two panels are in different units, so compare calls "
             "and widths across them, never positions.",
             ha="center", va="top", fontsize=10, color="#444")

    for ax, transform in zip(axes, ("log1p", "log")):
        methods = METHODS[transform]
        offs = np.linspace(-0.32, 0.32, len(methods))
        for i, s in enumerate(subjects):
            for off, (key, label, colour) in zip(offs, methods):
                if (transform, key, s) not in iv:
                    continue
                b, lo, hi = iv[(transform, key, s)]
                y = i + off
                clear = hi < 0 or lo > 0
                ax.plot([lo, hi], [y, y], color=colour, lw=2.2, solid_capstyle="round",
                        label=label if i == 0 else None)
                ax.plot(b, y, "o", ms=7, color=colour, mfc=colour if clear else "white",
                        mew=1.8, zorder=3)
        ax.axvline(0, color="#3d3c38", lw=1, ls=(0, (3, 3)))
        for i in range(len(subjects) - 1):
            ax.axhline(i + 0.5, color="#e6e5df", lw=0.8)
        ax.set_title(TITLES[transform], loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"practice slope b, log(attempt) coefficient ({transform} units)", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                  ncol=3, handlelength=1.6, columnspacing=1.2)
    axes[0].set_yticks(range(len(subjects)), [f"sub-{s}" for s in subjects], fontsize=11)
    axes[0].set_ylim(len(subjects) - 0.5, -0.5)

    # Count only the intervals drawn above.
    drawn = [(t, key) for t in METHODS for key, _, _ in METHODS[t]]
    calls = []
    for s in subjects:
        flags = [iv[(t, key, s)][2] < 0 or iv[(t, key, s)][1] > 0
                 for t, key in drawn if (t, key, s) in iv]
        calls.append(f"sub-{s} clear in {sum(flags)} of {len(flags)}")
    fig.text(0.5, 0.913, "Clear effect (interval clears zero), counted over every interval drawn:  "
             + ";  ".join(calls), ha="center", fontsize=10, fontweight="bold", color="#1f3f73")

    # p-value table: one column per method, same order as the intervals above.
    tax = fig.add_subplot(gs[1, :])
    tax.axis("off")
    cells, bold = [], []
    for s in subjects:
        row, brow = [], []
        for t, key, _ in TABLE_COLS:
            text, b = fmt_p(pv.get((t, key, s), float("nan")))
            row.append(text)
            brow.append(b)
        cells.append(row)
        bold.append(brow)
    table = tax.table(cellText=cells, rowLabels=[f"sub-{s}" for s in subjects],
                      colLabels=[c[2] for c in TABLE_COLS], loc="upper center", cellLoc="center",
                      bbox=[0.04, 0.0, 0.96, 0.86])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#dcdcd5")
        if r == 0:
            cell.set_text_props(fontweight="bold", fontsize=8.5)
            cell.set_facecolor("#f2f2ef")
            cell.set_height(cell.get_height() * 1.6)
        elif c >= 0 and bold[r - 1][c]:
            cell.set_text_props(fontweight="bold")
        if c >= 4 and r >= 0:
            cell.set_facecolor("#f7f7f4" if r > 0 else "#e9e9e4")
    tax.text(0.04 + 0.96 * 2 / 10, 0.95, "Original analysis: log(1 + d)", transform=tax.transAxes,
             ha="center", fontsize=10.5, fontweight="bold")
    tax.text(0.04 + 0.96 * 7 / 10, 0.95, "Main analysis: log(d)", transform=tax.transAxes,
             ha="center", fontsize=10.5, fontweight="bold")
    tax.set_title("p-values for the practice slope, by method", loc="left", fontsize=12,
                  fontweight="bold", pad=18)
    note = "\n".join(textwrap.fill(line, 190, subsequent_indent="      ") for line in P_NOTE.split("\n"))
    fig.text(0.07, 0.012, note, ha="left", va="bottom", fontsize=9, color="#333", linespacing=1.5)
    out = HERE / "practice_summary.png"
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
