"""Visualization helpers for clustered behavioural traces.

Pure plotting code shared by the clustering notebook. Kept out of the notebook so
the rendering logic lives in one importable place (mirrors the supervisor's
``mario_learning.behavior/src/mario_learning/traces.py``).

Rendering follows the supervisor's approach:
  * player x = ``player_x_posHi * 256 + player_x_posLo``
  * plotting y is *screen-corrected* (``load_traces(..., y_mode="screen")``), so
    multi-screen vertical scenes don't teleport and dead frames are hidden.
  * if a scene background PNG exists it is drawn with ``imshow`` and the y-axis is
    flipped in image space (``set_ylim(height, 0)``); x is offset by the scene's
    mastersheet entry point so traces land in scene-local pixel coordinates.
  * with no background, traces are drawn in absolute data coordinates with
    ``invert_yaxis()`` (NES screen coords: y=0 is the top of the screen).

Background images do not ship with the dataset and must be regenerated separately
(see the deprecated ``mario.scenes/code/deprecated/make_images/make_scene_img.py``);
until they exist, plots fall back to the plain-axis rendering automatically.
"""

from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from analysis.clustering import load_traces

# Validated categorical palette (dataviz skill), fixed hue order — never cycled.
# Paired with a distinct marker per slot as secondary encoding, since a trajectory
# scatter/line plot needs all-pairs separation, not just adjacent-pair separation.
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CLUSTER_MARKERS = ["o", "s", "^", "D", "P", "X", "*", "v"]
NOISE_COLOR = "#898781"   # muted ink — HDBSCAN noise (-1)
OTHER_COLOR = "#c3c2b7"   # muted ink — clusters beyond the 8 direct-labeled slots
N_DIRECT_CLUSTERS = 8      # clusters beyond this rank (by size) fold into "other"

# Outcome colors (match the supervisor's traces.py).
OUTCOME_COLORS = {"completed": "#2ca02c", "death": "#d62728"}
OUTCOME_OTHER = "#898781"

# Per-subject palette for coloring traces by player (plotting only). Kept distinct
# from CLUSTER_COLORS so a subject-colored panel can't be mistaken for a
# cluster-colored one. High-contrast qualitative hues (ColorBrewer Set1 order) so
# the handful of players stay easy to tell apart even when traces overlap.
SUBJECT_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
                  "#a65628", "#f781bf", "#00ced1", "#999999", "#dede00"]

# Time-ordered encodings use viridis — sequential, perceptually uniform, and the
# house sequential map in the sibling mario_learning.behavior repo (its stage
# colors are the same plt.get_cmap("viridis")(i / 5) samples).
#
# These are baked to hex rather than sampled from matplotlib at draw time because
# web/app.js must use the *identical* values — the dashboard and these figures are
# meant to be one figure in two media, and the browser has no colormap library.
# If you change either list, change it in web/app.js too.

# Stage colors: viridis at i/5, in STAGES order (early/middle/late x disc/practice).
STAGE_COLORS = ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"]

# 32-stop viridis ramp for attempt-ordered coloring (nearest-stop lookup). 32 steps
# is well past what is distinguishable in overlapping thin trajectories.
VIRIDIS_STOPS = [
    "#440154", "#470d60", "#48186a", "#482374", "#472e7c", "#453882", "#424186", "#3e4a89",
    "#3a548c", "#365d8d", "#32658e", "#2e6d8e", "#2b758e", "#287d8e", "#25848e", "#228c8d",
    "#1f948c", "#1e9c89", "#20a386", "#25ab82", "#2eb37c", "#3aba76", "#48c16e", "#58c765",
    "#6ccd5a", "#7fd34e", "#93d741", "#a8db34", "#c0df25", "#d5e21a", "#eae51a", "#fde725",
]


def subject_color_map(metadata) -> dict:
    """Stable ``Subject -> color`` dict, assigned by sorted subject id.

    A given subject gets the same color in every panel of a figure. Colors cycle
    if there are more subjects than palette entries.
    """
    subjects = sorted({m.get("Subject", "?") for m in metadata})
    return {s: SUBJECT_COLORS[i % len(SUBJECT_COLORS)] for i, s in enumerate(subjects)}


def default_background_dir(source_dir: Path) -> Path:
    """Where scene background PNGs are expected once generated."""
    return Path(source_dir) / "mario.scenes" / "sourcedata" / "scene_backgrounds"


def resolve_background(scene_id: str, background_dir: Path | None) -> Path | None:
    """Return ``<background_dir>/<scene_id>.png`` if it exists, else None."""
    if background_dir is None:
        return None
    path = Path(background_dir) / f"{scene_id}.png"
    return path if path.exists() else None


def scene_entry_map(source_dir: Path) -> dict[str, int]:
    """scene_id -> absolute world-x entry point, from scenes_mastersheet.csv.

    Used to offset a trace into scene-local pixel space when drawing on a scene
    background (mirrors the supervisor's ``_scene_start_map``).
    """
    df = pd.read_csv(Path(source_dir) / "scenes_mastersheet.csv").dropna(
        subset=["World", "Level", "Scene"]
    )
    df["scene_id"] = df.apply(
        lambda r: f"w{int(r['World'])}l{int(r['Level'])}s{int(r['Scene'])}", axis=1
    )
    return dict(zip(df["scene_id"], df["Entry point"].astype(int)))


# Unique per-clip identity within a scene, used to join stored cluster labels to
# freshly-loaded traces by content rather than by position. (Subject, Session, Run,
# ClipCode) is unique per scene; ClipCode alone collides across subjects and
# StartFrame is not unique within a run.
CLIP_KEY_FIELDS = ("Subject", "Session", "Run", "ClipCode")


def _clip_key(meta: dict) -> tuple:
    return tuple(meta.get(f) for f in CLIP_KEY_FIELDS)


def load_labeled_traces(scene_id, source_dir, output_dir, y_mode="screen"):
    """Load a scene's traces with cluster labels joined **by clip identity**.

    The cluster JSON stores ``labels`` and ``metadata`` in the order clustering
    ran; ``load_traces`` may return a different order, so we must not zip labels
    to traces positionally. Instead we build ``clip_key -> label`` from the stored
    (metadata, labels) pair and look each freshly-loaded trace up by its key. This
    is correct regardless of ordering — including the older cluster files written
    before the ordering fix.

    Returns:
        traces, metadata, labels (np.ndarray aligned to traces), method (str)
    """
    result = json.loads((Path(output_dir) / "clusters" / f"{scene_id}_clusters.json").read_text())
    key_to_label = {_clip_key(m): l for m, l in zip(result["metadata"], result["labels"])}

    traces, metadata = load_traces(scene_id, source_dir, y_mode=y_mode)
    labels = np.array([key_to_label.get(_clip_key(m), -1) for m in metadata])

    missing = sum(1 for m in metadata if _clip_key(m) not in key_to_label)
    if missing:
        print(f"  warning: {missing}/{len(metadata)} traces in {scene_id} had no "
              f"stored label (drawn as noise)")
    return traces, metadata, labels, result.get("method", "")


def _draw_by_cluster(ax, traces, labels, x_offset):
    """Draw traces colored by cluster label (rank-by-size direct colors)."""
    # rank clusters by size so the most common behaviours get direct colors
    sizes = {l: int((labels == l).sum()) for l in set(labels.tolist()) if l != -1}
    ranked = sorted(sizes, key=sizes.get, reverse=True)
    direct = {l: i for i, l in enumerate(ranked[:N_DIRECT_CLUSTERS])}

    seen_legend = set()
    for trace, label in zip(traces, labels):
        if label == -1:
            color, marker, name = NOISE_COLOR, "x", "noise"
        elif label in direct:
            slot = direct[label]
            color, marker, name = CLUSTER_COLORS[slot], CLUSTER_MARKERS[slot], f"cluster {label} (n={sizes[label]})"
        else:
            color, marker, name = OTHER_COLOR, ".", "other clusters"

        alpha = 0.35 if label == -1 or label not in direct else 0.85
        lw = 1.0 if label == -1 or label not in direct else 1.6
        label_kw = {"label": name} if name not in seen_legend else {}
        seen_legend.add(name)

        x = trace[:, 0] - x_offset
        ax.plot(x, trace[:, 1], color=color, alpha=alpha, linewidth=lw, **label_kw)
        ax.plot(x[0], trace[0, 1], color=color, marker=marker, markersize=5, alpha=alpha)

    return len(ranked)


def _draw_by_outcome(ax, traces, metadata, x_offset):
    """Draw traces colored by clip outcome (green completed / red death)."""
    seen_legend = set()
    for trace, meta in zip(traces, metadata):
        outcome = meta.get("Outcome", "")
        color = OUTCOME_COLORS.get(outcome, OUTCOME_OTHER)
        name = outcome or "unknown"
        label_kw = {"label": name} if name not in seen_legend else {}
        seen_legend.add(name)

        x = trace[:, 0] - x_offset
        ax.plot(x, trace[:, 1], color=color, alpha=0.5, linewidth=1.0, **label_kw)
    return None


def plot_scene_clusters(
    scene_id,
    source_dir,
    output_dir,
    ax=None,
    *,
    color_by="cluster",
    background_dir=None,
    title_fontsize=None,
    legend_fontsize=8,
):
    """Draw one scene's trajectories onto `ax` (own figure if None).

    Args:
        color_by:       "cluster" (rank-by-size cluster colors, default) or
                        "outcome" (green=completed, red=death).
        background_dir: directory holding ``<scene_id>.png`` scene backgrounds.
                        If a matching PNG exists it is drawn under the traces and
                        the y-axis is flipped in image space; otherwise a plain
                        y-flipped data axis is used.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    traces, metadata, labels, method = load_labeled_traces(scene_id, source_dir, output_dir)
    if not traces:
        print(f"No trace data found for scene {scene_id}, skipping plot.")
        return None

    bg_path = resolve_background(scene_id, background_dir)
    x_offset = 0
    if bg_path is not None:
        x_offset = scene_entry_map(source_dir).get(scene_id, 0)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 6))

    if bg_path is not None:
        bg = np.asarray(Image.open(bg_path).convert("RGB"))
        ax.imshow(bg)

    if color_by == "outcome":
        n_clusters = _draw_by_outcome(ax, traces, metadata, x_offset)
    elif color_by == "cluster":
        n_clusters = _draw_by_cluster(ax, traces, labels, x_offset)
    else:
        raise ValueError(f"Unknown color_by '{color_by}'. Use 'cluster' or 'outcome'.")

    if bg_path is not None:
        ax.set_xlim(0, bg.shape[1])
        ax.set_ylim(bg.shape[0], 0)  # image coords: y=0 at top
        ax.set_axis_off()
    else:
        ax.set_xlabel("player x position")
        ax.set_ylabel("player y position")
        ylo, yhi = playfield_ylim(traces)
        ax.set_ylim(yhi, ylo)  # inverted (y=0 at top); clipped to playfield so
        # rare off-screen death-falls run off the bottom instead of dominating

    n_clusters_txt = "" if n_clusters is None else f", {n_clusters} clusters ({method})"
    ax.set_title(f"{scene_id} — {len(traces)} traces{n_clusters_txt}", fontsize=title_fontsize)
    ax.legend(loc="best", fontsize=legend_fontsize, ncol=2)

    if standalone:
        fig.tight_layout()
        out_path = output_dir / "clusters" / f"figure_clustering_{scene_id}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {out_path}")
        plt.show()
    return ax


SCREEN_HEIGHT = 256  # one vertical NES screen, in the screen-corrected y units


def playfield_ylim(traces, lo_pct=1.0, hi_pct=95.0, max_screens=2, margin=0.05):
    """Robust (ylo, yhi) for the vertical view, ignoring masked frames.

    Uses percentiles rather than min/max so real-but-huge off-screen plunges
    (a pit death is a genuine fall through many stacked screens) don't stretch
    the axis and squash the actual gameplay — those traces just clip off the
    bottom edge, as they do on the real screen. The span is also capped at
    ``max_screens`` NES screens so a scene with a lot of falling can't inflate
    the view past the playfield.
    """
    ys = np.ma.concatenate([np.ma.masked_invalid(t[:, 1]) for t in traces])
    ys = ys.compressed()
    if ys.size == 0:
        return 0.0, 1.0
    ylo = float(np.percentile(ys, lo_pct))
    yhi = float(np.percentile(ys, hi_pct))
    yhi = min(yhi, ylo + max_screens * SCREEN_HEIGHT)  # cap to a few screens
    pad = (yhi - ylo) * margin or 1.0
    return ylo - pad, yhi + pad


def _shared_xlim(traces):
    """(xmin, xmax) across all traces (x has no screen-wrap problem)."""
    xmin = min(float(np.min(t[:, 0])) for t in traces)
    xmax = max(float(np.max(t[:, 0])) for t in traces)
    return xmin, xmax


def plot_scene_cluster_panels(
    scene_id,
    source_dir,
    output_dir,
    *,
    color_by="cluster",
    background_dir=None,
    include_noise=True,
    max_cols=4,
    panel_size=(4.5, 3.2),
):
    """One subplot per cluster for a single scene, all in one figure.

    Splits a scene's traces so each cluster is drawn on its own panel (with the
    others hidden), making individual cluster shapes easy to read. Panels share
    axis limits so shapes stay directly comparable. Backgrounds and the
    screen-corrected y coordinate work exactly as in ``plot_scene_clusters``.

    Args:
        color_by:      "cluster" (default) draws each cluster's panel in a single
                       cluster color; "subject" colors every trace by its player,
                       so each panel shows the cluster's per-subject composition.
        include_noise: add a final panel for HDBSCAN noise (label -1) if present.
        max_cols:      maximum columns in the panel grid.
        panel_size:    (width, height) inches per panel.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if color_by not in ("cluster", "subject"):
        raise ValueError(f"Unknown color_by '{color_by}'. Use 'cluster' or 'subject'.")

    traces, metadata, labels, method = load_labeled_traces(scene_id, source_dir, output_dir)
    if not traces:
        print(f"No trace data found for scene {scene_id}, skipping plot.")
        return None

    subj_colors = subject_color_map(metadata) if color_by == "subject" else {}

    # rank clusters by size (same ordering/colors as the combined plot)
    sizes = {l: int((labels == l).sum()) for l in set(labels.tolist()) if l != -1}
    ranked = sorted(sizes, key=sizes.get, reverse=True)
    panels = list(ranked)
    if include_noise and (labels == -1).any():
        panels.append(-1)
    if not panels:
        print(f"No clusters to panel for scene {scene_id}, skipping.")
        return None

    bg_path = resolve_background(scene_id, background_dir)
    x_offset = scene_entry_map(source_dir).get(scene_id, 0) if bg_path is not None else 0
    bg = np.asarray(Image.open(bg_path).convert("RGB")) if bg_path is not None else None

    if bg is None:
        xmin, xmax = _shared_xlim(traces)
        dx = (xmax - xmin) * 0.02 or 1.0
        ylo, yhi = playfield_ylim(traces)

    cols = min(max_cols, len(panels))
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(panel_size[0] * cols, panel_size[1] * rows),
                             squeeze=False)

    for i, label in enumerate(panels):
        ax = axes[i // cols][i % cols]
        if bg is not None:
            ax.imshow(bg)
        if label == -1:
            color, marker, name, n = NOISE_COLOR, "x", "noise", int((labels == -1).sum())
        else:
            slot = ranked.index(label)
            color = CLUSTER_COLORS[slot % len(CLUSTER_COLORS)]
            marker = CLUSTER_MARKERS[slot % len(CLUSTER_MARKERS)]
            name, n = f"cluster {label}", sizes[label]

        for trace, lab, meta in zip(traces, labels, metadata):
            if lab != label:
                continue
            x = trace[:, 0] - x_offset
            tcolor = subj_colors[meta.get("Subject", "?")] if color_by == "subject" else color
            ax.plot(x, trace[:, 1], color=tcolor, alpha=0.6, linewidth=1.2)
            ax.plot(x[0], trace[0, 1], color=tcolor, marker=marker, markersize=4, alpha=0.8)

        if color_by == "subject":
            # Per-subject composition of this cluster, as a share of its traces.
            panel_subs = [m.get("Subject", "?") for m, lab in zip(metadata, labels) if lab == label]
            comp = Counter(panel_subs)
            ty = 0.98
            for s in sorted(comp):
                pct = 100 * comp[s] / len(panel_subs)
                ax.text(0.02, ty, f"sub-{s}: {pct:.0f}% (n={comp[s]})",
                        transform=ax.transAxes, color=subj_colors[s], fontsize=7,
                        fontweight="bold", va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
                ty -= 0.085

        if bg is not None:
            ax.set_xlim(0, bg.shape[1])
            ax.set_ylim(bg.shape[0], 0)
            ax.set_axis_off()
        else:
            ax.set_xlim(xmin - dx, xmax + dx)
            ax.set_ylim(yhi, ylo)  # inverted (y=0 at top); clipped to playfield
        ax.set_title(f"{name} (n={n})", fontsize=10)

    for i in range(len(panels), rows * cols):
        axes[i // cols][i % cols].axis("off")

    color_txt = " colored by subject" if color_by == "subject" else ""
    fig.suptitle(f"{scene_id} — per-cluster traces{color_txt} ({len(ranked)} clusters, {method})",
                 fontsize=13, fontweight="bold")

    if color_by == "subject":
        handles = [plt.Line2D([0], [0], color=c, lw=2, label=f"sub-{s}")
                   for s, c in subj_colors.items()]
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8),
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout()

    suffix = "_by_subject" if color_by == "subject" else ""
    out_path = output_dir / "clusters" / f"figure_clustering_{scene_id}_panels{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.show()
    return fig
