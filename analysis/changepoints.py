from functools import partial
from pathlib import Path
import json
import re

import numpy as np
import scipy.misc
import scipy.special

# Backfill removed scipy.misc symbols so the library loads cleanly
if not hasattr(scipy.misc, "comb"):
    scipy.misc.comb = scipy.special.comb
if not hasattr(scipy.misc, "logsumexp"):
    scipy.misc.logsumexp = scipy.special.logsumexp


_SCENE_PATTERN = re.compile(r"_level-(\w+)_scene-(\d+)_")


def list_scenes(source_dir: Path) -> list[str]:
    """Return sorted unique scene IDs (e.g. 'w7l1s0') found in mario.scenes."""
    scenes_dir = source_dir / "mario.scenes"
    if not scenes_dir.exists():
        return []
    seen = set()
    for f in scenes_dir.glob("**/gamelogs/*_summary.json"):
        m = _SCENE_PATTERN.search(f.name)
        if m:
            seen.add(f"{m.group(1)}s{m.group(2)}")
    return sorted(seen)


def list_subjects(source_dir: Path) -> list[str]:
    """Return sorted unique subject IDs found in mario.scenes."""
    scenes_dir = source_dir / "mario.scenes"
    if not scenes_dir.exists():
        return []
    return sorted({p.name.split("-")[1] for p in scenes_dir.glob("sub-*") if p.is_dir()})


def load_all_attempts(subject: str, source_dir: Path) -> list[dict]:
    """Load the full chronological attempt sequence for one subject.

    Returns summary records sorted by (Session, Run, StartFrame).
    Bernoulli series: [1 if r["Outcome"] == "completed" else 0 for r in result]
    """
    scenes_dir = source_dir / "mario.scenes"
    records: list[dict] = []

    for f in (scenes_dir / f"sub-{subject}").glob("**/gamelogs/*_summary.json"):
        records.append(json.loads(f.read_text()))

    # ClipCode tiebreaker keeps ordering deterministic when several clips share
    # (Session, Run, StartFrame) — otherwise ties fall back to non-reproducible
    # filesystem glob order.
    records.sort(key=lambda d: (d["Session"], d["Run"], d["StartFrame"], d.get("ClipCode", "")))
    return records


def phase_changepoints(records: list[dict]) -> list[int]:
    """Return attempt indices where the Phase annotation changes.

    Reads the 'Phase' field from each summary record (e.g. 'discovery',
    'practice') and returns the index of the first attempt in each new phase.

    This is the primary changepoint method: the Phase annotations encode the
    experimenters' protocol boundaries (e.g. discovery → practice). The
    Bayesian changepoint detector cannot recover these from the completion
    signal alone because subjects' completion rates are near-ceiling (~90%)
    throughout both phases, leaving no detectable step-change.
    """
    phases = [r.get("Phase") for r in records]
    return [i + 1 for i, (a, b) in enumerate(zip(phases, phases[1:])) if a != b]


def bocd_pcp_curve(
    completions: list[int],
    block_size: int = 50,
    truncate: float = -np.inf,
) -> tuple[np.ndarray, int]:
    """Run offline Bayesian changepoint detection and return the Pcp marginal.

    Uses BetaBinomial marginal likelihood (exact for Bernoulli data) on
    block-aggregated counts. Returns a (n_blocks,) array where entry t is
    the posterior probability of a changepoint at block t, plus the block
    index of the highest-probability changepoint.

    Primarily a diagnostic: compare the Pcp curve against known Phase
    transitions to validate that the algorithm would find them if the signal
    were sharper.
    """
    from scipy.special import betaln
    from bayesian_changepoint_detection.offline_changepoint_detection import (
        offline_changepoint_detection,
        const_prior,
        dynamic_programming,
    )

    arr = np.array(completions, dtype=float)
    n_blocks = len(arr) // block_size
    if n_blocks < 4:
        return np.array([]), 0

    block_k = arr[:n_blocks * block_size].reshape(n_blocks, block_size).sum(axis=1)
    prefix_k = np.concatenate([[0], block_k.cumsum()])
    n_bk = len(block_k)
    N_per = block_size

    @dynamic_programming
    def _bb_llik(data, t, s):
        s += 1                           # library passes exclusive end; add 1 for slice
        sc = min(s, n_bk)
        k = float(prefix_k[sc] - prefix_k[t])
        total = float((sc - t) * N_per)
        return float(betaln(1.0 + k, 1.0 + total - k))

    _, _, Pcp = offline_changepoint_detection(
        block_k,
        partial(const_prior, l=n_blocks + 1),
        _bb_llik,
        truncate=truncate,
    )

    cp_marginal = np.exp(Pcp).sum(axis=0)
    return cp_marginal, int(cp_marginal.argmax())


_RUPTURES_ALGOS = ("pelt", "binseg", "dynp")


def _ruptures_detect(
    signal_1d,
    block_size: int = 50,
    beta: float = 1.0,
    n_bkps: int = 1,
    model: str = "l2",
) -> dict:
    """Run ruptures Pelt/Binseg/Dynp (CostL2) on a 1-D signal.

    Runs on BOTH the raw per-attempt series and its block-averaged version,
    so a jittery Bernoulli signal gets a smoother companion view. CostL2 is a
    mean-shift cost — the natural model for both a 0/1 completion rate and a
    continuous duration, since a changepoint in either is a shift in the mean.

    Pelt and Binseg run in penalty mode with an adaptive, per-signal penalty
    ``pen = beta * log(n) * signal.var()`` (a BIC-like scale that tracks each
    signal's own variance, so raw 0/1 and block-mean series stay comparable).
    Dynp requires a fixed changepoint count, so it runs with ``n_bkps``.

    Breakpoints are returned in **attempt-index space** (a block breakpoint b
    maps to attempt index ``b * block_size``) so they line up with
    phase_changepoints() output and the notebook x-axis. A series too short to
    segment yields empty lists rather than raising.

    Returns ``{"raw": {"pelt": [...], "binseg": [...], "dynp": [...]},
               "blocks": {...}}``.
    """
    import ruptures as rpt

    arr = np.asarray(signal_1d, dtype=float)

    def _segment(values, scale: int) -> dict:
        # values: 1-D signal already at the desired granularity;
        # scale: attempts per sample (1 for raw, block_size for blocks).
        # jump=1 keeps full sample resolution — ruptures defaults to jump=5,
        # which would snap block-signal breakpoints to a coarse 5*block_size grid.
        out = {a: [] for a in _RUPTURES_ALGOS}
        n = len(values)
        # Need enough samples to fit n_bkps+1 segments and a meaningful penalty.
        if n < max(4, n_bkps + 1):
            return out
        sig = values.reshape(-1, 1)
        var = float(sig.var())
        pen = beta * np.log(n) * var if var > 0 else np.inf

        def _to_attempts(bkps):
            # ruptures returns breakpoints ending in the series length n; drop it.
            return [int(b * scale) for b in bkps if b < n]

        if np.isfinite(pen):
            out["pelt"] = _to_attempts(rpt.Pelt(model=model, jump=1).fit(sig).predict(pen=pen))
            out["binseg"] = _to_attempts(rpt.Binseg(model=model, jump=1).fit(sig).predict(pen=pen))
        k = min(n_bkps, n - 1)
        if k >= 1:
            out["dynp"] = _to_attempts(rpt.Dynp(model=model, jump=1).fit(sig).predict(n_bkps=k))
        return out

    n_blocks = len(arr) // block_size
    blocks = (
        arr[:n_blocks * block_size].reshape(n_blocks, block_size).mean(axis=1)
        if n_blocks >= 1 else np.array([])
    )

    return {
        "raw": _segment(arr, 1),
        "blocks": _segment(blocks, block_size),
    }


def ruptures_changepoints(
    completions: list[int],
    block_size: int = 50,
    beta: float = 1.0,
    n_bkps: int = 1,
    model: str = "l2",
) -> dict:
    """Ruptures changepoints on a Bernoulli completion series (raw + blocks).

    Thin wrapper over _ruptures_detect: a completion changepoint is a shift in
    the mean completion probability p, which CostL2 detects. Non-Bayesian
    counterpart to bocd_pcp_curve, for comparison against the Phase annotations.
    """
    return _ruptures_detect(completions, block_size=block_size, beta=beta,
                            n_bkps=n_bkps, model=model)


def run_changepoint(
    subject: str,
    source_dir: Path,
    output_dir: Path,
    max_attempts: int | None = None,
    beta: float = 1.0,
    n_bkps: int | None = None,
    block_size: int = 50,
) -> None:
    """Detect learning-phase changepoints for one subject.

    Primary method: Phase-annotation transitions extracted from each record's
    'Phase' field (e.g. 'discovery' → 'practice'). These reflect the
    experimenters' protocol boundaries, which are the ground-truth learning
    phases for this dataset.

    Input:   all *_summary.json for sub-{subject} in mario.scenes/
    Output:  output_dir/<subject>_changepoints.json
             {
               "subject": str,
               "n_attempts": int,
               "changepoints": [<attempt indices>],
               "phases": [{"phase": str, "start": int, "end": int}, ...],
               "method": "phase_annotation",
               "ruptures": {  # non-Bayesian diagnostic, for comparison
                 "model": "l2", "block_size": int, "beta": float, "n_bkps": int,
                 "results": {"raw": {...}, "blocks": {...}}
               }
             }

    Args:
        max_attempts: if set, truncate to this many attempts (smoke testing)
        beta:         ruptures penalty scale (see _ruptures_detect)
        n_bkps:       fixed changepoint count for the Dynp algorithm
        block_size:   attempts per block for the aggregated ruptures signal
    """
    records = load_all_attempts(subject, source_dir)

    if not records:
        print(f"No data found for subject {subject}, skipping.")
        return

    if max_attempts is not None:
        records = records[:max_attempts]

    changepoints = phase_changepoints(records)

    # Build per-phase segments for the notebook
    boundaries = [0] + changepoints + [len(records)]
    phase_labels = [records[b]["Phase"] for b in boundaries[:-1]]
    phases = [
        {"phase": label, "start": start, "end": end}
        for label, start, end in zip(phase_labels, boundaries[:-1], boundaries[1:])
    ]

    # Non-Bayesian ruptures diagnostic on the completion signal. Defaulting
    # n_bkps to the phase-transition count makes Dynp directly comparable to
    # the ground-truth phases. Never let a detector hiccup sink the primary output.
    completions = [1 if r["Outcome"] == "completed" else 0 for r in records]
    rpt_n_bkps = max(1, len(changepoints)) if n_bkps is None else n_bkps
    try:
        rpt_results = ruptures_changepoints(
            completions, block_size=block_size, beta=beta, n_bkps=rpt_n_bkps
        )
    except Exception as e:  # pragma: no cover - diagnostic must not break pipeline
        print(f"  (ruptures diagnostic failed for subject {subject}: {e})")
        rpt_results = {"raw": {}, "blocks": {}}

    print(
        f"Subject {subject}: {len(records)} attempts → "
        f"{len(changepoints)} phase transition(s) at {changepoints} "
        f"({[p['phase'] for p in phases]})"
    )

    out = output_dir / f"{subject}_changepoints.json"
    out.write_text(json.dumps({
        "subject": subject,
        "n_attempts": len(records),
        "changepoints": changepoints,
        "phases": phases,
        "method": "phase_annotation",
        "ruptures": {
            "model": "l2",
            "block_size": block_size,
            "beta": beta,
            "n_bkps": rpt_n_bkps,
            "results": rpt_results,
        },
    }, indent=2))


def category_attempt_series(records: list[dict], source_dir: Path) -> dict[str, list[dict]]:
    """Split a subject's chronological records into per-category sub-sequences.

    A scene can carry several design-pattern tags, so a single attempt may land
    in more than one category (mirrors load_scene_patterns()'s many-to-many
    mapping). Chronological order is preserved within each category because the
    input records are already sorted by (Session, Run, StartFrame, ClipCode).

    Returns {category: [records...]} for every category the subject played.
    """
    from analysis.clustering import load_scene_patterns

    patterns = load_scene_patterns(source_dir)
    scenes_by_category: dict[str, set[str]] = {}
    for cat, grp in patterns.groupby("pattern"):
        scenes_by_category[cat] = set(grp["scene_id"])

    by_cat: dict[str, list[dict]] = {}
    for r in records:
        scene = r.get("SceneID")
        for cat, scenes in scenes_by_category.items():
            if scene in scenes:
                by_cat.setdefault(cat, []).append(r)
    return by_cat


def combined_timeline_series(
    records: list[dict],
    min_scene_attempts: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Scene-detrended duration on one continuous chronological timeline.

    The per-category series are short and choppy; pooling raw Duration across
    scenes is useless because scenes take intrinsically different times, so the
    pooled signal tracks *which scene* an attempt is, not learning. Fix: z-score
    each attempt's Duration against the **same subject's** other attempts at the
    **same scene**, then keep every attempt in true chronological order. What
    survives is how the subject's speed drifts over the whole session relative
    to their own per-scene norm — a single pooled learning signal. (Scene is the
    right normalizer: it is unambiguous per attempt, and strictly finer than
    category, which is many-to-many.) Attempts at scenes with fewer than
    ``min_scene_attempts`` repeats (or zero variance) contribute a neutral 0.

    Returns (normalized_z, raw_durations), both in chronological order and the
    same length as ``records``.
    """
    from collections import defaultdict

    raw = np.array([float(r.get("Duration", 0.0)) for r in records])
    normalized = np.zeros(len(records))

    by_scene: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_scene[r.get("SceneID")].append(i)

    for idxs in by_scene.values():
        vals = raw[idxs]
        std = vals.std()
        if len(vals) >= min_scene_attempts and std > 0:
            normalized[idxs] = (vals - vals.mean()) / std
    return normalized, raw


def run_changepoint_by_category(
    subject: str,
    source_dir: Path,
    output_dir: Path,
    max_attempts: int | None = None,
    beta: float = 1.0,
    n_bkps: int = 1,
    block_size: int = 50,
) -> None:
    """Detect changepoints in one subject's per-scene-category attempt series.

    For each design-pattern category the subject played, the subject's
    chronological attempts at scenes of that category are extracted and the
    ruptures detector is run on BOTH signals:
      - duration:   Duration (time-to-complete) — a continuous learning signal
                    with real dynamic range, expected to shrink with mastery.
      - completion: the Bernoulli completed/death signal (near ceiling here).

    Input:   all *_summary.json for sub-{subject} + scenes_mastersheet.csv
    Output:  output_dir/<subject>_changepoints_by_category.json
             {
               "subject": str, "method": "ruptures_by_category",
               "model": "l2", "block_size": int, "beta": float, "n_bkps": int,
               "categories": {
                 "<category>": {
                   "n_attempts": int,
                   "duration":   {"raw": {...}, "blocks": {...}},
                   "completion": {"raw": {...}, "blocks": {...}}
                 }, ...
               }
             }

    Args:
        max_attempts: if set, truncate the subject to this many attempts (smoke)
        beta:         ruptures penalty scale (see _ruptures_detect)
        n_bkps:       fixed changepoint count for the Dynp algorithm
        block_size:   attempts per block for the aggregated signal
    """
    records = load_all_attempts(subject, source_dir)

    if not records:
        print(f"No data found for subject {subject}, skipping.")
        return

    if max_attempts is not None:
        records = records[:max_attempts]

    by_cat = category_attempt_series(records, source_dir)
    if not by_cat:
        print(f"Subject {subject}: no category-tagged attempts, skipping.")
        return

    categories: dict[str, dict] = {}
    for cat in sorted(by_cat):
        cat_records = by_cat[cat]
        durations = [float(r.get("Duration", 0.0)) for r in cat_records]
        completions = [1 if r.get("Outcome") == "completed" else 0 for r in cat_records]
        try:
            duration_res = _ruptures_detect(durations, block_size=block_size,
                                            beta=beta, n_bkps=n_bkps)
            completion_res = _ruptures_detect(completions, block_size=block_size,
                                              beta=beta, n_bkps=n_bkps)
        except Exception as e:  # pragma: no cover - one bad category must not sink the rest
            print(f"  (ruptures failed for {subject}/{cat}: {e})")
            duration_res = completion_res = {"raw": {}, "blocks": {}}
        categories[cat] = {
            "n_attempts": len(cat_records),
            "duration": duration_res,
            "completion": completion_res,
        }

    # Combined view: all attempts pooled onto one chronological timeline, each
    # Duration z-scored against the subject's own attempts at the same scene, so
    # changepoints reflect session-wide learning rather than 24 short choppy
    # per-category series. This is the signal that gives meaningful breakpoints.
    normalized, _raw = combined_timeline_series(records)
    try:
        combined_res = _ruptures_detect(normalized, block_size=block_size,
                                        beta=beta, n_bkps=n_bkps)
    except Exception as e:  # pragma: no cover
        print(f"  (ruptures failed for {subject}/combined: {e})")
        combined_res = {"raw": {}, "blocks": {}}

    preview = ", ".join(f"{c}:{categories[c]['n_attempts']}" for c in list(categories)[:5])
    print(
        f"Subject {subject}: {len(records)} attempts → "
        f"{len(categories)} categories ({preview}, ...); "
        f"combined-timeline changepoints (blocks, dynp): "
        f"{combined_res.get('blocks', {}).get('dynp', [])}"
    )

    out = output_dir / f"{subject}_changepoints_by_category.json"
    out.write_text(json.dumps({
        "subject": subject,
        "method": "ruptures_by_category",
        "model": "l2",
        "block_size": block_size,
        "beta": beta,
        "n_bkps": n_bkps,
        "categories": categories,
        "combined_timeline": {
            "n_attempts": len(records),
            "normalize": "scene_zscore",
            "results": combined_res,
        },
    }, indent=2))
