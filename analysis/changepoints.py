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


def rep_attempt_series(records: list[dict]) -> list[dict]:
    """Collapse per-scene clips into one record per level repetition.

    A ``.bk2`` repetition is one playthrough of a whole level (identified by
    ``SourceBk2``); the scene clips are its segments, and a repetition may span
    several lives. Aggregating to this unit is what rescues the completion
    signal: a scene clip is a few seconds of Mario and is cleared ~92% of the
    time, so the per-clip series sits at the ceiling that defeats the detectors
    (see phase_changepoints()). Per repetition the cleared rate is ~53-81%
    across subjects, with the dynamic range a mean-shift detector needs.

    Two cleared definitions are returned because they differ a lot and answer
    different questions:
      - ``cleared``:  the repetition reached the end of the level (its final
                      clip completed). Deaths along the way are allowed, since
                      the player respawns and carries on.
      - ``flawless``: every clip completed, i.e. the level was cleared without
                      dying once. A strictly harder criterion (~0.22-0.54).

    Input records must already be in chronological order (as load_all_attempts
    returns them); repetitions come back ordered by their first clip.

    Returns [{"source_bk2", "level", "phase", "session", "run", "start_frame",
              "n_clips", "n_deaths", "cleared", "flawless", "duration"}, ...]
    """
    by_rep: dict[str, list[dict]] = {}
    for r in records:
        by_rep.setdefault(r.get("SourceBk2", ""), []).append(r)

    reps: list[dict] = []
    for source_bk2, clips in by_rep.items():
        clips = sorted(clips, key=lambda d: (d["Session"], d["Run"], d["StartFrame"],
                                             d.get("ClipCode", "")))
        first, last = clips[0], clips[-1]
        n_deaths = sum(1 for c in clips if c.get("Outcome") != "completed")
        reps.append({
            "source_bk2": source_bk2,
            "level": first.get("Level"),
            "phase": first.get("Phase"),
            "session": first["Session"],
            "run": first["Run"],
            "start_frame": first["StartFrame"],
            "n_clips": len(clips),
            "n_deaths": n_deaths,
            "cleared": 1 if last.get("Outcome") == "completed" else 0,
            "flawless": 1 if n_deaths == 0 else 0,
            "duration": sum(float(c.get("Duration", 0.0)) for c in clips),
        })

    reps.sort(key=lambda d: (d["session"], d["run"], d["start_frame"]))
    return reps


def phase_changepoints(records: list[dict], key: str = "Phase") -> list[int]:
    """Return attempt indices where the Phase annotation changes.

    Reads the 'Phase' field from each summary record (e.g. 'discovery',
    'practice') and returns the index of the first attempt in each new phase.

    This is the primary changepoint method: the Phase annotations encode the
    experimenters' protocol boundaries (e.g. discovery → practice). The
    Bayesian changepoint detector cannot recover these from the completion
    signal alone because subjects' per-scene-clip completion rates are
    near-ceiling (~90%) throughout both phases, leaving no detectable
    step-change. Aggregating to the level repetition removes that ceiling —
    see rep_attempt_series().

    ``key`` selects the field holding the phase label, so the same function
    serves both raw summary records ('Phase') and the repetition records built
    by rep_attempt_series() ('phase'); indices are then in rep-index space.
    """
    phases = [r.get(key) for r in records]
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


def level_adjusted_series(reps: list[dict], key: str = "cleared") -> tuple[np.ndarray, dict]:
    """Per-repetition outcome minus that level's own average — difficulty removed, time kept.

    The problem this solves: during discovery the subject walks through the game
    in world order, so "later in the session" and "harder level" are the same
    thing. A raw chronological cleared rate therefore *falls* (77% → 33% for
    sub-01) and reads as un-learning when it is really the difficulty gradient.

    Subtracting each level's own session-wide mean leaves, for every repetition,
    "how did this attempt compare to how this subject usually does on *this*
    level" — a residual in percentage points that keeps the true chronological
    order intact. Levels are averaged over the **whole session** (not per phase),
    so discovery and practice share one scale and the phase difference survives:
    all five subjects sit ~5-12 points below their level average in discovery and
    ~2-6 points above it in practice.

    Prefer this for *reporting* an effect size over time. For *detection* use the
    raw ``cleared`` series — the subtraction removes much of the mean shift the
    detector keys on, and pushes ruptures' breakpoint from within 0-14
    repetitions of the annotated boundary to 60-80 away.

    Returns (residuals in the same order as ``reps``, {level: mean}).
    """
    means: dict[str, list[int]] = {}
    for r in reps:
        means.setdefault(r.get("level"), []).append(r[key])
    level_means = {k: float(np.mean(v)) for k, v in means.items()}
    return (np.array([r[key] - level_means[r.get("level")] for r in reps]),
            level_means)


def phase_thirds(reps: list[dict], values) -> dict[str, list[float]]:
    """Split ``values`` into Early/Mid/Late by **true chronological order** per phase.

    Unlike phase_thirds_cleared(..., within_level=True), the bins here are real
    slabs of time: Early is the first third of the phase as it was actually
    played. Pair it with level_adjusted_series() to get a time axis that is both
    honest about chronology and free of the world-difficulty confound.

    **Only the practice thirds are comparable to each other.** In practice every
    third contains all 22 levels, so level mix is constant across bins and a
    change between them is a real change in performance. In discovery each third
    contains a *different, disjoint* set of levels (the subject walks the game in
    world order, playing each level once), so bins differ in what was played, not
    only in when — and no across-third comparison is valid there, adjusted or not.

    Adjusting does not rescue discovery, it just hides the problem differently.
    With level means taken per phase the discovery bins collapse to ~0.0 by
    construction. With session-wide means (what level_adjusted_series does) they
    do not collapse, and for some subjects appear to rise — but that rise tracks
    a floor effect, not learning: late-discovery levels are the hard ones, which
    have no room to improve (sub-02's world-8 levels gain -2/-8/+2 pp from
    discovery to practice while w3l2 gains +75). Level practice-rate vs
    discovery->practice gap correlates r=0.45; early-discovery levels average
    +36 pp of headroom against +13 pp for late-discovery ones.
    """
    values = np.asarray(values, dtype=float)
    out: dict[str, list[float]] = {}
    for phase in dict.fromkeys(r.get("phase") for r in reps):
        idx = np.array([i for i, r in enumerate(reps) if r.get("phase") == phase])
        if not len(idx):
            continue
        out[phase] = [float(values[part].mean()) if len(part) else float("nan")
                      for part in np.array_split(idx, 3)]
    return out


def phase_thirds_cleared(
    reps: list[dict],
    key: str = "cleared",
    within_level: bool = True,
) -> dict[str, list[float]]:
    """Early/Mid/Late cleared rate within each phase — the lab's summary view.

    Reproduces the "Cleared rate (%)" summary our collaborators plot: each phase
    (discovery, practice) split into three bins, one rate per bin.

    ``within_level=True`` (default) splits **each level's** repetitions into
    thirds and pools across levels. This is the binning that matters: during
    discovery the subject advances through progressively harder worlds, so plain
    chronological thirds make the cleared rate *fall* (77% → 66% → 33% for
    sub-01) — a difficulty effect masquerading as un-learning. Binning within
    level removes the confound and the rate rises (52% → 61% → 64%), matching
    the collaborators' figure. Pass ``within_level=False`` to get the raw
    chronological thirds and see the confound for yourself.

    NOT a changepoint input. The bins are cut *at* the phase boundary, so a
    detector run on this series would be handed the very transition it is meant
    to discover — and six points cannot be segmented anyway. Use the
    per-repetition series for detection and this for reporting.

    Returns {phase: [early, mid, late]} with rates in 0-1.
    """
    out: dict[str, list[float]] = {}
    for phase in dict.fromkeys(r.get("phase") for r in reps):
        in_phase = [r for r in reps if r.get("phase") == phase]
        if not in_phase:
            continue
        if within_level:
            groups: dict[str, list[int]] = {}
            for r in in_phase:
                groups.setdefault(r.get("level"), []).append(r[key])
            bins: list[list[int]] = [[], [], []]
            for seq in groups.values():
                for i, part in enumerate(np.array_split(np.array(seq), 3)):
                    bins[i].extend(part.tolist())
        else:
            bins = [part.tolist() for part in
                    np.array_split(np.array([r[key] for r in in_phase]), 3)]
        out[phase] = [float(np.mean(b)) if b else float("nan") for b in bins]
    return out


def _phase_segments(records: list[dict], changepoints: list[int], key: str) -> list[dict]:
    """Turn phase-transition indices into [{"phase", "start", "end"}, ...] spans."""
    boundaries = [0] + changepoints + [len(records)]
    return [
        {"phase": records[start].get(key), "start": start, "end": end}
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


def run_changepoint(
    subject: str,
    source_dir: Path,
    output_dir: Path,
    max_attempts: int | None = None,
    beta: float = 1.0,
    n_bkps: int | None = None,
    block_size: int = 50,
    rep_block_size: int = 10,
) -> None:
    """Detect learning-phase changepoints for one subject.

    Primary method: Phase-annotation transitions extracted from each record's
    'Phase' field (e.g. 'discovery' → 'practice'). These reflect the
    experimenters' protocol boundaries, which are the ground-truth learning
    phases for this dataset.

    Runs the ruptures detector on two units:
      - per scene clip (``ruptures``), near-ceiling at ~92% and largely flat;
      - per level repetition (``repetitions``), ~53-81% cleared with real
        dynamic range — the signal that actually tracks learning.

    Input:   all *_summary.json for sub-{subject} in mario.scenes/
    Output:  output_dir/<subject>_changepoints.json
             {
               "subject": str,
               "n_attempts": int,
               "changepoints": [<attempt indices>],
               "phases": [{"phase": str, "start": int, "end": int}, ...],
               "method": "phase_annotation",
               "ruptures": {  # per scene clip; non-Bayesian diagnostic
                 "model": "l2", "block_size": int, "beta": float, "n_bkps": int,
                 "results": {"raw": {...}, "blocks": {...}}
               },
               "repetitions": {  # per level repetition — the un-ceilinged signal
                 "n_reps": int, "block_size": int, "n_bkps": int,
                 "cleared_rate": float, "flawless_rate": float,
                 "changepoints": [<rep indices of Phase transitions>],
                 "phases": [{"phase": str, "start": int, "end": int}, ...],
                 "results": {"cleared": {...}, "flawless": {...}}
               }
             }

    Args:
        max_attempts:   if set, truncate to this many attempts (smoke testing)
        beta:           ruptures penalty scale (see _ruptures_detect)
        n_bkps:         fixed changepoint count for the Dynp algorithm
        block_size:     scene clips per block for the aggregated ruptures signal
        rep_block_size: repetitions per block for the repetition signal. Much
                        smaller than block_size because a subject has ~600-800
                        reps vs ~10-13k clips; 10 was picked by sweeping
                        {10,15,20,25,40} — it recovers the annotated phase
                        transition to within 0-14 reps for every subject, and
                        coarser blocks only quantize the answer away.
    """
    records = load_all_attempts(subject, source_dir)

    if not records:
        print(f"No data found for subject {subject}, skipping.")
        return

    if max_attempts is not None:
        records = records[:max_attempts]

    changepoints = phase_changepoints(records)
    phases = _phase_segments(records, changepoints, "Phase")

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

    # Level-repetition view: the same completion idea one level up, where the
    # rate is ~53-81% instead of pinned at ~92%. Phase boundaries are recomputed
    # in rep-index space so they overlay this signal's own x-axis.
    reps = rep_attempt_series(records)
    rep_changepoints = phase_changepoints(reps, key="phase")
    level_adjusted, level_means = level_adjusted_series(reps)
    cleared = [r["cleared"] for r in reps]
    flawless = [r["flawless"] for r in reps]
    rep_bkps = max(1, len(rep_changepoints)) if n_bkps is None else n_bkps
    try:
        rep_results = {
            "cleared": _ruptures_detect(cleared, block_size=rep_block_size,
                                        beta=beta, n_bkps=rep_bkps),
            "flawless": _ruptures_detect(flawless, block_size=rep_block_size,
                                         beta=beta, n_bkps=rep_bkps),
        }
    except Exception as e:  # pragma: no cover - diagnostic must not break pipeline
        print(f"  (repetition ruptures failed for subject {subject}: {e})")
        rep_results = {"cleared": {"raw": {}, "blocks": {}},
                       "flawless": {"raw": {}, "blocks": {}}}

    print(
        f"Subject {subject}: {len(records)} attempts → "
        f"{len(changepoints)} phase transition(s) at {changepoints} "
        f"({[p['phase'] for p in phases]})"
    )
    print(
        f"  {len(reps)} level repetitions: cleared {np.mean(cleared):.3f}, "
        f"flawless {np.mean(flawless):.3f}; "
        f"cleared changepoints (blocks, dynp): "
        f"{rep_results['cleared'].get('blocks', {}).get('dynp', [])} "
        f"vs phase transition(s) at {rep_changepoints}"
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
        "repetitions": {
            "n_reps": len(reps),
            "unit": "level_repetition",
            "block_size": rep_block_size,
            "beta": beta,
            "n_bkps": rep_bkps,
            "cleared_rate": float(np.mean(cleared)) if cleared else 0.0,
            "flawless_rate": float(np.mean(flawless)) if flawless else 0.0,
            "changepoints": rep_changepoints,
            "phases": _phase_segments(reps, rep_changepoints, "phase"),
            "results": rep_results,
            # Reporting views only — see phase_thirds_cleared() on why these bins
            # must never be fed to a detector.
            "phase_thirds": {
                # True chronological slabs: the primary reporting view.
                "chronological": {
                    "cleared": phase_thirds_cleared(reps, "cleared", within_level=False),
                    "flawless": phase_thirds_cleared(reps, "flawless", within_level=False),
                    # difficulty removed, chronology kept — flat in discovery by
                    # construction (see phase_thirds), real signal in practice
                    "level_adjusted": phase_thirds(reps, level_adjusted),
                },
                # NOT a time axis: bins are stages of practice at a level, pooled
                # from across the whole session. Kept because it is the only
                # within-discovery learning measure the design permits.
                "within_level": {
                    "cleared": phase_thirds_cleared(reps, "cleared"),
                    "flawless": phase_thirds_cleared(reps, "flawless"),
                },
            },
            "level_adjusted": {
                "definition": "cleared minus that level's session-wide mean, per repetition",
                "mean_by_phase": {
                    p["phase"]: float(level_adjusted[p["start"]:p["end"]].mean())
                    for p in _phase_segments(reps, rep_changepoints, "phase")
                },
                "level_means": level_means,
            },
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
