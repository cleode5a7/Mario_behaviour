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

    records.sort(key=lambda d: (d["Session"], d["Run"], d["StartFrame"]))
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


def run_changepoint(
    subject: str,
    source_dir: Path,
    output_dir: Path,
    max_attempts: int | None = None,
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
               "method": "phase_annotation"
             }

    Args:
        max_attempts: if set, truncate to this many attempts (smoke testing)
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
    }, indent=2))
