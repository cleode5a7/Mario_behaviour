"""Derive a per-clip time index and a phase-third label for a scene's traces.

Two quantities the dashboard needs and the raw ``*_summary.json`` records do not
carry:

  * **attempt index** — the *n*th entry this subject made into this scene, 1-based
    and chronological across the whole experiment.
  * **stage tertile** — early / middle / late *within* the clip's phase.

Both are derived from ``ClipCode``, which is
``session(3) ‖ run(2) ‖ bk2-ordinal(2) ‖ start-frame(7)`` and therefore monotone
in time within a subject. Order is derived explicitly rather than taken from the
incoming list order, so these functions are safe to run over an
already-exported payload whose order is whatever the exporter happened to use.

Why not the ``rep-NNN`` in ``SourceBk2``: it **restarts at zero every session**
(sub-01 plays w1l1 as rep 0->10 inside ses-001, then 0,1,2 again in ses-007, then
0,1 in ses-008), so it is not a time axis. It also collapses re-entries — one
``.bk2`` can produce several clips of the same scene when the player dies and
respawns back into it — whereas an attempt index counts each entry separately,
which is the unit the trajectory plot actually draws.
"""

from collections import defaultdict

import numpy as np

# Legend/report order. The stored field is only the tertile letter; the full
# stage name is ``tertile + "_" + phase``, since Phase travels with every record.
STAGES = ["early_discovery", "middle_discovery", "late_discovery",
          "early_practice", "middle_practice", "late_practice"]
TERTILES = ("e", "m", "l")
TERTILE_NAMES = {"e": "early", "m": "middle", "l": "late"}


def _ordering_values(metadata, idx: list[int]) -> list[float]:
    """Monotone-in-time sort key for the clips at ``idx``.

    ``ClipCode`` when every clip in the group has a parseable one; otherwise the
    group falls back to list position for *all* of its members, so a single bad
    record cannot interleave tiny positional keys with 14-digit clip codes.
    """
    codes = []
    for i in idx:
        try:
            codes.append(float(str(metadata[i].get("ClipCode", "")).strip()))
        except ValueError:
            return [float(i) for i in idx]
    return codes


def attempt_index(metadata) -> list[int]:
    """1-based nth attempt at this scene, numbered per subject.

    Every clip is one attempt, including a re-entry after a death inside the same
    playthrough. Returned aligned to ``metadata``.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, meta in enumerate(metadata):
        groups[meta.get("Subject", "?")].append(i)

    out = [0] * len(metadata)
    for idx in groups.values():
        values = _ordering_values(metadata, idx)
        order = sorted(range(len(idx)), key=lambda k: (values[k], idx[k]))
        for n, k in enumerate(order, start=1):
            out[idx[k]] = n
    return out


def stage_tertiles(metadata) -> list[str]:
    """Early/middle/late tertile within each clip's phase: "e", "m", "l" or "".

    Port of ``add_stage_column`` in the sibling mario_learning.behavior repo
    (``src/mario_learning/utils.py``): cut each group at the 1/3 and 2/3 quantiles
    of ``ClipCode``, strict ``<`` on both boundaries, so a clip sitting exactly on
    a boundary falls in the later bin.

    The sibling groups by ``(Subject, Phase, Level)``; here the key is just
    ``(Subject, Phase)`` because a scene sits inside exactly one level, so the
    Level term is already constant. That constancy is also what makes the
    early/middle/late split legitimate *within discovery* for this view — the
    level mix cannot shift between bins the way it does when discovery thirds are
    taken across a whole session (see the collinearity note in CLAUDE.md).

    Clips with no Phase get "". A group of one clip is degenerate (both quantiles
    equal its own code, so it lands in "l"); this matches the sibling's behaviour.
    """
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, meta in enumerate(metadata):
        phase = (meta.get("Phase") or "").strip()
        if phase:
            groups[(meta.get("Subject", "?"), phase)].append(i)

    out = [""] * len(metadata)
    for idx in groups.values():
        values = _ordering_values(metadata, idx)
        q33 = np.quantile(values, 1 / 3)
        q67 = np.quantile(values, 2 / 3)
        for k, i in enumerate(idx):
            v = values[k]
            out[i] = "e" if v < q33 else "m" if v < q67 else "l"
    return out
