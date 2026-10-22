"""Export clustered traces to a static payload the browser dashboard can read.

The stored cluster results (``output_data/clusters/<scene>_clusters.json``) hold
only ``labels`` + per-clip ``metadata`` — the x/y polylines are rebuilt on every
plot from ``source_data/mario.scenes/`` (a 23 GB datalad dataset). A browser
cannot read that, so this module bakes the geometry the figures already draw
into small per-scene JSON files.

Design rules:
  * **Export once, serve statically.** The dashboard must never touch
    ``mario.scenes`` at runtime — background PNGs are copied into the payload too.
  * **Same figure, different medium.** Geometry comes from
    ``load_labeled_traces`` (screen-corrected y, labels joined by clip key) and
    the axis limits from ``playfield_ylim``/``_shared_xlim``, so the dashboard
    and the PNGs in ``output_data/clusters/`` render the identical view.
  * **Labels are never re-derived.** The dashboard's subject/phase filters only
    hide traces; the cluster assignment shipped here is the one HDBSCAN produced
    over the full pooled set.
"""

from pathlib import Path
import json
import shutil

import numpy as np

from analysis.clustering import load_scene_patterns
from analysis.stages import attempt_index, stage_tertiles
from analysis.trace_plots import (
    _shared_xlim,
    default_background_dir,
    load_labeled_traces,
    playfield_ylim,
    resolve_background,
    scene_entry_map,
)
from analysis.stability import coassociation_path

# Per-clip summary fields carried into the payload for the hover tooltip, mapped
# to short keys (these arrays are the bulk of the file — every byte is repeated
# once per trace, and there are 55k traces in the corpus).
EXTRA_FIELDS = {
    "X_Traveled": "xt",
    "Hits_taken": "ht",
    "Enemies_killed": "ek",
    "Lives_lost": "ll",
}


def list_clustered_scenes(output_dir: Path) -> list[str]:
    """Scene IDs that have stored cluster labels, sorted.

    Not ``changepoints.list_scenes`` — a scene with no ``_clusters.json`` has
    nothing for the dashboard to show.
    """
    clusters_dir = Path(output_dir) / "clusters"
    if not clusters_dir.exists():
        return []
    return sorted(p.name[: -len("_clusters.json")]
                  for p in clusters_dir.glob("*_clusters.json"))


def add_derived(out_traces: list[dict]) -> None:
    """Attach the attempt index and stage tertile to payload records, in place.

    Depends only on ``s`` / ``ph`` / ``clip``, which are already on every exported
    record — that is what lets ``backfill_derived`` add these fields to an
    existing payload without re-reading the 23 GB source dataset.

    Adds:
        "a"  1-based nth attempt this subject made at this scene (chronological)
        "st" phase tertile, "e" / "m" / "l", or "" when Phase is missing. The full
             stage name is ``st + "_" + ph``; storing only the letter keeps the
             field one byte, and Phase and Stage stay two independent filters in
             the dashboard, which is exactly how the UI crosses them.
    """
    meta = [{"Subject": t.get("s", "?"), "Phase": t.get("ph", ""),
             "ClipCode": t.get("clip", "")} for t in out_traces]
    for t, a, st in zip(out_traces, attempt_index(meta), stage_tertiles(meta)):
        t["a"] = int(a)
        t["st"] = st


def _downsample(trace: np.ndarray, step: int, x_offset: int = 0) -> tuple[list, list]:
    """Thin a trace to every ``step``-th frame, always keeping first and last.

    Traces are sampled at 60 fps against a 224 px-tall background, so sub-pixel
    frame-by-frame fidelity is invisible and just inflates the payload. Masked
    frames (``player_state == 0``, hidden by the "screen" y-mode) become ``None``
    so the renderer can break the polyline there instead of drawing a false
    segment across the gap.
    """
    n = len(trace)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)

    xs = np.asarray(trace[:, 0])[idx]
    ys = trace[:, 1][idx]
    mask = np.ma.getmaskarray(ys) if np.ma.isMaskedArray(ys) else np.zeros(len(idx), bool)
    ys = np.ma.getdata(ys)

    x_out = [int(round(float(v))) - x_offset for v in xs]
    y_out = [None if m else int(round(float(v))) for v, m in zip(ys, mask)]
    return x_out, y_out


def export_coassociation(scene_id, source_dir, output_dir, web_dir, out_traces) -> bool:
    """Write one scene's co-association matrix for the browser. True if written.

    Two things happen here, both necessary.

    **Downcast to uint8.** The stored matrix is uint16 because the analysis copy has
    to re-derive the labels exactly; the browser only paints it, so one byte per cell
    halves the transfer and is still finer than a screen pixel can show.

    **Permute into payload order.** The matrix rows follow the cluster JSON's metadata
    order, and ``out_traces`` follows ``load_traces`` order. Those happen to coincide
    today, but the whole point of CLIP_KEY_FIELDS is that nothing may *rely* on that.
    So the permutation is built by clip key, and a scene whose keys do not line up is
    skipped rather than silently transposed into nonsense.
    """
    import json

    src = coassociation_path(scene_id, output_dir)
    if not src.exists():
        return False

    stored = np.load(src)
    clusters = Path(output_dir) / "clusters" / f"{scene_id}_clusters.json"
    meta = json.loads(clusters.read_text())["metadata"]
    row_of = {(m.get("Subject"), m.get("Session"), m.get("Run"), m.get("ClipCode")): i
              for i, m in enumerate(meta)}

    order = [row_of.get((t["s"], t["ses"], t["run"], t["clip"])) for t in out_traces]
    if any(o is None for o in order) or len(stored) != len(meta):
        print(f"  {scene_id}: co-association rows do not match the payload, skipping matrix.")
        return False

    order = np.asarray(order)
    small = (stored[np.ix_(order, order)] // 257).astype(np.uint8)   # 65535 -> 255

    out_dir = Path(web_dir) / "coassoc"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{scene_id}.bin").write_bytes(small.tobytes())
    return True


def _clip_row_map(scene_id, output_dir) -> dict:
    """clip key -> row index in the cluster file's own order."""
    path = Path(output_dir) / "clusters" / f"{scene_id}_clusters.json"
    meta = json.loads(path.read_text())["metadata"]
    return {(m.get("Subject"), m.get("Session"), m.get("Run"), m.get("ClipCode")): i
            for i, m in enumerate(meta)}


def leaf_order(scene_id, output_dir, out_traces, consensus) -> list[int] | None:
    """The dendrogram leaf order, expressed as **payload** indices.

    Two translations happen here, and both are the point.

    **Cluster-file order -> payload order.** ``leaves`` indexes the matrix the tree was
    built on, which follows the cluster file's metadata; the payload follows
    ``load_traces``. Those coincide today, but the whole reason CLIP_KEY_FIELDS exists
    is that nothing may rely on it — so the permutation goes through the clip key, the
    same way ``export_coassociation`` builds its own, and a scene whose keys do not line
    up returns None rather than a plausible-looking wrong ordering.

    **Reconstruction for older files.** A cluster file written before ``leaves`` existed
    has none, so the tree is rebuilt here from the stored co-association matrix with the
    settings that file recorded. ``_consensus_tree`` is a pure function of ``C`` and
    those two settings, so this is the same tree, not an approximation — the same reason
    ``run-coassociation`` can regenerate the matrix itself exactly. It costs one linkage
    on a few hundred points.
    """
    from analysis.stability import _consensus_tree, coassociation_path
    from scipy.cluster.hierarchy import leaves_list

    leaves = consensus.get("leaves")
    if leaves is None:
        src = coassociation_path(scene_id, output_dir)
        if not src.exists():
            return None
        C = np.load(src).astype(float) / 65535.0
        Z, _ = _consensus_tree(C,
                               input=consensus.get("input", "features"),
                               linkage_method=consensus.get("linkage", "ward"))
        leaves = [int(i) for i in leaves_list(Z)]

    row_of = _clip_row_map(scene_id, output_dir)
    payload_of = {}
    for p, t in enumerate(out_traces):
        row = row_of.get((t["s"], t["ses"], t["run"], t["clip"]))
        if row is None:
            return None
        payload_of[row] = p
    if len(payload_of) != len(leaves):
        return None
    try:
        return [payload_of[i] for i in leaves]
    except KeyError:
        return None


def add_candidate_cuts(scene_id, output_dir, report) -> None:
    """Give every candidate in the curve its ``bounds``, rebuilding them if needed.

    ``bounds`` is a cut expressed as ``k - 1`` split positions in the dendrogram leaf
    order (see ``stability.cut_bounds``), which is what lets the payload carry *every*
    candidate partition for a few hundred bytes and the dashboard offer a cut other
    than the one that was selected. Without it the browser could only draw the chosen k.

    Files written before the bounds were recorded get them rebuilt here from the stored
    co-association matrix, exactly as ``leaf_order`` rebuilds the leaf order and for the
    same reason: ``_consensus_tree`` and ``_candidate_cuts`` are pure functions of ``C``
    and the settings the file records, so this is the same candidate list, not a second
    one. A scene with no stored matrix keeps its curve and simply offers no alternative
    cuts — the dashboard falls back to the chosen labels.
    """
    from analysis.stability import consensus_cut_bounds

    rows = report.get("k_candidates") or []
    if not rows or all("bounds" in r for r in rows):
        return
    src = coassociation_path(scene_id, output_dir)
    if not src.exists():
        return
    C = np.load(src).astype(float) / 65535.0
    bounds = consensus_cut_bounds(
        C,
        input=report.get("input", "features"),
        linkage_method=report.get("linkage", "ward"),
        k_max=report.get("k_max", 15),
        min_cluster_size=report.get("min_cluster_size", 1),
    )
    for row in rows:
        if "bounds" not in row and row["k"] in bounds:
            row["bounds"] = bounds[row["k"]]


def selection_report(scene_id, output_dir, out_traces=None) -> dict | None:
    """The k-selection record for one scene, read back from its cluster file.

    Everything here already exists in ``<scene>_clusters.json`` — the ``params``
    block and ``stability.consensus`` report ``ward_consensus_labels`` returns. It is
    lifted into the payload so the dashboard can show *how* k was chosen without the
    browser ever seeing the cluster files, which is the same export-once rule the
    trace geometry follows.

    Returns None for a single-fit run, which has no ensemble and so no selection to
    report. The whole silhouette-vs-k curve is at most ``k_max`` rows, so this costs a
    few hundred bytes a scene.
    """
    path = Path(output_dir) / "clusters" / f"{scene_id}_clusters.json"
    if not path.exists():
        return None
    stored = json.loads(path.read_text())
    consensus = (stored.get("stability") or {}).get("consensus")
    if not consensus:
        return None

    stability = stored["stability"]
    report = dict(consensus)
    # Ensemble-level numbers the curve should be read against: a cut is only as
    # meaningful as the replicates it summarises.
    report["stable_share"] = stability.get("stable_share")
    report["k_replicates"] = stability.get("k_replicates")
    report["noise_frac"] = stability.get("noise_frac")
    params = stored.get("params") or {}
    report["bootstraps"] = params.get("bootstraps")
    report["frac"] = params.get("frac")
    report["scheme"] = params.get("scheme")

    # Re-expressed in payload indices, so the browser can apply it directly to the
    # matrix it was served without knowing anything about cluster-file ordering.
    if out_traces is not None and report.get("consensus_cluster") == "ward":
        report["leaves"] = leaf_order(scene_id, output_dir, out_traces, report)
        add_candidate_cuts(scene_id, output_dir, report)
        # The elbow is a pure function of the stored curve, so a file written before the
        # rule existed still gets one — the same reconstruction the leaf order gets, and
        # unlike the leaf order it needs nothing but the curve itself. It is a reading,
        # not a re-analysis: `selection` still says which rule made these labels.
        if report.get("elbow") is None and report.get("k_candidates"):
            from analysis.stability import elbow_k
            report["elbow"] = elbow_k(report["k_candidates"])
    else:
        report.pop("leaves", None)

    # The bounds are only usable alongside the leaf order they index into. Shipping
    # them without it would let the browser decode positions against payload order and
    # paint a partition that never existed.
    if not report.get("leaves"):
        for row in report.get("k_candidates") or []:
            row.pop("bounds", None)
    return report


def export_scene(scene_id, source_dir, output_dir, web_dir, *, step=3,
                 entry_map=None, background_dir=None) -> dict | None:
    """Write one scene's dashboard payload; return its index entry (None if empty)."""
    source_dir, output_dir, web_dir = Path(source_dir), Path(output_dir), Path(web_dir)
    traces, metadata, labels, method, stability = load_labeled_traces(scene_id, source_dir, output_dir)
    if not traces:
        print(f"No trace data found for scene {scene_id}, skipping export.")
        return None

    if entry_map is None:
        entry_map = scene_entry_map(source_dir)
    bg_path = resolve_background(scene_id, background_dir)

    # x is offset into scene-local pixel space only when a background is drawn —
    # exactly the condition plot_scene_clusters uses. The offset is applied to the
    # exported coordinates themselves (not left for the browser), so `x`, `xlim`
    # and `bg_size` are all in one consistent space.
    entry = int(entry_map.get(scene_id, 0)) if bg_path is not None else 0

    bg_size = None
    if bg_path is not None:
        from PIL import Image
        with Image.open(bg_path) as im:
            bg_size = [im.width, im.height]

    xmin, xmax = _shared_xlim(traces)
    ylo, yhi = playfield_ylim(traces)

    # Per-trace stability is NOT derivable from the payload (unlike `a`/`st` in
    # add_derived), so adding it is a real re-export, never a --backfill.
    stab = stability if stability is not None else [None] * len(traces)

    out_traces = []
    for trace, meta, label, sb in zip(traces, metadata, labels, stab):
        x, y = _downsample(trace, step, x_offset=entry)
        rec = {
            "s": meta.get("Subject", "?"),
            "ses": meta.get("Session", ""),
            "run": meta.get("Run", ""),
            "clip": meta.get("ClipCode", ""),
            "sf": meta.get("StartFrame", 0),
            "ph": meta.get("Phase", ""),
            "out": meta.get("Outcome", ""),
            "dur": meta.get("Duration", None),
            "c": int(label),
            "x": x,
            "y": y,
        }
        if sb is not None:
            rec["sb"] = round(float(sb), 2)
        rec.update({short: meta.get(field) for field, short in EXTRA_FIELDS.items()})
        out_traces.append(rec)

    add_derived(out_traces)

    payload = {
        "scene": scene_id,
        "method": method,
        "entry": entry,
        "bg": f"{scene_id}.png" if bg_path is not None else None,
        "bg_size": bg_size,
        "xlim": [round(xmin - entry, 2), round(xmax - entry, 2)],
        "ylim": [round(ylo, 2), round(yhi, 2)],
        "step": step,
        "traces": out_traces,
    }

    payload["cons"] = selection_report(scene_id, output_dir, out_traces)

    payload["coassoc"] = export_coassociation(
        scene_id, source_dir, output_dir, web_dir, out_traces)

    scenes_dir = web_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    out_path = scenes_dir / f"{scene_id}.json"
    out_path.write_text(json.dumps(payload, separators=(",", ":")))

    kb = out_path.stat().st_size / 1024
    n_clusters = len({int(l) for l in labels if l != -1})
    print(f"Exported {scene_id}: {len(out_traces)} traces, {n_clusters} clusters, {kb:.0f} KB")

    if bg_path is not None:
        bg_dir = web_dir / "backgrounds"
        bg_dir.mkdir(parents=True, exist_ok=True)
        dest = bg_dir / f"{scene_id}.png"
        if not dest.exists():
            shutil.copy2(bg_path, dest)

    return _index_entry(scene_id, out_traces, method, payload.get("cons"))


def _index_entry(scene_id: str, out_traces: list[dict], method: str,
                 cons: dict | None = None) -> dict:
    """Summary row for index.json, derived from the payload we just built."""
    labels = [t["c"] for t in out_traces]
    n_noise = sum(1 for l in labels if l == -1)
    stabilities = [t["sb"] for t in out_traces if t.get("sb") is not None]
    world, rest = scene_id[1:].split("l", 1)
    level, scene_no = rest.split("s", 1)
    return {
        "id": scene_id,
        "world": int(world),
        "level": int(level),
        "scene": int(scene_no),
        "n": len(out_traces),
        "k": len({l for l in labels if l != -1}),
        "noise": round(n_noise / len(out_traces), 3) if out_traces else 0.0,
        "sb": round(sum(stabilities) / len(stabilities), 3) if stabilities else None,
        "subjects": sorted({t["s"] for t in out_traces}),
        "phases": sorted({t["ph"] for t in out_traces if t["ph"]}),
        "method": method,
        # The silhouette of the cut that was taken, so the scene picker can sort and
        # filter on it without opening a payload.
        "sil": (cons or {}).get("silhouette"),
        "sel": (cons or {}).get("selection"),
    }


def _index_entry_from_file(path: Path) -> dict:
    """Rebuild an index row from an already-exported scene file (skip path)."""
    payload = json.loads(path.read_text())
    return _index_entry(payload["scene"], payload["traces"],
                        payload.get("method", ""), payload.get("cons"))


def write_index(web_dir: Path, source_dir: Path) -> Path:
    """(Re)write index.json from every exported scene file.

    Cheap enough to redo on every run, so the index can never drift from the
    scene files beside it. Pattern tags are joined in here so the scene picker
    can filter by design category without loading any scene payload.
    """
    web_dir, source_dir = Path(web_dir), Path(source_dir)
    scenes_dir = web_dir / "scenes"
    entries = [_index_entry_from_file(p) for p in sorted(scenes_dir.glob("*.json"))]

    tags: dict[str, list[str]] = {}
    try:
        patterns = load_scene_patterns(source_dir)
        for scene_id, group in patterns.groupby("scene_id"):
            tags[scene_id] = sorted(group["pattern"].tolist())
    except FileNotFoundError:
        print("  warning: scenes_mastersheet.csv not found — no pattern tags in index")

    for e in entries:
        e["tags"] = tags.get(e["id"], [])

    entries.sort(key=lambda e: (e["world"], e["level"], e["scene"]))
    index = {
        "scenes": entries,
        "subjects": sorted({s for e in entries for s in e["subjects"]}),
        "phases": sorted({p for e in entries for p in e["phases"]}),
        "tags": sorted({t for e in entries for t in e["tags"]}),
        "methods": sorted({e["method"] for e in entries if e["method"]}),
    }
    out_path = web_dir / "index.json"
    out_path.write_text(json.dumps(index, separators=(",", ":")))
    print(f"Wrote index -> {out_path} ({len(entries)} scenes)")
    return out_path


def _rounded(obj, nd=6):
    """Round every float in a nested JSON value — the payload is for drawing."""
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _rounded(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rounded(v, nd) for v in obj]
    return obj


# Bootstrap results are exploratory and live under docs/ (docs/practice_scene_bootstrap.py);
# the export reads them when present so the dashboard's summary view can show them.
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
SUMMARY_METHODS = ("lme4", "model", "nested", "scene_boot", "level_boot")


def _holm(p: dict) -> dict:
    """Holm step-down adjustment of {key: p}."""
    adj, running = {}, 0.0
    for rank, k in enumerate(sorted(p, key=p.get)):
        running = max(running, min(1.0, (len(p) - rank) * p[k]))
        adj[k] = running
    return adj


def practice_summary(output_dir: Path, docs_dir: Path = DOCS_DIR) -> dict | None:
    """Every CI and p-value for the primary measure, all attempts, per transform x subject.

    The table behind docs/practice_summary.png, for the dashboard's summary view:
    ``lme4`` (the scene-scale model: profile CI, Wald p, and ``p_holm`` across
    subjects), ``model`` (statsmodels scene-only model), ``nested`` (statsmodels nested, no
    scene correlation), and, when ``docs/practice_*_bootstrap*.csv`` exist, ``scene_boot``
    and ``level_boot`` (percentile CI; p = 2 x min(share >= 0, share <= 0), floored at the
    replicate resolution and flagged ``p_floor``). Only ``log`` practice form.
    """
    import pandas as pd

    path = Path(output_dir) / "practice" / "mixed_summary.csv"
    if not path.exists():
        return None
    s = pd.read_csv(path, dtype={"subject": str})
    primary = s.loc[s["primary"].astype(bool), "measure"]
    measure = primary.iloc[0] if len(primary) else None
    if measure is None:
        return None
    s = s[(s["measure"] == measure) & ~s["completed_only"].astype(bool) & (s["form"] == "log")]
    out: dict = {}
    names = {"lme4_nested": "lme4", "scene": "model", "level_scene": "nested"}
    for _, r in s.iterrows():
        key = names.get(r["structure"])
        if key:
            out.setdefault(r["transform"], {}).setdefault(r["subject"], {})[key] = {
                "b": r["slope"], "lo": r["slope_ci_low"], "hi": r["slope_ci_high"],
                "p": r["slope_p"]}
    # Holm across the five subjects, for the scene-scale model (lme4; the scene-only model when
    # lme4 has not been fitted).
    for t, by_subject in out.items():
        key = "lme4" if any("lme4" in m for m in by_subject.values()) else "model"
        ps = {subj: m[key]["p"] for subj, m in by_subject.items() if key in m}
        for subj, v in _holm(ps).items():
            by_subject[subj][key]["p_holm"] = v
    for unit in ("scene", "level"):
        summ = docs_dir / f"practice_{unit}_bootstrap.csv"
        draws = docs_dir / f"practice_{unit}_bootstrap_draws.csv"
        if not (summ.exists() and draws.exists()):
            continue
        dr = pd.read_csv(draws, dtype={"subject": str}).dropna(subset=["b"])
        for _, r in pd.read_csv(summ, dtype={"subject": str}).iterrows():
            b = dr[(dr["transform"] == r["transform"]) & (dr["subject"] == r["subject"])]["b"].to_numpy()
            if not len(b) or r["transform"] not in out:
                continue
            p = min(1.0, 2 * min((b >= 0).mean(), (b <= 0).mean()))
            out[r["transform"]].setdefault(r["subject"], {})[f"{unit}_boot"] = {
                "b": r["b"], "lo": r["boot_ci_low"], "hi": r["boot_ci_high"],
                "p": max(p, 2 / len(b)), "p_floor": bool(p == 0), "n_boot": int(len(b))}
    return {"measure": measure, "methods": list(SUMMARY_METHODS), "by_transform": out}


DIAGNOSTICS_SCRIPT = DOCS_DIR / "practice_residual_diagnostics.py"
# Diagnostic runs the dashboard can show: (key, model label, file stem under docs/).
DIAGNOSTIC_RUNS = [("lme4", "scene scale (lme4)", "practice_residual_diagnostics_lme4_{t}"),
                   ("route", "route scale (lme4)", "practice_residual_diagnostics_route_{t}"),
                   ("scene", "scene only (statsmodels)", "practice_residual_diagnostics_{t}")]
# The measure each run was checked on (the main result of that model).
DIAGNOSTIC_MEASURE = {"lme4": "d_nearest_norm_loso", "route": "d_own_loso",
                      "scene": "d_nearest_norm_loso"}
_STATUS = {"OK": "ok", "MILD": "mild", "MIXED": "mild", "CONCERN": "mild",
           "NOT RUN": "mild", "VIOLATED": "bad", "CHANGES": "bad"}


def assumption_checks(docs_dir: Path = DOCS_DIR) -> dict | None:
    """The residual and random-effect checks, as stored by docs/practice_residual_diagnostics.py.

    Reads the per-subject numbers in ``docs/practice_residual_diagnostics*.csv`` and grades
    them with that script's own ``verdicts()`` (imported, not copied), so the dashboard and
    the diagnostic figures can never disagree. Row 9 uses the same bootstrap comparison the
    figure does: the level bootstrap against the lme4 CI, or the scene (+ level) bootstrap
    against the statsmodels CI. Nothing is refitted. ``None`` when no run is on disk.
    """
    import importlib.util

    import pandas as pd

    if not DIAGNOSTICS_SCRIPT.exists():
        return None
    spec = importlib.util.spec_from_file_location("practice_residual_diagnostics",
                                                  DIAGNOSTICS_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def row(t):
        title, question, look, affects = t
        return {"title": title, "question": question, "look_for": look, "affects": affects}

    runs = {}
    for key, label, stem in DIAGNOSTIC_RUNS:
        for t in ("log", "log1p"):
            csv = Path(docs_dir) / f"{stem.format(t=t)}.csv"
            if not csv.exists():
                continue
            m = pd.read_csv(csv, dtype={"subject": str})
            boot, _ = (mod.load_route_vs_level(t) if key == "route" else
                       mod.load_lme4_vs_level(t) if key == "lme4" else mod.load_boot(t))
            rows = (mod.ROUTE_ROWS if key == "route" else
                    mod.ROWS[:-1] + [mod.LME4_ROW9 if key == "lme4" else mod.ROWS[-1]])
            unit = "route" if key == "route" else "scene"
            checks = []
            for spec_row, (text, _colour) in zip(rows, mod.verdicts(m, boot, unit)):
                word, _, detail = text.partition(" - ")
                checks.append({**row(spec_row), "verdict": word, "status": _STATUS.get(word, "mild"),
                               "detail": detail})
            png = Path(docs_dir) / f"{stem.format(t=t)}.png"
            runs.setdefault(key, {"label": label, "measure": DIAGNOSTIC_MEASURE[key]})[t] = {
                "checks": checks,
                "subjects": m.drop(columns=["transform"]).to_dict(orient="records"),
                "figure": f"diagnostics/{png.name}" if png.exists() else None,
            }
    return runs or None


FIT_DATA_POINTS = 800      # cloud points per panel; the bins are computed on every row
SPAGHETTI_SCENES = 12      # most-played scenes per subject in the dashboard's spaghetti panel


def export_fit_data(output_dir: Path, web_dir: Path) -> Path | None:
    """Write the data under the lme4 fits: ``practice_points.json`` and ``practice_spaghetti.json``.

    ``practice_points.json``: one entry per fit in ``lme4.json`` (scene scale, ``nested``
    model) and ``lme4_route.json``, keyed ``model|measure|transform|completed|subject`` with
    model ``lme4_nested`` / ``lme4_route`` as the dashboard names them, each holding
    ``fit_check_data()`` — pooled bins, and partial-residual bins when
    ``run-practice-lme4-effects`` has run — over every fitted row, plus a seeded thinning
    to ``FIT_DATA_POINTS`` points.

    ``practice_spaghetti.json``: for the primary setting of each model only (its primary
    measure, primary transform, all attempts), per subject, ``spaghetti_data()`` for their
    ``SPAGHETTI_SCENES`` most-played scenes — every attempt and each scene's (or route's)
    own fitted line — plus each scene's whole-scene route ranking (``rank``), so routes
    keep the colour slot they have everywhere else. Needs the effects; skipped without.

    Separate files, fetched only when their panel is drawn, so ``practice.json`` stays
    small. For drawing only — the cloud is a sample.
    """
    from analysis.practice import (fit_check_data, fit_id, load_attempts, load_effects,
                                   spaghetti_data)
    pdir = Path(output_dir) / "practice"
    sources = [("lme4_nested", "nested", pdir / "lme4.json"), ("lme4_route", "route", pdir / "lme4_route.json")]
    sources = [s for s in sources if s[2].exists()]
    if not sources or not (pdir / "attempts").is_dir():
        return None
    attempts = load_attempts(output_dir)
    effects = load_effects(output_dir)
    if effects is None:
        print("  no lme4 effects (invoke run-practice-lme4-effects) — fit check without partial "
              "residuals, no spaghetti panel")
    partial = effects[0].groupby("fid") if effects is not None else None
    out, spaghetti = {}, {}
    for key, model, path in sources:
        stored = json.loads(path.read_text())
        for f in stored["fits"]:
            if f["model"] != model:
                continue
            fid = fit_id(model, f["measure"], f["transform"], f["completed_only"], f["subject"])
            eff = partial.get_group(fid) if partial is not None and fid in partial.groups else None
            out[f"{key}|{fid.split('|', 1)[1]}"] = fit_check_data(
                attempts, model, f["measure"], f["subject"], f["transform"], f["completed_only"],
                effects=eff, n_points=FIT_DATA_POINTS)
        if effects is None:
            continue
        spaghetti[key] = {"measure": stored["primary"], "transform": stored["primary_transform"],
                          "subjects": {}}
        for s in stored["subjects"]:
            fid = fit_id(model, stored["primary"], stored["primary_transform"], False, s)
            if fid not in effects[2]:
                continue
            sp = spaghetti_data(*effects, fid, top=SPAGHETTI_SCENES)
            for scene, v in sp["scenes"].items():
                counts = attempts.loc[(attempts["scene"] == scene) & (attempts["cluster"] >= 0), "cluster"]
                v["rank"] = [int(c) for c in counts.value_counts().index]
            spaghetti[key]["subjects"][s] = sp
    dest = Path(web_dir) / "practice_points.json"
    dest.write_text(json.dumps(_rounded(out, 3), separators=(",", ":")))
    print(f"Wrote {len(out)} fit-check panels -> {dest}")
    if spaghetti:
        sdest = Path(web_dir) / "practice_spaghetti.json"
        sdest.write_text(json.dumps(_rounded(spaghetti, 4), separators=(",", ":")))
        print(f"Wrote per-scene fits -> {sdest}")
    return dest


def export_practice(output_dir: Path, web_dir: Path) -> Path | None:
    """Copy the practice models into the payload as ``practice.json``.

    The dashboard's stats section draws these; it may not read ``output_data/practice/``
    directly, so the fits travel through the export like everything else. Copied:
    ``mixed.json`` (the statsmodels models), and when present ``lme4.json`` (the primary,
    scene-scale model) as ``lme4``, ``lme4_route.json`` (the route-scale model) as
    ``lme4_route``, ``nested.json`` (statsmodels nested structures) as
    ``nested``, ``practice_summary()`` (every CI and p-value) as ``summary``, and
    ``assumption_checks()`` as ``diagnostics`` (the diagnostic figures are copied to
    ``diagnostics/`` beside it). Only the
    fitted records — coefficients, CIs, p-values, variances, AIC, fixed-effect curves —
    never the attempts, and nothing is refitted; the data under the lme4 curves goes to a
    separate files, ``export_fit_data()``: bins plus a thinned point sample, and the
    per-scene fits for the spaghetti panel. ``scenes`` lists are reduced to a count.
    Returns ``None`` when the practice models have not been run.
    """
    pdir = Path(output_dir) / "practice"
    src = pdir / "mixed.json"
    if not src.exists():
        print("  no practice models (output_data/practice/mixed.json) — stats section empty")
        return None
    mixed = json.loads(src.read_text())
    payload = {k: v for k, v in mixed.items() if k != "scenes"}
    payload["n_scenes"] = len(mixed.get("scenes", []))
    for name in ("lme4", "lme4_route", "nested"):
        path = pdir / f"{name}.json"
        if path.exists():
            payload[name] = {k: v for k, v in json.loads(path.read_text()).items() if k != "scenes"}
    summary = practice_summary(output_dir)
    if summary:
        payload["summary"] = summary
    checks = assumption_checks()
    if checks:
        payload["diagnostics"] = checks
        fig_dir = Path(web_dir) / "diagnostics"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for run in checks.values():
            for t in ("log", "log1p"):
                if t in run and run[t]["figure"]:
                    shutil.copy2(DOCS_DIR / Path(run[t]["figure"]).name, Path(web_dir) / run[t]["figure"])
    out = Path(web_dir) / "practice.json"
    out.write_text(json.dumps(_rounded(payload), separators=(",", ":")))
    print(f"Wrote practice models -> {out}")
    export_fit_data(output_dir, web_dir)
    return out


def sync_backgrounds(web_dir: Path, source_dir: Path, scene_ids=None) -> int:
    """Copy scene background PNGs into the payload so the served tree is self-contained."""
    web_dir, source_dir = Path(web_dir), Path(source_dir)
    background_dir = default_background_dir(source_dir)
    if scene_ids is None:
        scene_ids = [p.stem for p in (web_dir / "scenes").glob("*.json")]

    bg_dir = web_dir / "backgrounds"
    copied = 0
    for scene_id in scene_ids:
        src = resolve_background(scene_id, background_dir)
        if src is None:
            continue
        bg_dir.mkdir(parents=True, exist_ok=True)
        dest = bg_dir / f"{scene_id}.png"
        if not dest.exists():
            shutil.copy2(src, dest)
            copied += 1
    return copied


def backfill_derived(web_dir: Path) -> int:
    """Add the derived fields to already-exported scene payloads, in place.

    The attempt index and stage tertile are computed from ``s``/``ph``/``clip``,
    all of which the payload already carries, so this needs neither
    ``mario.scenes`` nor the cluster files — it rewrites 300-odd JSON files in
    seconds where a ``--force`` re-export would re-read tens of GB for nothing.
    Fresh exports get the fields from ``export_scene``; this is the migration
    path for payloads written before they existed.
    """
    scenes_dir = Path(web_dir) / "scenes"
    paths = sorted(scenes_dir.glob("*.json"))
    for path in paths:
        payload = json.loads(path.read_text())
        add_derived(payload["traces"])
        path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Backfilled attempt/stage fields in {len(paths)} scene payloads")
    return len(paths)


def backfill_selection(web_dir: Path, output_dir: Path) -> int:
    """Refresh each payload's ``cons`` block in place, from ``output_data`` only.

    A third category, distinct from both of the paths above it. ``cons`` is **not**
    derivable from the payload — ``backfill_derived``'s rule — but it is derivable
    without ``mario.scenes``: it lives in the cluster file, and the leaf order can be
    rebuilt from the stored co-association matrix. So a change to what the selection
    record carries costs a rewrite of 300-odd small JSON files rather than a ``--force``
    re-export that re-reads tens of GB to produce byte-identical geometry.

    The trace geometry and labels are untouched. Anything that changes *those* is a
    real re-export; this is only for the record of how k was chosen.
    """
    scenes_dir = Path(web_dir) / "scenes"
    paths = sorted(scenes_dir.glob("*.json"))
    updated = skipped = 0
    for path in paths:
        payload = json.loads(path.read_text())
        report = selection_report(payload["scene"], output_dir, payload["traces"])
        if report is None:
            skipped += 1
            continue
        payload["cons"] = report
        path.write_text(json.dumps(payload, separators=(",", ":")))
        updated += 1
    note = f", {skipped} without a consensus record" if skipped else ""
    print(f"Refreshed the selection record in {updated} scene payloads{note}")
    return updated


def backfill_labels(web_dir: Path, output_dir: Path) -> int:
    """Re-join the stored cluster labels (and per-trace confidence) to existing payloads.

    A fourth migration path, and it exists because of what a re-cluster actually
    changes. Cutting the same ensemble at a different k changes ``c`` and ``sb`` and
    **nothing else** — the geometry, the axis limits, the background, the derived time
    fields and the co-association matrix are all identical, since the traces and the
    replicate fits are untouched. A ``--force`` re-export would re-read tens of GB from
    ``mario.scenes`` to rebuild polylines byte-for-byte identical to the ones already on
    disk, so it is the wrong tool for a change of cut.

    The rule this follows is the one ``backfill_selection`` documents: ``c`` and ``sb``
    are not derivable from the payload, but they *are* derivable from ``output_data/``
    alone. The join is by **clip key**, never by position — the same key
    ``load_labeled_traces`` and ``export_coassociation`` use — so a payload whose keys
    do not line up with the cluster file is skipped rather than silently relabelled
    against the wrong rows.

    A re-export is still required when the traces themselves change (new source data,
    a different ``step``, a new y coordinate). This is only for a new partition over
    the same traces.
    """
    scenes_dir = Path(web_dir) / "scenes"
    paths = sorted(scenes_dir.glob("*.json"))
    updated = skipped = 0
    for path in paths:
        payload = json.loads(path.read_text())
        scene_id = payload["scene"]
        clusters = Path(output_dir) / "clusters" / f"{scene_id}_clusters.json"
        if not clusters.exists():
            skipped += 1
            continue
        stored = json.loads(clusters.read_text())
        labels = stored["labels"]
        per_trace = ((stored.get("stability") or {}).get("per_trace")) or None

        row_of = _clip_row_map(scene_id, output_dir)
        rows = [row_of.get((t["s"], t["ses"], t["run"], t["clip"])) for t in payload["traces"]]
        if any(r is None for r in rows) or len(labels) != len(payload["traces"]):
            print(f"  {scene_id}: clip keys do not line up with the cluster file, skipping.")
            skipped += 1
            continue

        for trace, row in zip(payload["traces"], rows):
            trace["c"] = int(labels[row])
            if per_trace is not None:
                trace["sb"] = round(float(per_trace[row]), 2)
            else:
                trace.pop("sb", None)
        payload["method"] = stored.get("method", payload.get("method", ""))
        path.write_text(json.dumps(payload, separators=(",", ":")))
        updated += 1
    note = f", {skipped} skipped" if skipped else ""
    print(f"Re-joined cluster labels in {updated} scene payloads{note}")
    return updated
