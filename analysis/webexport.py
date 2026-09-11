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
    else:
        report.pop("leaves", None)
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
