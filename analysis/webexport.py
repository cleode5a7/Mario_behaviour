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


def export_scene(scene_id, source_dir, output_dir, web_dir, *, step=3,
                 entry_map=None, background_dir=None) -> dict | None:
    """Write one scene's dashboard payload; return its index entry (None if empty)."""
    source_dir, output_dir, web_dir = Path(source_dir), Path(output_dir), Path(web_dir)
    traces, metadata, labels, method = load_labeled_traces(scene_id, source_dir, output_dir)
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

    out_traces = []
    for trace, meta, label in zip(traces, metadata, labels):
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

    return _index_entry(scene_id, out_traces, method)


def _index_entry(scene_id: str, out_traces: list[dict], method: str) -> dict:
    """Summary row for index.json, derived from the payload we just built."""
    labels = [t["c"] for t in out_traces]
    n_noise = sum(1 for l in labels if l == -1)
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
        "subjects": sorted({t["s"] for t in out_traces}),
        "phases": sorted({t["ph"] for t in out_traces if t["ph"]}),
        "method": method,
    }


def _index_entry_from_file(path: Path) -> dict:
    """Rebuild an index row from an already-exported scene file (skip path)."""
    payload = json.loads(path.read_text())
    return _index_entry(payload["scene"], payload["traces"], payload.get("method", ""))


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
