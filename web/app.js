/* Mario clustering dashboard.
 *
 * Reads the payload written by `invoke run-web-export` and renders the same
 * figure analysis/trace_plots.py renders to PNG — background overlay, entry-point
 * x-offset, screen-corrected y, rank-by-size cluster palette with marker shapes
 * as a secondary encoding, playfield y-clipping — plus subject/phase/outcome
 * filters the static figures cannot offer.
 *
 * The one rule that matters scientifically: filters NEVER re-derive cluster
 * labels. The consensus ran once over every trace in the scene — HDBSCAN refit on
 * 200 subsamples, a co-association matrix, then a Ward cut at the k with the best
 * silhouette — and a filter only changes which of those traces are drawn. Counts
 * recompute for the visible subset, and the title says "N of M" so a filtered view
 * can't be mistaken for the full one. The `k selection` view shows that cut being
 * made; the `matrix` view shows what it was made from.
 */

"use strict";

/* ---------- palettes: copied verbatim from analysis/trace_plots.py ---------- */

const CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                        "#e87ba4", "#008300", "#4a3aa7", "#e34948"];
const CLUSTER_MARKERS = ["o", "s", "^", "D", "P", "X", "*", "v"];
const NOISE_COLOR = "#898781";
const OTHER_COLOR = "#c3c2b7";
const N_DIRECT_CLUSTERS = 8;

const OUTCOME_COLORS = { completed: "#2ca02c", death: "#d62728" };
const OUTCOME_OTHER = "#898781";

const SUBJECT_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
                        "#a65628", "#f781bf", "#00ced1", "#999999", "#dede00"];

/* Time-ordered encodings use viridis. Both lists are baked hex copied verbatim
 * from analysis/trace_plots.py — change them there and here together. */
const STAGE_COLORS = ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"];
const VIRIDIS_STOPS = [
  "#440154", "#470d60", "#48186a", "#482374", "#472e7c", "#453882", "#424186", "#3e4a89",
  "#3a548c", "#365d8d", "#32658e", "#2e6d8e", "#2b758e", "#287d8e", "#25848e", "#228c8d",
  "#1f948c", "#1e9c89", "#20a386", "#25ab82", "#2eb37c", "#3aba76", "#48c16e", "#58c765",
  "#6ccd5a", "#7fd34e", "#93d741", "#a8db34", "#c0df25", "#d5e21a", "#eae51a", "#fde725",
];

/* Magma ramp for cluster stability, verbatim from STABILITY_STOPS in
 * analysis/trace_plots.py. Deliberately not viridis — that already means
 * "attempt #", and two sequential encodings sharing a palette is how a reader
 * mistakes one for the other. */
const STABILITY_STOPS = [
  "#000004", "#040414", "#0b0924", "#140e36", "#1f0c48", "#2b115a", "#381a6c", "#440f76",
  "#4f127b", "#5a167e", "#65197f", "#701f81", "#7b2382", "#862781", "#912b81", "#9c2e7f",
  "#a7327d", "#b3357a", "#be3d72", "#c8436a", "#d34b62", "#dc5459", "#e45d51", "#eb6a4a",
  "#f07948", "#f4894b", "#f79952", "#f9a95e", "#fbb96d", "#fcc97f", "#fdd893", "#fcfdbf",
];

/* The elbow rule's colour in the k-selection chart. Warm, so it cannot be mistaken
 * for either silhouette series — it marks one k, not a curve. */
const ELBOW_COLOR = "#eb6834";

/* Legend order for the six stages, mirroring STAGES in analysis/stages.py. The
 * payload stores only the tertile letter; the stage is that plus the phase. */
const STAGES = ["early_discovery", "middle_discovery", "late_discovery",
                "early_practice", "middle_practice", "late_practice"];
const TERTILE_NAMES = { e: "early", m: "middle", l: "late" };
const stageName = (t) => (t.st && t.ph ? `${TERTILE_NAMES[t.st]}_${t.ph}` : "");

/* Matplotlib linewidths are points at dpi=150; these are the CSS-pixel
 * equivalents that read the same on screen. */
const LW = { direct: 1.6, faint: 1.0, outcome: 1.0, subject: 1.2, attempt: 1.2, stage: 1.2, stability: 1.2 };
const ALPHA = { direct: 0.85, faint: 0.35, outcome: 0.5, subject: 0.6, attempt: 0.6, stage: 0.6, stability: 0.6 };
const MARKER_PX = { overlay: 7, panel: 6 };

/* ---------- state ---------- */

const state = {
  index: null,
  scene: null,          // the loaded scene payload
  bg: null,             // HTMLImageElement or null
  subjectColors: new Map(),
  attemptMax: new Map(),   // subject -> highest attempt index in this scene
  hasDerived: true,        // payload carries the attempt/stage fields
  filters: { subjects: new Set(), phase: "all", outcome: "all", stage: "all" },
  colorBy: "cluster",
  view: "overlay",
  showNoise: true,
  showBg: true,
  matrixOrder: "leaves",     // dendrogram leaf order; "cluster" = rank-by-size blocks
  showPartition: true,
  hiddenClusters: new Set(),
  hover: null,          // trace object under the cursor
  pinned: null,         // trace object clicked
  zoom: 1,
  pan: { x: 0, y: 0 },
  section: "clustering",   // or "stats": the practice-model section
  stats: { model: "lme4_nested", measure: "d_nearest_norm", transform: "log", form: "log", attempts: "all",
           spSubject: null, axisUnits: "units" },
  statsSubjectColors: new Map(),
};

const sceneCache = new Map();
const $ = (id) => document.getElementById(id);

/* ---------- marker shapes (matplotlib equivalents) ---------- */

function markerPath(ctx, kind, cx, cy, size) {
  const r = size / 2;
  ctx.beginPath();
  switch (kind) {
    case "o": ctx.arc(cx, cy, r, 0, Math.PI * 2); break;
    case ".": ctx.arc(cx, cy, r * 0.45, 0, Math.PI * 2); break;
    case "s": ctx.rect(cx - r, cy - r, size, size); break;
    case "^": poly(ctx, [[cx, cy - r], [cx + r, cy + r], [cx - r, cy + r]]); break;
    case "v": poly(ctx, [[cx, cy + r], [cx + r, cy - r], [cx - r, cy - r]]); break;
    case "D": poly(ctx, [[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]]); break;
    case "P": { // filled plus
      const t = r * 0.42;
      poly(ctx, [[cx - t, cy - r], [cx + t, cy - r], [cx + t, cy - t], [cx + r, cy - t],
                 [cx + r, cy + t], [cx + t, cy + t], [cx + t, cy + r], [cx - t, cy + r],
                 [cx - t, cy + t], [cx - r, cy + t], [cx - r, cy - t], [cx - t, cy - t]]);
      break;
    }
    case "X": { // filled cross
      const t = r * 0.42, d = r * 0.72;
      poly(ctx, [[cx - d, cy - d + t], [cx - d + t, cy - d], [cx, cy - t * 0.4],
                 [cx + d - t, cy - d], [cx + d, cy - d + t], [cx + t * 0.4, cy],
                 [cx + d, cy + d - t], [cx + d - t, cy + d], [cx, cy + t * 0.4],
                 [cx - d + t, cy + d], [cx - d, cy + d - t], [cx - t * 0.4, cy]]);
      break;
    }
    case "*": { // 5-pointed star
      const pts = [];
      for (let i = 0; i < 10; i++) {
        const rad = i % 2 ? r * 0.45 : r;
        const a = -Math.PI / 2 + (i * Math.PI) / 5;
        pts.push([cx + rad * Math.cos(a), cy + rad * Math.sin(a)]);
      }
      poly(ctx, pts);
      break;
    }
    default: ctx.arc(cx, cy, r, 0, Math.PI * 2);
  }
}

function poly(ctx, pts) {
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
}

/* "x" is a stroked marker in matplotlib, not a filled one. */
function drawMarker(ctx, kind, cx, cy, size, color, alpha) {
  ctx.globalAlpha = alpha;
  if (kind === "x") {
    const r = size / 2;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    ctx.moveTo(cx - r, cy - r); ctx.lineTo(cx + r, cy + r);
    ctx.moveTo(cx + r, cy - r); ctx.lineTo(cx - r, cy + r);
    ctx.stroke();
    return;
  }
  ctx.fillStyle = color;
  markerPath(ctx, kind, cx, cy, size);
  ctx.fill();
}

/* ---------- filtering & ranking ---------- */

function visibleTraces() {
  if (!state.scene) return [];
  const f = state.filters;
  return state.scene.traces.filter((t) =>
    f.subjects.has(t.s) &&
    (f.phase === "all" || t.ph === f.phase) &&
    (f.stage === "all" || t.st === f.stage) &&
    (f.outcome === "all" || t.out === f.outcome) &&
    (state.showNoise || t.c !== -1) &&
    !state.hiddenClusters.has(t.c));
}

/* Rank clusters by size, largest first — colors follow rank, not label id, which
 * is how _draw_by_cluster() in analysis/trace_plots.py assigns them.
 *
 * The ranking is computed over EVERY trace in the scene, never over the filtered
 * subset. Two reasons, and they agree:
 *   1. it is what the PNG does (the figures have no filters), so the dashboard
 *      shows the same cluster in the same hue;
 *   2. a filter must never repaint the traces that survive it — if selecting
 *      sub-01 turned cluster 4 from blue to green, the colour would be encoding
 *      the filter rather than the cluster.
 * Only the n= counts recompute for the visible subset ("recount live"). */
function sceneRanking() {
  const sizes = new Map();
  for (const t of state.scene.traces) {
    if (t.c === -1) continue;
    sizes.set(t.c, (sizes.get(t.c) || 0) + 1);
  }
  // label-id tiebreak so equal-sized clusters keep a stable order across redraws
  const ranked = [...sizes.keys()].sort((a, b) => (sizes.get(b) - sizes.get(a)) || (a - b));
  const direct = new Map(ranked.slice(0, N_DIRECT_CLUSTERS).map((l, i) => [l, i]));
  return { sizes, ranked, direct };
}

/* Per-label counts over the traces currently drawn — this is what the legend,
 * panel titles and composition pills report. */
function visibleCounts(traces) {
  const counts = new Map();
  for (const t of traces) counts.set(t.c, (counts.get(t.c) || 0) + 1);
  return counts;
}

/* The clusters that still have at least one visible trace, in stable rank order. */
function visibleRanked(rank, counts) {
  return rank.ranked.filter((l) => (counts.get(l) || 0) > 0);
}

/* ---------- which cut is on screen ---------- */

/* The one place the dashboard departs from "filters never change the labels", and
 * it does so loudly rather than quietly.
 *
 * Every candidate cut of the Ward tree was scored during the consensus run, and one
 * of them was selected by the k rule. The others are just as real — they are the same
 * tree cut at a different height — and being able to see them is how you tell a cut
 * that found structure from one that split a cluster in half. So the payload ships
 * all of them (`cons.k_candidates[].bounds`, k-1 split positions in the dendrogram
 * leaf order) and this control draws a different one.
 *
 * What it is NOT: a re-analysis. Nothing is refitted, no distance is recomputed, the
 * co-association matrix is untouched — the browser is still only drawing a stored
 * result, just not the selected one. The title says so whenever the cut on screen is
 * not the chosen one, because every other number in the interface (stable_share, the
 * per-trace confidence, the index row) was measured at the chosen k. */

function consensus() {
  return (state.scene && state.scene.cons) || {};
}

function chosenK() {
  const k = consensus().k;
  return typeof k === "number" ? k : null;
}

function activeK() {
  return state.k == null ? chosenK() : state.k;
}

function viewingAlternativeCut() {
  return state.k != null && state.k !== chosenK();
}

/* Candidates this browser can actually draw. A cut is only decodable alongside the
 * leaf order its positions index into, so a payload without one offers none — the
 * export drops the bounds in that case for exactly this reason. */
function cuttableCandidates() {
  const cons = consensus();
  const leaves = cons.leaves;
  if (!state.scene || !Array.isArray(leaves)) return [];
  if (leaves.length !== state.scene.traces.length) return [];
  return (cons.k_candidates || []).filter(
    (r) => Array.isArray(r.bounds) && r.bounds.length === r.k - 1);
}

/* Set every trace's label to the cut at k. Null (or the chosen k) restores the labels
 * the run itself produced, which are kept verbatim in `c0` — the stored partition is
 * authoritative and is never re-derived, even though decoding reproduces it exactly
 * (ARI 1.000 over the corpus). Only the cluster *ids* can differ, because equal-sized
 * clusters tie-break differently here than in _relabel_by_size, and that would repaint
 * the scene for no reason at the one k where there is a stored answer to defer to. */
function applyCut(k) {
  if (!state.scene) return;
  const traces = state.scene.traces;
  if (traces.length && traces[0].c0 === undefined) for (const t of traces) t.c0 = t.c;

  const row = k == null || k === chosenK()
    ? null
    : cuttableCandidates().find((r) => r.k === k);
  if (!row) {
    for (const t of traces) t.c = t.c0;
    state.k = null;
    return;
  }

  const leaves = consensus().leaves;
  const bounds = new Set(row.bounds);
  const group = new Int32Array(traces.length);
  const size = new Array(row.k).fill(0);
  let g = 0;
  for (let p = 0; p < leaves.length; p++) {
    if (bounds.has(p)) g++;
    group[leaves[p]] = g;
    size[g]++;
  }
  // Largest cluster becomes 0, as _relabel_by_size does, so the colour slots keep
  // meaning "rank by size" at every k. Ties break by leaf order, which is arbitrary
  // but deterministic — the point is that a redraw never reshuffles the palette.
  const rank = new Map(size.map((n, i) => [i, n])
    .sort((a, b) => (b[1] - a[1]) || (a[0] - b[0]))
    .map(([i], slot) => [i, slot]));
  traces.forEach((t, i) => { t.c = rank.get(group[i]); });
  state.k = k;
}

/* Viridis position for a trace, normalised WITHIN ITS SUBJECT: every player's
 * own first attempt is dark and their own last is bright. Players make very
 * different numbers of attempts at the same scene (44/34/27/33/29 on w1l1s0), and
 * a scene-wide denominator would make colour encode how much someone played as
 * well as when — so a light trace would mean "late" for one player and "played a
 * lot" for another. Per-subject keeps it reading purely as time. */
function attemptFrac(t) {
  const max = state.attemptMax.get(t.s) || 1;
  return max > 1 ? (t.a - 1) / (max - 1) : 1;
}

function attemptColor(t) {
  return VIRIDIS_STOPS[Math.round(attemptFrac(t) * (VIRIDIS_STOPS.length - 1))];
}

/* Stability is already a 0-1 confidence in the payload, so unlike the attempt
 * ramp it needs no per-subject normalisation — the scale is absolute and
 * comparable across scenes. */
function stabilityColor(t) {
  const v = Math.min(1, Math.max(0, t.sb == null ? 0 : t.sb));
  return STABILITY_STOPS[Math.round(v * (STABILITY_STOPS.length - 1))];
}

/* Painting order for sequential colour modes.
 *
 * Traces in a scene overlap almost completely — everyone runs the same corridor —
 * so at these alphas the top few strokes are the only ones visible, and painting
 * order decides what the plot says. Payload order is (Subject, Session, Run, ...),
 * i.e. blocked by subject, which for a time ramp is actively wrong: sub-06's first
 * attempt lands on top of sub-01's last, and the visible colour ends up encoding
 * subject order instead of time. Sort ascending so the latest attempts sit on top
 * and early ones show only where they actually diverged.
 *
 * Categorical modes (cluster / outcome / subject) keep payload order — nothing is
 * gained by reordering them, and leaving them alone keeps the overlay identical to
 * the PNG that analysis/trace_plots.py draws. */
function drawOrder(traces) {
  if (state.colorBy === "attempt") {
    return [...traces].sort((a, b) => attemptFrac(a) - attemptFrac(b));
  }
  if (state.colorBy === "stage") {   // STAGES is already in chronological order
    return [...traces].sort((a, b) => STAGES.indexOf(stageName(a)) - STAGES.indexOf(stageName(b)));
  }
  if (state.colorBy === "stability") {
    // Descending: the point of this view is to find the traces their cluster
    // barely agreed with, so they paint last. Same order as _draw_by_stability.
    return [...traces].sort((a, b) => (b.sb ?? 0) - (a.sb ?? 0));
  }
  return traces;
}

function styleFor(t, rank, counts) {
  if (state.colorBy === "outcome") {
    return {
      color: OUTCOME_COLORS[t.out] || OUTCOME_OTHER,
      marker: null, alpha: ALPHA.outcome, lw: LW.outcome,
      name: t.out || "unknown",
    };
  }
  let color, marker, name, alpha, lw;
  if (t.c === -1) {
    color = NOISE_COLOR; marker = "x"; name = "noise";
    alpha = ALPHA.faint; lw = LW.faint;
  } else if (rank.direct.has(t.c)) {
    const slot = rank.direct.get(t.c);
    color = CLUSTER_COLORS[slot]; marker = CLUSTER_MARKERS[slot];
    name = `cluster ${t.c} (n=${counts ? counts.get(t.c) || 0 : rank.sizes.get(t.c)})`;
    alpha = ALPHA.direct; lw = LW.direct;
  } else {
    color = OTHER_COLOR; marker = "."; name = "other clusters";
    alpha = ALPHA.faint; lw = LW.faint;
  }
  if (state.colorBy === "subject") {
    color = state.subjectColors.get(t.s) || OTHER_COLOR;
    alpha = ALPHA.subject; lw = LW.subject;
    name = `sub-${t.s}`;
  } else if (state.colorBy === "attempt") {
    color = attemptColor(t);
    marker = null;
    alpha = ALPHA.attempt; lw = LW.attempt;
    name = `attempt ${t.a}`;
  } else if (state.colorBy === "stage") {
    const i = STAGES.indexOf(stageName(t));
    color = i === -1 ? OTHER_COLOR : STAGE_COLORS[i];
    alpha = ALPHA.stage; lw = LW.stage;
    name = stageName(t) || "unknown stage";
  } else if (state.colorBy === "stability") {
    color = stabilityColor(t);
    marker = null;
    alpha = ALPHA.stability; lw = LW.stability;
    name = t.sb == null ? "no stability" : `stability ${t.sb.toFixed(2)}`;
  }
  return { color, marker, name, alpha, lw };
}

/* ---------- coordinate transform ---------- */

/* Data space always has y increasing downward — matplotlib flips the axis with
 * set_ylim(yhi, ylo) / set_ylim(bg_height, 0), which is canvas-native, so no flip
 * is needed here. With a background the extent is the image; without one it is the
 * exported xlim/ylim (from _shared_xlim and playfield_ylim). */
function dataExtent(usingBg) {
  const s = state.scene;
  if (usingBg) return { x0: 0, x1: s.bg_size[0], y0: 0, y1: s.bg_size[1] };
  return { x0: s.xlim[0], x1: s.xlim[1], y0: s.ylim[0], y1: s.ylim[1] };
}

/* Backgrounds are images and must keep their aspect ratio; the plain data axis
 * stretches to the box, as matplotlib's default aspect="auto" does. */
function baseTransform(w, h, usingBg) {
  const e = dataExtent(usingBg);
  const dx = (e.x1 - e.x0) || 1, dy = (e.y1 - e.y0) || 1;
  let sx, sy, ox, oy;
  if (usingBg) {
    const s = Math.min(w / dx, h / dy);
    sx = sy = s;
    ox = (w - dx * s) / 2;
    oy = (h - dy * s) / 2;
  } else {
    const pad = 26;
    sx = (w - pad * 2) / dx; sy = (h - pad * 2) / dy;
    ox = pad; oy = pad;
  }
  return { e, sx, sy, ox, oy };
}

function makeProjector(base, zoom, pan) {
  return {
    x: (v) => ((v - base.e.x0) * base.sx + base.ox) * zoom + pan.x,
    y: (v) => ((v - base.e.y0) * base.sy + base.oy) * zoom + pan.y,
    invX: (p) => ((p - pan.x) / zoom - base.ox) / base.sx + base.e.x0,
    invY: (p) => ((p - pan.y) / zoom - base.oy) / base.sy + base.e.y0,
  };
}

function usingBackground() {
  return !!(state.showBg && state.bg && state.scene && state.scene.bg_size);
}

/* ---------- canvas plumbing ---------- */

function sizeCanvas(canvas, cssW, cssH) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(cssW * dpr));
  canvas.height = Math.max(1, Math.round(cssH * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function drawTrace(ctx, t, proj, color, alpha, lw) {
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  let pen = false;
  for (let i = 0; i < t.x.length; i++) {
    const yv = t.y[i];
    if (yv === null) { pen = false; continue; }   // masked frame: break the line
    const px = proj.x(t.x[i]), py = proj.y(yv);
    if (pen) ctx.lineTo(px, py); else { ctx.moveTo(px, py); pen = true; }
  }
  ctx.stroke();
}

function firstPoint(t) {
  for (let i = 0; i < t.x.length; i++) if (t.y[i] !== null) return [t.x[i], t.y[i]];
  return null;
}

/* ---------- overlay view ---------- */

function renderOverlay() {
  const box = $("canvas-box");
  const canvas = $("canvas");
  const cssW = box.clientWidth, cssH = box.clientHeight;
  if (cssW < 2 || cssH < 2) return;

  const ctx = sizeCanvas(canvas, cssW, cssH);
  const usingBg = usingBackground();
  const base = baseTransform(cssW, cssH, usingBg);
  const proj = makeProjector(base, state.zoom, state.pan);

  ctx.clearRect(0, 0, cssW, cssH);
  ctx.globalAlpha = 1;

  if (usingBg) {
    ctx.imageSmoothingEnabled = false;   // NES art: keep the pixels crisp
    ctx.drawImage(state.bg, proj.x(0), proj.y(0),
                  state.scene.bg_size[0] * base.sx * state.zoom,
                  state.scene.bg_size[1] * base.sy * state.zoom);
  } else {
    drawAxes(ctx, cssW, cssH, base, proj);
  }

  const traces = visibleTraces();
  const rank = sceneRanking();                   // colours: stable, whole-scene
  const counts = visibleCounts(traces);          // labels: live, filtered subset
  const focus = state.pinned || state.hover;

  for (const t of drawOrder(traces)) {
    if (focus && t === focus) continue;          // drawn last, on top
    const st = styleFor(t, rank, counts);
    const dim = focus ? 0.12 : 1;                // ghost the rest while focused
    drawTrace(ctx, t, proj, st.color, st.alpha * dim, st.lw);
    if (st.marker) {
      const p = firstPoint(t);
      if (p) drawMarker(ctx, st.marker, proj.x(p[0]), proj.y(p[1]),
                        MARKER_PX.overlay, st.color, st.alpha * dim);
    }
  }

  if (focus && traces.includes(focus)) {
    const st = styleFor(focus, rank, counts);
    drawTrace(ctx, focus, proj, st.color, 1, Math.max(st.lw, 2.4));
    const p = firstPoint(focus);
    if (p) drawMarker(ctx, st.marker || "o", proj.x(p[0]), proj.y(p[1]),
                      MARKER_PX.overlay + 3, st.color, 1);
  }

  ctx.globalAlpha = 1;
  renderLegend(traces, rank, counts);
  renderTitle(traces, rank, counts);
}

/* Plain data axis for the handful of scenes with no background PNG. */
function drawAxes(ctx, w, h, base, proj) {
  const css = getComputedStyle(document.documentElement);
  const line = css.getPropertyValue("--line").trim();
  const ink3 = css.getPropertyValue("--ink-3").trim();
  ctx.save();
  ctx.strokeStyle = line;
  ctx.lineWidth = 1;
  ctx.strokeRect(base.ox, base.oy, w - base.ox * 2, h - base.oy * 2);
  ctx.fillStyle = ink3;
  ctx.font = "10px system-ui, sans-serif";

  const e = base.e;
  for (let i = 0; i <= 4; i++) {
    const xv = e.x0 + ((e.x1 - e.x0) * i) / 4;
    const yv = e.y0 + ((e.y1 - e.y0) * i) / 4;
    ctx.textAlign = "center";
    ctx.fillText(Math.round(xv), proj.x(xv), h - base.oy + 13);
    ctx.textAlign = "right";
    ctx.fillText(Math.round(yv), base.ox - 5, proj.y(yv) + 3);
  }
  ctx.textAlign = "center";
  ctx.fillText("player x position", w / 2, h - 3);
  ctx.restore();
}

function renderTitle(traces, rank, counts) {
  const s = state.scene;
  const total = s.traces.length;
  const k = visibleRanked(rank, counts).length;
  $("chart-title").textContent =
    `${s.scene} — ${traces.length} traces, ${k} clusters (${s.method})`;

  const filtered = traces.length !== total;
  const bits = [];
  if (filtered) bits.push(`<span class="filtered">${traces.length} of ${total} traces shown</span>`);
  // Which cut is on screen has to be said outright: it is the one thing in this
  // interface that changes the labels, and every stability number quoted elsewhere
  // (the index row, stable_share, the per-trace confidence) belongs to the chosen one.
  if (viewingAlternativeCut()) {
    bits.push(`<span class="filtered">cut at k=${state.k}, not the chosen k=${chosenK()}</span>`);
  } else {
    bits.push("cluster labels are fixed — filters only change what is drawn");
  }
  if (s.step > 1) bits.push(`polylines thinned to every ${s.step}${s.step === 3 ? "rd" : "th"} frame`);
  $("chart-sub").innerHTML = bits.join(" · ");
  $("clear-filters").hidden = !filtered && state.hiddenClusters.size === 0;
}

function renderLegend(traces, rank, counts) {
  const legend = $("legend");
  legend.innerHTML = "";

  if (state.colorBy === "outcome") {
    const counts = new Map();
    for (const t of traces) counts.set(t.out || "unknown", (counts.get(t.out || "unknown") || 0) + 1);
    for (const [name, n] of [...counts].sort((a, b) => b[1] - a[1])) {
      legend.appendChild(legendItem(OUTCOME_COLORS[name] || OUTCOME_OTHER, null,
                                    `${name} (n=${n})`, null));
    }
    return;
  }

  if (state.colorBy === "subject") {
    const counts = new Map();
    for (const t of traces) counts.set(t.s, (counts.get(t.s) || 0) + 1);
    for (const s of [...counts.keys()].sort()) {
      legend.appendChild(legendItem(state.subjectColors.get(s) || OTHER_COLOR, null,
                                    `sub-${s} (n=${counts.get(s)})`, null));
    }
    return;
  }

  if (state.colorBy === "stage") {
    const counts = new Map();
    for (const t of traces) {
      const name = stageName(t) || "unknown stage";
      counts.set(name, (counts.get(name) || 0) + 1);
    }
    // Fixed STAGES order, and every stage stays listed even at n=0 — a filter
    // that empties a third should say so rather than drop it from the legend.
    for (const name of STAGES) {
      legend.appendChild(legendItem(STAGE_COLORS[STAGES.indexOf(name)], null,
                                    `${name.replace("_", " ")} (n=${counts.get(name) || 0})`, null));
    }
    if (counts.has("unknown stage")) {
      legend.appendChild(legendItem(OTHER_COLOR, null,
                                    `unknown stage (n=${counts.get("unknown stage")})`, null));
    }
    return;
  }

  if (state.colorBy === "attempt") {
    // A row per attempt would be dozens of entries, so the ramp itself is the
    // legend. The scale is per-subject, which the caption has to say outright or
    // two same-coloured traces from different players look like the same attempt.
    const shown = traces.filter((t) => t.a != null);
    const hi = shown.length ? Math.max(...shown.map((t) => t.a)) : 0;
    legend.appendChild(legendRamp(`attempt 1 → last, per subject (max ${hi})`));
    return;
  }

  if (state.colorBy === "stability") {
    const shown = traces.filter((t) => t.sb != null);
    if (!shown.length) {
      legend.appendChild(legendItem(OTHER_COLOR, null,
                                    "no stability recorded — not a consensus run", null));
      return;
    }
    const mean = shown.reduce((a, t) => a + t.sb, 0) / shown.length;
    legend.appendChild(legendRamp(`unstable 0 → 1 stable (mean ${mean.toFixed(2)})`,
                                  STABILITY_STOPS));
    return;
  }

  // cluster mode: stable rank order, clickable to toggle. Every cluster in the
  // scene stays listed even at n=0, so a filter that empties one shows that
  // plainly instead of silently dropping it from the legend.
  for (const label of rank.ranked) {
    const slot = rank.direct.get(label);
    const color = slot === undefined ? OTHER_COLOR : CLUSTER_COLORS[slot];
    const marker = slot === undefined ? "." : CLUSTER_MARKERS[slot];
    legend.appendChild(legendItem(color, marker,
      `cluster ${label} (n=${counts.get(label) || 0})`, label));
  }
  if (state.showNoise && state.scene.traces.some((t) => t.c === -1)) {
    legend.appendChild(legendItem(NOISE_COLOR, "x", `noise (n=${counts.get(-1) || 0})`, -1));
  }
}

function legendItem(color, marker, text, toggleLabel) {
  const btn = document.createElement("button");
  btn.className = "legend-item";
  btn.type = "button";
  const hidden = toggleLabel !== null && state.hiddenClusters.has(toggleLabel);
  btn.setAttribute("aria-pressed", hidden ? "false" : "true");

  const mark = document.createElement("canvas");
  mark.className = "legend-mark";
  mark.width = 22 * (window.devicePixelRatio || 1);
  mark.height = 12 * (window.devicePixelRatio || 1);
  mark.style.width = "22px"; mark.style.height = "12px";
  const mctx = mark.getContext("2d");
  mctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  mctx.globalAlpha = 1;
  mctx.strokeStyle = color;
  mctx.lineWidth = 2;
  mctx.beginPath(); mctx.moveTo(1, 6); mctx.lineTo(21, 6); mctx.stroke();
  if (marker) drawMarker(mctx, marker, 6, 6, 7, color, 1);
  btn.appendChild(mark);

  const span = document.createElement("span");
  span.textContent = text;
  btn.appendChild(span);

  if (toggleLabel !== null) {
    btn.addEventListener("click", () => {
      if (state.hiddenClusters.has(toggleLabel)) state.hiddenClusters.delete(toggleLabel);
      else state.hiddenClusters.add(toggleLabel);
      render();
    });
  } else {
    btn.style.cursor = "default";
  }
  return btn;
}

/* Gradient swatch for a continuous ramp (attempt, stability) — same shape as
 * legendItem so it sits in the legend row without special-casing the layout. */
function legendRamp(text, stops = VIRIDIS_STOPS) {
  const btn = document.createElement("button");
  btn.className = "legend-item";
  btn.type = "button";
  btn.setAttribute("aria-pressed", "true");
  btn.style.cursor = "default";

  const dpr = window.devicePixelRatio || 1;
  const mark = document.createElement("canvas");
  mark.className = "legend-mark legend-mark--ramp";
  mark.width = 44 * dpr; mark.height = 12 * dpr;
  mark.style.width = "44px"; mark.style.height = "12px";
  const mctx = mark.getContext("2d");
  mctx.scale(dpr, dpr);
  const grad = mctx.createLinearGradient(0, 0, 44, 0);
  stops.forEach((c, i) => grad.addColorStop(i / (stops.length - 1), c));
  mctx.fillStyle = grad;
  mctx.fillRect(0, 2, 44, 8);
  btn.appendChild(mark);

  const span = document.createElement("span");
  span.textContent = text;
  btn.appendChild(span);
  return btn;
}

/* ---------- panels view ---------- */

/* Which composition pills a panel should carry, or null for modes where the
 * breakdown is not meaningful (cluster: it is the panel itself; attempt and
 * stability: continuous quantities, not categories). */
function compositionSpec(members) {
  if (!members.length) return null;
  if (state.colorBy === "subject") {
    const comp = new Map();
    for (const t of members) comp.set(t.s, (comp.get(t.s) || 0) + 1);
    return {
      entries: [...comp.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1)),
      label: (s) => `sub-${s}`,
      color: (s) => state.subjectColors.get(s) || OTHER_COLOR,
    };
  }
  if (state.colorBy === "stage") {
    const comp = new Map();
    for (const t of members) {
      const name = stageName(t) || "unknown";
      comp.set(name, (comp.get(name) || 0) + 1);
    }
    return {
      entries: STAGES.filter((n) => comp.has(n)).map((n) => [n, comp.get(n)])
        .concat(comp.has("unknown") ? [["unknown", comp.get("unknown")]] : []),
      label: (n) => n.replace("_", " "),
      color: (n) => (STAGES.indexOf(n) === -1 ? OTHER_COLOR : STAGE_COLORS[STAGES.indexOf(n)]),
    };
  }
  return null;
}

function renderPanels() {
  const host = $("panels");
  host.innerHTML = "";
  const traces = visibleTraces();
  const rank = sceneRanking();
  const counts = visibleCounts(traces);

  // Stable rank order, noise last — same panel order as
  // plot_scene_cluster_panels(). Clusters emptied by the filter are dropped
  // rather than shown as blank panels.
  const order = visibleRanked(rank, counts);
  if (state.showNoise && (counts.get(-1) || 0) > 0) order.push(-1);

  renderTitle(traces, rank, counts);
  $("legend").innerHTML = "";

  if (!order.length) {
    host.innerHTML = '<p class="note">No clusters to panel for the current filter.</p>';
    return;
  }

  const usingBg = usingBackground();
  const e = dataExtent(usingBg);
  const aspect = usingBg ? e.x1 / e.y1 : 1.4;

  for (const label of order) {
    const members = traces.filter((t) => t.c === label);
    const card = document.createElement("div");
    card.className = "panel-card";

    const h3 = document.createElement("h3");
    h3.innerHTML = `<span>${label === -1 ? "noise" : "cluster " + label}</span>` +
                   `<span class="n">n=${members.length}</span>`;
    card.appendChild(h3);

    const box = document.createElement("div");
    box.className = "panel-canvas-box";
    const canvas = document.createElement("canvas");
    box.appendChild(canvas);

    // Per-cluster composition pills, mirroring plot_scene_cluster_panels();
    // percentages are of the panel's visible traces. Shown for the two
    // categorical modes where "what is this cluster made of" is the question.
    const comp = compositionSpec(members);
    if (comp) {
      const wrap = document.createElement("div");
      wrap.className = "composition";
      for (const [key, n] of comp.entries) {
        const pct = Math.round((100 * n) / members.length);
        const span = document.createElement("span");
        span.textContent = `${comp.label(key)}: ${pct}% (n=${n})`;
        span.style.color = comp.color(key);
        wrap.appendChild(span);
      }
      box.appendChild(wrap);
    }

    card.appendChild(box);
    host.appendChild(card);

    // size after layout so clientWidth is real
    const cssW = box.clientWidth || 240;
    const cssH = Math.max(110, Math.round(cssW / aspect));
    canvas.style.height = cssH + "px";
    const ctx = sizeCanvas(canvas, cssW, cssH);
    const base = baseTransform(cssW, cssH, usingBg);
    const proj = makeProjector(base, 1, { x: 0, y: 0 });   // panels share fixed limits

    if (usingBg) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(state.bg, base.ox, base.oy, e.x1 * base.sx, e.y1 * base.sy);
    }

    const slot = rank.direct.get(label);
    const panelColor = label === -1 ? NOISE_COLOR
                     : slot === undefined ? OTHER_COLOR : CLUSTER_COLORS[slot];
    const marker = label === -1 ? "x"
                 : slot === undefined ? "." : CLUSTER_MARKERS[slot];

    for (const t of drawOrder(members)) {
      const color = state.colorBy === "cluster"
        ? panelColor
        : styleFor(t, rank, counts).color;
      drawTrace(ctx, t, proj, color, ALPHA.subject, LW.subject);
      const p = firstPoint(t);
      if (p) drawMarker(ctx, marker, proj.x(p[0]), proj.y(p[1]),
                        MARKER_PX.panel, color, 0.8);
    }
    ctx.globalAlpha = 1;
  }
}

/* ---------- hit testing & tooltip ---------- */

/* Geometric nearest-point search in screen space. Traces are densely sampled, so
 * point distance is as good as segment distance here, and this is exact — no
 * picking-buffer colour blending to misread. ~45k point tests worst case. */
function traceAt(px, py) {
  if (!state.scene || state.view !== "overlay") return null;
  const box = $("canvas-box");
  const base = baseTransform(box.clientWidth, box.clientHeight, usingBackground());
  const proj = makeProjector(base, state.zoom, state.pan);

  const THRESH = 6;
  let best = null, bestD = THRESH * THRESH;
  for (const t of visibleTraces()) {
    for (let i = 0; i < t.x.length; i++) {
      const yv = t.y[i];
      if (yv === null) continue;
      const dx = proj.x(t.x[i]) - px, dy = proj.y(yv) - py;
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = t; }
    }
  }
  return best;
}

function showTooltip(t, clientX, clientY) {
  const tip = $("tooltip");
  if (!t) { tip.dataset.show = "false"; return; }
  const vis = visibleTraces();
  const st = styleFor(t, sceneRanking(), visibleCounts(vis));
  const rows = [
    ["subject", `sub-${t.s}`],
    ["session/run", `${t.ses} / ${t.run}`],
    ["phase", t.ph],
    ["stage", stageName(t) ? stageName(t).replace("_", " ") : "—"],
    ["attempt", t.a == null ? "—" : `${t.a} of ${state.attemptMax.get(t.s) || t.a}`],
    ["outcome", t.out],
    ["duration", t.dur == null ? "—" : `${t.dur} s`],
    ["cluster", t.c === -1 ? "noise" : t.c],
    ["stability", t.sb == null ? "—"
      : viewingAlternativeCut() ? `${t.sb.toFixed(2)} (at k=${chosenK()})` : t.sb.toFixed(2)],
    ["x travelled", t.xt ?? "—"],
    ["hits taken", t.ht ?? "—"],
    ["enemies killed", t.ek ?? "—"],
    ["lives lost", t.ll ?? "—"],
  ];
  tip.innerHTML =
    `<div class="tt-head"><span class="tt-dot" style="background:${st.color}"></span>` +
    `<span>${t.clip}</span></div>` +
    `<dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>` +
    `<div class="tt-pin">${state.pinned === t ? "click to unpin" : "click to isolate"}</div>`;

  tip.dataset.show = "true";
  const r = tip.getBoundingClientRect();
  let x = clientX + 14, y = clientY + 14;
  if (x + r.width > window.innerWidth - 8) x = clientX - r.width - 14;
  if (y + r.height > window.innerHeight - 8) y = clientY - r.height - 14;
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}

/* ---------- sidebar ---------- */

function renderSceneList() {
  const host = $("scene-list");
  const q = $("scene-search").value.trim().toLowerCase();
  const tag = $("tag-filter").value;
  host.innerHTML = "";

  let group = null;
  let shown = 0;
  for (const s of state.index.scenes) {
    if (q && !s.id.includes(q)) continue;
    if (tag && !s.tags.includes(tag)) continue;
    const g = `World ${s.world} · Level ${s.level}`;
    if (g !== group) {
      group = g;
      const gh = document.createElement("div");
      gh.className = "scene-group";
      gh.textContent = g;
      host.appendChild(gh);
    }
    const btn = document.createElement("button");
    btn.className = "scene-row";
    btn.type = "button";
    btn.setAttribute("aria-current", state.scene && state.scene.scene === s.id ? "true" : "false");
    // The silhouette of the cut that was taken, straight from index.json, so the
    // picker says how well-separated each scene's partition is before it is opened.
    const sil = typeof s.sil === "number" ? ` · s=${s.sil.toFixed(2)}` : "";
    btn.innerHTML = `<span class="scene-id">${s.id}</span>` +
      `<span class="scene-meta">${s.n} traces · ${s.k} cl${sil}</span>`;
    btn.addEventListener("click", () => selectScene(s.id));
    host.appendChild(btn);
    shown++;
  }
  if (!shown) host.innerHTML = '<div class="scene-group">no scenes match</div>';
}

function renderSubjectList() {
  const host = $("subject-list");
  host.innerHTML = "";
  const subjects = state.scene
    ? [...new Set(state.scene.traces.map((t) => t.s))].sort()
    : state.index.subjects;

  for (const s of subjects) {
    const label = document.createElement("label");
    label.className = "check";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = state.filters.subjects.has(s);
    cb.addEventListener("change", () => {
      if (cb.checked) state.filters.subjects.add(s);
      else state.filters.subjects.delete(s);
      state.pinned = null; state.hover = null;
      render();
    });
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = state.subjectColors.get(s) || OTHER_COLOR;
    const txt = document.createElement("span");
    const n = state.scene ? state.scene.traces.filter((t) => t.s === s).length : null;
    txt.textContent = n === null ? `sub-${s}` : `sub-${s} (${n})`;
    label.append(cb, sw, txt);
    host.appendChild(label);
  }
}

function bindSeg(id, key, after) {
  const seg = $(id);
  seg.addEventListener("click", (ev) => {
    const btn = ev.target.closest("button");
    if (!btn) return;
    for (const b of seg.querySelectorAll("button")) {
      b.setAttribute("aria-pressed", String(b === btn));
    }
    const value = btn.dataset.value;
    if (key === "view") state.view = value;
    else if (key === "colorBy") state.colorBy = value;
    else if (key === "matrixOrder") state.matrixOrder = value;
    else state.filters[key] = value;
    state.pinned = null; state.hover = null;
    if (after) after();
    render();
  });
}

/* ---------- data loading ---------- */

/* Guards against out-of-order scene loads: clicking scene A then B fires two
 * awaited fetch/image chains, and without a token A's slower background could
 * land on top of B. Only the newest request is allowed to commit. */
let sceneToken = 0;

async function selectScene(id) {
  const token = ++sceneToken;
  let payload = sceneCache.get(id);
  if (!payload) {
    const res = await fetch(`data/scenes/${id}.json`, { cache: "no-cache" });
    if (!res.ok) { console.error("failed to load scene", id); return; }
    payload = await res.json();
    sceneCache.set(id, payload);
  }
  if (token !== sceneToken) return;      // superseded by a later click
  state.scene = payload;
  state.hiddenClusters.clear();
  state.pinned = null; state.hover = null;
  state.zoom = 1; state.pan = { x: 0, y: 0 };

  // Candidate lists differ per scene, so a k picked on the last one means nothing
  // here: every scene opens on the cut its own run selected.
  applyCut(null);
  applyCutAvailability();
  renderCutControl();

  // Subject -> color, assigned by sorted subject id over the SCENE's subjects,
  // exactly as subject_color_map() does in analysis/trace_plots.py.
  const subjects = [...new Set(payload.traces.map((t) => t.s))].sort();
  state.subjectColors = new Map(
    subjects.map((s, i) => [s, SUBJECT_COLORS[i % SUBJECT_COLORS.length]]));

  // Carry the subject selection across scene changes, restricted to whoever
  // actually played this scene. Phase and outcome filters already persist; the
  // subject filter used to silently reset, so comparing one player across
  // scenes quietly turned back into all five.
  const kept = [...state.filters.subjects].filter((s) => subjects.includes(s));
  state.filters.subjects = new Set(kept.length ? kept : subjects);

  // Denominator for the attempt ramp, per subject (see attemptColor).
  state.attemptMax = new Map();
  for (const t of payload.traces) {
    if (t.a != null) state.attemptMax.set(t.s, Math.max(state.attemptMax.get(t.s) || 0, t.a));
  }

  // A payload exported before the attempt/stage fields existed would silently
  // draw everything one colour and filter to nothing, so say so and lock the
  // controls instead.
  state.hasDerived = payload.traces.every((t) => t.a != null && t.st !== undefined);
  applyDerivedAvailability();

  state.bg = null;
  if (payload.bg) {
    const img = await loadImage(`data/backgrounds/${payload.bg}`).catch(() => null);
    if (token !== sceneToken) return;
    state.bg = img;
  }

  $("empty").hidden = true;
  $("reset-view").hidden = false;
  renderSceneList();
  renderSubjectList();
  render();
}

/* ---------- the k picker ---------- */

function renderCutControl() {
  const field = $("cut-field");
  const range = $("cut-range");
  const out = $("cut-value");
  const hint = $("cut-hint");
  const reset = $("cut-reset");
  const rows = cuttableCandidates();

  if (!rows.length) {
    field.classList.add("unavailable");
    range.disabled = reset.disabled = true;
    range.max = 0; range.value = 0;
    out.textContent = chosenK() == null ? "—" : `k=${chosenK()}`;
    out.className = "";
    hint.textContent = state.scene && state.scene.cons
      ? "This scene ships only the cut that was taken — re-export it to carry the others."
      : "Not a consensus run: there is no tree to cut at another k.";
    return;
  }

  field.classList.remove("unavailable");
  range.disabled = false;
  range.max = String(rows.length - 1);
  const k = activeK();
  let i = rows.findIndex((r) => r.k === k);
  if (i < 0) i = 0;
  range.value = String(i);
  const row = rows[i];
  const alt = viewingAlternativeCut();
  out.textContent = `k=${row.k}`;
  out.className = alt ? "alt" : "";
  reset.disabled = !alt;

  const dust = row.min_size != null && row.min_size <= 2
    ? ` · smallest cluster ${row.min_size} trace${row.min_size === 1 ? "" : "s"}`
    : "";
  const sil = typeof row.silhouette === "number"
    ? ` · silhouette ${row.silhouette.toFixed(4)}` : "";
  hint.textContent = alt
    ? `Showing the k=${row.k} cut${sil}${dust}. The run chose k=${chosenK()} — ` +
      `stability numbers and the scene list still describe that one.`
    : `The cut the run selected${sil}${dust}. Drag to see the same tree cut elsewhere.`;
}

/* The per-trace confidence was measured against the chosen partition, so it means
 * nothing about a cut the run did not make. Grey the mode out rather than paint a
 * number from one partition onto another. */
function applyCutAvailability() {
  const alt = viewingAlternativeCut();
  const btn = [...$("color-seg").querySelectorAll("button")]
    .find((b) => b.dataset.value === "stability");
  if (!btn) return;
  btn.disabled = alt;
  btn.title = alt
    ? `Per-trace confidence was measured at the chosen cut (k=${chosenK()}), not at this one.`
    : "";
  if (alt && state.colorBy === "stability") {
    state.colorBy = "cluster";
    for (const b of $("color-seg").querySelectorAll("button")) {
      b.setAttribute("aria-pressed", String(b.dataset.value === "cluster"));
    }
  }
}

function setCut(k) {
  applyCut(k);
  state.hiddenClusters.clear();
  state.pinned = null; state.hover = null;
  applyCutAvailability();
  renderCutControl();
  render();
}

function bindCutControl() {
  $("cut-range").addEventListener("input", (e) => {
    const rows = cuttableCandidates();
    const row = rows[Number(e.target.value)];
    if (row) setCut(row.k);
  });
  $("cut-reset").addEventListener("click", () => setCut(null));
}

/* Enable/disable the controls that need the derived fields. */
function applyDerivedAvailability() {
  const why = "Needs the attempt/stage fields — run: invoke run-web-export --backfill";
  const targets = [...$("stage-seg").querySelectorAll("button")].concat(
    [...$("color-seg").querySelectorAll("button")]
      .filter((b) => b.dataset.value === "attempt" || b.dataset.value === "stage"));
  for (const b of targets) {
    b.disabled = !state.hasDerived;
    b.title = state.hasDerived ? "" : why;
  }
  if (!state.hasDerived) {
    state.filters.stage = "all";
    if (state.colorBy === "attempt" || state.colorBy === "stage") state.colorBy = "cluster";
    for (const id of ["stage-seg", "color-seg"]) {
      for (const b of $(id).querySelectorAll("button")) {
        b.setAttribute("aria-pressed",
          String(b.dataset.value === (id === "stage-seg" ? "all" : state.colorBy)));
      }
    }
  }
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

/* ---------- render dispatch ---------- */

/* ---------- co-association matrix view ---------- */

/* The matrix the consensus labels were actually derived from: cell (i, j) is how
 * often traces i and j landed in the same cluster, over the replicates in which
 * both were drawn. Shipped as one uint8 per cell in payload order (see
 * webexport.export_coassociation), so the browser does the ordering.
 *
 * It is a *view*, like everything else here: the filters select a submatrix, they
 * never recompute anything. Rows are ordered by the whole-scene cluster ranking —
 * the same order that fixes the colour slots — so blocks line up with the legend
 * and narrowing to one subject cannot reshuffle them. */

const matrixCache = new Map();   // scene id -> Uint8Array

async function loadCoassoc(sceneId) {
  if (matrixCache.has(sceneId)) return matrixCache.get(sceneId);
  const res = await fetch(`data/coassoc/${sceneId}.bin`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`no co-association matrix for ${sceneId}`);
  const buf = new Uint8Array(await res.arrayBuffer());
  matrixCache.set(sceneId, buf);
  return buf;
}

function coassocColor(v) {
  return STABILITY_STOPS[Math.round((v / 255) * (STABILITY_STOPS.length - 1))];
}

/* Payload index of every visible trace, ordered by cluster (rank-by-size, noise
 * last) so the diagonal blocks are the clusters. */
/* Two orderings, and they answer different questions.
 *
 * "leaves" is the dendrogram order Ward itself produced, exported as `cons.leaves`
 * (payload indices). Cut clusters are contiguous in it — cut_tree cannot split a
 * leaf run — so the blocks still line up, but *within* a block the rows follow the
 * merge structure, so a cluster that is really two things shows as two sub-blocks.
 * That is structure the labels have already thrown away, and it is the only view
 * here that can show it.
 *
 * "cluster" is the older rank-by-size order. Inside a block rows fall in payload
 * order — subject, then time — so sub-structure is scrambled, but the blocks appear
 * in the same order as the legend and the panels, which makes them easy to match up.
 *
 * Both are whole-scene orders that filters only subset, never recompute, so
 * narrowing to one subject cannot reshuffle what you are looking at. */
function matrixOrder() {
  const visible = new Set(visibleTraces());
  const traces = state.scene.traces;
  const leaves = (state.scene.cons || {}).leaves;

  if (state.matrixOrder === "leaves" && Array.isArray(leaves) && leaves.length === traces.length) {
    const rows = [];
    for (const i of leaves) if (visible.has(traces[i])) rows.push({ t: traces[i], i });
    return rows;
  }

  const rank = sceneRanking();
  const slot = (t) => (t.c === -1 ? Number.MAX_SAFE_INTEGER : rank.ranked.indexOf(t.c));
  const rows = [];
  traces.forEach((t, i) => { if (visible.has(t)) rows.push({ t, i }); });
  rows.sort((a, b) => (slot(a.t) - slot(b.t)) || (a.i - b.i));
  return rows;
}

/* Does this scene actually have a leaf order to offer? */
function hasLeafOrder() {
  const leaves = (state.scene && state.scene.cons || {}).leaves;
  return Array.isArray(leaves) && leaves.length === state.scene.traces.length;
}

let matrixRows = [];      // current row order, for hit-testing
let matrixSide = 0;

async function renderMatrix() {
  const note = $("matrix-note");
  const canvas = $("matrix-canvas");

  if (!state.scene.coassoc) {
    note.textContent =
      "No co-association matrix exported for this scene — it was not clustered with " +
      "the consensus method, or `invoke run-coassociation` has not been run.";
    canvas.width = canvas.height = 0;
    $("matrix-legend").innerHTML = "";
    return;
  }

  let data;
  try {
    data = await loadCoassoc(state.scene.scene);
  } catch (err) {
    note.textContent = String(err.message);
    return;
  }

  const n = state.scene.traces.length;
  matrixRows = matrixOrder();
  const m = matrixRows.length;
  matrixSide = m;
  if (!m) { note.textContent = "No traces match the current filters."; return; }

  // One image pixel per cell, drawn at native size; CSS scales it with
  // image-rendering: pixelated, so no interpolation invents values.
  canvas.width = m;
  canvas.height = m;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(m, m);

  // Pre-resolve the 256 ramp entries to RGB once, rather than per cell.
  const lut = new Uint8Array(256 * 3);
  for (let v = 0; v < 256; v++) {
    const hex = coassocColor(v);
    lut[v * 3] = parseInt(hex.slice(1, 3), 16);
    lut[v * 3 + 1] = parseInt(hex.slice(3, 5), 16);
    lut[v * 3 + 2] = parseInt(hex.slice(5, 7), 16);
  }

  for (let r = 0; r < m; r++) {
    const src = matrixRows[r].i * n;
    const dst = r * m * 4;
    for (let cIdx = 0; cIdx < m; cIdx++) {
      const v = data[src + matrixRows[cIdx].i];
      const o = dst + cIdx * 4;
      img.data[o] = lut[v * 3];
      img.data[o + 1] = lut[v * 3 + 1];
      img.data[o + 2] = lut[v * 3 + 2];
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  const usingLeaves = state.matrixOrder === "leaves" && hasLeafOrder();

  renderPartition();
  renderMatrixLegend();

  // The caption has to separate the two, because they come from opposite sides of the
  // clustering step: the *values* are the matrix Ward was handed and are untouched by
  // it, while every *ordering* on offer is derived from what Ward returned — `leaves`
  // is its dendrogram, `cluster` is its cut ranked by size. There is no useful
  // pre-Ward arrangement to offer instead: in payload order (subject, then time) the
  // block structure is invisible, which is the whole reason such a figure is permuted
  // at all. So the caption names both halves rather than claiming one for the pair.
  // The leaf order is the tree's, not the cut's, so it is the same at every k — which
  // is what makes scrubbing k readable here: the rows hold still and only the blocks
  // on the right redraw.
  $("matrix-caption").textContent = usingLeaves
    ? "co-association — values as measured, rows in Ward's leaf order"
    : "co-association — values as measured, rows grouped by Ward's cut";
  $("partition-caption").innerHTML = viewingAlternativeCut()
    ? `partition &mdash; the tree cut at k=${state.k}`
    : "partition &mdash; the cut Ward returned";

  const total = state.scene.traces.length;
  const ordering = usingLeaves
    ? "Rows follow the dendrogram, so a cluster that is really two things shows as two " +
      "sub-blocks — structure the labels no longer carry."
    : (state.matrixOrder === "leaves"
        ? "No leaf order exported for this scene, so rows fall back to cluster order."
        : "Rows ordered by cluster, largest first, noise last — the same order as the legend.");
  const pair = state.showPartition
    ? " Left is what Ward was given; right is the cut it returned. A block that is solid " +
      "on the right but mottled on the left is a cluster the cut asserted over weak agreement."
    : "";
  note.textContent =
    `${m} of ${total} traces. Cell (i, j) = share of resampling replicates in which ` +
    `both traces landed in the same cluster, among replicates that drew both. ` +
    ordering + pair;
}

/* The partition as a matrix: 1 where two traces share a cluster, 0 otherwise —
 * `part2adj` in the BASC reference, drawn in the same row order as the matrix beside
 * it so the two are directly comparable. Painted in each cluster's own legend colour
 * rather than a flat binary, so a block can be matched to the trajectories that made
 * it without counting along an axis.
 *
 * Derived from the labels in the browser: it is a picture of `cons.k`, not a second
 * measurement, and nothing about it needs exporting. */
function renderPartition() {
  const panel = $("partition-panel");
  const grid = $("matrix-grid");
  panel.hidden = !state.showPartition;
  grid.classList.toggle("single", !state.showPartition);
  if (!state.showPartition) return;

  const canvas = $("partition-canvas");
  const m = matrixRows.length;
  canvas.width = m;
  canvas.height = m;
  if (!m) return;

  const rank = sceneRanking();
  const css = getComputedStyle(document.documentElement);
  const ground = css.getPropertyValue("--surface").trim() || "#fcfcfb";
  const rgb = (hex) => [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16),
                        parseInt(hex.slice(5, 7), 16)];
  const bg = rgb(ground);
  const colorOf = (label) => {
    if (label === -1) return rgb(NOISE_COLOR);
    const slot = rank.direct.get(label);
    return rgb(slot === undefined ? OTHER_COLOR : CLUSTER_COLORS[slot]);
  };

  const fill = new Map();
  for (const row of matrixRows) if (!fill.has(row.t.c)) fill.set(row.t.c, colorOf(row.t.c));

  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(m, m);
  for (let r = 0; r < m; r++) {
    const lr = matrixRows[r].t.c;
    const dst = r * m * 4;
    for (let c = 0; c < m; c++) {
      const same = matrixRows[c].t.c === lr && lr !== -1;
      const [R, G, B] = same ? fill.get(lr) : bg;
      const o = dst + c * 4;
      img.data[o] = R; img.data[o + 1] = G; img.data[o + 2] = B; img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function renderMatrixLegend() {
  const legend = $("matrix-legend");
  legend.innerHTML = "";
  const dpr = window.devicePixelRatio || 1;
  const bar = document.createElement("canvas");
  bar.width = 120 * dpr; bar.height = 12 * dpr;
  bar.style.width = "120px"; bar.style.height = "12px";
  const bctx = bar.getContext("2d");
  bctx.scale(dpr, dpr);
  const grad = bctx.createLinearGradient(0, 0, 120, 0);
  STABILITY_STOPS.forEach((c, i) => grad.addColorStop(i / (STABILITY_STOPS.length - 1), c));
  bctx.fillStyle = grad;
  bctx.fillRect(0, 0, 120, 12);

  const lo = document.createElement("span"); lo.textContent = "never together (0)";
  const hi = document.createElement("span"); hi.textContent = "always (1)";
  legend.append(lo, bar, hi);
}

/* Hit-testing is index arithmetic on the canvas, not a colour read: the same
 * reason the overlay uses a geometric nearest-point search rather than a picking
 * buffer — a scaled pixel must never be decoded back into a value. */
function bindMatrix() {
  const canvas = $("matrix-canvas");
  const box = $("matrix-box");

  canvas.addEventListener("mousemove", (e) => {
    if (!matrixSide || !matrixRows.length) return;
    const r = canvas.getBoundingClientRect();
    const col = Math.floor(((e.clientX - r.left) / r.width) * matrixSide);
    const row = Math.floor(((e.clientY - r.top) / r.height) * matrixSide);
    if (row < 0 || col < 0 || row >= matrixSide || col >= matrixSide) return;

    const a = matrixRows[row].t, b = matrixRows[col].t;
    const data = matrixCache.get(state.scene.scene);
    const n = state.scene.traces.length;
    const v = data[matrixRows[row].i * n + matrixRows[col].i] / 255;
    showMatrixTooltip(a, b, v, e.clientX, e.clientY);
  });
  canvas.addEventListener("mouseleave", () => { $("tooltip").dataset.show = "false"; });
  box.addEventListener("mouseleave", () => { $("tooltip").dataset.show = "false"; });

  // The partition canvas is the same grid in the same order, so the same index
  // arithmetic locates the same pair. It reports the co-association value too: the
  // whole reason to show the two side by side is to find cells where the cut says
  // "same cluster" and the ensemble does not agree.
  const pcanvas = $("partition-canvas");
  pcanvas.addEventListener("mousemove", (e) => {
    if (!matrixSide || !matrixRows.length) return;
    const r = pcanvas.getBoundingClientRect();
    const col = Math.floor(((e.clientX - r.left) / r.width) * matrixSide);
    const row = Math.floor(((e.clientY - r.top) / r.height) * matrixSide);
    if (row < 0 || col < 0 || row >= matrixSide || col >= matrixSide) return;
    const a = matrixRows[row].t, b = matrixRows[col].t;
    const data = matrixCache.get(state.scene.scene);
    const n = state.scene.traces.length;
    const v = data ? data[matrixRows[row].i * n + matrixRows[col].i] / 255 : null;
    showPartitionTooltip(a, b, v, e.clientX, e.clientY);
  });
  pcanvas.addEventListener("mouseleave", () => { $("tooltip").dataset.show = "false"; });
  $("partition-box").addEventListener("mouseleave", () => { $("tooltip").dataset.show = "false"; });
}

function showPartitionTooltip(a, b, v, clientX, clientY) {
  const tip = $("tooltip");
  const same = a.c === b.c && a.c !== -1;
  const rank = sceneRanking();
  const slot = rank.direct.get(a.c);
  const dot = a.c === -1 ? NOISE_COLOR
            : (slot === undefined ? OTHER_COLOR : CLUSTER_COLORS[slot]);
  const rows = [
    ["row", traceLabel(a)],
    ["column", traceLabel(b)],
    ["the cut says", same ? `same cluster (${a.c})` : "different clusters"],
    ["the ensemble says", v == null ? "—" : `${(v * 100).toFixed(0)}% of shared replicates`],
  ];
  tip.innerHTML =
    `<div class="tt-head"><span class="tt-dot" style="background:${same ? dot : OTHER_COLOR}"></span>` +
    `<span>${same ? "in the same cluster" : "split apart"}</span></div>` +
    `<dl>${rows.map(([k, val]) => `<dt>${k}</dt><dd>${val}</dd>`).join("")}</dl>`;
  tip.dataset.show = "true";
  const r = tip.getBoundingClientRect();
  let x = clientX + 14, y = clientY + 14;
  if (x + r.width > window.innerWidth - 8) x = clientX - r.width - 14;
  if (y + r.height > window.innerHeight - 8) y = clientY - r.height - 14;
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}

/* Who a trace is, in terms a reader can act on.
 *
 * NOT the ClipCode. That field is session-run-ordinal-startframe concatenated
 * (00101010000122 = ses-001, run-01, 0th bk2, frame 122), so putting it beside a
 * label like "row" makes a 14-digit identifier look like a quantity. It stays
 * available on the diagonal, where the tooltip is describing one trace and the id
 * is the thing you would copy. */
function traceLabel(t) {
  const who = `sub-${t.s}`;
  const when = t.a != null ? `attempt ${t.a}` : `${t.ses}/${t.run}`;
  const cluster = t.c === -1 ? "noise" : `cluster ${t.c}`;
  return `${who} · ${when} · ${cluster}`;
}

function showMatrixTooltip(a, b, v, clientX, clientY) {
  const tip = $("tooltip");
  const name = (t) => (t.c === -1 ? "noise" : `cluster ${t.c}`);
  const rows = a === b
    ? [["trace", `sub-${a.s} · ${a.a != null ? `attempt ${a.a}` : `${a.ses}/${a.run}`}`],
       ["cluster", name(a)],
       ["clip id", a.clip],
       ["clustered (not noise) in", `${(v * 100).toFixed(0)}% of replicates`]]
    : [["row", traceLabel(a)], ["column", traceLabel(b)],
       ["same cluster in", `${(v * 100).toFixed(0)}% of shared replicates`]];
  tip.innerHTML =
    `<div class="tt-head"><span class="tt-dot" style="background:${coassocColor(v * 255)}"></span>` +
    `<span>co-association ${v.toFixed(2)}</span></div>` +
    `<dl>${rows.map(([k, val]) => `<dt>${k}</dt><dd>${val}</dd>`).join("")}</dl>`;
  tip.dataset.show = "true";
  const r = tip.getBoundingClientRect();
  let x = clientX + 14, y = clientY + 14;
  if (x + r.width > window.innerWidth - 8) x = clientX - r.width - 14;
  if (y + r.height > window.innerHeight - 8) y = clientY - r.height - 14;
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}


/* ---------- k-selection view ---------- */

/* How k was chosen, drawn from the record the run itself stored.
 *
 * Ward returns a whole tree and no k: `linkage` gives n-1 merges and nothing says
 * where to stop. Every cut from k=2 to k_max is therefore scored by silhouette on
 * 1 - C (the co-association distance the ensemble lives in) and the best one taken.
 * That curve is `cons.k_candidates`, written by ward_consensus_labels — this view
 * only draws it, exactly like every other view here draws a stored result.
 *
 * The second, dashed series is the same cut scored on the raw Fréchet distances.
 * It is recorded and NOT used to select, because it systematically prefers coarser
 * cuts and coarser flatters stable_share for free. Drawing both is the point: where
 * the two disagree is where the ensemble and the raw trajectory geometry disagree.
 */

const SELECTION_PAD = { l: 48, r: 18, t: 20, b: 38 };
let selectionPoints = [];        // {k, cx, cy, row} for hit-testing

function selectionRows() {
  const cons = state.scene && state.scene.cons;
  return cons && Array.isArray(cons.k_candidates) ? cons.k_candidates : [];
}

function renderSelection() {
  const note = $("selection-note");
  const canvas = $("selection-canvas");
  const facts = $("selection-facts");
  const cons = state.scene.cons;
  selectionPoints = [];

  if (!cons) {
    canvas.width = canvas.height = 0;
    facts.innerHTML = "";
    $("selection-legend").innerHTML = "";
    note.textContent =
      "No selection record for this scene — it was not clustered with the consensus " +
      "method, so there was no ensemble to cut and no k to choose.";
    return;
  }

  renderSelectionFacts(cons);
  const rows = selectionRows();

  if (!rows.length) {
    canvas.width = canvas.height = 0;
    $("selection-legend").innerHTML = "";
    note.textContent = selectionExplainer(cons);
    return;
  }

  const box = $("selection-box");
  const w = box.clientWidth, h = box.clientHeight;
  const ctx = sizeCanvas(canvas, w, h);
  const css = getComputedStyle(document.documentElement);
  const ink2 = css.getPropertyValue("--ink-2").trim() || "#55554e";
  const ink3 = css.getPropertyValue("--ink-3").trim() || "#8a8a80";
  const line = css.getPropertyValue("--line").trim() || "#dcdcd5";
  const accent = css.getPropertyValue("--accent").trim() || "#2a78d6";

  ctx.clearRect(0, 0, w, h);

  const x0 = SELECTION_PAD.l, x1 = w - SELECTION_PAD.r;
  const y0 = SELECTION_PAD.t, y1 = h - SELECTION_PAD.b;

  // Silhouette is bounded in [-1, 1] and everything here is positive, so the axis
  // is fixed at 0..1 rather than auto-scaled. Auto-scaling would magnify a 0.005
  // difference into a visible cliff and make scenes incomparable side by side.
  const px = (i) => (rows.length === 1
    ? (x0 + x1) / 2
    : x0 + ((x1 - x0) * i) / (rows.length - 1));
  const py = (v) => y1 - (y1 - y0) * Math.max(0, Math.min(1, v));

  ctx.font = "10px system-ui, sans-serif";
  ctx.strokeStyle = line;
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const v = g / 4;
    const y = py(v);
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
    ctx.stroke();
    ctx.fillStyle = ink3;
    ctx.textAlign = "right";
    ctx.fillText(v.toFixed(2), x0 - 6, y + 3);
  }

  // The cut that was taken: a rule through it, drawn under the series.
  const chosenIdx = rows.findIndex((r) => r.k === cons.k);
  if (chosenIdx >= 0) {
    ctx.save();
    ctx.strokeStyle = accent;
    ctx.globalAlpha = 0.35;
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px(chosenIdx), y0);
    ctx.lineTo(px(chosenIdx), y1);
    ctx.stroke();
    ctx.restore();
  }

  // The elbow: the last k before the curve falls off a cliff. Drawn as a rule of its
  // own because it is a *different reading of the same curve*, not a second series —
  // there is nothing to plot at each k, only one k that the shape of the curve points
  // at. Whether it selected anything is `cons.rule`; the legend says which.
  const elbow = cons.elbow || {};
  const elbowIdx = elbow.k == null ? -1 : rows.findIndex((r) => r.k === elbow.k);
  if (elbowIdx >= 0) {
    ctx.save();
    ctx.strokeStyle = ELBOW_COLOR;
    ctx.globalAlpha = 0.7;
    ctx.setLineDash([2, 3]);
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(px(elbowIdx), y0);
    ctx.lineTo(px(elbowIdx), y1);
    ctx.stroke();
    // An arrow into the drop it names, so the rule reads as "and then it falls".
    if (elbowIdx + 1 < rows.length) {
      ctx.setLineDash([]);
      ctx.globalAlpha = 0.9;
      const xa = px(elbowIdx), xb = px(elbowIdx + 1);
      ctx.beginPath();
      ctx.moveTo(xa, py(rows[elbowIdx].silhouette));
      ctx.lineTo(xb, py(rows[elbowIdx + 1].silhouette));
      ctx.lineWidth = 3;
      ctx.stroke();
    }
    ctx.restore();
  }

  const series = (key, color, dashed) => {
    const pts = rows
      .map((r, i) => ({ i, v: r[key] }))
      .filter((p) => typeof p.v === "number");
    if (!pts.length) return false;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = dashed ? 1.4 : 2;
    if (dashed) ctx.setLineDash([5, 4]);
    ctx.beginPath();
    pts.forEach((p, n) => (n ? ctx.lineTo(px(p.i), py(p.v)) : ctx.moveTo(px(p.i), py(p.v))));
    ctx.stroke();
    ctx.setLineDash([]);
    // A cut whose smallest cluster is 1-2 traces is drawn hollow. This is the whole
    // reason the curve has a flat tail: a trace that never co-clustered sits at
    // distance exactly 1.0 from everything, so peeling it off costs the silhouette
    // nothing (sklearn scores a singleton as 0, and it already scored ~0 inside its
    // parent) while tightening every cluster it left. The metric cannot object to
    // dust, so the marker says what the score will not.
    for (const p of pts) {
      const dusty = (rows[p.i].min_size ?? 99) <= 2;
      ctx.beginPath();
      ctx.arc(px(p.i), py(p.v), dashed ? 2 : 3.2, 0, Math.PI * 2);
      if (dusty && !dashed) {
        ctx.fillStyle = css.getPropertyValue("--surface-2").trim() || "#f2f2ef";
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.6;
        ctx.stroke();
      } else {
        ctx.fillStyle = color;
        ctx.fill();
      }
    }
    ctx.restore();
    return true;
  };

  const hasFrechet = series("silhouette_frechet", ink3, true);
  series("silhouette", accent, false);

  // Ring the chosen point so the cut is findable without reading the axis.
  if (chosenIdx >= 0 && typeof rows[chosenIdx].silhouette === "number") {
    ctx.save();
    ctx.strokeStyle = accent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px(chosenIdx), py(rows[chosenIdx].silhouette), 7, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // ...and a second, wider ring on the cut currently drawn everywhere else, when that
  // is not the chosen one. Two rings is the honest picture: the run made one choice,
  // and you are looking at another.
  const viewedIdx = rows.findIndex((r) => r.k === activeK());
  if (viewingAlternativeCut() && viewedIdx >= 0
      && typeof rows[viewedIdx].silhouette === "number") {
    ctx.save();
    ctx.strokeStyle = accent;
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px(viewedIdx), py(rows[viewedIdx].silhouette), 10, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // Axis frame and k labels.
  ctx.strokeStyle = line;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x0, y1);
  ctx.lineTo(x1, y1);
  ctx.stroke();

  ctx.textAlign = "center";
  const cuttable = new Set(cuttableCandidates().map((r) => r.k));
  rows.forEach((r, i) => {
    const chosen = r.k === cons.k;
    const viewed = r.k === activeK();
    ctx.fillStyle = chosen || viewed ? accent : ink3;
    ctx.font = `${chosen || viewed ? "600 " : ""}10px system-ui, sans-serif`;
    ctx.fillText(String(r.k), px(i), y1 + 14);
    selectionPoints.push({
      k: r.k, cx: px(i), cy: py(r.silhouette), row: r, chosen, viewed,
      elbow: r.k === elbow.k, cuttable: cuttable.has(r.k),
    });
  });

  ctx.fillStyle = ink2;
  ctx.font = "10px system-ui, sans-serif";
  ctx.fillText("k (clusters in the cut)", (x0 + x1) / 2, h - 6);
  ctx.save();
  ctx.translate(12, (y0 + y1) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("silhouette", 0, 0);
  ctx.restore();

  renderSelectionLegend(accent, ink3, hasFrechet, elbowIdx >= 0);
  note.textContent = selectionExplainer(cons)
    + (cuttable.size ? "  Click a k to draw that cut in every other view." : "");
}

function renderSelectionLegend(accent, ink3, hasFrechet, hasElbow) {
  const legend = $("selection-legend");
  legend.innerHTML = "";
  const item = (color, dashed, text) => {
    const el = document.createElement("span");
    el.className = "legend-item";
    el.style.cursor = "default";
    const mark = document.createElement("canvas");
    const dpr = window.devicePixelRatio || 1;
    mark.className = "legend-mark";
    mark.width = 22 * dpr; mark.height = 12 * dpr;
    mark.style.width = "22px"; mark.style.height = "12px";
    const c = mark.getContext("2d");
    c.scale(dpr, dpr);
    c.strokeStyle = color;
    c.lineWidth = dashed ? 1.4 : 2;
    if (dashed) c.setLineDash([4, 3]);
    c.beginPath(); c.moveTo(1, 6); c.lineTo(21, 6); c.stroke();
    el.appendChild(mark);
    const span = document.createElement("span");
    span.textContent = text;
    el.appendChild(span);
    legend.appendChild(el);
  };
  const cons = state.scene.cons || {};
  const rule = cons.rule || "silhouette";
  item(accent, false, `silhouette on 1 − C  (${rule === "silhouette" ? "selects k" : "recorded"})`);
  if (hasFrechet) item(ink3, true, "silhouette on Fréchet distance  (recorded only)");
  if (hasElbow) {
    item(ELBOW_COLOR, true, `elbow — last k before the drop  (${
      rule === "silhouette" ? "recorded" : "selects k"})`);
  }
  hollowNote(accent);
}

/* Hollow marker key — drawn the same way the chart draws it, so the two agree. */
function hollowNote(accent) {
  const legend = $("selection-legend");
  const el = document.createElement("span");
  el.className = "legend-item";
  el.style.cursor = "default";
  const dpr = window.devicePixelRatio || 1;
  const mark = document.createElement("canvas");
  mark.className = "legend-mark";
  mark.width = 22 * dpr; mark.height = 12 * dpr;
  mark.style.width = "22px"; mark.style.height = "12px";
  const c = mark.getContext("2d");
  c.scale(dpr, dpr);
  c.beginPath();
  c.arc(11, 6, 3.2, 0, Math.PI * 2);
  c.fillStyle = getComputedStyle(document.documentElement)
    .getPropertyValue("--surface-2").trim() || "#f2f2ef";
  c.fill();
  c.strokeStyle = accent;
  c.lineWidth = 1.6;
  c.stroke();
  el.appendChild(mark);
  const span = document.createElement("span");
  span.textContent = "hollow: smallest cluster ≤ 2 traces";
  el.appendChild(span);
  legend.appendChild(el);
}

/* One sentence saying what happened, in the terms of the run that happened. */
function selectionExplainer(cons) {
  const argmax = cons.tol > 0
    ? `smallest k scoring within ${cons.tol} of the best`
    : "highest silhouette (ties to the smaller k)";
  const elbow = cons.elbow || {};
  const cliff = elbow.k == null
    ? "no drop in this curve reaches that depth, so it has no elbow"
    : `the curve's steepest drop is ${elbow.steepest}, and k=${elbow.k} is the last ` +
      `cut before the first drop of at least ${elbow.threshold}`;
  const rules = {
    silhouette: argmax,
    elbow: `last k before the curve drops off a cliff — ${cliff}`,
    elbow_capped: `highest silhouette at or below the elbow — ${cliff}`,
    elbow_no_drop: `the elbow rule found no cliff, so it fell back to the argmax (${argmax})`,
  };
  const rule = rules[cons.selection] || argmax;
  const floor = cons.min_cluster_size > 1
    ? `Cuts whose smallest cluster fell below ${cons.min_cluster_size} traces were not eligible.`
    : "Every cut from k=2 upward was eligible — no minimum cluster size was imposed on the Ward cut.";
  const aside = cons.selection === "silhouette" && elbow.k != null && elbow.k !== cons.k
    ? ` The elbow rule would have cut at k=${elbow.k} instead; it is drawn but did not select.`
    : "";

  switch (cons.selection) {
    case "silhouette":
    case "elbow":
    case "elbow_capped":
    case "elbow_no_drop":
      return `Ward linkage over the co-association matrix (${cons.input} form), cut at ` +
        `k=${cons.k} of the ${(cons.k_candidates || []).length} candidates from 2 to ${cons.k_max}. ` +
        `Rule: ${rule}. ${floor}${aside}`;
    case "fallback_k2":
      return `No cut cleared the minimum cluster size of ${cons.min_cluster_size}, so the ` +
        `k=2 cut was taken rather than dropping the scene. The silhouette did not choose here.`;
    case "no_ensemble_signal":
      return "Every replicate called every trace noise, so no pair ever co-clustered and " +
        "1 − C is uniformly 1. Ward would carve that flat matrix into arbitrary blocks, " +
        "so the scene is reported as all-noise instead — the honest answer, not a failure to tune.";
    case "too_small":
      return "Too few traces to select between cuts: the silhouette needs 2 ≤ k ≤ n − 1.";
    default:
      return "The ensemble was cut by HDBSCAN on 1 − C, which chooses no k and records no " +
        "silhouette curve. Re-run with --consensus-cluster ward to get one.";
  }
}

function renderSelectionFacts(cons) {
  const facts = $("selection-facts");
  facts.innerHTML = "";
  const reps = cons.k_replicates || {};
  const pct = (v) => (typeof v === "number" ? `${(v * 100).toFixed(0)}%` : "—");
  const num = (v, d = 4) => (typeof v === "number" ? v.toFixed(d) : "—");

  const elbow = cons.elbow || {};
  const rows = [
    ["clusters chosen (k)", String(cons.k ?? "—"), true],
    ["silhouette there", num(cons.silhouette), true],
    ["cut on screen", viewingAlternativeCut() ? `k=${state.k}` : "the chosen cut",
      viewingAlternativeCut()],
    ["silhouette on Fréchet", num(cons.silhouette_frechet), false],
    ["elbow k", elbow.k == null ? "— (no cliff)" : String(elbow.k),
      cons.selection === "elbow" || cons.selection === "elbow_capped"],
    ["drop / steepest", elbow.k == null ? "—" : `${num(elbow.drop)} / ${num(elbow.steepest)}`, false],
    ["selection", cons.selection || "—", false],
    ["aggregation", `${cons.consensus_cluster || "—"} / ${cons.input || "—"} / ${cons.linkage || "—"}`, false],
    ["k searched", `2 … ${cons.k_max ?? "—"}`, false],
    ["k rule", cons.rule || "silhouette", false],
    ["plateau tolerance", cons.tol > 0 ? String(cons.tol) : "0 (off)", false],
    ["min cluster size (cut)", cons.min_cluster_size > 1 ? String(cons.min_cluster_size) : "1 (off)", false],
    ["replicate k", reps.mean != null ? `${reps.mean} ± ${reps.sd}` : "—", false],
    ["replicates", cons.bootstraps != null ? `${cons.bootstraps} × ${cons.frac} (${cons.scheme})` : "—", false],
    ["traces in a stable cluster", pct(cons.stable_share), false],
    ["noise", pct(cons.noise_frac), false],
  ];

  for (const [term, value, highlight] of rows) {
    const wrap = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    if (highlight) dd.className = "chosen";
    wrap.append(dt, dd);
    facts.appendChild(wrap);
  }
}

function nearestSelectionPoint(clientX) {
  const canvas = $("selection-canvas");
  if (!selectionPoints.length) return null;
  const r = canvas.getBoundingClientRect();
  const mx = clientX - r.left;
  let best = null, bestD = Infinity;
  for (const p of selectionPoints) {
    const d = Math.abs(p.cx - mx);
    if (d < bestD) { bestD = d; best = p; }
  }
  return best && bestD <= 40 ? best : null;
}

function bindSelectionCanvas() {
  const canvas = $("selection-canvas");
  canvas.addEventListener("mousemove", (ev) => {
    const best = nearestSelectionPoint(ev.clientX);
    canvas.style.cursor = best && best.cuttable ? "pointer" : "default";
    if (!best) { $("tooltip").dataset.show = "false"; return; }
    showSelectionTooltip(best, ev.clientX, ev.clientY);
  });
  // Clicking the curve draws that cut everywhere else. The chart is where the choice
  // is argued about, so it is also where the alternative can be taken up.
  canvas.addEventListener("click", (ev) => {
    const best = nearestSelectionPoint(ev.clientX);
    if (!best || !best.cuttable) return;
    setCut(best.k === chosenK() ? null : best.k);
    showSelectionTooltip(best, ev.clientX, ev.clientY);
  });
  canvas.addEventListener("mouseleave", () => { $("tooltip").dataset.show = "false"; });
}

function showSelectionTooltip(point, clientX, clientY) {
  const tip = $("tooltip");
  const r0 = point.row;
  // Read live rather than from the point's stored flags: a click changes which cut is
  // drawn before the chart is repainted, and a tooltip saying otherwise would be wrong
  // for one frame at exactly the moment it is being read.
  const rows = [
    ["silhouette (1 − C)", typeof r0.silhouette === "number" ? r0.silhouette.toFixed(4) : "—"],
    ["silhouette (Fréchet)", typeof r0.silhouette_frechet === "number" ? r0.silhouette_frechet.toFixed(4) : "—"],
    ["smallest cluster", r0.min_size != null ? `${r0.min_size} traces` : "—"],
  ];
  if (point.elbow) rows.push(["elbow", "last k before the drop"]);
  const tags = [];
  if (point.k === chosenK()) tags.push("chosen");
  if (point.k === activeK()) tags.push("on screen");
  tip.innerHTML =
    `<div class="tt-head"><span>k = ${point.k}</span>` +
    `<span>${tags.join(" · ") || (point.cuttable ? "click to draw" : "not chosen")}</span></div>` +
    `<dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;
  tip.dataset.show = "true";
  const r = tip.getBoundingClientRect();
  let x = clientX + 14, y = clientY + 14;
  if (x + r.width > window.innerWidth - 8) x = clientX - r.width - 14;
  if (y + r.height > window.innerHeight - 8) y = clientY - r.height - 14;
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}

let rafPending = false;
function render() {
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => {
    rafPending = false;
    const stats = state.section === "stats";
    $("stats-wrap").hidden = !stats;
    if (stats) {
      for (const id of ["empty", "overlay-wrap", "panels", "matrix-wrap", "selection-wrap"]) {
        $(id).hidden = true;
      }
      renderStats();
      return;
    }
    if (!state.scene) { $("empty").hidden = false; return; }
    const view = state.view;
    // The stats section owns the title while it is open; take it back on return.
    { const traces = visibleTraces(); renderTitle(traces, sceneRanking(), visibleCounts(traces)); }
    $("overlay-wrap").hidden = view !== "overlay";
    $("panels").hidden = view !== "panels";
    $("matrix-wrap").hidden = view !== "matrix";
    $("selection-wrap").hidden = view !== "selection";
    $("reset-view").hidden = view !== "overlay";
    if (view === "overlay") renderOverlay();
    else if (view === "panels") renderPanels();
    else if (view === "selection") renderSelection();
    else renderMatrix();
  });
}

/* ---------- events ---------- */

function bindCanvas() {
  const box = $("canvas-box");
  const canvas = $("canvas");
  let dragging = false, dragStart = null, moved = false;

  canvas.addEventListener("mousemove", (ev) => {
    const r = canvas.getBoundingClientRect();
    if (dragging) {
      moved = true;
      state.pan.x = dragStart.panX + (ev.clientX - dragStart.x);
      state.pan.y = dragStart.panY + (ev.clientY - dragStart.y);
      render();
      return;
    }
    const hit = traceAt(ev.clientX - r.left, ev.clientY - r.top);
    if (hit !== state.hover) { state.hover = hit; render(); }
    showTooltip(state.pinned || hit, ev.clientX, ev.clientY);
  });

  canvas.addEventListener("mouseleave", () => {
    state.hover = null;
    if (!state.pinned) $("tooltip").dataset.show = "false";
    render();
  });

  canvas.addEventListener("mousedown", (ev) => {
    dragging = true; moved = false;
    dragStart = { x: ev.clientX, y: ev.clientY, panX: state.pan.x, panY: state.pan.y };
    box.classList.add("panning");
  });

  window.addEventListener("mouseup", (ev) => {
    if (!dragging) return;
    dragging = false;
    box.classList.remove("panning");
    if (moved) return;                       // a drag is not a click
    const r = canvas.getBoundingClientRect();
    const hit = traceAt(ev.clientX - r.left, ev.clientY - r.top);
    state.pinned = state.pinned === hit ? null : hit;
    showTooltip(state.pinned, ev.clientX, ev.clientY);
    render();
  });

  canvas.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const r = canvas.getBoundingClientRect();
    const mx = ev.clientX - r.left, my = ev.clientY - r.top;
    const factor = Math.exp(-ev.deltaY * 0.0015);
    const next = Math.min(20, Math.max(1, state.zoom * factor));
    state.pan.x = mx - ((mx - state.pan.x) * next) / state.zoom;
    state.pan.y = my - ((my - state.pan.y) * next) / state.zoom;
    state.zoom = next;
    render();
  }, { passive: false });

  window.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") {
      state.pinned = null;
      $("tooltip").dataset.show = "false";
      render();
    }
  });
}

function resetView() {
  state.zoom = 1;
  state.pan = { x: 0, y: 0 };
  render();
}

function clearFilters() {
  if (state.scene) {
    state.filters.subjects = new Set(state.scene.traces.map((t) => t.s));
  }
  state.filters.phase = "all";
  state.filters.stage = "all";
  state.filters.outcome = "all";
  state.hiddenClusters.clear();
  state.pinned = null; state.hover = null;
  for (const id of ["phase-seg", "stage-seg", "outcome-seg"]) {
    for (const b of $(id).querySelectorAll("button")) {
      b.setAttribute("aria-pressed", String(b.dataset.value === "all"));
    }
  }
  renderSubjectList();
  render();
}

/* ---------- stats analysis section ---------- */

/* Draws the per-subject practice models written by `invoke run-practice-models` and
 * copied into data/practice.json by `invoke run-web-export`. Like every other view
 * it only draws a stored result: the controls choose which fit is shown, nothing is
 * refitted in the browser. Colours match notebooks/figure_practice.ipynb so the
 * dashboard and figure_practice_*.png are the same figures: raw vs leave-subject-out
 * are CLUSTER_COLORS[0]/[1], linear vs log CLUSTER_COLORS[2]/[3], subjects take
 * SUBJECT_COLORS by sorted subject id. */

const STATS_SERIES = [
  { suffix: "", label: "all-subject medoids", color: CLUSTER_COLORS[0] },
  { suffix: "_loso", label: "leave-subject-out", color: CLUSTER_COLORS[1] },
];
const SLOPE_TERM = { linear: "t", quadratic: "t", log: "logt", log_phase: "logt" };
const MEASURE_INFO = {
  d_nearest_norm: { name: "distance to nearest medoid / its spread",
                    unit: "ratio", ref: 1, digits: 2 },
  d_own: { name: "Fréchet distance to own medoid", unit: "px", ref: null, digits: 1 },
};
const STATS_AT = 30;               // the "at attempt 30" column, as in mixed_summary.csv
/* Back-transforms, mirroring TRANSFORMS in analysis/practice.py. */
const INVERSE = { log: Math.exp, log1p: Math.expm1 };
const SVG_NS = "http://www.w3.org/2000/svg";
let statsData = null;              // practice.json once loaded; false when unavailable
let statsLoading = false;

function svgEl(tag, attrs, parent) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}

function niceTicks(lo, hi, n) {
  const span = hi - lo || 1;
  const mag = 10 ** Math.floor(Math.log10(span / n));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    out.push(Number(v.toFixed(10)));
  }
  return out;
}

const signed = (v, d = 4) => (v == null ? "—" : `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(d)}`);
const fmtP = (p) => (p == null ? "—" : p < 0.0001 ? "<0.0001" : p.toFixed(4));
const subjColor = (s) => state.statsSubjectColors.get(s) || NOISE_COLOR;

/* The three models the Model control switches between. Only the main statsmodels model
 * was fitted in every practice form and for the own-medoid measure; the nested ones are
 * log form and nearest/spread only, so the other controls are disabled for them. */
const STATS_MODELS = {
  lme4_nested: { label: "scene-scale lme4", short: "scene scale (lme4)",
                 formula: "distance to the nearest route prototype against attempt number, with " +
                          "(1 + log(attempt) | level) + (1 + log(attempt) | level:scene), fitted in R's lme4" },
  scene: { label: "scene only (statsmodels)", short: "scene only",
           formula: "(1 + practice | scene), the first model, fitted in statsmodels" },
  level_scene: { label: "nested, no scene correlation (statsmodels)", short: "nested, no corr.",
                 formula: "levels + scenes, scene intercept and slope uncorrelated (all statsmodels can nest)" },
  lme4_route: { label: "route-scale lme4", short: "route scale (lme4)",
                formula: "distance in px to the attempt's OWN route prototype against the nth use of that " +
                         "route, with (1 + log(use) | level) + (1 + log(use) | level:scene) + " +
                         "(1 + log(use) | level:scene:route), fitted in R's lme4; Wald CIs" },
  route_scene: { label: "own route, scene & level effects (lme4)", short: "own route, scene & level effects",
                 formula: "distance in px to the attempt's OWN route prototype against the nth use of that " +
                          "route, with (1 + log(use) | level) + (1 + log(use) | level:scene) and NO route " +
                          "effect, fitted in R's lme4 (run-practice-route-checks); Wald CIs" },
};
// Models on the route clock: x is the nth use of a route, y the distance to its own prototype.
const ROUTE_CLOCK = new Set(["lme4_route", "route_scene"]);
const onRouteClock = () => ROUTE_CLOCK.has(state.stats.model);
const SUMMARY_COLORS = { lme4: CLUSTER_COLORS[4], model: CLUSTER_COLORS[0], nested: CLUSTER_COLORS[6],
                         lme4_level_boot: CLUSTER_COLORS[3], scene_boot: CLUSTER_COLORS[1],
                         level_boot: CLUSTER_COLORS[2] };
const SUMMARY_LABELS = { lme4: "scene scale (lme4) — main", model: "scene only (statsmodels)",
                         nested: "nested, no scene corr. (statsmodels)",
                         lme4_level_boot: "level bootstrap of the lme4 model",
                         scene_boot: "scene bootstrap (scene-only model)",
                         level_boot: "level bootstrap (scene-only model)" };

/* One fit of the selected model as {meta, r}: r carries coef / ci / p (keyed by the
 * form's slope term), curve and converged, whatever model it came from. */
function modelFit(measure, subject) {
  const { model, form, transform } = state.stats;
  const completed = state.stats.attempts === "completed";
  if (model === "scene") {
    const f = statsFit(measure, subject);
    return f ? { meta: f, r: f.forms[form] } : null;
  }
  if (form !== "log") return null;
  if (model === "lme4_route") {
    const f = statsData.lme4_route && statsData.lme4_route.fits.find((x) => x.measure === measure
      && x.subject === subject && x.transform === transform && x.completed_only === completed);
    if (!f) return null;
    return {
      meta: { n_obs: f.n_obs, n_groups: f.n_routes, n_levels: f.n_levels,
              n_zero_dropped: f.n_zero_dropped, best_form: null, ci_method: f.ci_method,
              singular: f.singular },
      r: { coef: { Intercept: f.intercept, logt: f.b }, ci: { logt: f.ci }, p: { logt: f.p_wald },
           curve: f.curve, converged: !f.warnings },
    };
  }
  if (model === "route_scene") {
    const f = statsData.route_checks && statsData.route_checks.fits.find((x) => x.correction === "scene"
      && x.shape === "linear" && x.measure === measure && x.subject === subject
      && x.transform === transform && x.completed_only === completed);
    if (!f) return null;
    const zeros = statsData.lme4_route && statsData.lme4_route.fits.find((x) => x.measure === measure
      && x.subject === subject && x.transform === transform && x.completed_only === completed);
    return {
      meta: { n_obs: f.n_obs, n_groups: f.n_scenes, n_levels: f.n_levels,
              n_zero_dropped: zeros ? zeros.n_zero_dropped : null, best_form: null, ci_method: "wald",
              singular: f.singular, p_bh: f.p_bh },
      r: { coef: { Intercept: f.intercept, logt: f.b }, ci: { logt: [f.wald_low, f.wald_high] },
           p: { logt: f.p_wald }, curve: { ...f.curve, attempt: f.curve.use }, converged: !f.warnings },
    };
  }
  if (model === "lme4_nested") {
    const f = statsData.lme4 && statsData.lme4.fits.find((x) => x.model === "nested"
      && x.measure === measure && x.subject === subject && x.transform === transform
      && x.completed_only === completed);
    if (!f) return null;
    return {
      meta: { n_obs: f.n_obs, n_groups: f.n_scenes, n_levels: f.n_levels,
              n_zero_dropped: f.n_zero_dropped, best_form: null, ci_method: f.ci_method,
              singular: f.singular, lrt_p: f.lrt_p },
      r: { coef: { Intercept: f.intercept, logt: f.b }, ci: { logt: f.ci }, p: { logt: f.p_wald },
           curve: f.curve, converged: !f.warnings },
    };
  }
  const f = statsData.nested && statsData.nested.fits.find((x) => x.structure === model
    && x.measure === measure && x.subject === subject && x.transform === transform
    && x.completed_only === completed);
  return f && f.coef ? { meta: { ...f, best_form: null }, r: f } : null;
}

function statsFit(measure, subject) {
  const completed = state.stats.attempts === "completed";
  return statsData.fits.find((f) => f.measure === measure && f.subject === subject
                                    && f.completed_only === completed
                                    && f.transform === state.stats.transform);
}

/* Fixed-effect prediction at one attempt, back-transformed — the same arithmetic as
 * _predict() in analysis/practice.py followed by the transform's inverse. */
function statsPredict(coef, form, attempt) {
  const t = attempt / statsData.t_scale;
  let y = coef.Intercept;
  if (form === "linear" || form === "quadratic") {
    y += coef.t * t + (coef["I(t ** 2)"] || 0) * t * t;
  } else {
    y += coef.logt * Math.log(attempt);
  }
  return INVERSE[state.stats.transform](y);
}

function showStatsTooltip(head, color, rows, clientX, clientY) {
  const tip = $("tooltip");
  tip.innerHTML =
    `<div class="tt-head">${color ? `<span class="tt-dot" style="background:${color}"></span>` : ""}` +
    `<span>${head}</span></div>` +
    `<dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;
  tip.dataset.show = "true";
  const r = tip.getBoundingClientRect();
  let x = clientX + 14, y = clientY + 14;
  if (x + r.width > window.innerWidth - 8) x = clientX - r.width - 14;
  if (y + r.height > window.innerHeight - 8) y = clientY - r.height - 14;
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}
const hideStatsTooltip = () => { $("tooltip").dataset.show = "false"; };

function legendHTML(items, dot) {
  return items.map(([color, text]) =>
    `<span><i class="${dot ? "dot" : ""}" style="background:${color}"></i>${text}</span>`).join("");
}

/* Coefficient plot: one row per subject, raw and leave-subject-out side by side. */
function renderStatsCoef() {
  const { measure, form } = state.stats;
  const term = SLOPE_TERM[form];
  const host = $("stats-coef");
  host.innerHTML = "";
  const subjects = statsData.subjects;
  const pts = [];
  subjects.forEach((s, i) => {
    STATS_SERIES.forEach((ser, j) => {
      const fit = modelFit(measure + ser.suffix, s);
      const r = fit && fit.r;
      if (!r || !r.coef) return;
      const f = { ...fit.meta, measure: measure + ser.suffix };
      pts.push({ s, i, j, ser, f, r, b: r.coef[term], lo: r.ci[term][0], hi: r.ci[term][1] });
    });
  });
  if (!pts.length) { host.textContent = "No fits for this selection."; return; }

  const w = host.clientWidth || 700, rowH = 38;
  const pad = { l: 64, r: 22, t: 14, b: 44 };
  const h = pad.t + subjects.length * rowH + pad.b;
  let lo = Math.min(0, ...pts.map((p) => p.lo)), hi = Math.max(0, ...pts.map((p) => p.hi));
  const m = (hi - lo) * 0.06; lo -= m; hi += m;
  const px = (v) => pad.l + ((w - pad.l - pad.r) * (v - lo)) / (hi - lo);
  const rowY = (i) => pad.t + rowH * (i + 0.5);
  const svg = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, height: h, role: "img",
    "aria-label": `${form}-form practice coefficient per subject, raw and leave-subject-out` }, host);

  for (const v of niceTicks(lo, hi, 6)) {
    svgEl("line", { x1: px(v), x2: px(v), y1: pad.t, y2: h - pad.b, class: "grid" }, svg);
    svgEl("text", { x: px(v), y: h - pad.b + 14, "text-anchor": "middle", class: "tick" }, svg)
      .textContent = v === 0 ? "0" : signed(v, 3).replace("+", "");
  }
  svgEl("line", { x1: px(0), x2: px(0), y1: pad.t - 4, y2: h - pad.b, class: "zero" }, svg);
  svgEl("text", { x: (pad.l + w - pad.r) / 2, y: h - 8, "text-anchor": "middle", class: "axis-label" }, svg)
    .textContent = (onRouteClock() ? "log(nth use of route) coefficient"
                    : form === "log" ? "log(attempt) coefficient" : "coefficient per 10 attempts") +
      `, ${state.stats.transform} units — left of 0 = closer to the prototypes with practice`;
  subjects.forEach((s, i) => {
    svgEl("text", { x: pad.l - 10, y: rowY(i) + 4, "text-anchor": "end", class: "row-label" }, svg)
      .textContent = `sub-${s}`;
  });

  for (const p of pts) {
    const y = rowY(p.i) + (p.j === 0 ? -7 : 7);
    const g = svgEl("g", {}, svg);
    svgEl("line", { x1: px(p.lo), x2: px(p.hi), y1: y, y2: y, stroke: p.ser.color,
                    "stroke-width": 2, "stroke-linecap": "round" }, g);
    for (const v of [p.lo, p.hi]) {
      svgEl("line", { x1: px(v), x2: px(v), y1: y - 3.5, y2: y + 3.5, stroke: p.ser.color,
                      "stroke-width": 2, "stroke-linecap": "round" }, g);
    }
    svgEl("circle", { cx: px(p.b), cy: y, r: 4.5, fill: p.ser.color, class: "dot-ring" }, g);
    const hit = svgEl("rect", { x: px(p.lo) - 6, y: y - 7, width: px(p.hi) - px(p.lo) + 12,
                                height: 14, class: "hit" }, g);
    const clear = p.hi < 0 || p.lo > 0;
    const rows = [
      ["medoids", p.ser.label],
      ["coefficient", signed(p.b)],
      ["95% CI", `${signed(p.lo)} to ${signed(p.hi)}`],
      ["p", fmtP(p.r.p[term])],
      ["CI excludes 0", clear ? "yes" : "no"],
      ["attempts", p.f.n_obs.toLocaleString()],
      [measure === "d_own" ? "scene × cluster series" : "scenes", p.f.n_groups.toLocaleString()],
      ["CI method", p.f.ci_method === "wald" ? "Wald (profile failed)" : p.f.ci_method ? "profile likelihood" : "Wald"],
      ["converged", p.r.converged ? "yes" : "no"],
    ];
    hit.addEventListener("mousemove", (ev) =>
      showStatsTooltip(`sub-${p.s} · ${p.f.measure}`, p.ser.color, rows, ev.clientX, ev.clientY));
    hit.addEventListener("mouseleave", hideStatsTooltip);
  }

  $("stats-coef-legend").innerHTML = legendHTML(STATS_SERIES.map((s) => [s.color, s.label]), true);
  $("stats-coef-cap").textContent =
    `Each dot is one subject's practice coefficient b in the ${form} form under the ` +
    `${STATS_MODELS[state.stats.model].label} model, with its 95% ` +
    "confidence interval. Left of the dashed zero line means attempts moved closer to the " +
    "cluster prototypes with practice; an interval that crosses it is no clear change. Blue " +
    "measures against medoids chosen from all five players, orange against medoids chosen " +
    "from the other four only — read an effect as real where the two agree.";
}

/* A line chart with a crosshair tooltip. Series need not share x values: each is
 * read at its own nearest point, and skipped past its own last attempt. */
function drawLineChart(host, { series, yDomain, yLabel, xLabel, ref, refLabel, height, title, digits, aria }) {
  host.innerHTML = "";
  if (title) {
    const d = document.createElement("div");
    d.className = "panel-title";
    d.innerHTML = title;
    host.appendChild(d);
  }
  const w = host.clientWidth || 600;
  const pad = { l: yLabel ? 54 : 40, r: 16, t: 12, b: xLabel ? 42 : 26 };
  const svg = svgEl("svg", { viewBox: `0 0 ${w} ${height}`, height, role: "img", "aria-label": aria || "" }, host);
  const xMax = Math.max(...series.flatMap((s) => s.points.map((p) => p[0])));
  const xMin = 1;
  const [ylo, yhi] = yDomain;
  const px = (x) => pad.l + ((w - pad.l - pad.r) * (x - xMin)) / (xMax - xMin || 1);
  const py = (y) => height - pad.b - ((height - pad.t - pad.b) * (y - ylo)) / (yhi - ylo || 1);

  for (const v of niceTicks(ylo, yhi, 4)) {
    svgEl("line", { x1: pad.l, x2: w - pad.r, y1: py(v), y2: py(v), class: "grid" }, svg);
    svgEl("text", { x: pad.l - 6, y: py(v) + 3.5, "text-anchor": "end", class: "tick" }, svg)
      .textContent = v.toFixed(digits);
  }
  for (const v of niceTicks(xMin, xMax, 5)) {
    svgEl("text", { x: px(v), y: height - pad.b + 14, "text-anchor": "middle", class: "tick" }, svg)
      .textContent = v;
  }
  if (xLabel) {
    svgEl("text", { x: (pad.l + w - pad.r) / 2, y: height - 8, "text-anchor": "middle",
                    class: "axis-label" }, svg).textContent = xLabel;
  }
  if (yLabel) {
    const cy = (pad.t + height - pad.b) / 2;
    svgEl("text", { x: 13, y: cy, "text-anchor": "middle", class: "axis-label",
                    transform: `rotate(-90 13 ${cy})` }, svg).textContent = yLabel;
  }
  if (ref != null && ref >= ylo && ref <= yhi) {
    svgEl("line", { x1: pad.l, x2: w - pad.r, y1: py(ref), y2: py(ref), class: "ref" }, svg);
    if (refLabel) {
      svgEl("text", { x: w - pad.r - 4, y: py(ref) - 4, "text-anchor": "end", class: "ref-label" }, svg)
        .textContent = refLabel;
    }
  }
  for (const s of series) {
    svgEl("polyline", { points: s.points.map(([x, y]) => `${px(x)},${py(y)}`).join(" "),
                        fill: "none", stroke: s.color, "stroke-width": 2,
                        "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
  }

  const cross = svgEl("g", { visibility: "hidden" }, svg);
  const vline = svgEl("line", { y1: pad.t, y2: height - pad.b, class: "crosshair" }, cross);
  const dots = series.map((s) => svgEl("circle", { r: 4, fill: s.color, class: "dot-ring" }, cross));
  const hit = svgEl("rect", { x: pad.l, y: pad.t, width: w - pad.l - pad.r,
                              height: height - pad.t - pad.b, class: "hit" }, svg);
  hit.style.cursor = "crosshair";
  hit.addEventListener("mousemove", (ev) => {
    const r = svg.getBoundingClientRect();
    const sx = ((ev.clientX - r.left) / r.width) * w;
    const a = Math.round(xMin + ((sx - pad.l) / (w - pad.l - pad.r)) * (xMax - xMin));
    const rows = [];
    let shownX = null;
    series.forEach((s, i) => {
      const last = s.points[s.points.length - 1][0];
      if (a > last) { dots[i].setAttribute("visibility", "hidden"); return; }
      const p = s.points.reduce((best, q) => (Math.abs(q[0] - a) < Math.abs(best[0] - a) ? q : best));
      dots[i].setAttribute("visibility", "visible");
      dots[i].setAttribute("cx", px(p[0]));
      dots[i].setAttribute("cy", py(p[1]));
      shownX = shownX == null ? p[0] : shownX;
      rows.push([s.label, p[1].toFixed(digits + 1)]);
    });
    if (!rows.length) { cross.setAttribute("visibility", "hidden"); hideStatsTooltip(); return; }
    vline.setAttribute("x1", px(shownX)); vline.setAttribute("x2", px(shownX));
    cross.setAttribute("visibility", "visible");
    showStatsTooltip(onRouteClock() ? `use ${shownX} of the route`
                       : `attempt ${shownX} at a scene`, null, rows, ev.clientX, ev.clientY);
  });
  hit.addEventListener("mouseleave", () => {
    cross.setAttribute("visibility", "hidden");
    hideStatsTooltip();
  });
}

function curvePoints(r) {
  const inv = INVERSE[state.stats.transform];
  return r.curve.attempt.map((a, i) => [a, inv(r.curve.y[i])]);
}

function paddedDomain(values, ref) {
  let lo = Math.min(...values), hi = Math.max(...values);
  if (ref != null) { lo = Math.min(lo, ref); hi = Math.max(hi, ref); }
  const m = (hi - lo) * 0.06 || 0.05;
  return [lo - m, hi + m];
}

function renderStatsCurves() {
  const { measure, form } = state.stats;
  const info = MEASURE_INFO[measure];
  const series = [];
  for (const s of statsData.subjects) {
    const fit = modelFit(`${measure}_loso`, s);
    const r = fit && fit.r;
    if (r && r.curve) series.push({ label: `sub-${s}`, color: subjColor(s), points: curvePoints(r) });
  }
  const host = $("stats-curves");
  if (!series.length) { host.textContent = "No fitted curves for this selection."; return; }
  drawLineChart(host, {
    series, height: 340, digits: info.digits,
    yDomain: paddedDomain(series.flatMap((s) => s.points.map((p) => p[1])), info.ref),
    yLabel: `${info.name} (${info.unit})`,
    xLabel: onRouteClock()
      ? "nth use of the route at that scene (each route's own count)"
      : "attempt at a scene (restarts at 1 for every scene)",
    ref: info.ref, refLabel: info.ref != null ? "1.0 = typical cluster member" : null,
    aria: `${form}-form fitted curve per subject`,
  });
  $("stats-curve-legend").innerHTML = legendHTML(series.map((s) => [s.color, s.label]));
  $("stats-curve-cap").textContent =
    `Each line is one subject's fitted ${form}-form line under the ${STATS_MODELS[state.stats.model].label} ` +
    "model — the average scene" + (state.stats.model === "scene" ? "" : " and level") +
    `, with their own adjustments set to zero — converted back from ${state.stats.transform} to ` +
    (measure === "d_nearest_norm"
      ? "the distance ratio (0 would be the medoid itself, 1 a typical member). "
      : "Fréchet distance in pixels. ") +
    (state.stats.model === "lme4_route"
      ? "Here practice is the nth use of a route, and the distance is always to that route's own " +
        "prototype, so the origin never changes along a series: a fall is tighter execution of a route. "
      : state.stats.model === "route_scene"
      ? "Here practice is the nth use of a route and the distance is to that route's own prototype, " +
        "but routes get no effect of their own: a fall mixes tighter execution of a route with players " +
        "moving to routes that were tight from the start. "
      : "") +
    "Each stops at that subject's 95th percentile of practice, past which few scenes " +
    "still contribute. The y-axis is zoomed to the data, so drops look steeper than they are.";
}

/* Fit against the data: the stored lme4 curve over the rows it was fitted on, from
 * data/practice_points.json (export_fit_data, fetched only when this block is drawn).
 * Two rows per subject: pooled bins, and the same bins over the ggeffects partial
 * residuals (run-practice-lme4-effects: each attempt with its level's, scene's and
 * route's own intercept and slope removed). The fixed slope is a within-unit slope, so
 * the second row is the like-for-like check; the pooled row also carries between-unit
 * differences (routes kept for many uses start tighter). Bins are geometric means under
 * log, the same quantity the curve is. Drawing only: the points are a seeded sample. */
let fitPoints = null;              // practice_points.json once loaded; false when unavailable
let fitPointsLoading = false;
const FIT_POINT_COLOR = "#898781";

function renderStatsFitData() {
  const block = $("stats-fitdata-block");
  const model = state.stats.model;
  const lme4 = model === "lme4_nested" || model === "lme4_route";
  block.hidden = !lme4 || state.stats.form !== "log";
  if (block.hidden) return;
  if (fitPoints === null) {
    $("stats-fitdata-cap").textContent = "loading the data under the fits…";
    if (!fitPointsLoading) {
      fitPointsLoading = true;
      fetch("data/practice_points.json", { cache: "no-cache" })
        .then((res) => (res.ok ? res.json() : false))
        .catch(() => false)
        .then((data) => { fitPoints = data; render(); });
    }
    return;
  }
  const pooledHost = $("stats-fitdata-pooled"), withinHost = $("stats-fitdata-within");
  pooledHost.innerHTML = ""; withinHost.innerHTML = "";
  if (!fitPoints) {
    $("stats-fitdata-cap").textContent =
      "No data/practice_points.json in this export. Run  invoke run-web-export --backfill  and reload.";
    return;
  }
  const { measure, transform } = state.stats;
  const completed = state.stats.attempts === "completed" ? 1 : 0;
  const route = model === "lme4_route";
  const panels = [];
  for (const s of statsData.subjects) {
    const fit = modelFit(`${measure}_loso`, s);
    const data = fitPoints[`${model}|${measure}_loso|${transform}|${completed}|${s}`];
    if (fit && fit.r && fit.r.curve && data && data.bins.length) panels.push({ s, fit: fit.r, data });
  }
  const hasPartial = panels.some((p) => p.data.partial_bins);
  $("stats-fitdata-partial-label").hidden = !hasPartial;
  withinHost.hidden = !hasPartial;
  if (!panels.length) {
    $("stats-fitdata-cap").textContent = "No fit for this selection.";
    return;
  }
  // Every box is inserted before any is measured, or the first ones read the row as their width.
  const boxes = panels.map((p) => {
    const mk = (host) => { const b = document.createElement("div"); b.className = "stats-chart"; host.appendChild(b); return b; };
    return { ...p, pooledBox: mk(pooledHost), withinBox: mk(withinHost) };
  });
  const inv = INVERSE[transform];
  for (const within of hasPartial ? [false, true] : [false]) {
    const col = within ? 2 : 1;
    const ys = boxes.flatMap((b) => b.data.points.map((p) => p[col]).filter((v) => v != null && v > 0))
      .sort((a, b) => a - b);
    const q = (f) => ys[Math.min(ys.length - 1, Math.floor(f * ys.length))];
    let [ylo, yhi] = [q(0.01), q(0.99)];
    for (const b of boxes) {
      for (const bin of (within ? b.data.partial_bins : b.data.bins) || []) { ylo = Math.min(ylo, bin.lo); yhi = Math.max(yhi, bin.hi); }
    }
    for (const b of boxes) {
      if (within && !b.data.partial_bins) continue;
      const curve = b.fit.curve.attempt.map((a, i) => [a, inv(b.fit.curve.y[i])]);
      drawFitPanel(within ? b.withinBox : b.pooledBox, {
        title: within ? null : `sub-${b.s} <span class="muted">n=${b.data.n_obs}</span>`,
        points: b.data.points.filter((p) => p[col] != null).map((p) => [p[0], p[col]]),
        bins: within ? b.data.partial_bins : b.data.bins, curve, color: subjColor(b.s),
        yDomain: [ylo, yhi], ref: measure === "d_nearest_norm" ? 1 : null,
        digits: MEASURE_INFO[measure].digits, route, within, subject: b.s,
        yLabel: measure === "d_nearest_norm" ? "distance / spread" : "distance (px)",
      });
    }
  }
  $("stats-fitdata-legend").innerHTML =
    legendHTML([[FIT_POINT_COLOR, "attempts (a sample)"]], true) +
    legendHTML([["var(--ink)", "fitted curve"]]) +
    legendHTML(panels.map((p) => [subjColor(p.s), `sub-${p.s} binned, 95% CI`]), true);
  $("stats-fitdata-cap").innerHTML =
    `The ${STATS_MODELS[model].label} fixed-effect curve (dark line), drawn over the rows it was fitted on ` +
    `(leave-subject-out medoids, ${transform}(d), ${completed ? "completed attempts only" : "all attempts"}). ` +
    "Grey points are a seeded sample of the attempts. The coloured bins use every attempt. They are " +
    `the mean of ${transform}(d) per bin, converted back, which is the same quantity the curve is ` +
    "(a plain mean of the distances would sit above it from skew alone). " +
    `<strong>Read the second row.</strong> The fixed slope is a <em>within-${route ? "route" : "scene"}</em> slope, ` +
    `because every ${route ? "route" : "scene"} has its own intercept and slope. The pooled row also mixes ` +
    (route
      ? "<em>different</em> routes: a route that ends up reused 20+ times is already about 30% tighter on its " +
        "first use than one dropped after a few, so only the tight routes are left in the late pooled bins, " +
        "and those bins fall well below the line even though the model is right. "
      : "different scenes: only scenes played many times reach the late bins, so those bins could drift " +
        "from the line if those scenes differed from the average. For this model the two rows come out " +
        "almost the same (bins within 2-5%, 15% at one bin for sub-05): scenes differ little next to the " +
        "attempt-to-attempt noise, and the scenes played longest are not systematically easier or harder, " +
        "so here the raw data was already a fair check. ") +
    (hasPartial
      ? "The second row is the standard mixed-model check, the partial residuals (from the R package " +
        "<code>ggeffects</code>): each attempt with its own level's, scene's" + (route ? " and route's" : "") +
        " intercept and slope taken out, so it is compared with the line on the model's own terms. "
      : "The partial-residual row needs <code>invoke run-practice-lme4-effects</code> (R with ggeffects), " +
        "then <code>invoke run-web-export --backfill</code>. ") +
    "Hover a bin for its n and how many scenes it holds. The error bars treat attempts as independent, " +
    "which they are not within a scene, so read them as where the bin sits, not as a test. " +
    "Both axes are on a log scale, where the model's curve is a straight line, so a bend in the bins " +
    "is what a wrong shape would look like. The y-axes are shared within each row. " +
    "<em>Axis labels</em> switches the ruler, not the picture: ordinary units, or the natural-log values " +
    `the model is fitted on, where the line's slope is b` +
    (transform === "log" ? "." : ` (under ${transform} the model's scale is ln(1 + d), so only approximately).`);
}

function drawFitPanel(host, { title, points, bins, curve, color, yDomain, ref, digits, route, within, subject, yLabel }) {
  if (title) {
    const d = document.createElement("div");
    d.className = "panel-title";
    d.innerHTML = title;
    host.appendChild(d);
  }
  const w = host.clientWidth || 240, height = 200;
  const pad = { l: 52, r: 8, t: 8, b: 30 };
  const svg = svgEl("svg", { viewBox: `0 0 ${w} ${height}`, height, role: "img",
                             "aria-label": `sub-${subject} data under the fitted curve` }, host);
  const xMax = Math.max(curve[curve.length - 1][0], ...bins.map((b) => b.x));
  // Both axes log: the model is linear in log(x) on log(d), so its curve is a straight line.
  const fx = Math.log;
  const [x0, x1] = [fx(0.8), fx(xMax * 1.15)];
  const [y0, y1] = [Math.log(yDomain[0]), Math.log(yDomain[1])];
  const px = (x) => pad.l + ((w - pad.l - pad.r) * (fx(x) - x0)) / (x1 - x0);
  const py = (y) => height - pad.b - ((height - pad.t - pad.b) * (Math.log(y) - y0)) / (y1 - y0 || 1);
  const inside = (x, y) => y >= yDomain[0] && y <= yDomain[1] && fx(x) <= x1;

  drawLogAxes(svg, { px, py, pad, w, height, xMax: xMax * 1.15, ylo: yDomain[0], yhi: yDomain[1],
                    xName: route ? "nth use of the route" : "attempt at scene", yName: yLabel });
  if (ref != null && ref >= yDomain[0] && ref <= yDomain[1]) {
    svgEl("line", { x1: pad.l, x2: w - pad.r, y1: py(ref), y2: py(ref), class: "ref" }, svg);
  }
  // Integer x stacks into columns; a fixed seeded jitter (in the axis's own scale) spreads them.
  const cloud = svgEl("g", { fill: FIT_POINT_COLOR, "fill-opacity": 0.22 }, svg);
  let seed = 7;
  const rand = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };
  for (const [x, y] of points) {
    const jx = x * Math.exp(0.22 * rand());
    if (!inside(jx, y)) continue;
    svgEl("circle", { cx: px(jx).toFixed(1), cy: py(y).toFixed(1), r: 1.3 }, cloud);
  }
  svgEl("polyline", { points: curve.filter(([, y]) => y > 0).map(([x, y]) => `${px(x)},${py(y)}`).join(" "),
                      fill: "none", stroke: "var(--ink)", "stroke-width": 2,
                      "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
  for (const b of bins) {
    const g = svgEl("g", {}, svg);
    svgEl("line", { x1: px(b.x), x2: px(b.x), y1: py(b.lo), y2: py(b.hi), stroke: color, "stroke-width": 1.5 }, g);
    svgEl("circle", { cx: px(b.x), cy: py(b.mean), r: 3.5, fill: color, class: "dot-ring" }, g);
    const hit = svgEl("circle", { cx: px(b.x), cy: py(b.mean), r: 8, class: "hit" }, g);
    hit.addEventListener("mousemove", (ev) => showStatsTooltip(
      `sub-${subject} · ${route ? "use" : "attempt"} ≈ ${b.x.toFixed(b.x < 10 ? 1 : 0)}`, color,
      [[within ? "partial residual, geometric mean" : "geometric mean", b.mean.toFixed(digits + 1)],
       ["95% CI", `${b.lo.toFixed(digits + 1)} – ${b.hi.toFixed(digits + 1)}`],
       ["attempts", b.n_obs], ["scenes", b.n_scenes]], ev.clientX, ev.clientY));
    hit.addEventListener("mouseleave", hideStatsTooltip);
  }
}

/* One line per scene (per route, for the route model): the "spaghetti" view of a mixed
 * model, from data/practice_spaghetti.json (export_fit_data). For one subject, their most-
 * played scenes, every attempt as a point and each unit's own line — the fixed effects plus
 * its level's, scene's (and route's) random intercept and slope from the stored fit — with
 * the fixed-effect line dashed. Exported for each model's primary setting only. Routes are
 * coloured by their whole-scene rank, the slot they have in every other view. */
let spaghettiData = null;          // practice_spaghetti.json once loaded; false when unavailable
let spaghettiLoading = false;

function renderStatsSpaghetti() {
  const block = $("stats-spaghetti-block");
  const model = state.stats.model;
  block.hidden = !(model === "lme4_nested" || model === "lme4_route");
  if (block.hidden) return;
  if (spaghettiData === null) {
    $("stats-spaghetti-cap").textContent = "loading the per-scene fits…";
    if (!spaghettiLoading) {
      spaghettiLoading = true;
      fetch("data/practice_spaghetti.json", { cache: "no-cache" })
        .then((res) => (res.ok ? res.json() : false))
        .catch(() => false)
        .then((data) => { spaghettiData = data; render(); });
    }
    return;
  }
  const host = $("stats-spaghetti");
  host.innerHTML = "";
  const src = spaghettiData && spaghettiData[model];
  if (!src) {
    $("stats-spaghetti-cap").innerHTML =
      "No per-scene fits in this export. Run <code>invoke run-practice-lme4-effects</code> (R with ggeffects), " +
      "then <code>invoke run-web-export --backfill</code>, and reload.";
    $("stats-spaghetti-subject").innerHTML = "";
    $("stats-spaghetti-legend").innerHTML = "";
    return;
  }
  const route = model === "lme4_route";
  const subjects = Object.keys(src.subjects).sort();
  if (!subjects.includes(state.stats.spSubject)) state.stats.spSubject = subjects[0];
  const subject = state.stats.spSubject;
  $("stats-spaghetti-subject").innerHTML = subjects.map((s) =>
    `<button type="button" data-subject="${s}" aria-pressed="${s === subject}">sub-${s}</button>`).join("");
  $("stats-spaghetti-unit").textContent = route ? "route" : "scene";
  const sp = src.subjects[subject];
  const inv = INVERSE[sp.transform];
  const scenes = Object.entries(sp.scenes);
  // Every box is inserted before any is measured, or the first ones read the row as their width.
  const boxes = scenes.map(() => {
    const b = document.createElement("div");
    b.className = "stats-chart";
    host.appendChild(b);
    return b;
  });
  const colorOf = (v) => {
    if (!route) return () => subjColor(subject);
    const slot = new Map(v.rank.map((c, i) => [c, i < N_DIRECT_CLUSTERS ? CLUSTER_COLORS[i] : OTHER_COLOR]));
    return (r) => slot.get(r) || OTHER_COLOR;
  };
  const lineOf = (a, b, x1) => {
    const n = 24, out = [];
    for (let k = 0; k <= n; k++) {
      const x = Math.exp((Math.log(Math.max(x1, 1.01)) * k) / n);
      out.push([x, inv(a + b * Math.log(x))]);
    }
    return out;
  };
  scenes.forEach(([scene, v], i) => {
    const color = colorOf(v);
    const xMax = Math.max(...v.points.map((p) => p[0]));
    drawSpaghettiPanel(boxes[i], {
      title: `${scene} <span class="muted">n=${v.n}` +
             (route ? ` · ${v.lines.length} route${v.lines.length === 1 ? "" : "s"}` : "") + "</span>",
      points: v.points.map(([x, d, r]) => [x, d, color(r)]),
      lines: v.lines.map((l) => ({ color: color(l.route), pts: lineOf(l.a, l.b, l.x1), route: l.route, b: l.b })),
      fixed: lineOf(sp.fixed.a, sp.fixed.b, xMax),
      route, digits: MEASURE_INFO[src.measure.replace("_loso", "")].digits,
      ref: src.measure.startsWith("d_nearest_norm") ? 1 : null,
    });
  });
  $("stats-spaghetti-legend").innerHTML =
    legendHTML([[NOISE_COLOR, "fixed effect (average scene)"]]) +
    (route ? legendHTML([[CLUSTER_COLORS[0], "routes, coloured by whole-scene size as everywhere else"]])
           : legendHTML([[subjColor(subject), `sub-${subject}: attempts and the scene's own line`]]));
  const notMain = state.stats.attempts !== "all" || state.stats.transform !== sp.transform
    || `${state.stats.measure}_loso` !== src.measure;
  $("stats-spaghetti-cap").innerHTML =
    `The ${STATS_MODELS[model].label} model scene by scene, for sub-${subject}'s ${scenes.length} most-played ` +
    "scenes: every attempt as a point, and " +
    (route ? "one line per <strong>route</strong> used there, because this model gives every route its own " +
             "intercept and slope. Each line is coloured like its route's points and stops at that route's last use. "
           : "that scene's own fitted line. ") +
    "A unit's line is the fixed effect plus its level's, scene's" + (route ? " and route's" : "") +
    " random intercept and slope (the stored fit's BLUPs; rebuilding them reproduces lme4's fitted values). " +
    "The dashed line is the fixed effect, the average scene. A line that misses its points is worth a look. " +
    "A short or flat one usually means the unit has few attempts, and the model pulled it toward the average. " +
    `Shown for this model's main setting (${src.measure}, ${sp.transform}(d), all attempts)` +
    (notMain ? ', <span class="filtered">not the setting chosen above</span>.' : ".");
}

function drawSpaghettiPanel(host, { title, points, lines, fixed, route, digits, ref }) {
  const t = document.createElement("div");
  t.className = "panel-title";
  t.innerHTML = title;
  host.appendChild(t);
  const w = host.clientWidth || 260, height = 190;
  const pad = { l: 52, r: 8, t: 6, b: 30 };
  const svg = svgEl("svg", { viewBox: `0 0 ${w} ${height}`, height, role: "img",
                             "aria-label": t.textContent }, host);
  const ys = points.map((p) => p[1]).filter((y) => y > 0).sort((a, b) => a - b);
  const ylo = ys[Math.floor(0.01 * (ys.length - 1))], yhi = ys[Math.ceil(0.99 * (ys.length - 1))];
  const xMax = Math.max(...points.map((p) => p[0]));
  const [x0, x1] = [Math.log(0.8), Math.log(xMax * 1.15)];
  const [y0, y1] = [Math.log(ylo / 1.15), Math.log(yhi * 1.15)];
  const px = (x) => pad.l + ((w - pad.l - pad.r) * (Math.log(x) - x0)) / (x1 - x0);
  const py = (y) => height - pad.b - ((height - pad.t - pad.b) * (Math.log(y) - y0)) / (y1 - y0 || 1);
  const clipY = (y) => Math.min(Math.max(y, Math.exp(y0)), Math.exp(y1));
  drawLogAxes(svg, { px, py, pad, w, height, xMax: xMax * 1.15, ylo: Math.exp(y0), yhi: Math.exp(y1),
                    xName: route ? "nth use of the route" : "attempt at scene",
                    yName: ref != null ? "distance / spread" : "distance (px)" });
  if (ref != null && ref >= Math.exp(y0) && ref <= Math.exp(y1)) {
    svgEl("line", { x1: pad.l, x2: w - pad.r, y1: py(ref), y2: py(ref), class: "ref" }, svg);
  }
  const cloud = svgEl("g", { "fill-opacity": 0.55 }, svg);
  for (const [x, y, c] of points) {
    if (y < Math.exp(y0) || y > Math.exp(y1)) continue;
    svgEl("circle", { cx: px(x).toFixed(1), cy: py(y).toFixed(1), r: 1.8, fill: c }, cloud);
  }
  const poly = (pts, attrs) => svgEl("polyline", {
    points: pts.map(([x, y]) => `${px(x)},${py(clipY(y))}`).join(" "), fill: "none",
    "stroke-linejoin": "round", "stroke-linecap": "round", ...attrs }, svg);
  poly(fixed, { stroke: NOISE_COLOR, "stroke-width": 1.3, "stroke-dasharray": "4 3" });
  for (const l of lines) {
    poly(l.pts, { stroke: l.color, "stroke-width": 2.2 });
    const hit = poly(l.pts, { stroke: "transparent", "stroke-width": 10, class: "hit" });
    hit.addEventListener("mousemove", (ev) => showStatsTooltip(
      route ? `route ${l.route}` : "this scene's own line", l.color,
      [["slope on log(x)", signed(l.b, 3)],
       ["at x = 1", l.pts[0][1].toFixed(digits + 1)],
       ["at its last x", l.pts[l.pts.length - 1][1].toFixed(digits + 1)]], ev.clientX, ev.clientY));
    hit.addEventListener("mouseleave", hideStatsTooltip);
  }
}

/* Ticks and axis titles for a log-log panel. Spacing is always logarithmic; the
 * "axis labels" control only picks the ruler: ordinary units (attempt 1, 2, 5, 10; distance
 * 10, 20, 50 px) or the natural-log values the model is fitted on, evenly spaced
 * (ln attempt 0, 1, 2, 3), where the fixed line's slope reads directly as b. */
function drawLogAxes(svg, { px, py, pad, w, height, xMax, ylo, yhi, xName, yName }) {
  const ln = state.stats.axisUnits === "ln";
  const fmt = (v) => (v >= 10 ? v.toFixed(0) : String(Number(v.toPrecision(2))));
  const yTicks = ln ? niceTicks(Math.log(ylo), Math.log(yhi), 6).map((t) => [Math.exp(t), String(t)])
                    : logTicks(ylo, yhi).map((v) => [v, fmt(v)]);
  for (const [v, label] of yTicks) {
    svgEl("line", { x1: pad.l, x2: w - pad.r, y1: py(v), y2: py(v), class: "grid" }, svg);
    svgEl("text", { x: pad.l - 5, y: py(v) + 3.5, "text-anchor": "end", class: "tick" }, svg).textContent = label;
  }
  const xTicks = ln ? niceTicks(0, Math.log(xMax), 4).filter((t) => t >= 0).map((t) => [Math.exp(t), String(t)])
                    : [1, 2, 5, 10, 20, 50, 100, 200, 500].filter((v) => v <= xMax).map((v) => [v, String(v)]);
  for (const [v, label] of xTicks) {
    svgEl("text", { x: px(v), y: height - pad.b + 13, "text-anchor": "middle", class: "tick" }, svg).textContent = label;
  }
  svgEl("text", { x: (pad.l + w - pad.r) / 2, y: height - 4, "text-anchor": "middle", class: "axis-label" }, svg)
    .textContent = ln ? `ln(${xName})` : `${xName} (log scale)`;
  const cy = (pad.t + height - pad.b) / 2;
  svgEl("text", { x: 11, y: cy, "text-anchor": "middle", class: "axis-label",
                  transform: `rotate(-90 11 ${cy})` }, svg).textContent = ln ? `ln(${yName})` : `${yName} (log scale)`;
}

/* Ticks for a log axis: 1-2-5 per decade, or the decades alone when that is too many. */
function logTicks(lo, hi) {
  const at = (mantissas) => {
    const out = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      for (const m of mantissas) { const v = Number((m * 10 ** e).toPrecision(6)); if (v >= lo && v <= hi) out.push(v); }
    }
    return out;
  };
  const fine = at([1, 2, 5]);
  const decades = at([1]);
  if (fine.length < 3) return at([1, 1.5, 2, 3, 4, 5, 7]);   // a narrow range: label it densely
  return fine.length > 6 && decades.length >= 2 ? decades : fine;
}

/* The Benjamini-Hochberg adjusted p of one fit, as exported (statsData.bh, from mixed_summary.csv's
 * slope_p_bh, computed by statsmodels' multipletests). The browser never adjusts p-values itself:
 * one family is one fit setting across the five subjects, as in _bh_by_family. */
const BH_STRUCTURE = { lme4_nested: "lme4_nested", lme4_route: "lme4_route", scene: "scene",
                       level_scene: "level_scene" };
function statsBh(measure, subject) {
  const { model, transform, form } = state.stats;
  if (model === "route_scene") { const f = modelFit(measure, subject); return f ? f.meta.p_bh : undefined; }
  const completed = state.stats.attempts === "completed" ? 1 : 0;
  const key = `${BH_STRUCTURE[model]}|${measure}|${transform}|${completed}|${model === "scene" ? form : "log"}|${subject}`;
  return statsData.bh ? statsData.bh[key] : undefined;
}

/* CI check by bootstrap: for the selected model, each subject's model CI against the percentile
 * CI of 200 refits on resampled levels (or scenes), from statsData.bootstraps — copied from the
 * summary CSVs under docs/ by practice_bootstraps(); nothing is resampled in the browser. */
function renderStatsBootstrap() {
  const block = $("stats-boot-block");
  const model = state.stats.model;
  const data = statsData.bootstraps && statsData.bootstraps[model];
  const t = state.stats.transform;
  const host = $("stats-boot-table");
  if (!data) {
    host.innerHTML = "";
    $("stats-boot-cap").textContent = statsData.bootstraps
      ? `No bootstrap was run for the ${STATS_MODELS[model].label} model. The two main models ` +
        "(scene-scale lme4 and route-scale lme4) and the scene-only model have one."
      : "No bootstrap results in this export: run the scripts under docs/ (see the decisions log), " +
        "then invoke run-web-export --backfill.";
    return;
  }
  block.hidden = false;
  const ci = (lo, hi) => `${signed(lo)} to ${signed(hi)}`;
  const yesNo = (v) => (v ? "<strong>yes</strong>" : "no");
  const rows = [];
  for (const run of data.runs) {
    for (const r of run.rows.filter((x) => x.transform === t).sort((a, b) => a.subject.localeCompare(b.subject))) {
      const same = r.model_excludes_zero === r.boot_excludes_zero;
      rows.push(`<tr><td class="subj"><i style="background:${subjColor(r.subject)}"></i>sub-${r.subject}</td>` +
        `<td>${run.label}</td><td>${signed(r.b)}</td><td>${ci(r.model_ci_low, r.model_ci_high)}</td>` +
        `<td>${ci(r.boot_ci_low, r.boot_ci_high)}</td><td>${r.width_ratio.toFixed(2)}</td>` +
        `<td>${yesNo(r.model_excludes_zero)} / ${yesNo(r.boot_excludes_zero)}${same ? "" : " — differs"}</td>` +
        `<td>${Math.round(100 * r.share_boot_below_zero)}%</td>` +
        `<td>${r.share_singular != null ? Math.round(100 * r.share_singular) + "%" : "—"}</td>` +
        `<td>${r.n_boot}${r.n_failed ? ` (${r.n_failed} failed)` : ""}</td></tr>`);
    }
  }
  host.innerHTML =
    "<thead><tr><th>subject</th><th>bootstrap</th><th>slope b</th><th>model 95% CI</th>" +
    "<th>bootstrap 95% CI</th><th>width, bootstrap / model</th><th>excludes 0? model / bootstrap</th>" +
    "<th>replicate slopes &lt; 0</th><th>singular refits</th><th>replicates</th></tr></thead>" +
    `<tbody>${rows.join("")}</tbody>`;
  const ratios = data.runs.flatMap((run) => run.rows.filter((x) => x.transform === t).map((x) => x.width_ratio));
  const changed = data.runs.flatMap((run) => run.rows.filter((x) => x.transform === t &&
    x.model_excludes_zero !== x.boot_excludes_zero));
  const notMain = `${state.stats.measure}_loso` !== data.measure || state.stats.attempts !== "all";
  $("stats-boot-cap").innerHTML =
    "Is the model's confidence interval right? For each subject, whole " +
    (model === "scene" ? "scenes (or whole levels)" : "levels") + " were redrawn with replacement, the model " +
    "refitted 200 times (seed 0), and the 2.5th–97.5th percentile of the refitted slopes taken as the bootstrap CI. " +
    "Resampling whole units keeps each one's own noise, so the bootstrap does not assume the normal, " +
    "equal-variance residuals the model's CI formula does. " +
    `Here (${t}(d)) the bootstrap CIs are ${Math.min(...ratios).toFixed(2)}–${Math.max(...ratios).toFixed(2)} ` +
    "times the model's width, and " + (changed.length ? `${changed.length} call(s) change.` : "no subject's call changes.") +
    " Width ratios between about 0.9 and 1.1 mean the same, given 200 replicates" +
    (model === "scene" ? "; these two runs were made when the scene-only model was the main model." : ".") +
    ` Run on ${data.measure}, all attempts, log form` +
    (notMain ? ', <span class="filtered">not the setting chosen above</span>.' : ".") +
    ` Source: ${data.runs.map((r) => `<code>${r.file}</code>`).join(", ")}.`;
}

/* Scene effects or route effects, and the shape: the route-scale data (own-route distance against
 * nth use) fitted at three corrections — pooled least squares, scene + level random effects, and
 * scene + level + route (the route-scale model) — each linear on log-log axes and with a
 * log(use)^2 term. From statsData.route_checks (output_data/practice/route_checks.json, fitted by
 * analysis/practice_route_checks.R); the browser only draws and tabulates it. The bins under the
 * lines are the pooled bins of the fit-check panel (practice_points.json). */
const RCHK = [
  ["pooled", "nothing removed (plain line)", FIT_POINT_COLOR],
  ["scene", "scene & level differences removed", CLUSTER_COLORS[0]],
  ["route_int", "+ route baselines removed (route intercepts)", CLUSTER_COLORS[2]],
  ["route", "scene, level & route differences removed (route model)", CLUSTER_COLORS[1]],
];

function renderStatsRouteChecks() {
  const block = $("stats-rchk-block");
  block.hidden = !onRouteClock() || state.stats.form !== "log";
  if (block.hidden) return;
  const rc = statsData.route_checks;
  const table = $("stats-rchk-table"), quad = $("stats-rchk-quad"), host = $("stats-rchk-panels");
  table.innerHTML = ""; quad.innerHTML = ""; host.innerHTML = "";
  $("stats-rchk-legend").innerHTML = ""; $("stats-rchk-quad-cap").textContent = "";
  if (!rc) {
    $("stats-rchk-cap").innerHTML = "Not in this export. Run <code>invoke run-practice-route-checks</code>, " +
      "then <code>invoke run-web-export --backfill</code>, and reload.";
    return;
  }
  const { measure, transform } = state.stats;
  const m = `${measure}_loso`;
  const completed = state.stats.attempts === "completed";
  const fit = (s, correction, shape) => rc.fits.find((f) => f.measure === m && f.transform === transform &&
    f.completed_only === completed && f.subject === s && f.correction === correction && f.shape === shape);
  const doubling = transform === "log";
  const pct = (b) => `${b < 0 ? "−" : "+"}${Math.abs((2 ** b - 1) * 100).toFixed(1)}%`;
  const eff = (b) => (doubling ? pct(b) : signed(b, 3));
  const subjects = statsData.subjects.filter((s) => fit(s, "route", "linear"));
  // A file fitted before route_int existed (design v1) lacks that step; show what it has.
  const shown = RCHK.filter(([c]) => rc.fits.some((f) => f.correction === c));
  const has = (c) => shown.some(([x]) => x === c);

  // 1. The slope at each correction: does the pooled decrease survive scene effects, then routes?
  const rows = subjects.map((s) => {
    const cells = shown.map(([c]) => {
      const f = fit(s, c, "linear");
      const clear = f.wald_high < 0 || f.wald_low > 0;
      return `<td class="${clear ? "clear" : "unclear"}">${eff(f.b)} <span class="muted">(${eff(f.wald_low)} to ` +
             `${eff(f.wald_high)})</span></td><td>${c === "pooled" ? "—" : fmtP(f.p_bh)}</td>`;
    });
    return `<tr><td class="subj"><i style="background:${subjColor(s)}"></i>sub-${s}</td>${cells.join("")}</tr>`;
  });
  table.innerHTML = "<thead><tr><th>subject</th>" +
    shown.map(([c, label]) => `<th>${label}: ${doubling ? "per doubling" : "slope"} (95% CI)</th>` +
                             `<th>p, BH-FDR</th>`).join("") +
    `</tr></thead><tbody>${rows.join("")}</tbody>`;

  // 2. Every line over the pooled bins, one panel per subject.
  const inv = INVERSE[transform];
  const panels = subjects.map((s) => {
    const box = document.createElement("div");
    box.className = "stats-chart";
    host.appendChild(box);
    const data = fitPoints && fitPoints[`lme4_route|${m}|${transform}|${completed ? 1 : 0}|${s}`];
    const lines = shown.flatMap(([c, , color]) => ["linear", "quadratic"].map((shape) => {
      const f = fit(s, c, shape);
      return { color, shape, label: c, pts: f.curve.use.map((u, i) => [u, inv(f.curve.y[i])]) };
    }));
    return { s, box, bins: (data && data.bins) || [], lines };
  });
  const ys = panels.flatMap((p) => [...p.bins.flatMap((b) => [b.lo, b.hi]),
                                    ...p.lines.flatMap((l) => l.pts.map(([, y]) => y))]).filter((y) => y > 0);
  const yDomain = [Math.min(...ys) * 0.95, Math.max(...ys) * 1.05];
  for (const p of panels) drawRouteCheckPanel(p, yDomain, MEASURE_INFO[measure].digits);
  $("stats-rchk-legend").innerHTML =
    legendHTML(shown.map(([, label, color]) => [color, label])) +
    '<span class="muted">solid: linear on log-log · dashed: + quadratic term</span> ' +
    legendHTML([["var(--ink)", "pooled bins, 95% CI"]], true);
  if (fitPoints === null) renderStatsFitData();   // starts the lazy load of the bins

  // 3. Linear against quadratic, per correction.
  const qrows = subjects.map((s) => {
    const cells = shown.map(([c]) => {
      const lin = fit(s, c, "linear"), q = fit(s, c, "quadratic");
      const dAic = q.aic - lin.aic;
      return `<td>${dAic >= 0 ? "+" : "−"}${Math.abs(dAic).toFixed(1)}</td>` +
             `<td>${c === "pooled" ? fmtP(q.lrt_p) : fmtP(q.lrt_p_bh)}</td>` +
             `<td>${eff(q.slope_at_1)} → ${eff(q.slope_at_50)}</td>`;
    });
    return `<tr><td class="subj"><i style="background:${subjColor(s)}"></i>sub-${s}</td>${cells.join("")}</tr>`;
  });
  quad.innerHTML = "<thead><tr><th>subject</th>" +
    shown.map(([c, label]) => `<th>${label}: ΔAIC</th><th>curvature p${c === "pooled" ? "" : ", BH-FDR"}</th>` +
                            `<th>local slope, use 1 → 50</th>`).join("") +
    `</tr></thead><tbody>${qrows.join("")}</tbody>`;

  const lin = (c) => subjects.map((s) => fit(s, c, "linear"));
  const range = (fs) => {
    const v = fs.map((f) => (doubling ? (2 ** f.b - 1) * 100 : f.b));
    return `${Math.min(...v).toFixed(1)} to ${Math.max(...v).toFixed(1)}${doubling ? "%" : ""}`;
  };
  const sceneClear = lin("scene").filter((f) => f.wald_high < 0).length;
  const notMain = measure !== "d_own" || transform !== "log" || completed;
  $("stats-rchk-cap").innerHTML =
    "The same data as the route model — each attempt's distance to its own route's prototype, against the nth " +
    `use of that route, leave-subject-out medoids — fitted ${shown.length} ways, each removing more. Every line is ` +
    "the fixed-effect slope, the trend left once the differences named are taken out. " +
    "<strong>Nothing removed</strong> (grey) is a plain straight line through every attempt, no random effects: " +
    "the trend the pooled row above shows by eye. Its CI treats attempts as independent and is far too " +
    "narrow, so it gets no test. <strong>Scene &amp; level differences removed</strong> (blue): every level and " +
    "scene gets its own intercept and slope (random effects), so easy or often-played scenes cannot make the " +
    "trend, but all routes in a scene are treated alike: does the pooled decrease survive? " +
    "<strong>+ route baselines removed</strong> (green): each route also gets its own intercept only, so routes " +
    "may differ in how tight they are but all improve at the same rate. What it takes away is structure: " +
    "some routes are simply tighter. Whatever slope survives it is change along a route. " +
    "<strong>Scene, level &amp; route differences removed</strong> (orange) is the route model itself: each route " +
    "gets its own intercept and slope, so it is also allowed its own rate of improvement. " +
    `Here the decrease is ${range(lin("pooled"))} per ${doubling ? "doubling" : "log unit"} with nothing removed, ` +
    `${range(lin("scene"))} with scene and level differences removed (CI below 0 for ${sceneClear} of ` +
    `${subjects.length}), ` +
    (has("route_int") ? `${range(lin("route_int"))} with route baselines removed as well, and ` : "and ") +
    `${range(lin("route"))} with route intercepts and slopes. ` +
    "What disappears once routes are separated is a difference " +
    "<em>between routes</em> in the same scene, not one between scenes. " +
    "p, BH-FDR: Benjamini-Hochberg across the five subjects, per column (statsmodels). " +
    (notMain ? '<span class="filtered">Not the main setting (d_own, log, all attempts).</span> ' : "") +
    "Source: <code>output_data/practice/route_checks.json</code>.";
  $("stats-rchk-quad-cap").innerHTML =
    "<strong>Is a straight line on log-log axes the right shape?</strong> Each model refitted with a " +
    "log(use)² term (a fixed effect only: scenes and routes keep their own straight slope) and compared " +
    "with the straight line by AIC and a likelihood-ratio test (1 df). ΔAIC below about −2 favours the " +
    "curve; the local slope shows which way it bends: towards 0 means the decrease flattens with use, away " +
    "from 0 that it steepens. The pooled test treats attempts as independent, so it says the curve is real " +
    "far too readily.";
}

function drawRouteCheckPanel({ s, box, bins, lines }, yDomain, digits) {
  const d = document.createElement("div");
  d.className = "panel-title";
  d.innerHTML = `sub-${s}`;
  box.appendChild(d);
  const w = box.clientWidth || 240, height = 200;
  const pad = { l: 52, r: 8, t: 8, b: 30 };
  const svg = svgEl("svg", { viewBox: `0 0 ${w} ${height}`, height, role: "img",
                             "aria-label": `sub-${s}: the route data under three corrections` }, box);
  const xMax = Math.max(...lines.map((l) => l.pts[l.pts.length - 1][0]), ...bins.map((b) => b.x)) * 1.15;
  const [x0, x1] = [Math.log(0.8), Math.log(xMax)];
  const [y0, y1] = [Math.log(yDomain[0]), Math.log(yDomain[1])];
  const px = (x) => pad.l + ((w - pad.l - pad.r) * (Math.log(x) - x0)) / (x1 - x0);
  const py = (y) => height - pad.b - ((height - pad.t - pad.b) * (Math.log(y) - y0)) / (y1 - y0 || 1);
  drawLogAxes(svg, { px, py, pad, w, height, xMax, ylo: yDomain[0], yhi: yDomain[1],
                    xName: "nth use of the route", yName: "distance (px)" });
  for (const l of lines) {
    svgEl("polyline", { points: l.pts.map(([x, y]) => `${px(x)},${py(y)}`).join(" "), fill: "none",
                        stroke: l.color, "stroke-width": l.shape === "linear" ? 2 : 1.6,
                        ...(l.shape === "linear" ? {} : { "stroke-dasharray": "5 3" }),
                        "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
  }
  for (const b of bins) {
    const g = svgEl("g", {}, svg);
    svgEl("line", { x1: px(b.x), x2: px(b.x), y1: py(b.lo), y2: py(b.hi), stroke: "var(--ink)", "stroke-width": 1.2 }, g);
    svgEl("circle", { cx: px(b.x), cy: py(b.mean), r: 3, fill: "var(--ink)", class: "dot-ring" }, g);
    const hit = svgEl("circle", { cx: px(b.x), cy: py(b.mean), r: 8, class: "hit" }, g);
    hit.addEventListener("mousemove", (ev) => showStatsTooltip(
      `sub-${s} · use ≈ ${b.x.toFixed(b.x < 10 ? 1 : 0)}`, "var(--ink)",
      [["pooled geometric mean", b.mean.toFixed(digits + 1)],
       ["95% CI", `${b.lo.toFixed(digits + 1)} – ${b.hi.toFixed(digits + 1)}`],
       ["attempts", b.n_obs], ["scenes", b.n_scenes]], ev.clientX, ev.clientY));
    hit.addEventListener("mouseleave", hideStatsTooltip);
  }
}

function renderStatsTable() {
  const { measure, form } = state.stats;
  const info = MEASURE_INFO[measure];
  const term = SLOPE_TERM[form];
  // A log-log slope b is a proportional effect: each doubling of practice multiplies the
  // distance by 2^b. Only under log(d) — log1p(d) slopes are not a clean ratio.
  const doubling = form === "log" && state.stats.transform === "log";
  const pctDoubling = (b) => `${b < 0 ? "−" : "+"}${Math.abs((2 ** b - 1) * 100).toFixed(1)}%`;
  const head = ["subject", "medoids", "coefficient", "95% CI",
                ...(doubling ? ["per doubling (95% CI)"] : []), "p", "p, BH-FDR (5 subjects)", "CI excludes 0",
                `at ${onRouteClock() ? "use" : "attempt"} 1 (${info.unit})`,
                `at ${onRouteClock() ? "use" : "attempt"} ${STATS_AT} (${info.unit})`,
                "attempts", "zeros dropped",
                state.stats.model === "lme4_route" ? "routes"
                  : measure === "d_own" && state.stats.model === "scene" ? "series" : "scenes",
                "converged"];
  const rows = [];
  for (const s of statsData.subjects) {
    for (const ser of [...STATS_SERIES].reverse()) {
      const fit = modelFit(measure + ser.suffix, s);
      const r = fit && fit.r;
      if (!r || !r.coef) continue;
      const f = fit.meta;
      const [lo, hi] = r.ci[term];
      const clear = hi < 0 || lo > 0;
      rows.push(`<tr class="${ser.suffix ? "loso" : ""}">` +
        `<td class="subj"><i style="background:${subjColor(s)}"></i>sub-${s}</td>` +
        `<td>${ser.suffix ? "leave-subject-out" : "all-subject"}</td>` +
        `<td>${signed(r.coef[term])}</td><td>${signed(lo)} to ${signed(hi)}</td>` +
        (doubling ? `<td>${pctDoubling(r.coef[term])} (${pctDoubling(lo)} to ${pctDoubling(hi)})</td>` : "") +
        `<td>${fmtP(r.p[term])}</td><td>${fmtP(statsBh(measure + ser.suffix, s))}</td>` +
        `<td class="${clear ? "clear" : "unclear"}">${clear ? "yes" : "no"}</td>` +
        `<td>${statsPredict(r.coef, form, 1).toFixed(info.digits)}</td>` +
        `<td>${statsPredict(r.coef, form, STATS_AT).toFixed(info.digits)}</td>` +
        `<td>${f.n_obs.toLocaleString()}</td><td>${(f.n_zero_dropped || 0).toLocaleString()}</td>` +
        `<td>${f.n_groups.toLocaleString()}</td>` +
        `<td>${r.converged ? (f.singular ? "yes (singular)" : "yes") : "no"}</td></tr>`);
    }
  }
  $("stats-table").innerHTML =
    `<thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody>`;
  $("stats-table-cap").textContent =
    "p, BH-FDR: the p-value corrected with the Benjamini-Hochberg procedure for testing the same model " +
    "on five subjects, separately for all-subject and leave-subject-out medoids; significant after " +
    "correction when below 0.05. It controls the false discovery rate, the expected share of false " +
    "positives among the subjects called significant. Computed by statsmodels' multipletests " +
    "(method fdr_bh) in the analysis, not in the browser. " +
    (doubling
    ? `Per doubling: the change in distance each time the number of ` +
      `${onRouteClock() ? "uses of a route" : "attempts at a scene"} doubles, ` +
      "(2^b − 1) × 100, with the CI converted the same way. It is a proportional change, so it " +
      "holds at any point on the curve: 1→2, 10→20 and 30→60 all bring the same percentage."
    : "Per-doubling percentages are shown only for the log form under log(d); under " +
      (form === "log" ? "log(1 + d)" : "the linear form") + " the slope is not a proportional change.");
}

/* Assumption checks for the selected model and transform. Read from the payload's
 * `diagnostics` block, which grades docs/practice_residual_diagnostics*.csv with that
 * script's own verdicts() — the dashboard never re-judges a check. Only the scene-scale
 * lme4 model and the scene-only statsmodels model have been checked; for the others the
 * block says so rather than borrowing another model's verdicts. */
const CHECK_MODELS = { lme4_nested: "lme4", lme4_route: "route", scene: "scene" };
const VERDICT_MARK = { ok: "✓", mild: "~", bad: "✕" };
const CHECK_COLUMNS = [
  ["n_obs", "attempts", 0], ["n_zero_dropped", "zeros dropped", 0],
  ["max_bin_bias", "1. largest bin bias", 3], ["lag1_r", "2. lag-1 r", 3],
  ["skew", "3-4. skew", 2], ["kurtosis", "3-4. kurtosis", 2], ["sd_ratio", "5. SD ratio", 2],
  ["max_abs_z_re", "6. most extreme scene (SD)", 1],
  ["share_scenes_outside_funnel", "7. scenes outside funnel", 2],
  ["outside_shuffled_p95", "7. shuffled 95th pct", 2],
  ["world_sd_ratio", "8. world SD ratio", 2], ["world_sd_ratio_shuffled_p95", "8. shuffled 95th pct", 2],
];

function renderStatsChecks() {
  const diag = statsData.diagnostics;
  const block = $("stats-checks-block");
  block.hidden = !diag;
  if (!diag) return;
  const { transform, measure, attempts } = state.stats;
  const key = CHECK_MODELS[state.stats.model];
  const run = key && diag[key] && diag[key][transform];
  const table = $("stats-checks"), subj = $("stats-checks-subj");
  $("stats-checks-subj-wrap").hidden = !run;
  $("stats-checks-fig-wrap").hidden = !(run && run.figure);
  if (!run) {
    const have = Object.values(diag).map((d) => d.label).join(" and ");
    $("stats-checks-cap").textContent =
      `No assumption checks have been run for the ${STATS_MODELS[state.stats.model].label} model` +
      (key ? ` under ${transform}(d)` : "") + `. Checks exist for the ${have} models; select one ` +
      "to see them. They are produced by docs/practice_residual_diagnostics.py.";
    table.innerHTML = "";
    subj.innerHTML = "";
    return;
  }
  const checked = diag[key].measure || statsData.primary;
  const shown = `${measure}_loso` === checked && attempts === "all";
  $("stats-checks-cap").innerHTML =
    `Residual and random-effect checks for the ${diag[key].label} model on ` +
    `<code>${checked}</code>, <code>${transform}(d)</code>, all attempts` +
    (shown ? "" : ' <span class="filtered">— not the measure selected above; the checks were run on this one only</span>') +
    ". Each row is one assumption. The third column says what a violation would damage: the " +
    "<b>slope</b> itself, or only its <b>CI</b> and p-value. Rows marked CI matter only if the CI " +
    "actually moves, which is what the last row tests by comparing with a bootstrap that does " +
    "not make these assumptions. Verdicts come from docs/practice_residual_diagnostics.py, " +
    "not from this page.";
  table.innerHTML =
    "<thead><tr><th>check</th><th>question</th><th>if violated, it affects</th><th>verdict</th></tr></thead><tbody>" +
    run.checks.map((c) => {
      const [what, ...rest] = c.affects.split(" - ");
      return `<tr><td class="check">${c.title}</td>` +
        `<td>${c.question}<span class="verdict-detail">look for: ${c.look_for}</span></td>` +
        `<td class="affects"><b>${what}</b>${rest.length ? " — " + rest.join(" - ") : ""}</td>` +
        `<td><span class="verdict ${c.status}"><span aria-hidden="true">${VERDICT_MARK[c.status]}</span>${c.verdict}</span>` +
        `<span class="verdict-detail">${c.detail}</span></td></tr>`;
    }).join("") + "</tbody>";
  const cols = CHECK_COLUMNS.filter(([k]) => run.subjects.some((r) => r[k] != null));
  subj.innerHTML =
    `<thead><tr><th>subject</th>${cols.map(([, h]) => `<th>${h}</th>`).join("")}</tr></thead><tbody>` +
    run.subjects.map((r) => `<tr><td class="subj"><i style="background:${subjColor(r.subject)}"></i>sub-${r.subject}</td>` +
      cols.map(([k, , d]) => `<td>${r[k] == null ? "—" : Number(r[k]).toFixed(d)}</td>`).join("") + "</tr>").join("") +
    "</tbody>";
  // The figure is ~1 MB: only fetched when the reader opens it.
  const fig = $("stats-checks-fig"), wrap = $("stats-checks-fig-wrap");
  fig.dataset.src = `data/${run.figure}`;
  if (wrap.open) fig.src = fig.dataset.src;
}

/* The summary view: every CI and p-value for the scene-scale main result (primary measure,
 * all attempts, log form), per transform — the dashboard twin of docs/practice_summary.png. */
function renderStatsSummary() {
  const sum = statsData.summary;
  const block = $("stats-summary-block");
  block.hidden = !sum;
  if (!sum) return;
  const host = $("stats-summary");
  host.innerHTML = "";
  const order = ["log", "log1p"].filter((t) => sum.by_transform[t]);
  const methods = sum.methods.filter((m) => order.some((t) =>
    Object.values(sum.by_transform[t]).some((x) => x[m])));
  const boxes = order.map((t) => {
    const box = document.createElement("div");
    box.className = "stats-chart";
    host.appendChild(box);
    return [t, box];
  });
  for (const [t, box] of boxes) {
    const bt = sum.by_transform[t];
    const subjects = Object.keys(bt).sort();
    const title = document.createElement("div");
    title.className = "panel-title";
    title.innerHTML = t === "log" ? "Main analysis: log(d)" : "Original analysis: log(1 + d)";
    box.appendChild(title);
    const w = box.clientWidth || 600, rowH = 22 * methods.length + 16;
    const pad = { l: 58, r: 14, t: 8, b: 34 };
    const h = pad.t + subjects.length * rowH + pad.b;
    const all = subjects.flatMap((s) => methods.filter((m) => bt[s][m]).flatMap((m) => [bt[s][m].lo, bt[s][m].hi]));
    let lo = Math.min(0, ...all), hi = Math.max(0, ...all);
    const mg = (hi - lo) * 0.05; lo -= mg; hi += mg;
    const px = (v) => pad.l + ((w - pad.l - pad.r) * (v - lo)) / (hi - lo);
    const svg = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, height: h, role: "img",
                               "aria-label": `practice slope by method, ${t}` }, box);
    for (const v of niceTicks(lo, hi, 5)) {
      svgEl("line", { x1: px(v), x2: px(v), y1: pad.t, y2: h - pad.b, class: "grid" }, svg);
      svgEl("text", { x: px(v), y: h - pad.b + 14, "text-anchor": "middle", class: "tick" }, svg)
        .textContent = v === 0 ? "0" : signed(v, 3).replace("+", "");
    }
    svgEl("line", { x1: px(0), x2: px(0), y1: pad.t, y2: h - pad.b, class: "zero" }, svg);
    svgEl("text", { x: (pad.l + w - pad.r) / 2, y: h - 6, "text-anchor": "middle", class: "axis-label" }, svg)
      .textContent = `practice slope b (${t} units)`;
    subjects.forEach((s, i) => {
      const y0 = pad.t + i * rowH;
      svgEl("text", { x: pad.l - 8, y: y0 + rowH / 2 + 4, "text-anchor": "end", class: "row-label" }, svg)
        .textContent = `sub-${s}`;
      if (i) svgEl("line", { x1: 0, x2: w, y1: y0, y2: y0, class: "grid" }, svg);
      methods.forEach((m, k) => {
        const v = bt[s][m];
        if (!v) return;
        const y = y0 + 8 + k * 22 + 4, col = SUMMARY_COLORS[m];
        const clear = v.hi < 0 || v.lo > 0;
        svgEl("line", { x1: px(v.lo), x2: px(v.hi), y1: y, y2: y, stroke: col, "stroke-width": 2.2,
                        "stroke-linecap": "round" }, svg);
        svgEl("circle", { cx: px(v.b), cy: y, r: 4.5, fill: clear ? col : "var(--surface-2)",
                          stroke: col, "stroke-width": 1.8 }, svg);
        const hit = svgEl("rect", { x: px(v.lo) - 6, y: y - 7, width: px(v.hi) - px(v.lo) + 12,
                                    height: 14, class: "hit" }, svg);
        const rows = [["slope b", signed(v.b)], ["95% CI", `${signed(v.lo)} to ${signed(v.hi)}`],
                      ["p", v.p_floor ? `< ${v.p.toFixed(2)}` : fmtP(v.p)],
                      ["clear effect", clear ? "yes" : "no"]];
        if (v.p_bh != null) rows.push(["p, BH-FDR adjusted", fmtP(v.p_bh)]);
        hit.addEventListener("mousemove", (ev) =>
          showStatsTooltip(`sub-${s} · ${SUMMARY_LABELS[m]}`, col, rows, ev.clientX, ev.clientY));
        hit.addEventListener("mouseleave", hideStatsTooltip);
      });
    });
  }
  $("stats-summary-legend").innerHTML =
    legendHTML(methods.map((m) => [SUMMARY_COLORS[m], SUMMARY_LABELS[m]]));
  $("stats-sum-cap").textContent =
    `${sum.measure}, log practice form, all attempts. Every bar is a 95% confidence interval for ` +
    "the practice slope b from one method; filled = it clears zero (a clear effect), hollow = it " +
    "does not. The two panels are in different units: compare calls and widths, never positions.";

  // p-value table: transforms side by side, methods as columns, BH-FDR after the scene model.
  const cols = [];
  for (const t of order) {
    for (const m of methods) {
      cols.push([t, m, "p"]);
      if (Object.values(sum.by_transform[t]).some((x) => x[m] && x[m].p_bh != null)) cols.push([t, m, "p_bh"]);
    }
  }
  const subjects = Object.keys(sum.by_transform[order[0]]).sort();
  const groupHead = order.map((t) => {
    const n = cols.filter((c) => c[0] === t).length;
    return `<th colspan="${n}" class="grp">${t === "log" ? "main: log(d)" : "original: log(1 + d)"}</th>`;
  }).join("");
  const colHead = cols.map(([, m, k]) =>
    `<th>${k === "p_bh" ? `${m === "lme4" ? "scene scale" : "scene only"},<br>BH-FDR adjusted`
                          : SUMMARY_LABELS[m].replace(" (", "<br>(").replace(" — main", "")}</th>`).join("");
  const body = subjects.map((s) => `<tr><td class="subj"><i style="background:${subjColor(s)}"></i>sub-${s}</td>` +
    cols.map(([t, m, k]) => {
      const v = (sum.by_transform[t][s] || {})[m];
      if (!v || v[k] == null) return "<td>—</td>";
      const p = v[k];
      const txt = v.p_floor && k === "p" ? `< ${p.toFixed(2)}` : fmtP(p);
      return `<td class="${p < 0.05 ? "clear" : "unclear"}">${txt}</td>`;
    }).join("") + "</tr>").join("");
  $("stats-ptable").innerHTML =
    `<thead><tr><th></th>${groupHead}</tr><tr><th>subject</th>${colHead}</tr></thead><tbody>${body}</tbody>`;
  $("stats-p-note").innerHTML =
    "<strong>How each p-value is computed</strong> — every one a two-sided test of no practice effect (b = 0). " +
    "<em>Fitted models</em> (scene-scale lme4, scene only, nested statsmodels): Wald z test, z = b / SE with the SE " +
    "from the model, p from a normal curve; lme4's bars above are profile-likelihood CIs. " +
    "<em>BH-FDR adjusted</em>: the scene-scale model's p-values corrected for testing five subjects with " +
    "the Benjamini-Hochberg procedure, which controls the expected share of false positives among the " +
    "subjects called significant. " +
    "<em>Bootstraps</em>: the share of 200 resampled slopes on the far side of zero, doubled; no normality " +
    "assumed, and with 200 replicates the smallest it can show is 0.01. The lme4 level bootstrap refits the " +
    "main scene-scale model itself; the scene and level bootstraps of the scene-only model were made when that " +
    "was the main model, and are kept as a second check. Bold = p < 0.05, the same call as an " +
    "interval clearing zero. The columns are different methods for the same question, not repeated tests.";
}

/* Disable what the selected model was not fitted for, and fall back off it. */
function applyStatsAvailability() {
  const hasLme4 = !!(statsData && statsData.lme4);
  const hasNested = !!(statsData && statsData.nested);
  const hasRoute = !!(statsData && statsData.lme4_route);
  const avail = { lme4_nested: hasLme4, scene: true, level_scene: hasNested, lme4_route: hasRoute,
                  route_scene: !!(statsData && statsData.route_checks) };
  if (!avail[state.stats.model]) state.stats.model = "scene";
  const route = onRouteClock();
  const nestedModel = state.stats.model !== "scene";
  if (nestedModel && state.stats.form !== "log") state.stats.form = "log";
  // The within-route model exists for the own-medoid distance only; the other nested
  // models for nearest/spread only.
  if (route) state.stats.measure = "d_own";
  else if (nestedModel && state.stats.measure === "d_own") state.stats.measure = "d_nearest_norm";
  const sync = (id, key, disabledFor) => {
    for (const b of $(id).querySelectorAll("button")) {
      b.disabled = disabledFor(b.dataset.value);
      b.setAttribute("aria-pressed", String(b.dataset.value === state.stats[key]));
    }
  };
  sync("stats-model-seg", "model", (v) => !avail[v]);
  sync("stats-form-seg", "form", (v) => nestedModel && v !== "log");
  sync("stats-measure-seg", "measure", (v) => (route ? v !== "d_own" : nestedModel && v === "d_own"));
  sync("stats-transform-seg", "transform", () => false);
  sync("stats-attempts-seg", "attempts", () => false);
}

function renderStats() {
  let { measure, form, attempts, transform } = state.stats;
  $("chart-title").textContent = "Practice effect — one mixed model per subject";
  $("reset-view").hidden = true;
  $("clear-filters").hidden = true;

  if (statsData === null) {
    $("chart-sub").textContent = "loading practice models…";
    if (!statsLoading) {
      statsLoading = true;
      fetch("data/practice.json", { cache: "no-cache" })
        .then((res) => (res.ok ? res.json() : false))
        .catch(() => false)
        .then((data) => {
          statsData = data;
          if (data) {
            state.statsSubjectColors = new Map(
              data.subjects.map((s, i) => [s, SUBJECT_COLORS[i % SUBJECT_COLORS.length]]));
          }
          render();
        });
    }
    return;
  }
  $("stats-missing").hidden = !!statsData;
  $("stats-body").hidden = !statsData;
  if (!statsData) {
    $("chart-sub").textContent = "no practice models in this export";
    $("stats-missing").textContent =
      "No practice models found in data/practice.json. Run  invoke run-practice-models  and " +
      "then  invoke run-web-export --backfill  (a small copy, no re-export), and reload.";
    return;
  }

  applyStatsAvailability();
  ({ measure, form, attempts, transform } = state.stats);   // the model may have switched the measure
  // Two main results, one per scale: the scene scale on the primary measure (the scene-only
  // statsmodels model stands in without R), and the route scale on the own-route distance.
  const primaryModel = statsData.lme4 ? "lme4_nested" : "scene";
  const mainSetting = state.stats.form === "log" && attempts === "all"
                      && transform === statsData.primary_transform;
  const mainScale = !mainSetting ? null
    : state.stats.model === primaryModel && `${measure}_loso` === statsData.primary ? "scene scale"
    : state.stats.model === "lme4_route" && measure === "d_own" ? "route scale" : null;
  $("chart-sub").innerHTML =
    `${STATS_MODELS[state.stats.model].short} · ${MEASURE_INFO[measure].name} · ${transform}(d) · ` +
    `${state.stats.form} form · ` +
    `${attempts === "all" ? "all attempts" : "completed attempts only"}` +
    ` · ${statsData.n_scenes} scenes · ` +
    (mainScale ? `a main result (${mainScale})` : '<span class="filtered">a robustness check, not a main result</span>');
  renderStatsSummary();
  renderStatsCoef();
  renderStatsCurves();
  renderStatsFitData();
  renderStatsSpaghetti();
  renderStatsRouteChecks();
  renderStatsTable();
  renderStatsBootstrap();
  renderStatsChecks();
  $("stats-note").innerHTML =
    "Each subject is fitted separately, by maximum likelihood. There are <strong>two main models</strong>, " +
    "one per scale, both in lme4. <strong>Scene scale</strong>: distance to the nearest route prototype, " +
    "<code>log(d) ~ log(attempt) + (1 + log(attempt) | level) + (1 + log(attempt) | level:scene)</code> — " +
    "a level's scenes are played back to back, so scenes are nested in levels. <strong>Route scale</strong>: " +
    "distance in px to the attempt's own route prototype against the nth use of that route, with routes " +
    "nested in scenes in levels. The first asks whether players settle into routes, the second whether " +
    "they execute a route more precisely. " +
    `Selected here: <em>${STATS_MODELS[state.stats.model].formula}</em>. ` +
    "The scene-only statsmodels model (random intercept and slope on the same time term per scene, per " +
    "scene × cluster for the own-medoid measure) is the one fitted in every form and measure. Coefficients are in the transform's " +
    "units, so compare <code>log</code> and <code>log1p</code> through the curves, not the slopes; " +
    "curves and the <em>at attempt</em> columns are converted back. Leave-subject-out " +
    "measures each subject against medoids and spreads chosen from the other subjects only. " +
    `The main results are <code>${statsData.primary}</code> at the scene scale and <code>d_own_loso</code> at ` +
    `the route scale, on <code>${statsData.primary_transform}(d)</code> ` +
    "(attempts at exactly 0 dropped), log form, all attempts; <code>log1p</code> is the original " +
    "analysis, kept because its residuals were heavily right-skewed and the switch should stay traceable; " +
    "everything else here is a check on them. Sources: <code>output_data/practice/lme4.json</code>, " +
    "<code>lme4_route.json</code>, " +
    "<code>mixed.json</code>, <code>nested.json</code> (all flattened in <code>mixed_summary.csv</code>) and the " +
    "bootstraps under <code>docs/</code>.";
}

function bindStats() {
  $("stats-checks-fig-wrap").addEventListener("toggle", (ev) => {
    const fig = $("stats-checks-fig");
    if (ev.target.open && fig.dataset.src && fig.getAttribute("src") !== fig.dataset.src) fig.src = fig.dataset.src;
  });
  const section = $("section-seg");
  section.addEventListener("click", (ev) => {
    const btn = ev.target.closest("button");
    if (!btn) return;
    for (const b of section.querySelectorAll("button")) {
      b.setAttribute("aria-pressed", String(b === btn));
    }
    state.section = btn.dataset.value;
    for (const el of document.querySelectorAll("[data-section]")) {
      el.hidden = el.dataset.section !== state.section;
    }
    hideStatsTooltip();
    render();
  });
  for (const seg of document.querySelectorAll("[data-axis-units]")) {
    seg.addEventListener("click", (ev) => {
      const btn = ev.target.closest("button");
      if (!btn) return;
      state.stats.axisUnits = btn.dataset.units;
      for (const b of document.querySelectorAll("[data-axis-units] button")) {
        b.setAttribute("aria-pressed", String(b.dataset.units === state.stats.axisUnits));
      }
      hideStatsTooltip();
      renderStatsFitData();
      renderStatsSpaghetti();
      renderStatsRouteChecks();
    });
  }
  $("stats-spaghetti-subject").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button");
    if (!btn) return;
    state.stats.spSubject = btn.dataset.subject;
    hideStatsTooltip();
    renderStatsSpaghetti();
  });
  for (const [id, key] of [["stats-model-seg", "model"],
                           ["stats-measure-seg", "measure"], ["stats-transform-seg", "transform"],
                           ["stats-form-seg", "form"],
                           ["stats-attempts-seg", "attempts"]]) {
    const seg = $(id);
    seg.addEventListener("click", (ev) => {
      const btn = ev.target.closest("button");
      if (!btn || btn.disabled) return;
      for (const b of seg.querySelectorAll("button")) {
        b.setAttribute("aria-pressed", String(b === btn));
      }
      state.stats[key] = btn.dataset.value;
      hideStatsTooltip();
      render();
    });
  }
}

/* ---------- boot ---------- */

async function init() {
  const res = await fetch("data/index.json", { cache: "no-cache" });
  state.index = await res.json();
  state.filters.subjects = new Set(state.index.subjects);
  state.subjectColors = new Map(
    state.index.subjects.map((s, i) => [s, SUBJECT_COLORS[i % SUBJECT_COLORS.length]]));

  const totalTraces = state.index.scenes.reduce((a, s) => a + s.n, 0);
  $("corpus-note").textContent =
    `${state.index.scenes.length} scenes · ${totalTraces.toLocaleString()} traces · ` +
    `${state.index.subjects.length} subjects`;

  const tagSel = $("tag-filter");
  for (const t of state.index.tags) {
    const opt = document.createElement("option");
    opt.value = t; opt.textContent = t;
    tagSel.appendChild(opt);
  }

  renderSceneList();
  renderSubjectList();

  $("scene-search").addEventListener("input", renderSceneList);
  tagSel.addEventListener("change", renderSceneList);
  bindSeg("phase-seg", "phase");
  bindSeg("stage-seg", "stage");
  bindSeg("outcome-seg", "outcome");
  bindSeg("view-seg", "view");
  bindMatrix();
  bindSelectionCanvas();
  bindSeg("order-seg", "matrixOrder");
  $("toggle-partition").addEventListener("change", (e) => {
    state.showPartition = e.target.checked;
    render();
  });
  bindSeg("color-seg", "colorBy");
  bindCutControl();
  $("toggle-noise").addEventListener("change", (e) => {
    state.showNoise = e.target.checked; render();
  });
  $("toggle-bg").addEventListener("change", (e) => {
    state.showBg = e.target.checked; resetView();
  });
  $("reset-view").addEventListener("click", resetView);
  $("clear-filters").addEventListener("click", clearFilters);
  bindCanvas();
  bindStats();
  window.addEventListener("resize", render);

  if (state.index.scenes.length) selectScene(state.index.scenes[0].id);
  // index.html#stats opens straight onto the stats section, so it can be linked to.
  if (location.hash === "#stats") $("section-seg").querySelector('[data-value="stats"]').click();
}

init();
