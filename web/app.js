/* Mario clustering dashboard.
 *
 * Reads the payload written by `invoke run-web-export` and renders the same
 * figure analysis/trace_plots.py renders to PNG — background overlay, entry-point
 * x-offset, screen-corrected y, rank-by-size cluster palette with marker shapes
 * as a secondary encoding, playfield y-clipping — plus subject/phase/outcome
 * filters the static figures cannot offer.
 *
 * The one rule that matters scientifically: filters NEVER re-derive cluster
 * labels. HDBSCAN ran once over every trace in the scene; a filter only changes
 * which of those traces are drawn. Counts recompute for the visible subset, and
 * the title says "N of M" so a filtered view can't be mistaken for the full one.
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
  hiddenClusters: new Set(),
  hover: null,          // trace object under the cursor
  pinned: null,         // trace object clicked
  zoom: 1,
  pan: { x: 0, y: 0 },
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
  bits.push("cluster labels are fixed — filters only change what is drawn");
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
    ["stability", t.sb == null ? "—" : t.sb.toFixed(2)],
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
    btn.innerHTML = `<span class="scene-id">${s.id}</span>` +
      `<span class="scene-meta">${s.n} traces · ${s.k} cl</span>`;
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
    const res = await fetch(`data/scenes/${id}.json`);
    if (!res.ok) { console.error("failed to load scene", id); return; }
    payload = await res.json();
    sceneCache.set(id, payload);
  }
  if (token !== sceneToken) return;      // superseded by a later click
  state.scene = payload;
  state.hiddenClusters.clear();
  state.pinned = null; state.hover = null;
  state.zoom = 1; state.pan = { x: 0, y: 0 };

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
  const res = await fetch(`data/coassoc/${sceneId}.bin`);
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
function matrixOrder() {
  const rank = sceneRanking();
  const slot = (t) => (t.c === -1 ? Number.MAX_SAFE_INTEGER : rank.ranked.indexOf(t.c));
  const visible = new Set(visibleTraces());
  const rows = [];
  state.scene.traces.forEach((t, i) => { if (visible.has(t)) rows.push({ t, i }); });
  rows.sort((a, b) => (slot(a.t) - slot(b.t)) || (a.i - b.i));
  return rows;
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

  renderMatrixLegend();
  const total = state.scene.traces.length;
  note.textContent =
    `${m} of ${total} traces. Cell (i, j) = share of resampling replicates in which ` +
    `both traces landed in the same cluster, among replicates that drew both. ` +
    `Rows ordered by cluster, noise last — so each bright diagonal block is a cluster.`;
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
}

function showMatrixTooltip(a, b, v, clientX, clientY) {
  const tip = $("tooltip");
  const name = (t) => (t.c === -1 ? "noise" : `cluster ${t.c}`);
  const rows = a === b
    ? [["trace", a.clip], ["subject", `sub-${a.s}`], [name(a), ""],
       ["clustered (not noise) in", `${(v * 100).toFixed(0)}% of replicates`]]
    : [["row", `${a.clip} — ${name(a)}`], ["column", `${b.clip} — ${name(b)}`],
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

let rafPending = false;
function render() {
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => {
    rafPending = false;
    if (!state.scene) return;
    const view = state.view;
    $("overlay-wrap").hidden = view !== "overlay";
    $("panels").hidden = view !== "panels";
    $("matrix-wrap").hidden = view !== "matrix";
    $("reset-view").hidden = view !== "overlay";
    if (view === "overlay") renderOverlay();
    else if (view === "panels") renderPanels();
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

/* ---------- boot ---------- */

async function init() {
  const res = await fetch("data/index.json");
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
  bindSeg("color-seg", "colorBy");
  $("toggle-noise").addEventListener("change", (e) => {
    state.showNoise = e.target.checked; render();
  });
  $("toggle-bg").addEventListener("change", (e) => {
    state.showBg = e.target.checked; resetView();
  });
  $("reset-view").addEventListener("click", resetView);
  $("clear-filters").addEventListener("click", clearFilters);
  bindCanvas();
  window.addEventListener("resize", render);

  if (state.index.scenes.length) selectScene(state.index.scenes[0].id);
}

init();
