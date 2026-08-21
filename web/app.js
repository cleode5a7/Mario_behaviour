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

/* Matplotlib linewidths are points at dpi=150; these are the CSS-pixel
 * equivalents that read the same on screen. */
const LW = { direct: 1.6, faint: 1.0, outcome: 1.0, subject: 1.2 };
const ALPHA = { direct: 0.85, faint: 0.35, outcome: 0.5, subject: 0.6 };
const MARKER_PX = { overlay: 7, panel: 6 };

/* ---------- state ---------- */

const state = {
  index: null,
  scene: null,          // the loaded scene payload
  bg: null,             // HTMLImageElement or null
  subjectColors: new Map(),
  filters: { subjects: new Set(), phase: "all", outcome: "all" },
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

  for (const t of traces) {
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

/* ---------- panels view ---------- */

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

    // Per-cluster subject composition, mirroring the pills in
    // plot_scene_cluster_panels(); percentages are of the panel's visible traces.
    if (state.colorBy === "subject" && members.length) {
      const comp = new Map();
      for (const t of members) comp.set(t.s, (comp.get(t.s) || 0) + 1);
      const wrap = document.createElement("div");
      wrap.className = "composition";
      for (const s of [...comp.keys()].sort()) {
        const pct = Math.round((100 * comp.get(s)) / members.length);
        const span = document.createElement("span");
        span.textContent = `sub-${s}: ${pct}% (n=${comp.get(s)})`;
        span.style.color = state.subjectColors.get(s) || OTHER_COLOR;
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

    for (const t of members) {
      const color = state.colorBy === "subject"
        ? (state.subjectColors.get(t.s) || OTHER_COLOR)
        : state.colorBy === "outcome"
          ? (OUTCOME_COLORS[t.out] || OUTCOME_OTHER)
          : panelColor;
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
    ["outcome", t.out],
    ["duration", t.dur == null ? "—" : `${t.dur} s`],
    ["cluster", t.c === -1 ? "noise" : t.c],
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

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

/* ---------- render dispatch ---------- */

let rafPending = false;
function render() {
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => {
    rafPending = false;
    if (!state.scene) return;
    const overlay = state.view === "overlay";
    $("overlay-wrap").hidden = !overlay;
    $("panels").hidden = overlay;
    $("reset-view").hidden = !overlay;
    if (overlay) renderOverlay(); else renderPanels();
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
  state.filters.outcome = "all";
  state.hiddenClusters.clear();
  state.pinned = null; state.hover = null;
  for (const id of ["phase-seg", "outcome-seg"]) {
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
  bindSeg("outcome-seg", "outcome");
  bindSeg("view-seg", "view");
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
