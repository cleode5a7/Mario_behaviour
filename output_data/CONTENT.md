# Output Data

After running the full pipeline (`invoke run`), this folder contains:

## `changepoints/`

One JSON file per subject produced by `invoke run-changepoints`:

- `<subject_id>_changepoints.json` — detected learning-phase changepoints in that subject's full chronological attempt sequence (all scenes combined).
  Fields: `{"subject": str, "n_attempts": int, "changepoints": [<attempt indices>], "phases": [...], "method": "phase_annotation"}`.
  Also carries a `"ruptures"` block — a non-Bayesian diagnostic that runs the `ruptures` PELT/Binseg/Dynp algorithms (CostL2) on the completion signal, on both the raw per-attempt series and a 50-attempt block-averaged series, for comparison against the Phase annotations:
  `{"model": "l2", "block_size": int, "beta": float, "n_bkps": int, "results": {"raw": {"pelt": [...], "binseg": [...], "dynp": [...]}, "blocks": {...}}}` (breakpoints in attempt-index space).
  And a `"repetitions"` block — the same detectors run one unit up, per **level repetition** (one `.bk2` playthrough) instead of per scene clip. This is the signal that actually tracks learning: a scene clip is cleared ~92% of the time (a ceiling no detector can segment), whereas a level repetition is cleared 53–81% depending on subject. Two nested criteria are reported — `cleared` (reached the end of the level, deaths allowed) and `flawless` (cleared without dying once, 22–54%):
  `{"n_reps": int, "unit": "level_repetition", "block_size": int, "beta": float, "n_bkps": int, "cleared_rate": float, "flawless_rate": float, "changepoints": [...], "phases": [...], "results": {"cleared": {"raw": {...}, "blocks": {...}}, "flawless": {...}}}`. All indices are in **rep-index space**, including the `changepoints`/`phases` re-derived from the Phase annotations, so detected and annotated boundaries share one axis.
  The `"repetitions"` block also carries `"phase_thirds"` — an Early/Mid/Late summary within each phase, mirroring the lab's figure. `"chronological"` holds true slabs of time and is the primary reporting view: `cleared`, `flawless`, and `level_adjusted` (each repetition minus that level's session-wide mean, so difficulty is removed while chronology is kept). `"within_level"` splits each level's repetitions into thirds and pools them — difficulty-controlled, but **not a time axis**: its bins are scattered across the whole session (chronological medians 110/113/116 of 230), so it measures improvement with repetition at a level rather than over time. **Reporting only** — all these bins are cut at the phase boundary and must never be fed to a changepoint detector.
  A sibling `"level_adjusted"` block records the definition, the per-phase means, and the per-level means behind that adjustment. **Only the practice thirds are comparable to each other.** Practice contains all 22 levels in every third, so level mix is constant and the rise is a real effect. Discovery's thirds each hold a *different, disjoint* set of levels (the subject walks the game in world order), so they differ in what was played rather than only when — and adjusting does not fix it: what looks like a within-discovery rise is a floor effect, since the hard late-discovery levels have no room to improve.
  `notebooks/figure_changepoints.ipynb` writes three figures here: `figure_changepoints.png` (per scene clip), `figure_changepoints_repetitions.png` (per level repetition, `cleared` and `flawless` panels side by side), and `figure_changepoints_phase_thirds.png` (Early/Mid/Late summary, both binnings side by side).

## `changepoints_by_category/`

One JSON file per subject produced by `invoke run-changepoints-by-category`:

- `<subject_id>_changepoints_by_category.json` — changepoints in each subject's attempt series **restricted to scenes sharing a design-pattern category** (from `source_data/scenes_mastersheet.csv`). For every category the subject played, the `ruptures` detector runs on **both** the `Duration` (time-to-complete) signal — continuous, expected to shrink with mastery — and the Bernoulli completion signal, each on raw + block-averaged series.
  Fields: `{"subject": str, "method": "ruptures_by_category", "model": "l2", "block_size": int, "beta": float, "n_bkps": int, "categories": {"<category>": {"n_attempts": int, "duration": {"raw": {...}, "blocks": {...}}, "completion": {"raw": {...}, "blocks": {...}}}, ...}}`.
  `notebooks/figure_changepoints_by_category.ipynb` writes one small-multiples figure per subject here (`figure_changepoints_by_category_<subject>.png`): duration vs. completion curves for the busiest categories, with per-algorithm changepoint lines.

## `clusters/`

One JSON file per scene produced by `invoke run-clustering`:

- `<scene_id>_clusters.json` — cluster label per behavioural trace for that scene.
  Fields: `{"scene": str, "method": str, "params": {...}, "labels": [<per-trace cluster
  label>], "metadata": [...], "stability": {...}}`
  `labels`, `metadata` and `stability.per_trace` are positionally aligned, in the order
  clustering ran.

  `params` records the settings that produced the partition — method, `min_cluster_size`,
  `min_samples`, `cluster_selection_epsilon`, and for the consensus method `bootstraps`,
  `frac`, `seed`, `scheme` and `consensus_cluster` (how the ensemble itself was turned
  into labels). On `consensus_cluster: "ward"` — the default — it also carries
  `consensus_input`, `consensus_linkage`, `consensus_k_max`, `consensus_tol`,
  `consensus_min_cluster_size` (the eligibility floor on a Ward cut — a different knob
  from `min_cluster_size`, which is the HDBSCAN density parameter for the replicate
  fits; both ship at their off values, `0.0` and `1`), and `consensus_k_rule` with its
  two shape parameters `consensus_elbow_frac` / `consensus_elbow_min_drop` (which
  reading of the silhouette-vs-k curve picked k, and what counts as a cliff in it). On
  `consensus_cluster: "hdbscan"` it carries `consensus_epsilon` instead (the merge
  threshold for that fit, in co-association distance rather than Fréchet units — see
  CLAUDE.md). The two sets are mutually exclusive on purpose, so a file says which
  aggregation made it rather than recording a setting that had no effect.
  Files written before this field existed carried no parameters at all,
  and the corpus was not made uniformly (an audit recovered 14 distinct settings across
  308 scenes), so `analysis.stability.recover_params` exists to reconstruct them by grid
  search against the stored labels.

  `stability` is the resampling report from the consensus method, `null` for the
  single-fit ones: `per_cluster` (mean bootstrap Jaccard per cluster), `per_trace`
  (0-1 consensus confidence per trace), `stable_share`, `k_replicates` (mean/sd of the
  cluster count across replicates), `noise_frac`, and `min_co_sampled` — the smallest
  number of replicates any pair was drawn into together, which is the diagnostic for
  whether `bootstraps` was large enough.

  `stability.consensus` records how the ensemble was cut. For Ward: the selected `k`,
  `selection` — `"elbow"` normally, meaning the cut taken was the last one before the
  silhouette curve drops off a cliff; `"elbow_no_drop"` when that curve had no cliff to
  find, so the rule fell back to the argmax; `"silhouette"` when `consensus_k_rule`
  asked for the plain argmax, or `"elbow_capped"` for the argmax at or below the cliff;
  `"fallback_k2"` when no cut cleared the minimum cluster size (only
  reachable when `consensus_min_cluster_size` is raised above its default of 1); `"no_ensemble_signal"` when no pair of traces ever co-clustered in any
  replicate, in which case every trace is labelled `-1` and `k` is 0, because cutting an
  informationless matrix would invent structure the resampling did not find — the chosen
  `silhouette` and `silhouette_frechet`, `rule` (which reading was in force),
  `elbow` — where the cliff is (`k`, the `drop` there, the `steepest` drop in the curve
  and the `threshold` a drop had to clear), recorded on **every** run whichever rule
  selected, exactly as `silhouette_frechet` is recorded without selecting —
  `k_candidates` — the whole silhouette-vs-k curve, so the cut can be second-guessed
  without re-running the ensemble, each candidate carrying `bounds`, its cut expressed
  as `k - 1` split positions in the leaf order below (which is what lets the dashboard
  draw a cut other than the one taken without re-deriving it) — and `leaves`, the dendrogram leaf order
  (`scipy.cluster.hierarchy.leaves_list`, i.e. `dendrogram(..., no_plot=True)["leaves"]`)
  in this file's own row order. The tree carries structure the labels discard, and the
  matrix only shows it when permuted into this order. Note `noise_frac` is 0 on any Ward run that produced
  clusters at all: Ward assigns every trace, so `-1` appears only in the
  `no_ensemble_signal` case above.
  A label of `-1` is HDBSCAN noise. Each `metadata` entry is the trace's
  `*_summary.json` record verbatim (21 fields: `Subject`, `Session`, `Run`, `Level`,
  `SceneID`, `ClipCode`, `StartFrame`, `EndFrame`, `Duration`, `Outcome`, `Phase`,
  `SourceBk2`, `GameName`, and the gameplay counters), so subject/phase/outcome are
  available without re-reading the source dataset.

  **No trace geometry is stored here** — the x/y polylines are rebuilt from
  `source_data/mario.scenes/` on demand by `load_traces()`. See `web/` below for the
  exported copy the dashboard reads.

## `clusters_by_category/`

One JSON file per scene category (design-pattern tag from `source_data/scenes_mastersheet.csv`,
e.g. `2-Horde`, `Gap stair valley`) produced by `invoke run-clustering-by-category`:

- `<category-slug>_clusters.json` — cluster label per behavioural trace, pooled across every
  scene tagged with that category. Traces are re-based to a common origin before clustering
  since raw traces from different scenes sit at different absolute level positions.
  Fields: as `clusters/` above but keyed `"category"` instead of `"scene"` (each
  `metadata` entry carries `SceneID` so pooled traces stay attributable to their scene).

## `stability/`

Resampling diagnostics written by `invoke run-cluster-audit`, `invoke run-cluster-sweep`,
`invoke select-cluster-params`, `invoke run-scheme-comparison` and
`invoke run-consensus-comparison`. All three read only `output_data/` — the stored labels
and the cached Fréchet matrices — and never open `source_data/mario.scenes/`.

- `audit.json` — one entry per already-clustered scene: the parameters behind it, plus
  `n`, `k`, `noise_frac`, `per_cluster` Jaccard and `stable_share` measured *at those
  parameters*. Scoring with any other settings depresses the Jaccard through a parameter
  mismatch rather than through real instability, which is why the parameters come first.

  `params_source` says how they were obtained. `"recorded"` means the file carried its own
  consensus `params` block and its own stability report, both used as-is. `"recovered"`
  means the file predates `params` and the settings were reconstructed by grid search
  against its labels (`recovery_ari`, or `null` when nothing in the grid reproduces them) —
  this path applies only to single-fit partitions, since a consensus partition has no
  single-fit equivalent to recover.

  **The file accumulates and never overwrites**: a scene already present is skipped, so an
  `audit.json` written before a re-clustering stays a record of the *previous* corpus. That
  is the point — it is what makes superseded results reproducible — but it means the file
  can describe labels that no longer exist on disk. Delete it (or use `clean-stability`) to
  re-audit what is there now.
- `<scene_id>_sweep.json` — one row per `(min_cluster_size, epsilon)` grid point:
  `k`, `noise_frac`, `stable_share`, `per_cluster` Jaccard, and `eps_quantile` (where that
  epsilon falls in this scene's own pairwise-distance distribution — the check on whether a
  single global epsilon means the same thing scene to scene). Scenes too small for the
  grid are written with `"skipped"` set and no rows.
- `scheme_comparison.json` — every scene clustered under both resampling schemes
  (subagging and Breiman bootstrap) at identical parameters, `B` and seed, so the only
  difference is how replicates are drawn. `rows` carries per-scene `k`, `stable_share`,
  `noise_frac`, replicate-`k` spread and the ARI between the two partitions; `summary`
  aggregates each scheme. Written by `invoke run-scheme-comparison`.
- `consensus_comparison.json` — every scene clustered under all three consensus
  aggregations (Ward on the co-association matrix as features, Ward on `1 - C` as a
  precomputed distance, and the older HDBSCAN fit on `1 - C`) from **one shared
  ensemble per scene**, so nothing but the last step differs. `rows` carries per-scene
  `k`, `stable_share`, `noise_frac`, both silhouettes and the pairwise ARI between the
  three partitions; `summary` aggregates each. Written by
  `invoke run-consensus-comparison`.
- `selected_params.json` — the corpus-wide choice. `rows` aggregates every sweep
  trace-weighted; `selected` is the winner under the rule "largest median k whose stable
  share clears `floor`", and `cleared_floor` says whether the floor was reached at all.

**`stable_share` is the headline number throughout**: the share of a scene's traces
sitting in a cluster whose mean bootstrap Jaccard exceeds 0.75. It is deliberately
per-trace rather than per-cluster, so a scene with forty unreproducible clusters and one
solid one is scored by how many traces ended up somewhere trustworthy, and noise counts
against it for free. It is **not** sufficient alone — collapsing a scene to k=2 scores
well — which is why the selection rule maximises k subject to a floor on it rather than
maximising it directly.

## `coassociation/`

The consensus matrices, written by `invoke run-coassociation`:

- `<scene_id>.npy` — symmetric `n x n` **uint16**, in the same row order as that scene's
  `clusters/<scene_id>_clusters.json` metadata. Divide by 65535 for the probability:
  `C[i,j]` = how often traces i and j landed in the same cluster, over the replicates in
  which *both* were drawn. The diagonal is how often a trace was clustered at all rather
  than called noise.

`cluster_traces` builds this matrix, reads the labels and the stability numbers off it,
and then drops it, so nothing downstream could display it. This task regenerates it from
the cached Fréchet matrix plus the `params` each cluster file records — the draws are
seeded, so it reproduces the matrix that produced the stored labels exactly (re-deriving
the labels from it gives ARI 1.000).

uint16 rather than uint8 because this is the analysis copy: at uint8 the rounding
perturbs enough borderline distances to shift the partition (ARI 0.998). ~26 MB for the
corpus, against ~106 MB for float64. The web export downsamples to uint8, which is a
display copy and is not exact.

## `practice/`

The practice effect, measured as distance to cluster medoids. Written by
`invoke run-practice-distances` and `invoke run-practice-models` from `clusters/` and
`frechet/` only.

- `attempts/<scene_id>.csv` — one row per attempt: `scene`, `subject`, `clip`
  (ClipCode), `phase`, `stage` (e/m/l tertile within phase), `outcome`, `cluster`,
  `attempt` (the subject's nth entry into this scene), `cluster_use` (nth use of this
  cluster by this subject), `switch` (1 if the cluster differs from the subject's
  previous attempt, empty on the first), `is_medoid`, and the distances:
  - `d_own` — Fréchet distance to the medoid of the attempt's own cluster.
  - `d_nearest`, `nearest_cluster` — distance to the nearest medoid among clusters of
    at least 3 members, and which cluster that was.
  - `d_nearest_norm` — `d_nearest` divided by that cluster's spread (median
    member-to-medoid distance); 1.0 = a typical member.
  - `*_loso` — the same with each subject measured against medoids chosen among the
    *other* subjects' members only (empty where a cluster holds only that subject).

  A medoid is the member with the smallest summed distance to the rest of its cluster.
  Medoid rows score 0 by definition and are excluded from every raw-medoid fit (kept
  in the `_loso` fits, where they are measured against other subjects' medoids). Scenes whose
  ensemble carried no signal (all `-1`) have no file.
- `curves.csv` — one row per series with at least 10 non-medoid attempts (`d_own`:
  subject × scene × cluster; `d_nearest_norm`: subject × scene), for each measure and its
  `_loso` twin, under each `transform` (`log`, zeros dropped, and `log1p`): AICc of
  linear, quadratic, power (`a + b·t^-c`) and exponential (`a + b·exp(-c(t-1))`) fits
  to the transformed distance vs attempt, the `best` model, its `delta_aicc` over the
  runner-up, the linear `slope`, and `change` (best model's prediction at the last
  attempt minus the first, in that transform's units; negative = fell).
- `mixed.json` — one fit per subject × measure × raw/`_loso` × transform × all/completed
  attempts (80): a MixedLM on the transformed distance fitted to that subject alone —
  `log(d)` (the primary; attempts at exactly `d = 0` dropped and counted in
  `n_zero_dropped`) and `log1p(d)` (the original analysis, kept for comparison) — with a random
  intercept and a random slope per scene (`d_nearest_norm`) or per scene × cluster
  (`d_own`), in four practice forms (`linear`, `quadratic`, `log`, `log_phase`). The
  random slope is on the form's own time term — `t = attempt / 10` for linear and
  quadratic (no random `t²`), `log(attempt)` for the log forms — recorded as
  `re_formula`. Each form records AIC, coefficients, CIs, p-values, `re` (random
  intercept and slope variances, their correlation, residual variance) and (except
  `log_phase`) `curve`, the fixed-effect prediction on an attempt grid (`y`, in the
  fit's `transform` units). `best_form` is the lowest-AIC form among the first three.
  Top level: `design` (`per_subject_v2`), `primary` (`d_nearest_norm_loso`),
  `primary_transform` (`log`), `transforms`, `scenes`, `subjects`; the task refits when
  the scenes, measures or design change.
- `lme4.json` — the **scene-scale practice model**, one of the two main models, written by `invoke run-practice-lme4`
  (`analysis/practice_lme4.R`, needs R with lme4): per subject × measure
  (`d_nearest_norm` and its `_loso` twin) × transform × all/completed attempts, the
  scene-scale model `y ~ logt + (1 + logt | level) + (1 + logt | level:scene)` (`model`
  `nested`) and its scene-only twin (`scene`, the statsmodels main model refitted as a
  cross-check), by maximum likelihood. Each fit records `intercept`, slope `b`, `se`, the
  Wald and profile-likelihood CIs, `ci` (profile, Wald where the profile failed —
  `ci_method`), `p_wald`, AIC, log-likelihood, `resid_var`, scene and (nested) level
  variances and correlations, `singular` (a variance estimated at zero), `warnings`, the
  level likelihood-ratio test (`lrt_chisq`, `lrt_p`, 3 df), `n_zero_dropped` and `curve`
  (the fixed-effect prediction on an attempt grid, transform units). Top level: `design`,
  `formula`, `primary`, `primary_transform`, `scenes`, `subjects`, `measures`, `transforms`.
- `lme4_route.json` — the **route-scale (within-route) practice model**, the other main model, also from
  `invoke run-practice-lme4` (`analysis/practice_lme4_route.R`): for `d_own` and `d_own_loso`
  — the Fréchet distance in px to the attempt's **own** route (cluster) prototype, so the
  origin never changes along a series — against the nth use of that route (`cluster_use`),
  `y ~ log_use + (1 + log_use | level) + (1 + log_use | level:scene) + (1 + log_use |
  level:scene:route)` per subject × transform × all/completed attempts. Same fields as
  `lme4.json` (Wald CIs: `ci_method` `wald`), plus `n_routes` and route-level variances
  (`route_var_*`); its `curve` is on the nth-use axis (`use`, mirrored in `attempt`).
- `lme4_effects/` — from `invoke run-practice-lme4-effects` (`analysis/practice_lme4_effects.R`),
  which refits every fit in `lme4.json` (nested model) and `lme4_route.json` and checks the
  refit reproduces the stored coefficients. Not tracked in git (~18 MB, regenerated).
  `partial.csv.gz`: one row per fitted attempt — `fid` (`model|measure|transform|completed
  (0/1)|subject`), `subject`, `scene`, `route` (-1 for the scene model), `x` (log attempt or
  log nth use), `y` (the transformed distance) and `partial`, the ggeffects partial residual
  (fixed prediction + conditional residual). `ranef.csv.gz`: every level's, scene's and
  route's random intercept `u` and slope `v` per fit (`kind`, `id` as lme4 names it:
  `w1l1`, `w1l1:w1l1s0`, `w1l1:w1l1s0:3`). `fixef.json`: the refit fixed effects per fit and
  the largest gaps to the stored fits and between ggeffects and lme4. `smoke/` holds a
  `--smoke` run (first subject's two primary fits), never replacing the full one.
- `nested.json` — alternative random-effect structures for the `log` practice form, fitted
  beside the main model by `analysis.practice.fit_structure`, for `d_nearest_norm` and
  its `_loso` twin × transform × all/completed × subject (80 fits). `structure` is
  `level_scene` (a correlated random intercept + slope per **level**, `w1l1`, with each
  scene's intercept and slope as independent variance components nested inside) or
  `scene_uncorr` (the scene effects alone, under the same no-correlation restriction, so
  the two compare like for like). Each fit records coefficients, CIs, p-values, AIC,
  `converged`, `re` (`level` and `scene` variances, `resid_var`), `curve`, `fit_note`
  (which optimiser and start won) and, for `level_scene`, `llf_gain_over_scene_uncorr`
  (should never be negative: the nested model contains the simpler one). Top level:
  `design` (`v2`), `scenes`, `structures`, `subjects`.
- `mixed_summary.csv` — `mixed.json` flattened for reading: one row per subject ×
  measure × transform × all/completed × form (320 rows), plus the 80 `nested.json` fits,
  plus the 80 `lme4.json` fits and the 40 `lme4_route.json` fits (`structure` `lme4_route`,
  `clock` `route_use`, `n_groups` = routes), told apart by `structure` (`lme4_nested` — the scene-scale model —
  `scene` for the statsmodels main model, `scene_uncorr`, `level_scene`, `lme4_scene`;
  nested rows add `n_levels`, `var_level_intercept`, `var_level_slope`, `corr_level`; lme4
  rows add `ci_method`, `singular` and `lrt_p_levels`), with `transform` and
  `n_zero_dropped` columns. `slope_p_bh` is `slope_p` adjusted with the Benjamini-Hochberg
  (FDR) procedure across the subjects of each fit setting (`structure` × `measure` ×
  `transform` × `completed_only` × `form`), so the route-scale rows carry their own
  correction, not the scene scale's. It replaced a Holm-adjusted `slope_p_holm` column. `primary` marks the `lme4_nested` rows when `lme4.json` exists,
  the statsmodels main model's otherwise. `primary` marks the five scene-scale main results
  (`d_nearest_norm_loso` on `log(d)`, all attempts, `log` form) and sorts them first; the
  route-scale main results are the `lme4_route` rows for `d_own_loso` on `log(d)`, all attempts.
  Slopes are in each row's transform units, so compare transforms through the ratio
  columns, never the slopes. `slope` is
  the form's practice coefficient (`t`, per 10 attempts, for linear/quadratic;
  `log(attempt)` for the log forms) with `slope_ci_low`/`_high`, `slope_p` and
  `ci_excludes_zero`; `t2` and `phase_step` are the quadratic and phase terms where
  present; `ratio_first`/`ratio_at_30` back-transform the fixed-effect line to the
  distance ratio at attempts 1 and 30 with the row's own inverse transform (discovery
  level for `log_phase`); then the
  random-effect variances, `corr`, `resid_var`, `aic`, `converged`, `is_best_form`.
  Written from `mixed.json` alone, so it never needs a refit.
- `switching.json` — switch rate per subject per bin of attempts (`binned`) and a
  logistic GLM `switch ~ C(subject) + log(attempt)` with errors clustered by
  subject × scene (`trend`).
- `figure_practice_*.png` — from `notebooks/figure_practice.ipynb`: the main analysis, the
  scene-scale lme4 model's slopes with BH-FDR adjusted p and its fitted curves (`lme4`); both lme4
  models' curves drawn over the attempts they were fitted on, pooled and as binned partial
  residuals (`lme4_data`); each subject's six most-played scenes with each scene's own fitted
  line (`lme4_spaghetti_scene`) and each route's (`lme4_spaghetti_route`); one example
  participant (`example`), per-subject mixed-model slopes and curves, raw vs
  leave-subject-out in the log form (`mixed`) and the same in the linear form
  (`mixed_linear`), linear against log per subject for the primary
  measure (`forms`), best-curve shares per series (`curves`), and switch rate over attempts
  (`switching`), and the practice slope and AIC under each random-effect structure
  (`structures`, both transforms). These are drawn on the primary `log(d)`; the `mixed`, `mixed_linear` and
  `forms` figures are drawn again on `log1p(d)` with a `_log1p` suffix.

## `route_usage/`

Route usage: does a player concentrate on fewer routes across the stages of play?
Written by `invoke run-route-usage` (`analysis/route_usage.py`, models in
`analysis/route_usage_lme4.R`) from `practice/attempts/` alone. A cell is one
subject × scene × stage; the stages are the halves of each phase by attempt order
(`early_discovery`, `late_discovery`, `early_practice`, `late_practice`; an odd count puts
the extra attempt in the early half). Routes are the scene's cluster labels.

- `cells.csv` — one row per cell with at least 2 attempts: `subject`, `scene`, `level`,
  `stage`, `n` (attempts), `routes_used`, `simpson` (the unbiased Simpson index,
  `sum n_i (n_i - 1) / (N (N - 1))`: the chance two different attempts share a route) and
  `shannon_rarefied` (expected Shannon entropy in bits of 4 attempts drawn without
  replacement, computed exactly; empty below 4 attempts).
- `models.json` — per subject × measure (`simpson`, `shannon_rarefied`) × weighting (`n`:
  weighted by attempts per cell, the primary; `none`: unweighted), the lme4 fit
  `y ~ stage + (1 | level) + (1 | level:scene)`: cell counts, singularity and warnings,
  `stage_means` (estimate, Wald SE and CI for the average scene; for `simpson` also
  `effective_routes` = `1 / D` and its CI), and three `contrasts`
  (`late_vs_early_discovery`, `late_vs_early_practice`, `practice_vs_discovery`), each
  with Wald CI and p, a chance level from route labels shuffled across stages within a
  scene (`null`: mean, sd, permutation `p_perm`) and a level bootstrap (`boot`: percentile
  CI and its width relative to the Wald CI). Top-level keys record the scenes, design and
  resampling settings the fit was made with; a failed fit is stored as `null` with the
  reason in `warnings`.
- `figure_route_usage_*.png` — from `notebooks/figure_route_usage.ipynb`: stage means per
  subject as effective routes and rarefied entropy (`stages`), and the three contrasts
  against the chance level with the bootstrap and unweighted fits beside them
  (`contrasts`).

## `frechet/`

Cached pairwise Fréchet distance matrices, one per clustering unit, written by
`invoke run-clustering` and `invoke run-clustering-by-category`:

- `<scene_id>.npy` / `<category-slug>.npy` — the symmetric `n x n` float distance
  matrix for that scene's (or category's) traces, in the deterministic load order of
  `load_traces()`. The cache is validated by trace count: a matrix whose size no longer
  matches the traces on disk is recomputed.

This is the expensive half of clustering (O(n^2) Fréchet, quadratic in pool size), and
it depends only on the traces — never on the method, `k` or epsilon. That is why
`clean-frechet` is a separate task from `clean-clustering`: deleting the labels to
re-cluster with a different epsilon reuses these, and only a full `invoke clean` or an
explicit `invoke clean-frechet` throws them away.

## Trace figures

`figure_clustering_silhouette.png` shows how the consensus cut is chosen: silhouette
against k for three representative scenes with the selected k marked, every scene's curve
overlaid, whether the chosen k is an interior peak or the largest candidate offered
(276 vs 24, the latter all scenes that hit the `k_max = 15` ceiling), and how many traces
are in the smallest cluster of each chosen cut — the check the silhouette itself cannot
perform, since it scores a singleton at 0 and so cannot penalise a cut for peeling one
off. Drawn from the `stability.consensus.k_candidates` block, so it needs no
recomputation.

`notebooks/figure_clustering.ipynb` also writes trajectory figures into `clusters/`
(`figure_clustering_<scene_id>.png`, `figure_clustering_category_<slug>.png`, etc.).
Traces use a screen-corrected vertical coordinate and are colored by cluster (default)
or by outcome. `figure_clustering_<scene_id>_panels.png` shows the same scene split into
one subplot per cluster (shared axes) so individual cluster shapes are easy to compare.
`figure_clustering_<scene_id>_by_stability.png` colours every trace by its consensus
confidence on a magma ramp (dark = its cluster-mates rarely agreed with it), drawn
low-confidence-last so the unreliable traces are the visible ones.
`figure_clustering_<scene_id>_panels_by_subject.png` is the same panel grid with every
trace colored by player instead of by cluster, and each panel captioned with its
per-subject composition (`sub-01: 45% (n=12)`) — this is how a cluster is checked for
being one player's idiosyncrasy rather than a shared strategy.

`notebooks/figure_cluster_stability.ipynb` writes `figure_stability_audit.png`,
`figure_stability_sweep.png`, `figure_stability_clusters.png`,
`figure_stability_scheme.png` (subagging vs bootstrap) and
`figure_stability_<scene_id>.png` (co-association matrix plus the spread of `k` across the
ensemble) into `clusters/` as well.

Each scene figure overlays a scene background image when one exists at
`source_data/mario.scenes/sourcedata/scene_backgrounds/<scene_id>.png`, with x offset by
the scene's mastersheet `Entry point`; otherwise it falls back to a plain y-flipped data
axis. These images are **not** part of the dataset and must be generated separately, but
310 of them are present in this working copy, so the overlay is active by default here.

## `web/`

The static payload and app behind `invoke dashboard` — an interactive version of the
figures above, with scene / subject / phase / stage / outcome filters. Written by
`invoke run-web-export`; roughly 50 MB in total.

- `output_data/web/data/index.json` — one row per exported scene (`id`, `world`, `level`, `scene`, `n`
  traces, `k` clusters, `noise` fraction, `sb` mean stability, `sil` — the silhouette of
  the cut that was taken — `sel` — how it was chosen — `subjects`, `phases`,
  `method`, pattern `tags`),
  plus the corpus-wide subject/phase/tag lists. Drives the scene picker without loading
  any scene payload.
- `output_data/web/data/practice.json` — the per-subject practice models for the dashboard's **Stats
  analysis** section: `output_data/practice/mixed.json` copied by
  `analysis.webexport.export_practice`, floats rounded to 6 decimals and the `scenes`
  list reduced to `n_scenes`, with `practice/lme4.json` as `lme4`, `practice/lme4_route.json` as
  `lme4_route`, `practice/nested.json` as
  `nested`, and `summary` — every CI and p-value for the primary measure (lme4 with
  BH-FDR adjusted p, scene-only model, statsmodels nested, and the scene and level bootstraps read
  from `docs/practice_*_bootstrap*.csv` when present), per transform and subject, `bh` —
  every BH-FDR adjusted slope p from `mixed_summary.csv`'s `slope_p_bh`, keyed
  `structure|measure|transform|completed (0/1)|form|subject`, so the browser never adjusts —
  `bootstraps` — per dashboard model (`lme4_nested`, `lme4_route`, `scene`), the measure and
  each CI-check bootstrap's per-subject rows copied from its summary CSV under `docs/` — and
  `diagnostics` — the assumption checks for the scene-scale lme4, route-scale lme4 and
  scene-only statsmodels models, per transform, each on its own main measure (`measure`): each check's question, what a violation affects, and the verdict,
  graded by `docs/practice_residual_diagnostics.py`'s own `verdicts()` from its
  `docs/practice_residual_diagnostics*.csv`, plus the per-subject numbers behind them.
  Nothing is refitted and no attempt rows are included.
- `output_data/web/data/diagnostics/` — the full diagnostic figures
  (`docs/practice_residual_diagnostics*.png`), copied by `export_practice` so the Stats
  section can show them; ~1 MB each, fetched only when the reader opens them.
  Rewritten by every `run-web-export`, `--backfill` included.
- `output_data/web/data/practice_points.json` — the data under each lme4 fit, for the Stats
  section's *Fit against the data* panel: `analysis.practice.fit_check_data()` per fit in
  `lme4.json` (`nested` model) and `lme4_route.json`, keyed
  `lme4_nested|lme4_route` `|measure|transform|completed (0/1)|subject`. Each holds `n_obs`
  (equal to the fit's), `bins` (geometric means with 95% CI, `n_obs`, `n_scenes`),
  `partial_bins` (the same over the partial residuals, when `lme4_effects/` exists), and
  `points` — a seeded sample of 800 `[x, d, partial d]` rows, for drawing only. Written by
  `export_fit_data` from `export_practice`, `--backfill` included; ~1.4 MB, fetched only when
  the panel is drawn.
- `output_data/web/data/practice_spaghetti.json` — the dashboard's *One line per scene* panel:
  for each lme4 model's primary setting (`lme4_nested`: `d_nearest_norm_loso`; `lme4_route`:
  `d_own_loso`; `log`, all attempts), per subject, `analysis.practice.spaghetti_data()` for
  their 12 most-played scenes — `points` (`[x, d, route]`, every attempt), `lines` (each
  scene's, or route's, own intercept `a` and slope `b` on the log scale, drawn to `x1`),
  `fixed`, and each scene's whole-scene route ranking `rank` for colour slots. Written with
  `practice_points.json`; only when `lme4_effects/` exists. ~0.2 MB.
- `data/scenes/<scene_id>.json` — the drawable scene: `method`, `entry` (the applied
  x-offset), `bg` + `bg_size`, `xlim`/`ylim` (precomputed from `_shared_xlim` and
  `playfield_ylim`), `step`, `cons` (below), and a `traces` array. Each trace carries its cluster label
  `c`, the filterable metadata (`s` subject, `ses`, `run`, `clip`, `sf`, `ph` phase,
  `out` outcome, `dur`), its consensus confidence `sb` (below), two derived time fields
  (`a`, `st` — below), tooltip counters (`xt`, `ht`, `ek`, `ll`), and the polyline as
  parallel `x`/`y` integer arrays. A
  `null` in `y` is a masked (dead) frame and must break the line rather than be
  interpolated across.
- `a` — **attempt index**: the nth entry this subject made into this scene, 1-based and
  chronological across the whole experiment. Each clip is one attempt, re-entries after
  a death included. Not the `rep-NNN` from `SourceBk2`, which restarts every session.
- `st` — **phase tertile**: `"e"`, `"m"` or `"l"`, cutting each `(subject, phase)` group
  at the 1/3 and 2/3 `ClipCode` quantiles. The full stage name is `st + "_" + ph`, so
  only the letter is stored. Both fields come from `analysis/stages.py` and can be added
  to an existing payload with `invoke run-web-export --backfill`.
- `cons` — the **k-selection record**, lifted verbatim out of the scene's cluster file
  by `analysis.webexport.selection_report`: the aggregation used, the chosen `k` and its
  `silhouette`, the `selection` route and the `rule` that produced it, the `elbow`
  block (where the curve falls off a cliff, recorded whether or not it selected), the
  whole `k_candidates` curve (one row per
  candidate cut: `k`, `silhouette`, `silhouette_frechet`, `min_size`, `bounds`), plus the ensemble
  numbers the curve should be read against (`stable_share`, `k_replicates`,
  `bootstraps`, `frac`, `scheme`), and `leaves` — the dendrogram leaf order,
  re-expressed in **payload** indices so the browser can apply it directly to the
  matrix it was served. It is what the dashboard's **k selection** view draws, and
  `bounds` + `leaves` together are what its **Clusters (k)** control decodes to draw a
  candidate cut other than the one that was taken — a few kilobytes a scene, and `null`
  for a single-fit run, which has no ensemble and therefore no k to choose. Not
  derivable from the payload, but derivable from `output_data/` alone, so
  `invoke run-web-export --backfill` refreshes it without re-reading `mario.scenes`
  (`bounds` for files predating the field are rebuilt from the stored co-association
  matrix, which reproduces the same tree exactly).
- `sb` — **consensus confidence**, 0-1: how often this trace's cluster-mates agreed with
  it across the bootstrap replicates. Present only for scenes clustered with the
  consensus method. Unlike `a` and `st` it is **not** derivable from what the payload
  already carries, so adding it is a real re-export — `--backfill` cannot supply it.
- `data/coassoc/<scene_id>.bin` — the scene's co-association matrix as raw `n x n`
  **uint8**, row-major, in the **payload's** trace order (permuted from the analysis copy
  by clip key, never by position). One byte per cell; the browser fetches it only when the
  Co-association view is opened. `coassoc: true` in the scene payload says one exists.
  Display precision only — for numbers use `output_data/coassociation/`.
- `data/backgrounds/<scene_id>.png` — copies of the scene backgrounds, so the served
  tree never reaches back into the 23 GB `mario.scenes` dataset at runtime.
- `index.html`, `app.js`, `style.css` — copied from `web/` by `invoke dashboard`.

Two caveats worth carrying:

- **Coordinates are downsampled.** `step` (default 3) means only every 3rd frame was
  kept, plus the first and last, and coordinates are rounded to integers. Fine for
  drawing at 224 px tall; **do not** use this payload for quantitative analysis — go
  back to `load_traces()` for that.
- **Cluster labels are the pooled ones.** The ensemble ran once over every trace in the
  scene. The dashboard's filters only change which traces are drawn; they never
  re-cluster, and cluster colors stay fixed to the whole-scene rank-by-size order so a
  filter cannot repaint the traces that survive it. The same holds for `sb`: it is the
  whole-scene consensus confidence and does not recompute under a filter.

## `PROVENANCE.json`

Written by `invoke run` (see `airoh.provenance`). Records what produced the current
outputs: the project's own git commit and dirty flag, the environment, a checksum of
`source_data/MANIFEST.json` plus an inlined copy of each input asset's commit and
checksum, and the size and checksum of every output file.

Git-tracked, and it **changes on every run** — that churn is the record working, not
noise to suppress. It attests, it does not retrieve: it can tell you a figure came from
commit `abc123` of an input dataset with a dirty working tree, which is what "why do my
numbers differ from the paper's" actually needs, but only datalad can bring that state
back.

> Note: apart from `changepoints/`, `changepoints_by_category/` and this record, output files are ignored by Git. Run `invoke run` to regenerate them.
