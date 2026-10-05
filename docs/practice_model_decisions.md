# Practice effect: model decisions

A record of how the practice-effect model was chosen, what was checked, and what is still
open. Every number here comes from files in this repository, listed under each section, so
it can be checked against them.

## Question

Do players' attempts at a scene get closer to the scene's typical routes with practice?

## Outcome measure

`d_nearest_norm_loso`, one value per attempt:

- the Fréchet distance from the attempt's trace to the **nearest cluster medoid** (the most
  central real trace of a cluster), among clusters of at least 3 traces;
- divided by that cluster's **spread** (median member-to-medoid distance), so `1` means
  "as far out as a typical member" and `0` means "identical to the medoid";
- **leave-subject-out (LOSO)**: the medoid and the spread are chosen from the *other*
  subjects' traces only, so no player is measured against a reference their own attempts
  helped define. (The clusters themselves were still formed from everyone's traces.)

The normalisation makes scenes and clusters comparable; LOSO removes a circularity that
was visible in the data (sub-01 shows a clear decline against all-subject medoids and none
against LOSO medoids).

Time is the **attempt number at a scene** (the nth time this player entered this scene),
not time in the study, so the effect is "more practice on the same scene".

## Model

One **linear mixed model per subject**, fitted by maximum likelihood (statsmodels MixedLM):

```
transform(d) = (a + u_scene) + (b + v_scene) · log(attempt) + ε
```

- `a`, `b` — the subject's fixed intercept and practice slope (the typical scene);
- `u_scene`, `v_scene` — random intercept and random slope per scene;
- `ε` — residual noise.

**Why per subject, not pooled.** Five subjects are too few to estimate a between-subject
variance, and the earlier pooled model (subject as a fixed intercept, one shared slope)
forced all five to learn at the same rate. The claim is about these five players,
replicated five times.

**Why a random slope, not only a random intercept.** Scenes differ in learning rate more
than the average rate itself (SD of scene slopes 0.07-0.11 against average slopes
0.007-0.043, under `log1p`). Dropping the random slope worsens AIC by 79-454 per subject
(likelihood-ratio p < 1e-18) and narrows the CI on `b` by 30-48% — overconfident, because it
treats ~300 scenes as agreeing on one slope. Conclusions were unchanged, but the random
slope gives the honest uncertainty. Checked on `log1p`, before the transform switch.

**Practice form.** `log(attempt)` is reported as primary for every subject so slopes are
comparable. It is not the AIC-best form for every subject: linear, quadratic and a
phase-step form are fitted too and stored, and the linear-against-log comparison is drawn
per subject (`figure_practice_forms*.png`). The conclusions do not depend on the form.

## The transform: `log(d)` replaces `log1p(d)`

The model was first fitted on `log1p(d) = log(1 + d)`. Residual diagnostics showed:

| check | `log1p(d)` | `log(d)` |
|---|---|---|
| residual skewness | 1.7 to 2.6 | 0.4 to 1.0 |
| residual excess kurtosis | 10 to 17 | 4 to 5 |
| residual SD, lowest to highest prediction | x2.6 to x3.3 | x1.7 to x2.0 |
| largest standardised scene effect | 6 to 11 | 5 to 9 |
| trend shape (largest bin bias) | <= 0.03 | <= 0.04 |
| lag-1 residual autocorrelation | <= 0.06 | <= 0.04 |

`log1p` compresses the 0-1 range where most ratios sit against a long upper tail, which
produced the skew. `log` treats "half" and "double" symmetrically. It also makes a slope a
proportional change in the ratio itself rather than in `1 + d`.

**Cost:** `log(0)` is undefined, so attempts with `d = 0` exactly (a trace identical to a
medoid) are dropped: 44-79 per subject, about 0.6%.

**Both are kept.** Every fit is stored under both transforms, labelled `transform`, so the
original analysis is reproducible. The `log1p` results were refitted after the change and
match the earlier run exactly.

## Results (`d_nearest_norm_loso`, `log` form, all attempts)

| subject | `log(d)`: b [95% CI] | ratio at attempt 1 → 30 | `log1p(d)`: b [95% CI] |
|---|---|---|---|
| 01 | −0.011 [−0.031, +0.010] | 1.47 → 1.42 | −0.007 [−0.022, +0.008] |
| 02 | −0.028 [−0.047, −0.008] | 1.24 → 1.13 | −0.017 [−0.029, −0.004] |
| 03 | −0.037 [−0.060, −0.013] | 1.35 → 1.19 | −0.030 [−0.046, −0.013] |
| 05 | −0.064 [−0.084, −0.045] | 1.43 → 1.15 | −0.043 [−0.056, −0.030] |
| 06 | −0.060 [−0.080, −0.040] | 1.41 → 1.15 | −0.040 [−0.054, −0.027] |

**Four of five subjects show a clear practice effect; sub-01 does not, under either
transform.** Slopes are in each transform's units and cannot be compared across columns;
the ratios can. With completed attempts only, the count is 4 of 5 under `log` and 3 of 5
under `log1p`.

**The effect is clear but small.** Under `log1p`, practice explains at most ~1% of the
variance in a subject's attempts (marginal R²), scenes 15-37%, and attempt-to-attempt
noise the rest. Statistically clear and large are different claims.

## What each diagnostic protects

A failed assumption does not always mean a wrong result. Some assumptions protect the
**slope estimate `b` itself**; others only the **confidence interval** (CI) around it,
where `b` stays right on average but the model misjudges how certain it is. The CI is
computed by a formula that assumes normal, equal-variance, independent noise; when those
fail, the formula's CI can be too narrow or too wide.

| row in the diagnostics figure | assumption | if violated, affects | `log(d)` |
|---|---|---|---|
| 0 classic residual plot | centred and equally wide | both (see rows 1, 5, 7) | mixed |
| 1 shape of the trend | the curve form is right | **slope** — `b` itself wrong; no CI fix helps | OK |
| 2 independence | attempts independent given scene | CI (mainly) — too narrow | OK |
| 3-4 normality | residuals bell-shaped | CI (mainly); heavy tails can pull `b` | violated (moderate) |
| 5 constant spread | same noise at every prediction | CI; `b` unbiased but less precise | mild |
| 6 random-effect shape | `u`, `v` bell-shaped | scene variances; extreme scenes can pull `b` | concern |
| 7 equal variance across scenes | same noise in every scene (a residual check, grouped by scene) | CI | violated |
| 8 equal random-effect variance | one `var(u)`, one `var(v)` across worlds | scene variances and CI | OK |
| 9 bootstrap (scene and level) | — not an assumption: whether the CI problems above matter | CI | mild — calls unchanged, level CI up to 1.34x |

The rows that protect the slope pass. The failures sit in the rows that affect the CI,
and row 9 — the scene and level bootstraps below — shows they do not change the conclusions.

## Diagnostics

After the switch, under `log(d)`:

- **Normality** — much improved but a moderate right tail remains. sub-05 sits just past
  the rule-of-thumb limits (skew 1.03, kurtosis 5.4), so the computed verdict reads
  "violated". The thresholds were not adjusted to change that.
- **Constant spread** — mild: residual SD rises up to x2.0 at the highest predictions.
- **Extreme scenes** — a few scenes 5-9 SD out in their random effects.
- **Unequal noise between scenes** — violated. The model assumes one residual variance
  for every scene. 58-64% of scenes have a residual SD outside the range expected under
  equal variance, against 31-34% when the same residuals are shuffled across scenes
  (95th percentile 35-39%). Shuffling keeps the residuals' real heavy-tailed shape, so
  this is not just the tails: **some scenes are genuinely noisier than others.** Under
  `log1p` it was 67-74% against 45-51%.
- **Equal random-effect variance across worlds** — passes. The spread of the scene
  effects `u` and `v` differs between worlds by x1.8 to x2.8, but every subject is within
  what shuffling world labels produces (95th percentile x2.2 to x3.4).

The trend shape, independence and world checks pass, so `b` itself is trustworthy; the
others — above all the unequal noise between scenes — put the **confidence intervals**
at risk. A model giving each scene its own residual variance would address it directly,
but statsmodels cannot fit one (R's `glmmTMB` can). The bootstrap below checks whether
the risk materialises.

## Scene bootstrap: are the confidence intervals right?

**This check is about the CI only.** The model's CI comes from a formula that relies on
the assumptions above; the bootstrap measures the uncertainty directly instead. For each
subject, it draws that subject's scenes **with replacement** (as many as they played —
some scenes twice, some not at all, every attempt of a drawn scene kept), refits the
main model, and repeats 200 times. The spread of the 200 slopes is the uncertainty, and
its 2.5th-97.5th percentile is the bootstrap CI. Resampling whole scenes keeps each
scene's own noise level and shape, so it does not assume normal or equal-variance noise.

| subject | b (`log(d)`) | model 95% CI | bootstrap 95% CI | width, bootstrap / model | clear? model / bootstrap |
|---|---|---|---|---|---|
| 01 | −0.011 | [−0.031, +0.010] | [−0.033, +0.009] | 0.99 | no / no |
| 02 | −0.028 | [−0.047, −0.008] | [−0.044, −0.009] | 0.91 | yes / yes |
| 03 | −0.037 | [−0.060, −0.013] | [−0.061, −0.009] | 1.08 | yes / yes |
| 05 | −0.064 | [−0.084, −0.045] | [−0.082, −0.047] | 0.87 | yes / yes |
| 06 | −0.060 | [−0.080, −0.040] | [−0.079, −0.042] | 0.95 | yes / yes |

`docs/practice_scene_bootstrap.png` shows this: the two intervals side by side per subject, and each subject's 200 slopes with the cut points that make the bootstrap CI.

**The model's intervals hold.** The bootstrap intervals are 0.87-1.08 times as wide as
the model's — no systematic widening — and every subject's clear/not-clear call is the
same. Under `log1p` the same holds (0.88-1.11, same calls). So the assumption problems
found above are real but do not change the uncertainty in practice; the model's CIs can
be reported, with the bootstrap as the check. For sub-01, 82% of bootstrap slopes are
below zero: a lean towards improvement, but not a clear one.

With 200 replicates the bootstrap percentiles themselves carry Monte Carlo error of a few
percent of the width, so ratios between ~0.9 and ~1.1 should be read as "the same".

## Level bootstrap: are scenes of the same level independent?

Both the model and the scene bootstrap assume **scenes are independent of each other**.
The mixed model handles the dependence it is told about (attempts within a scene, through
the scene random effects), but nothing tells it that scenes belong to levels, and the
scenes of one level are played back to back in the same run. If they move together, the
~300 scenes are fewer independent pieces of evidence than they look, and both CIs are too
narrow.

The **level bootstrap** resamples whole **levels** (22 per subject) instead of scenes —
every scene of a drawn level comes along, together — and refits the *same* model. It
assumes only that levels are independent. If scenes within a level are correlated, its
CI comes out wider than the scene bootstrap's; if not, the two match. It cannot come out
meaningfully narrower.

| subject | model 95% CI | scene bootstrap | level bootstrap | level width / model | clear? |
|---|---|---|---|---|---|
| 01 | [−0.031, +0.010] | [−0.033, +0.009] | [−0.028, +0.011] | 0.95 | no (84% of slopes < 0) |
| 02 | [−0.047, −0.008] | [−0.044, −0.009] | [−0.044, −0.010] | 0.86 | yes |
| 03 | [−0.060, −0.013] | [−0.061, −0.009] | [−0.071, −0.008] | **1.34** | yes |
| 05 | [−0.084, −0.045] | [−0.082, −0.047] | [−0.087, −0.044] | 1.10 | yes |
| 06 | [−0.080, −0.040] | [−0.080, −0.042] | [−0.083, −0.038] | 1.12 | yes |

(`log(d)`; under `log1p` the level widths are 0.74-1.56 of the model's, sub-03 again the
widest, with the same calls.)

**Every clear/not-clear call survives.** The level CIs are somewhat wider for three
subjects — up to 1.34x for sub-03 — so there is **some** correlation between scenes of a
level, most visible for sub-03. With only 22 levels the level percentiles are noisier
than the scene ones (sub-02 coming out *narrower*, at 0.86, is that noise), so the
size of the widening is approximate. It does not reach the conclusions: even the widest
interval stays clear of zero.

**Reporting.** The model's CIs with both bootstraps as checks is defensible; the more
conservative choice is to report the **level-bootstrap CIs** as the primary intervals,
since they make the weakest independence assumption. The model-based alternative —
level as a further random effect, scenes nested in levels — has not been fitted; it
would show directly how much learning rates differ between levels (`var(v_level)`).

Not modelled, by choice:

- **Outcome (completed/death)** improves fit (AIC −55 to −251) but barely changes the
  residual shape, and it partly *is* the learning (dying less). Adding it changes the
  question from "total improvement" to "improvement in execution beyond dying less".
  Undecided; checked on `log1p`.
- **Phase** adds nothing over the practice term (ΔAIC ≈ 0 for 4 of 5 subjects).

## Nested model: scenes within levels

The level bootstrap corrects the CI for scenes of a level being related; the **nested
model** builds that relation into the model itself:

```
transform(d) = (a + u_level + u_scene) + (b + v_level + v_scene) · log(attempt) + ε
```

Every scene still has its own intercept and slope (`u_scene`, `v_scene`), now nested in
its level, which gets one too (`u_level`, `v_level`). Level only would have forced all
scenes of a level onto one line, though scenes differ a lot within a level. The main
model is kept; the nested one is fitted beside it (`output_data/practice/nested.json`).

**A limitation of the software.** statsmodels can nest the scene effects only as
independent terms, so the nested model cannot have the correlation between a scene's
intercept and slope that the main model estimates (−0.6 to −0.9). To compare like for
like, a scene-only model with the same restriction (`scene_uncorr`) is fitted too. The
model with both levels and that correlation needs R's `lme4` and has not been fitted.

| subject (`log(d)`) | main b | nested b [95% CI] | nested CI width / main | level bootstrap width / main | levels vs `scene_uncorr`, ΔAIC | clear? |
|---|---|---|---|---|---|---|
| 01 | −0.011 | −0.011 [−0.032, +0.011] | 1.03 | 0.95 | −13 | no |
| 02 | −0.028 | −0.026 [−0.042, −0.010] | 0.83 | 0.86 | +6 | yes |
| 03 | −0.037 | −0.035 [−0.068, −0.002] | 1.37 | 1.34 | −30 | yes |
| 05 | −0.064 | −0.067 [−0.091, −0.042] | 1.24 | 1.10 | −17 | yes |
| 06 | −0.060 | −0.061 [−0.085, −0.038] | 1.16 | 1.12 | +1 | yes |

- **The two approaches agree.** The nested model's CIs widen by about as much as the
  level bootstrap's, per subject — the model-based and the resampling answer to the same
  question match.
- **Levels matter for some subjects.** Like for like, adding levels improves AIC for
  sub-01, 03 and 05 and not for 02 or 06. For sub-03 the level slope variance (0.0043)
  is as large as the scene slope variance (0.0039): learning rate differs as much
  between levels as between scenes within one.
- **The main model still fits best overall** (AIC lower by 22-85 than the nested model):
  the scene intercept-slope correlation it carries is worth more than the level terms.
  So the nested model is not a replacement for it here, only a check.
- **Conclusions unchanged**: 4 of 5 subjects improve clearly, sub-01 does not, under
  every structure and both transforms.
- **Fitting note.** Fitted from a cold start, 9 of 40 nested fits stopped in a worse
  solution than the simpler model they contain (impossible at the true optimum), so each
  is now warm-started from the simpler model's solution and the best of several
  optimisers kept; every nested fit now reaches at least the simpler model's likelihood.
  Two non-primary fits (raw `d_nearest_norm`, `log1p`, sub-03) report non-convergence;
  their estimates are in line with the others and are flagged in the files.

**Reporting.** The two corrections for levels — level bootstrap and nested model — agree,
and neither changes a conclusion. The level-aware intervals (either one) are the
conservative choice for the CIs; the main model remains the best-fitting description.
`output_data/practice/figure_practice_structures.png` shows the slopes and AIC side by
side.

## Full nested model in lme4: levels and the scene correlation together

The statsmodels nested model above had to drop the scene intercept-slope correlation.
R's `lme4` fits the model with both — `y ~ log(attempt) + (1 + log(attempt) | level) +
(1 + log(attempt) | level:scene)` — by maximum likelihood, like every other fit here
(`docs/practice_lme4.R`, run through `docs/practice_lme4.py`; needs R with lme4).

**Cross-check first.** lme4 also refits the main scene-only model; it matches the
statsmodels fit to within 1.4e-6 on the slope and 0.0002 on AIC, for every subject,
transform and attempt set. The two programs agree, so the new model can be trusted on
the same footing.

| subject (`log(d)`) | main b | full nested b | nested 95% CI (profile) | CI width vs main | levels add anything? (LRT p) | clear? |
|---|---|---|---|---|---|---|
| 01 | −0.011 | −0.011 | [−0.033, +0.011] | 1.02 | yes (0.0002) | no |
| 02 | −0.028 | −0.028 | [−0.047, −0.008] | 1.00 | no (1.0; level variance 0) | yes |
| 03 | −0.037 | −0.037 | [−0.070, −0.003] | 1.35 | yes (0.04) | yes |
| 05 | −0.064 | −0.066 | [−0.091, −0.043] | 1.14 | borderline (0.055) | yes |
| 06 | −0.060 | −0.062 | [−0.086, −0.039] | 1.11 | no (0.44) | yes |

- **Conclusions unchanged**, under both transforms and with completed attempts only.
- **The widening matches** the level bootstrap and the statsmodels nested model (sub-03:
  1.35 here, 1.34 level bootstrap, 1.37 statsmodels nested) — three methods agree.
- **This resolves the earlier AIC puzzle.** The statsmodels nested model fit worse than
  the main model only because it lacked the scene correlation; with it, the full nested
  model is as good or better (AIC lower for sub-01, 03, 05; within ~6 for 02, 06).
- **Singular fits** (sub-02 throughout, and some `log1p` / completed-only fits): lme4
  estimates the level variances at zero, i.e. levels do not differ for that subject.
  That is an answer, not a failure — for sub-02 the nested model reduces exactly to the
  main one. The profile CI could not be computed for three of these fits; the Wald CI is
  used there (the summary plot does the same).
- Profile-likelihood CIs are reported because they do not rely on the slope's estimate
  being normally distributed; they agree with the Wald CIs to within ~0.002.

**Assumption checks for this model.** The same nine checks were run on the full nested
model (`uv run python docs/practice_residual_diagnostics.py lme4`, figures
`docs/practice_residual_diagnostics_lme4_log.png` / `_log1p.png`):

| check | main model (`log(d)`) | full nested lme4 (`log(d)`) |
|---|---|---|
| trend shape (largest bin bias) | 0.035 | 0.035 |
| independence (largest lag-1 r) | 0.04 | 0.04 |
| residual skew / kurtosis | 0.44-1.03 / 3.9-5.4 | 0.44-1.03 / 3.9-5.4 |
| spread, lowest to highest prediction | up to x1.97 | up to x1.98 |
| most extreme scene effect | 9.4 SD | 9.4 SD |
| scenes outside equal-variance funnel (shuffled p95) | 58-64% (35-39%) | 58-64% (35-39%) |
| random-effect spread across worlds | within chance | within chance |
| **level** effects bell-shaped (row 6) | — | yes, on the line |
| row 9: level bootstrap width / this model's CI | — | **0.86-0.96**, every call unchanged |

- **The residual picture is the same as the main model's**, to two or three decimals:
  adding levels re-divides the scene differences between level and scene effects but
  leaves each attempt's residual almost unchanged. So everything said above about the
  main model's residuals — slope-protecting checks pass, CI-related ones partly fail —
  holds for this model too.
- **The level effects themselves look normal**, the one assumption this model adds.
- **Row 9 compares this model's own CI with the level bootstrap** (the bootstrap that,
  like this model, does not assume a level's scenes are independent). The bootstrap
  intervals are slightly *narrower* (0.86-0.96x), so the nested model's CI is, if
  anything, a little conservative — and no call changes. A bootstrap of the nested
  model itself was not run (it would take hours of refits); the level bootstrap of the
  scene-only model is the closest available check.

**For the model choice** this is option C from the discussion: design-faithful (scenes
within levels) and keeping the correlation the data need. It changes no conclusion and
gives level-aware CIs from the model itself, so it is the natural primary model if R is
acceptable in the analysis; the statsmodels main model with a level-bootstrap CI is the
Python-only equivalent.

## Where everything is

**Start with `docs/practice_summary.png`**: every confidence interval for the main result in
one figure — the original `log1p` model, the full nested lme4 model and the bootstraps,
then the same for `log(d)` plus the statsmodels nested model — with each interval's
clear/not-clear call marked.
Regenerate with `uv run python docs/practice_summary_plot.py` (seconds; no refitting).

| what | file |
|---|---|
| summary of every CI | `docs/practice_summary.png`, script `docs/practice_summary_plot.py` |
| full nested model (lme4) | `docs/practice_lme4.csv`; scripts `docs/practice_lme4.py` (entry point) and `docs/practice_lme4.R` |
| its assumption checks | `docs/practice_residual_diagnostics_lme4_log.png` / `_lme4_log1p.png` (+ `.csv`); residuals from `docs/practice_lme4_resid.R` |
| model code | `analysis/practice.py` (`fit_mixed`, `TRANSFORMS`, `PRIMARY_TRANSFORM`) |
| all fits, both transforms | `output_data/practice/mixed.json` |
| nested (level + scene) and like-for-like fits | `output_data/practice/nested.json`; figure `output_data/practice/figure_practice_structures.png` |
| flat results table | `output_data/practice/mixed_summary.csv` (`primary` marks the main rows; `structure` tells the models apart) |
| main figures | `output_data/practice/figure_practice_mixed.png`, `figure_practice_mixed_linear.png`, `figure_practice_forms.png` |
| same figures, original `log1p` | `output_data/practice/figure_practice_*_log1p.png` |
| residual diagnostics (9 checks: trend, independence, normality x2, spread, random-effect normality, scene variance, world variance) | `docs/practice_residual_diagnostics_log.png`, `docs/practice_residual_diagnostics_log1p.png` (numbers in the matching `.csv`) |
| diagnostics script | `docs/practice_residual_diagnostics.py` |
| bootstraps (CI check) | `docs/practice_scene_bootstrap.png` (figure, scene and level), `docs/practice_level_bootstrap.csv` and `_draws.csv` (level), `docs/practice_scene_bootstrap.csv` (summary), `docs/practice_scene_bootstrap_draws.csv` (all 2,000 slopes), script `docs/practice_scene_bootstrap.py` |
| interactive view | dashboard → Stats analysis (`invoke dashboard`, then `#stats`); the transform switch flips between the two |

Reproduce: `uv run invoke run-practice-models --force`, then the notebook
(`invoke run-notebooks`), `uv run python docs/practice_residual_diagnostics.py` (~2 min)
`uv run python docs/practice_scene_bootstrap.py` and `... level` (~11 min each on 12
cores; seeded), then the diagnostics again so row 9 reads the new bootstraps.
