# Partial residuals and per-unit random effects of the two main lme4 practice models,
# for drawing the fits against the data (analysis.practice.run_practice_lme4_effects).
#
# Refits, per fit, exactly the model analysis/practice_lme4.R ("nested") or
# analysis/practice_lme4_route.R ("route") fitted — same frames, formula, ML, optimiser —
# without the CIs, and writes:
#   * partial residuals from ggeffects (residualize_over_grid on the fixed-effect
#     prediction): the fixed-effect prediction at each attempt plus that attempt's
#     conditional residual, i.e. the outcome with its level's, scene's (and route's)
#     own intercept and slope taken out. Checked here against lme4's own
#     predict(re.form = NA) + resid(); the fit stops if they disagree by more than
#     GG_TOLERANCE. They are not bit-identical: ggeffects predicts on its own grid of x
#     values, and gaps up to ~2.5e-4 (sub-03's route fits) were seen — 0.025% of a
#     distance on the log scale, invisible in any plot. The largest gap is recorded.
#   * every level's, scene's (and route's) random intercept u and slope v (BLUPs), so a
#     unit's own line is (b0 + sum u) + (b1 + sum v) * log(x).
#   * the fixed intercept and slope, for the caller to check against the stored fit.
# Rows are matched by the frames' `rid`.
#
#   Rscript analysis/practice_lme4_effects.R <frames.csv> <partial.csv> <ranef.csv> <fixef.csv> [n_cores]

suppressPackageStartupMessages({
  library(lme4)
  library(ggeffects)
  library(parallel)
})

args <- commandArgs(trailingOnly = TRUE)
frames <- read.csv(args[1], colClasses = c(subject = "character", fid = "character",
                                           scene = "character", level = "character"))
n_cores <- if (length(args) >= 5) as.integer(args[5]) else max(1, detectCores() - 1)

FORMULAS <- list(
  nested = y ~ x + (1 + x | level) + (1 + x | level:scene),
  route = y ~ x + (1 + x | level) + (1 + x | level:scene) + (1 + x | level:scene:route))
MAXFUN <- list(nested = 2e5, route = 3e5)
GG_TOLERANCE <- 1e-3

effects <- function(re, grp, kind, fid) {
  e <- re[[grp]]
  if (is.null(e)) return(NULL)
  data.frame(fid = fid, kind = kind, id = rownames(e), u = e[, "(Intercept)"], v = e[, "x"])
}

one <- function(g) {
  model <- g$model[1]
  fid <- g$fid[1]
  m <- suppressMessages(suppressWarnings(
    lmer(FORMULAS[[model]], data = g, REML = FALSE,
         control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = MAXFUN[[model]])))))
  pr <- predict_response(m, terms = "x [all]", type = "fixed")
  partial <- residualize_over_grid(pr, m)$predicted
  manual <- predict(m, re.form = NA) + resid(m)
  gap <- max(abs(partial - manual))
  if (length(partial) != nrow(g) || gap > GG_TOLERANCE)
    stop(sprintf("%s: ggeffects partial residuals disagree with lme4 (max |diff| %g)", fid, gap))
  re <- ranef(m)
  fe <- fixef(m)
  list(
    partial = data.frame(fid = fid, rid = g$rid, partial = partial),
    ranef = rbind(effects(re, "level", "level", fid), effects(re, "level:scene", "scene", fid),
                  effects(re, "level:scene:route", "route", fid)),
    fixef = data.frame(fid = fid, intercept = fe[["(Intercept)"]], b = fe[["x"]],
                       n_obs = nrow(g), max_gap = gap, singular = isSingular(m)))
}

res <- mclapply(split(frames, frames$fid), function(g) try(one(g)), mc.cores = n_cores)
bad <- vapply(res, inherits, logical(1), "try-error")
if (any(bad)) stop("effects fit failed for: ", paste(names(res)[bad], collapse = ", "),
                   "\n", paste(unique(unlist(res[bad])), collapse = "\n"))
write.csv(do.call(rbind, lapply(res, `[[`, "partial")), args[2], row.names = FALSE)
write.csv(do.call(rbind, lapply(res, `[[`, "ranef")), args[3], row.names = FALSE)
write.csv(do.call(rbind, lapply(res, `[[`, "fixef")), args[4], row.names = FALSE)
