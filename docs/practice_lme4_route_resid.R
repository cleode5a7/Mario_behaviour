# Residuals, random effects and simplified refits of the route-scale practice model.
#
# Called by docs/practice_residual_diagnostics.py (``route`` mode) with the model's exact
# rows (analysis.practice.route_frame, plus a row id). Per subject, refits
#   full : y ~ log_use + (1 + log_use | level) + (1 + log_use | level:scene)
#                      + (1 + log_use | level:scene:route)            (the route-scale model)
# and writes, in the frames' row order, fitted values and conditional residuals (y minus
# the fixed AND every random part), plus each level's, scene's and route's random
# intercept and slope.
#
# The singular-fit check: most full fits are singular, with level and scene slope
# variances at zero or correlations at +-1. Two simpler models drop exactly those terms:
#   no_upper_slopes : (1 | level) + (1 | level:scene) + (1 + log_use | level:scene:route)
#   uncorrelated    : (1 + log_use || level) + (1 + log_use || level:scene)
#                     + (1 + log_use || level:scene:route)
# If the slope b and its SE barely move, the singularity did not shape the result.
#
#   Rscript docs/practice_lme4_route_resid.R <frames.csv> <resid.csv> <ranef.csv> <simplified.csv>

suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
frames <- read.csv(args[1], colClasses = c(subject = "character"))
ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 3e5))
FORMULAS <- list(
  full = y ~ log_use + (1 + log_use | level) + (1 + log_use | level:scene)
             + (1 + log_use | level:scene:route),
  no_upper_slopes = y ~ log_use + (1 | level) + (1 | level:scene) + (1 + log_use | level:scene:route),
  uncorrelated = y ~ log_use + (1 + log_use || level) + (1 + log_use || level:scene)
                     + (1 + log_use || level:scene:route))

fit <- function(f, g) suppressMessages(suppressWarnings(
  lmer(f, data = g, REML = FALSE, control = ctrl)))

effects <- function(re, grp, kind, s) {
  x <- re[[grp]]
  data.frame(subject = s, kind = kind, id = rownames(x),
             u = x[, "(Intercept)"], v = x[, "log_use"])
}

resid_rows <- list(); ranef_rows <- list(); simple_rows <- list()
for (g in split(frames, frames$subject)) {
  s <- g$subject[1]
  for (name in names(FORMULAS)) {
    m <- fit(FORMULAS[[name]], g)
    se <- sqrt(vcov(m)["log_use", "log_use"])
    simple_rows[[length(simple_rows) + 1]] <- data.frame(
      subject = s, model = name, b = fixef(m)[["log_use"]], se = se,
      ci_low = fixef(m)[["log_use"]] - 1.96 * se, ci_high = fixef(m)[["log_use"]] + 1.96 * se,
      aic = AIC(m), singular = isSingular(m))
    if (name != "full") next
    resid_rows[[length(resid_rows) + 1]] <- data.frame(subject = s, rid = g$rid,
                                                       fit = fitted(m), res = resid(m))
    re <- ranef(m)
    ranef_rows[[length(ranef_rows) + 1]] <- rbind(
      effects(re, "level", "level", s),
      effects(re, "level:scene", "scene", s),
      effects(re, "level:scene:route", "route", s))
  }
  cat(sprintf("sub-%s done\n", s))
}
write.csv(do.call(rbind, resid_rows), args[2], row.names = FALSE)
write.csv(do.call(rbind, ranef_rows), args[3], row.names = FALSE)
write.csv(do.call(rbind, simple_rows), args[4], row.names = FALSE)
