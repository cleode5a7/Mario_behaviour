# Residuals and random effects of the full nested practice model, for the diagnostics.
#
# Called by docs/practice_residual_diagnostics.py (``lme4`` mode) with the exact model
# frames it would fit itself. Refits, per subject, the same model as docs/practice_lme4.R
#   y ~ logt + (1 + logt | level) + (1 + logt | level:scene),   REML = FALSE
# and writes, in the frames' row order, the fitted values and conditional residuals
# (y minus fixed AND random parts, as statsmodels' resid), plus every level's and
# scene's random intercept and slope. No profile CIs, so it runs in about a minute.
#
#   Rscript docs/practice_lme4_resid.R <frames.csv> <resid.csv> <ranef.csv>

suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
frames <- read.csv(args[1], colClasses = c(subject = "character"))
resid_rows <- list()
ranef_rows <- list()
for (g in split(frames, frames$subject)) {
  m <- suppressMessages(suppressWarnings(
    lmer(y ~ logt + (1 + logt | level) + (1 + logt | level:scene), data = g, REML = FALSE,
         control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))))
  resid_rows[[length(resid_rows) + 1]] <- data.frame(
    subject = g$subject, rid = g$rid, fit = fitted(m), res = resid(m))
  re <- ranef(m)
  lv <- re$level
  sc <- re$`level:scene`
  ranef_rows[[length(ranef_rows) + 1]] <- rbind(
    data.frame(subject = g$subject[1], kind = "level", id = rownames(lv),
               u = lv[, "(Intercept)"], v = lv[, "logt"]),
    data.frame(subject = g$subject[1], kind = "scene", id = sub("^[^:]*:", "", rownames(sc)),
               u = sc[, "(Intercept)"], v = sc[, "logt"]))
  cat(sprintf("sub-%s: singular=%s\n", g$subject[1], isSingular(m)))
}
write.csv(do.call(rbind, resid_rows), args[2], row.names = FALSE)
write.csv(do.call(rbind, ranef_rows), args[3], row.names = FALSE)
