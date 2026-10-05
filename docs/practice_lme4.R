# Full nested practice model in lme4: levels AND correlated scene effects.
#
# statsmodels can nest scene effects inside levels only as independent variance
# components, so its nested model (analysis/practice.py, "level_scene") lacks the scene
# intercept-slope correlation the main model carries. lme4 fits both at once. Called by
# docs/practice_lme4.py, which exports the exact model frames the statsmodels fits used.
#
#   Rscript docs/practice_lme4.R <frames.csv> <out.csv>
#
# Per subject x transform x attempts, two models, both by maximum likelihood (REML = FALSE,
# as in statsmodels, so AIC and likelihood-ratio tests compare like with like):
#   scene  : y ~ logt + (1 + logt | scene)
#            the main statsmodels model, refitted here as a cross-check between programs
#   nested : y ~ logt + (1 + logt | level) + (1 + logt | level:scene)
#            the model statsmodels could not fit
# For each: slope, Wald CI (b +/- 1.96 SE, as statsmodels reports), profile-likelihood CI,
# variance components at every level, AIC, convergence warnings; and a likelihood-ratio
# test of nested against scene (one level adds 3 parameters).

suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
frames <- read.csv(args[1], colClasses = c(subject = "character"))
out_path <- args[2]

fit <- function(formula, d) {
  warn <- character(0)
  m <- withCallingHandlers(
    lmer(formula, data = d, REML = FALSE,
         control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))),
    warning = function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning") })
  list(m = m, warn = paste(unique(warn), collapse = " | "))
}

varcomp <- function(m, group) {
  vc <- VarCorr(m)[[group]]
  c(var_intercept = vc[1, 1], var_slope = vc[2, 2],
    corr = attr(vc, "correlation")[1, 2])
}

summarise <- function(f, label, key) {
  m <- f$m
  b <- fixef(m)[["logt"]]
  se <- sqrt(vcov(m)["logt", "logt"])
  prof <- tryCatch(confint(m, parm = "logt", method = "profile", quiet = TRUE),
                   error = function(e) matrix(NA_real_, 1, 2))
  row <- data.frame(key, model = label, b = b, se = se,
                    wald_low = b - 1.96 * se, wald_high = b + 1.96 * se,
                    profile_low = prof[1, 1], profile_high = prof[1, 2],
                    p_wald = 2 * pnorm(-abs(b / se)),
                    aic = AIC(m), loglik = as.numeric(logLik(m)),
                    resid_var = sigma(m)^2, singular = isSingular(m),
                    warnings = f$warn)
  sc <- if (label == "scene") "scene" else "level:scene"
  s <- varcomp(m, sc)
  row$scene_var_intercept <- s[["var_intercept"]]
  row$scene_var_slope <- s[["var_slope"]]
  row$scene_corr <- s[["corr"]]
  if (label == "nested") {
    l <- varcomp(m, "level")
    row$level_var_intercept <- l[["var_intercept"]]
    row$level_var_slope <- l[["var_slope"]]
    row$level_corr <- l[["corr"]]
  } else {
    row$level_var_intercept <- NA; row$level_var_slope <- NA; row$level_corr <- NA
  }
  row
}

rows <- list()
for (g in split(frames, list(frames$subject, frames$transform, frames$completed_only), drop = TRUE)) {
  key <- data.frame(subject = g$subject[1], transform = g$transform[1],
                    completed_only = g$completed_only[1], n_obs = nrow(g),
                    n_scenes = length(unique(g$scene)), n_levels = length(unique(g$level)))
  cat(sprintf("sub-%s %s%s ... ", key$subject, key$transform,
              ifelse(key$completed_only, " (completed)", "")))
  s <- fit(y ~ logt + (1 + logt | scene), g)
  n <- fit(y ~ logt + (1 + logt | level) + (1 + logt | level:scene), g)
  rs <- summarise(s, "scene", key)
  rn <- summarise(n, "nested", key)
  lrt <- anova(s$m, n$m)
  rn$lrt_chisq <- lrt$Chisq[2]
  rn$lrt_p <- lrt$`Pr(>Chisq)`[2]
  rs$lrt_chisq <- NA; rs$lrt_p <- NA
  rows[[length(rows) + 1]] <- rs
  rows[[length(rows) + 1]] <- rn
  cat(sprintf("b scene %.4f nested %.4f, LRT p %.2g\n", rs$b, rn$b, rn$lrt_p))
}
write.csv(do.call(rbind, rows), out_path, row.names = FALSE)
cat("Wrote", out_path, "\n")
