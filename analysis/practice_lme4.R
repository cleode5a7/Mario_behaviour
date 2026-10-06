# Full nested practice model in lme4 — the primary practice-effect model.
#
# Called by analysis.practice.run_practice_lme4 with the exact model frames the
# statsmodels fits use (same transforms, same dropped zeros, same distances). Per
# subject x measure x transform x attempts, two models, both by maximum likelihood
# (REML = FALSE, as in statsmodels, so AIC and the likelihood-ratio test are comparable):
#   scene  : y ~ logt + (1 + logt | scene)
#            the statsmodels main model, refitted here as a cross-check between programs
#   nested : y ~ logt + (1 + logt | level) + (1 + logt | level:scene)
#            the primary model: levels AND correlated scene effects, which statsmodels
#            cannot nest
# For each: intercept and slope, Wald CI (b +/- 1.96 SE, as statsmodels reports) and
# profile-likelihood CI, Wald p, variance components at every level, AIC, singularity,
# and the fixed-effect curve on an attempt grid (1 to the 95th-percentile attempt, 60
# points, as fit_mixed draws it); plus a likelihood-ratio test of nested against scene.
#
#   Rscript analysis/practice_lme4.R <frames.csv> <fits.csv> <curves.csv> [n_cores]

suppressPackageStartupMessages({
  library(lme4)
  library(parallel)
})

args <- commandArgs(trailingOnly = TRUE)
frames <- read.csv(args[1], colClasses = c(subject = "character"))
n_cores <- if (length(args) >= 4) as.integer(args[4]) else max(1, detectCores() - 1)

fit <- function(formula, d) {
  warn <- character(0)
  m <- withCallingHandlers(
    lmer(formula, data = d, REML = FALSE,
         control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))),
    warning = function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning") },
    message = function(m) invokeRestart("muffleMessage"))
  list(m = m, warn = paste(unique(warn), collapse = " | "))
}

varcomp <- function(m, group) {
  vc <- VarCorr(m)[[group]]
  c(var_intercept = vc[1, 1], var_slope = vc[2, 2], corr = attr(vc, "correlation")[1, 2])
}

summarise <- function(f, label, key, grid) {
  m <- f$m
  fe <- fixef(m)
  b <- fe[["logt"]]
  se <- sqrt(vcov(m)["logt", "logt"])
  prof <- tryCatch(suppressMessages(confint(m, parm = "logt", method = "profile", quiet = TRUE)),
                   error = function(e) matrix(NA_real_, 1, 2))
  row <- data.frame(key, model = label, intercept = fe[["(Intercept)"]], b = b, se = se,
                    wald_low = b - 1.96 * se, wald_high = b + 1.96 * se,
                    profile_low = prof[1, 1], profile_high = prof[1, 2],
                    p_wald = 2 * pnorm(-abs(b / se)),
                    aic = AIC(m), loglik = as.numeric(logLik(m)),
                    resid_var = sigma(m)^2, singular = isSingular(m), warnings = f$warn)
  s <- varcomp(m, if (label == "scene") "scene" else "level:scene")
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
  curve <- data.frame(key[c("measure", "subject", "transform", "completed_only")], model = label,
                      attempt = grid, y = fe[["(Intercept)"]] + b * log(grid))
  list(row = row, curve = curve)
}

one <- function(g) {
  key <- data.frame(measure = g$measure[1], subject = g$subject[1], transform = g$transform[1],
                    completed_only = g$completed_only[1], n_obs = nrow(g),
                    n_scenes = length(unique(g$scene)), n_levels = length(unique(g$level)))
  grid <- unique(round(seq(1, quantile(g$attempt, 0.95, type = 7), length.out = 60)))
  s <- summarise(fit(y ~ logt + (1 + logt | scene), g), "scene", key, grid)
  n <- summarise(fit(y ~ logt + (1 + logt | level) + (1 + logt | level:scene), g), "nested", key, grid)
  ll <- 2 * (n$row$loglik - s$row$loglik)
  n$row$lrt_chisq <- ll
  n$row$lrt_p <- pchisq(max(ll, 0), df = 3, lower.tail = FALSE)
  s$row$lrt_chisq <- NA; s$row$lrt_p <- NA
  list(rows = rbind(s$row, n$row), curves = rbind(s$curve, n$curve))
}

groups <- split(frames, list(frames$measure, frames$subject, frames$transform,
                             frames$completed_only), drop = TRUE)
res <- mclapply(groups, one, mc.cores = n_cores)
bad <- vapply(res, inherits, logical(1), "try-error")
if (any(bad)) stop("lme4 fit failed for: ", paste(names(groups)[bad], collapse = ", "),
                   "\n", paste(unique(unlist(res[bad])), collapse = "\n"))
write.csv(do.call(rbind, lapply(res, `[[`, "rows")), args[2], row.names = FALSE)
write.csv(do.call(rbind, lapply(res, `[[`, "curves")), args[3], row.names = FALSE)
cat(sprintf("lme4: %d groups fitted on %d cores\n", length(groups), n_cores))
