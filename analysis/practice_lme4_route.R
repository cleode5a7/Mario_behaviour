# Within-route practice model in lme4 — the secondary (mechanism) analysis.
#
# The primary model asks whether attempts become more typical of *some* route; its
# distance is measured against the nearest prototype, which changes from attempt to
# attempt. This model fixes the origin: each attempt's Fréchet distance (px) to its OWN
# route's prototype, with practice counted as the nth use of that route, and routes
# nested in scenes nested in levels:
#   y ~ log_use + (1 + log_use | level) + (1 + log_use | level:scene)
#                + (1 + log_use | level:scene:route)
# A negative slope means each further use of a route is executed closer to its typical
# form. Called by analysis.practice.run_practice_lme4 with the exact model frames.
# Maximum likelihood (REML = FALSE). Wald CI only by default: with three nested levels a
# profile CI is hours of refits, and on the two-level model it agreed with Wald to ~0.002;
# pass "profile" as the fifth argument to compute it anyway.
#
#   Rscript analysis/practice_lme4_route.R <frames.csv> <fits.csv> <curves.csv> [n_cores] [profile]

suppressPackageStartupMessages({
  library(lme4)
  library(parallel)
})

args <- commandArgs(trailingOnly = TRUE)
frames <- read.csv(args[1], colClasses = c(subject = "character"))
n_cores <- if (length(args) >= 4) as.integer(args[4]) else max(1, detectCores() - 1)
with_profile <- length(args) >= 5 && args[5] == "profile"

varcomp <- function(m, group) {
  vc <- VarCorr(m)[[group]]
  c(var_intercept = vc[1, 1], var_slope = vc[2, 2], corr = attr(vc, "correlation")[1, 2])
}

one <- function(g) {
  key <- data.frame(measure = g$measure[1], subject = g$subject[1], transform = g$transform[1],
                    completed_only = g$completed_only[1], n_obs = nrow(g),
                    n_routes = length(unique(paste(g$scene, g$route))),
                    n_scenes = length(unique(g$scene)), n_levels = length(unique(g$level)))
  warn <- character(0)
  m <- withCallingHandlers(
    lmer(y ~ log_use + (1 + log_use | level) + (1 + log_use | level:scene)
           + (1 + log_use | level:scene:route), data = g, REML = FALSE,
         control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 3e5))),
    warning = function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning") },
    message = function(m) invokeRestart("muffleMessage"))
  fe <- fixef(m)
  b <- fe[["log_use"]]
  se <- sqrt(vcov(m)["log_use", "log_use"])
  prof <- if (with_profile) tryCatch(
    suppressMessages(confint(m, parm = "log_use", method = "profile", quiet = TRUE)),
    error = function(e) matrix(NA_real_, 1, 2)) else matrix(NA_real_, 1, 2)
  row <- data.frame(key, model = "route", intercept = fe[["(Intercept)"]], b = b, se = se,
                    wald_low = b - 1.96 * se, wald_high = b + 1.96 * se,
                    profile_low = prof[1, 1], profile_high = prof[1, 2],
                    p_wald = 2 * pnorm(-abs(b / se)), aic = AIC(m),
                    loglik = as.numeric(logLik(m)), resid_var = sigma(m)^2,
                    singular = isSingular(m), warnings = paste(unique(warn), collapse = " | "))
  for (grp in c("level", "level:scene", "level:scene:route")) {
    v <- varcomp(m, grp)
    tag <- c(level = "level", `level:scene` = "scene", `level:scene:route` = "route")[[grp]]
    row[[paste0(tag, "_var_intercept")]] <- v[["var_intercept"]]
    row[[paste0(tag, "_var_slope")]] <- v[["var_slope"]]
    row[[paste0(tag, "_corr")]] <- v[["corr"]]
  }
  grid <- unique(round(seq(1, quantile(g$use, 0.95, type = 7), length.out = 60)))
  curve <- data.frame(key[c("measure", "subject", "transform", "completed_only")], model = "route",
                      use = grid, y = fe[["(Intercept)"]] + b * log(grid))
  list(row = row, curve = curve)
}

groups <- split(frames, list(frames$measure, frames$subject, frames$transform,
                             frames$completed_only), drop = TRUE)
res <- mclapply(groups, function(g) tryCatch(one(g), error = function(e) e), mc.cores = n_cores)
bad <- vapply(res, function(r) inherits(r, "error") || inherits(r, "try-error"), logical(1))
if (any(bad)) stop("route fit failed for: ", paste(names(groups)[bad], collapse = ", "), "\n",
                   paste(vapply(res[bad], conditionMessage, ""), collapse = "\n"))
write.csv(do.call(rbind, lapply(res, `[[`, "row")), args[2], row.names = FALSE)
write.csv(do.call(rbind, lapply(res, `[[`, "curve")), args[3], row.names = FALSE)
cat(sprintf("lme4 route: %d groups fitted on %d cores\n", length(groups), n_cores))
