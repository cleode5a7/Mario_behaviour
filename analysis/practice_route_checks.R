# Checks on the route-scale practice effect: scene vs route effects, and its shape.
#
# The route-scale data (distance in px to the attempt's own route prototype, against the
# nth use of that route) fitted at four levels of correction, each as a straight line on
# log-log axes and with a quadratic term in log(use):
#   pooled    : y ~ log_use                                     (plain least squares)
#   scene     : y ~ log_use + (1 + log_use | level) + (1 + log_use | level:scene)
#   route_int : scene + (1 | level:scene:route)                 (route baselines only)
#   route     : scene + (1 + log_use | level:scene:route)      (= lme4_route.json's model)
# and the same four with "+ lu2" (lu2 = log_use^2). Going pooled -> scene asks whether the
# pooled decrease survives scene and level effects; scene -> route_int asks how much of what
# is left is routes simply differing in how tight they are (structure) rather than change
# along a route; route_int -> route lets each route also improve at its own rate. The curvature
# term is a fixed effect only: scenes and routes keep their own linear slope. Pooled SEs
# treat attempts as independent and are too small; they are kept for description only.
# Maximum likelihood (REML = FALSE), so the linear and quadratic fits compare by AIC and a
# likelihood-ratio test (1 df). Wald CIs. Called by
# analysis.practice.run_practice_route_checks with route_frame()'s rows.
#
#   Rscript analysis/practice_route_checks.R <frames.csv> <fits.csv> <curves.csv> [n_cores]

suppressPackageStartupMessages({
  library(lme4)
  library(parallel)
})

args <- commandArgs(trailingOnly = TRUE)
frames <- read.csv(args[1], colClasses = c(subject = "character"))
n_cores <- if (length(args) >= 4) as.integer(args[4]) else max(1, detectCores() - 1)
frames$lu2 <- frames$log_use^2
SLOPE_AT <- c(1, 5, 20, 50)    # nth use at which the quadratic's local slope is reported

re <- list(
  pooled = "",
  scene = " + (1 + log_use | level) + (1 + log_use | level:scene)",
  route_int = " + (1 + log_use | level) + (1 + log_use | level:scene) + (1 | level:scene:route)",
  route = " + (1 + log_use | level) + (1 + log_use | level:scene) + (1 + log_use | level:scene:route)")

fit <- function(correction, quadratic, g) {
  f <- as.formula(paste0("y ~ log_use", if (quadratic) " + lu2" else "", re[[correction]]))
  warn <- character(0)
  m <- withCallingHandlers(
    if (correction == "pooled") lm(f, data = g) else
      lmer(f, data = g, REML = FALSE,
           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 3e5))),
    warning = function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning") },
    message = function(m) invokeRestart("muffleMessage"))
  list(m = m, warn = paste(unique(warn), collapse = " | "))
}

summarise <- function(f, correction, quadratic, key, grid) {
  m <- f$m
  fe <- if (correction == "pooled") coef(m) else fixef(m)
  V <- as.matrix(vcov(m))
  b <- fe[["log_use"]]
  se <- sqrt(V["log_use", "log_use"])
  b2 <- if (quadratic) fe[["lu2"]] else 0
  row <- data.frame(key, correction = correction, shape = if (quadratic) "quadratic" else "linear",
                    intercept = fe[["(Intercept)"]], b = b, se = se,
                    wald_low = b - 1.96 * se, wald_high = b + 1.96 * se,
                    p_wald = 2 * pnorm(-abs(b / se)),
                    b2 = if (quadratic) b2 else NA, se2 = if (quadratic) sqrt(V["lu2", "lu2"]) else NA,
                    aic = AIC(m), loglik = as.numeric(logLik(m)),
                    singular = if (correction == "pooled") FALSE else isSingular(m),
                    warnings = f$warn)
  row$p2_wald <- if (quadratic) 2 * pnorm(-abs(row$b2 / row$se2)) else NA
  for (u in SLOPE_AT) {          # local slope d y / d log(use) at use u, and its SE
    L <- if (quadratic) c(1, 2 * log(u)) else 1
    nm <- if (quadratic) c("log_use", "lu2") else "log_use"
    row[[paste0("slope_at_", u)]] <- sum(L * fe[nm])
    row[[paste0("slope_se_at_", u)]] <- sqrt(drop(t(L) %*% V[nm, nm, drop = FALSE] %*% L))
  }
  curve <- data.frame(key[c("measure", "subject", "transform", "completed_only")],
                      correction = correction, shape = row$shape, use = grid,
                      y = fe[["(Intercept)"]] + b * log(grid) + b2 * log(grid)^2)
  list(row = row, curve = curve)
}

one <- function(g) {
  key <- data.frame(measure = g$measure[1], subject = g$subject[1], transform = g$transform[1],
                    completed_only = g$completed_only[1], n_obs = nrow(g),
                    n_routes = length(unique(paste(g$scene, g$route))),
                    n_scenes = length(unique(g$scene)), n_levels = length(unique(g$level)))
  grid <- unique(round(seq(1, quantile(g$use, 0.95, type = 7), length.out = 60)))
  out <- list()
  for (correction in names(re)) {
    lin <- summarise(fit(correction, FALSE, g), correction, FALSE, key, grid)
    quad <- summarise(fit(correction, TRUE, g), correction, TRUE, key, grid)
    chisq <- 2 * (quad$row$loglik - lin$row$loglik)
    quad$row$lrt_chisq <- chisq
    quad$row$lrt_p <- pchisq(max(chisq, 0), df = 1, lower.tail = FALSE)
    lin$row$lrt_chisq <- NA; lin$row$lrt_p <- NA
    out <- c(out, list(lin, quad))
  }
  list(rows = do.call(rbind, lapply(out, `[[`, "row")),
       curves = do.call(rbind, lapply(out, `[[`, "curve")))
}

groups <- split(frames, list(frames$measure, frames$subject, frames$transform,
                             frames$completed_only), drop = TRUE)
res <- mclapply(groups, function(g) tryCatch(one(g), error = function(e) e), mc.cores = n_cores)
bad <- vapply(res, function(r) inherits(r, "error") || inherits(r, "try-error"), logical(1))
if (any(bad)) stop("route check failed for: ", paste(names(groups)[bad], collapse = ", "), "\n",
                   paste(vapply(res[bad], conditionMessage, ""), collapse = "\n"))
write.csv(do.call(rbind, lapply(res, `[[`, "rows")), args[2], row.names = FALSE)
write.csv(do.call(rbind, lapply(res, `[[`, "curves")), args[3], row.names = FALSE)
cat(sprintf("route checks: %d groups x %d fits on %d cores\n", length(groups), 2 * length(re), n_cores))
