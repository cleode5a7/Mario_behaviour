# Route-usage model in lme4: does route choice concentrate across the four stages?
#
# Called by analysis.route_usage.run_route_usage with one row per cell (subject x scene
# x stage) for the observed data (rep 0) and for each chance-level shuffle (rep >= 1).
# Per subject x measure x weighting, by maximum likelihood (REML = FALSE, as the
# practice models):
#   y ~ stage + (1 | level) + (1 | level:scene)        early discovery = reference
# weighted by the cell's attempt count (normalised to mean 1) or unweighted. From the
# fixed effects and their covariance: the four stage means and three contrasts, each
# with a Wald SE. On rep 0 also a level bootstrap: whole levels resampled with
# replacement (a level drawn twice enters as two distinct levels), refitted, contrasts
# kept.
#
#   Rscript analysis/route_usage_lme4.R <frames.csv> <out.csv> <n_cores> <n_boot> <seed> <subjects>

suppressPackageStartupMessages({
  library(lme4)
  library(parallel)
})

args <- commandArgs(trailingOnly = TRUE)
frames <- read.csv(args[1], colClasses = c(subject = "character"))
n_cores <- as.integer(args[3])
n_boot <- as.integer(args[4])
seed <- as.integer(args[5])
subjects <- strsplit(args[6], ",")[[1]]

STAGES <- c("early_discovery", "late_discovery", "early_practice", "late_practice")
MEASURES <- c("simpson", "shannon_rarefied")
WEIGHTINGS <- c("n", "none")
# Rows: what each quantity is in terms of (intercept, late_disc, early_prac, late_prac).
L <- rbind(
  early_discovery         = c(1, 0, 0, 0),
  late_discovery          = c(1, 1, 0, 0),
  early_practice          = c(1, 0, 1, 0),
  late_practice           = c(1, 0, 0, 1),
  late_vs_early_discovery = c(0, 1, 0, 0),
  late_vs_early_practice  = c(0, 0, -1, 1),
  practice_vs_discovery   = c(0, -0.5, 0.5, 0.5))
CONTRASTS <- c("late_vs_early_discovery", "late_vs_early_practice", "practice_vs_discovery")

fit_one <- function(d, measure, weighting, rows) {
  d <- d[!is.na(d[[measure]]), ]
  d$y <- d[[measure]]
  d$stage <- factor(d$stage, levels = STAGES)
  d$w <- if (weighting == "n") d$n / mean(d$n) else 1
  empty <- data.frame(quantity = rows, est = NA_real_, se = NA_real_, p = NA_real_,
                      singular = NA, warnings = "")
  if (any(table(d$stage) == 0)) {
    empty$warnings <- "a stage has no cells"
    return(empty)
  }
  warn <- character(0)
  m <- tryCatch(withCallingHandlers(
    lmer(y ~ stage + (1 | level) + (1 | level:scene), data = d, weights = w, REML = FALSE,
         control = lmerControl(optimizer = "bobyqa")),
    warning = function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning") },
    message = function(m) invokeRestart("muffleMessage")),
    error = function(e) e)
  if (inherits(m, "error")) {
    empty$warnings <- conditionMessage(m)
    return(empty)
  }
  Lr <- L[rows, , drop = FALSE]
  est <- as.vector(Lr %*% fixef(m))
  se <- sqrt(diag(Lr %*% as.matrix(vcov(m)) %*% t(Lr)))
  data.frame(quantity = rows, est = est, se = se, p = 2 * pnorm(-abs(est / se)),
             singular = isSingular(m), warnings = paste(unique(warn), collapse = " | "))
}

fit_all <- function(d, kind, rep, subject, rows) {
  out <- list()
  for (measure in MEASURES) for (weighting in WEIGHTINGS) {
    r <- fit_one(d, measure, weighting, rows)
    out[[length(out) + 1]] <- cbind(kind = kind, rep = rep, subject = subject,
                                    measure = measure, weighting = weighting, r)
  }
  do.call(rbind, out)
}

resample_levels <- function(d, b) {
  set.seed(seed * 100003 + b)
  levs <- unique(d$level)
  pick <- sample(levs, length(levs), replace = TRUE)
  do.call(rbind, lapply(seq_along(pick), function(j) {
    x <- d[d$level == pick[j], ]
    x$level <- paste0(x$level, "#", j)
    x$scene <- paste0(x$scene, "#", j)
    x
  }))
}

jobs <- list()
for (s in subjects) {
  for (rep in sort(unique(frames$rep[frames$subject == s]))) jobs[[length(jobs) + 1]] <- list("fit", s, rep)
  for (b in seq_len(n_boot)) jobs[[length(jobs) + 1]] <- list("boot", s, b)
}

run_job <- function(job) {
  kind <- job[[1]]; s <- job[[2]]; i <- job[[3]]
  if (kind == "fit") {
    d <- frames[frames$subject == s & frames$rep == i, ]
    if (i == 0) fit_all(d, "fit", i, s, rownames(L)) else fit_all(d, "null", i, s, CONTRASTS)
  } else {
    d <- frames[frames$subject == s & frames$rep == 0, ]
    fit_all(resample_levels(d, i), "boot", i, s, CONTRASTS)
  }
}

res <- mclapply(jobs, function(j) tryCatch(run_job(j), error = function(e) e), mc.cores = n_cores)
bad <- vapply(res, function(r) inherits(r, "error"), logical(1))
if (any(bad)) stop("route-usage fits failed: ",
                   paste(unique(vapply(res[bad], conditionMessage, "")), collapse = "\n"))
write.csv(do.call(rbind, res), args[2], row.names = FALSE)
cat(sprintf("route usage: %d jobs on %d cores\n", length(jobs), n_cores))
