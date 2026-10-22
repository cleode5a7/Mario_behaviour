# Level bootstrap of the route-scale practice model, for row 9 of its diagnostics.
#
# Called by docs/practice_route_bootstrap.py with the model's exact rows
# (analysis.practice.route_frame). Per subject x transform and replicate: draw the
# subject's levels with replacement (a level drawn twice enters as two distinct levels,
# its scenes and routes renamed to match), refit
#   y ~ log_use + (1 + log_use | level) + (1 + log_use | level:scene)
#               + (1 + log_use | level:scene:route),   REML = FALSE
# and keep the slope. Levels are the resampling unit because they are what this model,
# like the scene-scale one, assumes independent. Replicates run in parallel over cores;
# each draw is seeded by (seed, replicate), so the answer does not depend on the cores.
#
#   Rscript docs/practice_lme4_route_boot.R <frames.csv> <draws.csv> <n_boot> <seed> <n_cores>

suppressPackageStartupMessages({
  library(lme4)
  library(parallel)
})

args <- commandArgs(trailingOnly = TRUE)
frames <- read.csv(args[1], colClasses = c(subject = "character"))
n_boot <- as.integer(args[3])
seed <- as.integer(args[4])
n_cores <- as.integer(args[5])

one <- function(d, b) {
  set.seed(seed * 100003 + b)
  levs <- unique(d$level)
  pick <- sample(levs, length(levs), replace = TRUE)
  x <- do.call(rbind, lapply(seq_along(pick), function(j) {
    y <- d[d$level == pick[j], ]
    y$level <- paste0(y$level, "#", j)
    y$scene <- paste0(y$scene, "#", j)
    y
  }))
  m <- tryCatch(suppressMessages(suppressWarnings(
    lmer(y ~ log_use + (1 + log_use | level) + (1 + log_use | level:scene)
           + (1 + log_use | level:scene:route), data = x, REML = FALSE,
         control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 3e5))))),
    error = function(e) NULL)
  if (is.null(m)) return(c(b = NA_real_, singular = NA))
  c(b = fixef(m)[["log_use"]], singular = isSingular(m))
}

groups <- split(frames, list(frames$subject, frames$transform), drop = TRUE)
jobs <- expand.grid(g = names(groups), b = seq_len(n_boot), stringsAsFactors = FALSE)
res <- mclapply(seq_len(nrow(jobs)), function(i) {
  d <- groups[[jobs$g[i]]]
  r <- one(d, jobs$b[i])
  data.frame(transform = d$transform[1], subject = d$subject[1], replicate = jobs$b[i],
             b = r[["b"]], singular = as.logical(r[["singular"]]))
}, mc.cores = n_cores)
write.csv(do.call(rbind, res), args[2], row.names = FALSE)
cat(sprintf("route-scale level bootstrap: %d refits on %d cores\n", nrow(jobs), n_cores))
