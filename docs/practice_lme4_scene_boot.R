# Level bootstrap of the scene-scale practice model (the main lme4 model), for its CI check.
#
# Called by docs/practice_lme4_scene_bootstrap.py with the model's exact rows
# (analysis.practice._mixed_frame, as analysis/practice_lme4.R fits them). Per subject x
# transform and replicate: draw the subject's levels with replacement (a level drawn twice
# enters as two distinct levels, its scenes renamed to match), refit
#   y ~ logt + (1 + logt | level) + (1 + logt | level:scene),   REML = FALSE
# with the optimiser analysis/practice_lme4.R uses, and keep the slope. Levels are the
# resampling unit because they are what this model assumes independent. The twin of
# docs/practice_lme4_route_boot.R; each draw is seeded by (seed, replicate) the same way,
# so the answer does not depend on the cores.
#
#   Rscript docs/practice_lme4_scene_boot.R <frames.csv> <draws.csv> <n_boot> <seed> <n_cores>

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
    lmer(y ~ logt + (1 + logt | level) + (1 + logt | level:scene), data = x, REML = FALSE,
         control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))))),
    error = function(e) NULL)
  if (is.null(m)) return(c(b = NA_real_, singular = NA))
  c(b = fixef(m)[["logt"]], singular = isSingular(m))
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
cat(sprintf("scene-scale level bootstrap: %d refits on %d cores\n", nrow(jobs), n_cores))
