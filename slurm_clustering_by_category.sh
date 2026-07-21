#!/bin/bash
#SBATCH --job-name=mario-clustering-by-category
#SBATCH --account=def-YOURPI          # ← replace with your allocation account
#SBATCH --array=0-27                  # one job per category (28 categories total)
#SBATCH --time=12:00:00               # worst category ("Enemy", ~19k pooled traces) needs hours
#SBATCH --mem=8G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/clustering_by_category_%A_%a.out
#SBATCH --error=logs/clustering_by_category_%A_%a.err

# Pairwise Fréchet distance is O(n^2) in pooled trace count, and categories pool
# traces across ALL scenes carrying that tag — unlike per-scene clustering, sizes
# vary hugely (154 traces for "Gap stair valley" vs ~19,000 for "Enemy"). The
# resource request above is sized for the worst case; small categories will finish
# in minutes and just return the node early.

# ── change to project directory ───────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"

# ── load environment ──────────────────────────────────────────────────────────
module load python/3.12   # adjust to what's available: module spider python
source .venv/bin/activate

# ── select category for this array task ───────────────────────────────────────
CATEGORIES_FILE="categories.txt"

if [ ! -f "$CATEGORIES_FILE" ]; then
    echo "categories.txt not found. Generate it first with:"
    echo "  invoke list-categories-task > categories.txt"
    exit 1
fi

CATEGORY=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CATEGORIES_FILE")

if [ -z "$CATEGORY" ]; then
    echo "No category for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Array task $SLURM_ARRAY_TASK_ID → category $CATEGORY"

# ── run clustering for this category ──────────────────────────────────────────
invoke run-clustering-by-category --categories "$CATEGORY"
