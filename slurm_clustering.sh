#!/bin/bash
#SBATCH --job-name=mario-clustering
#SBATCH --account=def-YOURPI          # ← replace with your allocation account
#SBATCH --array=0-307                 # one job per scene (308 scenes total)
#SBATCH --time=01:00:00               # 1 h; worst scene (~3 h single-core) ÷ 8 CPUs
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/clustering_%A_%a.out
#SBATCH --error=logs/clustering_%A_%a.err

# ── change to project directory ───────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"

# ── load environment ──────────────────────────────────────────────────────────
module load python/3.12   # adjust to what's available: module spider python
source .venv/bin/activate

# ── select scene for this array task ─────────────────────────────────────────
SCENES_FILE="scenes.txt"

if [ ! -f "$SCENES_FILE" ]; then
    echo "scenes.txt not found. Generate it first with:"
    echo "  invoke list-scenes-task > scenes.txt"
    exit 1
fi

SCENE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SCENES_FILE")

if [ -z "$SCENE" ]; then
    echo "No scene for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Array task $SLURM_ARRAY_TASK_ID → scene $SCENE"

# ── run clustering for this scene ─────────────────────────────────────────────
invoke run-clustering --scenes "$SCENE"
