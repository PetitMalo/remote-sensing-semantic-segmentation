#!/bin/bash
#SBATCH --job-name=rsss-dummy-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Dummy training run on PANDO: one epoch of the U-Net baseline over the FLAIR#1
# toy split (156 train / 44 val tiles, 512x512, 5 bands). Nothing here measures
# quality -- it proves the GPU path end to end: data, forward/backward, metrics
# and a checkpoint on disk.
#
# PANDO's compute nodes have no outbound network, so stage everything from the
# login node first:
#   uv sync --frozen
#   uv run python -m rsss.data.download --data-dir "$HOME/rsss-data"
#
# Then, from the repository root:
#   sbatch scripts/pando/job.sh
#
# Override the defaults with the environment:
#   DATA_DIR=/scratch/me/data RUN_DIR=/scratch/me/runs EPOCHS=2 sbatch scripts/pando/job.sh

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

# Root the datasets are extracted into (it holds flair_1_toy_dataset/).
DATA_DIR="${DATA_DIR:-$HOME/rsss-data}"
# Run artefacts: never the repository, which may sit on a small quota.
RUN_DIR="${RUN_DIR:-$HOME/rsss-runs/dummy-${SLURM_JOB_ID:-local}}"
EPOCHS="${EPOCHS:-1}"

# Slurm opens --output/--error before this script runs, so logs/ must exist.
mkdir -p logs "$RUN_DIR"

if [ ! -d "$DATA_DIR/flair_1_toy_dataset/CSVs" ]; then
    echo "No FLAIR#1 toy dataset under $DATA_DIR." >&2
    echo "Stage it on the login node first:" >&2
    echo "  uv run python -m rsss.data.download --data-dir $DATA_DIR" >&2
    exit 1
fi

echo "Job $SLURM_JOB_ID on $(hostname); data=$DATA_DIR runs=$RUN_DIR epochs=$EPOCHS"
uv run --no-sync python -m rsss.train \
    --data-dir "$DATA_DIR" \
    --output-dir "$RUN_DIR" \
    --max-epochs "$EPOCHS" \
    --seed 0
