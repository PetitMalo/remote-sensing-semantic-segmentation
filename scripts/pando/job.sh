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
# PANDO runs RHEL 7.8 (glibc 2.17), driver 550.54.15 (CUDA 12.4) and Tesla V100
# GPUs, so its commands select the `pando` dependency group: it is the only fork
# of uv.lock that installs on that glibc, and torch 2.6/cu124 is the newest build
# that the driver and the V100s both support. Its compute nodes have no outbound
# network, so stage everything from the login node first:
#
#   uv sync --frozen --no-default-groups --no-dev --group pando
#   uv run --no-default-groups --group pando python -m rsss.data.download \
#       --data-dir "$HOME/rsss-data"
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

# The group selection lives here because every uv command needs it: the default
# groups are `dev` and `gcp`, whose wheels need glibc 2.28 and would fail to
# install (or import) on PANDO.
UV_GROUP_ARGS=(--no-default-groups --no-dev --group pando)

# Slurm opens --output/--error before this script runs, so logs/ must exist.
mkdir -p logs "$RUN_DIR"

if [ ! -d "$DATA_DIR/flair_1_toy_dataset/CSVs" ]; then
    echo "No FLAIR#1 toy dataset under $DATA_DIR." >&2
    echo "Stage it on the login node first:" >&2
    echo "  uv run ${UV_GROUP_ARGS[*]} python -m rsss.data.download --data-dir $DATA_DIR" >&2
    exit 1
fi

echo "Job ${SLURM_JOB_ID:-local} on $(hostname); data=$DATA_DIR runs=$RUN_DIR epochs=$EPOCHS"
uv run "${UV_GROUP_ARGS[@]}" --no-sync python -m rsss.train \
    --data-dir "$DATA_DIR" \
    --output-dir "$RUN_DIR" \
    --max-epochs "$EPOCHS" \
    --seed 0
