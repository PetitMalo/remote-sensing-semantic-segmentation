#!/bin/bash
#SBATCH --job-name=rsss-unet-full-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-user=mael.dacher.90@gmail.com
#SBATCH --mail-type=END
#
# Full training run on PANDO: the U-Net baseline over the FLAIR#1 toy split
# (156 train / 44 val tiles, 512x512, 5 bands) for the recipe's epochs, then an
# evaluation on the test split. It reports per-class IoU in the log and writes
# the metrics and checkpoints under RUN_DIR.
#
# PANDO runs RHEL 7.8 (glibc 2.17), driver 550.54.15 (CUDA 12.4) and Tesla V100
# GPUs, so its commands select the `pando` dependency group: it is the only fork
# of uv.lock that installs on that glibc, and torch 2.6/cu124 is the newest build
# that the driver and the V100s both support. Its compute nodes have no outbound
# network, so stage the environment, the dataset and the encoder weights from
# the login node first:
#
#   uv sync --frozen --no-default-groups --no-dev --group pando
#   SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
#       uv run --no-default-groups --group pando python -m rsss.data.download \
#       --data-dir "$HOME/rsss-data"
#   SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
#       uv run --no-default-groups --group pando python -m rsss.models.weights \
#       --encoder resnet34
#
# Then, from the repository root:
#   sbatch scripts/pando/job.sh
#
# Override the defaults with the environment:
#   DATA_DIR=/scratch/me/data RUN_DIR=/scratch/me/runs EPOCHS=2 sbatch scripts/pando/job.sh
# A smoke run caps the batches instead of the epochs:
#   EPOCHS=1 LIMIT_TRAIN_BATCHES=2 sbatch scripts/pando/job.sh

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

# Root the datasets are extracted into (it holds flair_1_toy_dataset/).
DATA_DIR="${DATA_DIR:-$HOME/rsss-data}"
# Run artefacts: never the repository, which may sit on a small quota.
RUN_DIR="${RUN_DIR:-$HOME/rsss-runs/unet-${SLURM_JOB_ID:-local}}"
# The recipe's own max_epochs is used when EPOCHS is unset; keep the default in
# step with configs/train.yaml.
EPOCHS="${EPOCHS:-30}"
# Encoder whose cached weights the model build needs (smp's `encoder_name`).
ENCODER="${ENCODER:-resnet34}"
# Cache the pretrained weights resolve from: the login-node fetch has to land
# where the compute node reads it, and `$HOME` is shared between the two.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# The group selection lives here because every uv command needs it: the default
# groups are `dev` and `gcp`, whose wheels need glibc 2.28 and would fail to
# install (or import) on PANDO.
UV_GROUP_ARGS=(--no-default-groups --no-dev --group pando)

# Slurm opens --output/--error before this script runs, so logs/ must exist.
mkdir -p logs "$RUN_DIR"

if [ ! -d "$DATA_DIR/flair_1_toy_dataset/CSVs" ]; then
    echo "No FLAIR#1 toy dataset under $DATA_DIR." >&2
    echo "Stage it on the login node first:" >&2
    echo "  SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \\" >&2
    echo "    uv run ${UV_GROUP_ARGS[*]} python -m rsss.data.download --data-dir $DATA_DIR" >&2
    exit 1
fi

if ! ls -d "$HF_HOME"/hub/models--*--"$ENCODER".* > /dev/null 2>&1; then
    echo "No cached $ENCODER weights under $HF_HOME/hub." >&2
    echo "Stage them on the login node first:" >&2
    echo "  SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \\" >&2
    echo "    uv run ${UV_GROUP_ARGS[*]} python -m rsss.models.weights --encoder $ENCODER" >&2
    exit 1
fi

EXTRA_ARGS=()
if [ -n "${LIMIT_TRAIN_BATCHES:-}" ]; then
    EXTRA_ARGS+=(--limit-train-batches "$LIMIT_TRAIN_BATCHES")
fi

echo "Job ${SLURM_JOB_ID:-local} on $(hostname); data=$DATA_DIR runs=$RUN_DIR epochs=$EPOCHS"
# Guard the optional-array expansion: PANDO's bash 4.2 treats an empty
# "${EXTRA_ARGS[@]}" as an unbound variable under `set -u` (fixed in 4.4).
uv run "${UV_GROUP_ARGS[@]}" --no-sync python -m rsss.train \
    --data-dir "$DATA_DIR" \
    --output-dir "$RUN_DIR" \
    --max-epochs "$EPOCHS" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    --seed 0
