#!/bin/bash
#SBATCH --job-name=rsss-subset-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-user=mael.dacher.90@gmail.com
#SBATCH --mail-type=END
#
# Training run on PANDO over the FLAIR#1 subset: whole spatial domains,
# alphabetically first per split (5,275 train / 3,550 val / 2,250 test tiles,
# 512x512, 5 bands, 13.6 GiB), then an evaluation on the test split. It reports
# per-class IoU in the log and writes the metrics and checkpoints under RUN_DIR.
# The subset exists because PANDO's session cannot hold the full tree (95 GiB
# extracted, ~146 GiB free at peak).
#
# TRAIN_CONFIG selects the recipe and therefore the backbone: the U-Net
# baseline (configs/train.yaml, 30 epochs) by default, or the DINOv3 linear
# probe (configs/train_dinov3.yaml, 10 epochs). Each recipe's own max_epochs is
# used unless EPOCHS overrides it, and the cache check below follows the choice:
#
#   TRAIN_CONFIG=configs/train_dinov3.yaml RUN_DIR=... sbatch scripts/pando/job.sh
#
# PANDO runs RHEL 7.8 (glibc 2.17), driver 550.54.15 (CUDA 12.4) and Tesla V100
# GPUs, so its commands select the `pando` dependency group: it is the only fork
# of uv.lock that installs on that glibc, and torch 2.6/cu124 is the newest build
# that the driver and the V100s both support. Its compute nodes have no outbound
# network, so stage the environment, the dataset and the pretrained weights from
# the login node first:
#
#   uv sync --frozen --no-default-groups --no-dev --group pando
#   # The wheel cache outgrows the session: drop it once the env is synced.
#   uv cache clean
#   # Size the subset first (CSVs only), then stage its tiles.
#   SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
#       uv run --no-default-groups --group pando python -m rsss.data.download \
#       --dataset flair_1_subset --data-dir "$HOME/rsss-data" --report-only
#   SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
#       uv run --no-default-groups --group pando python -m rsss.data.download \
#       --dataset flair_1_subset --data-dir "$HOME/rsss-data"
#   # U-Net: smp's ImageNet encoder. DINOv3: timm's SAT-493M ViT (~1.2 GB).
#   SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
#       uv run --no-default-groups --group pando python -m rsss.models.weights \
#       --encoder resnet34
#   SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
#       uv run --no-default-groups --group pando python -m rsss.models.weights \
#       --dinov3
#
# The subset fetches only the kept tiles, one member at a time out of IGN's
# archives, so it needs 13.6 GiB of free space and no more at peak; point DATA_DIR
# wherever that much is available. --num-domains TRAIN,VALIDATION,TEST changes
# the recipe (default 3,2,2) and re-stages an existing tree to it in place --
# domains dropped by the new recipe are pruned, kept tiles are reused. The CSVs
# and the chosen domains are recorded in
# $DATA_DIR/flair_1_subset_dataset/subset_manifest.json.
#
# Then, from the repository root:
#   sbatch scripts/pando/job.sh
#
# A 30-epoch run can outlast one allocation. If it hits the walltime, submit
# again with the same RUN_DIR and resume from the last checkpoint:
#   DATA_DIR=/scratch/me/rsss-data RUN_DIR=/scratch/me/rsss-runs/unet-01 \
#       RESUME=last sbatch scripts/pando/job.sh
#
# Override the defaults with the environment:
#   DATA_DIR=/scratch/me/data RUN_DIR=/scratch/me/runs EPOCHS=2 sbatch scripts/pando/job.sh
# A full-dataset run, where the quota allows it, selects the full config:
#   DATA_CONFIG=configs/data_full.yaml DATASET=flair_1_full sbatch scripts/pando/job.sh
# A toy smoke run selects the toy config and caps the batches:
#   DATA_CONFIG=configs/data.yaml DATASET=flair_1_toy EPOCHS=1 \
#       LIMIT_TRAIN_BATCHES=2 sbatch scripts/pando/job.sh

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

# Data source: the subset by default, or configs/data_full.yaml for the full set.
DATA_CONFIG="${DATA_CONFIG:-configs/data_subset.yaml}"
DATASET="${DATASET:-flair_1_subset}"
case "$DATASET" in
    flair_1_toy) DATASET_DIR=flair_1_toy_dataset ;;
    flair_1_full) DATASET_DIR=flair_1_full_dataset ;;
    flair_1_subset) DATASET_DIR=flair_1_subset_dataset ;;
    *)
        echo "Unknown DATASET=$DATASET (expected flair_1_toy, flair_1_full or flair_1_subset)." >&2
        exit 1
        ;;
esac
# Root the datasets are assembled under (it holds $DATASET_DIR/).
DATA_DIR="${DATA_DIR:-$HOME/rsss-data}"
# Recipe: the U-Net baseline, or configs/train_dinov3.yaml for the DINOv3 probe.
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train.yaml}"
# Leave EPOCHS empty to use the recipe's own max_epochs (30 for the U-Net
# baseline, 10 for the probe); set it to override that budget.
EPOCHS="${EPOCHS:-}"
# Resume from "last" (or an explicit checkpoint) to finish a run across
# allocations; leave empty for a fresh fit.
RESUME="${RESUME:-}"
# U-Net encoder whose cached ImageNet weights the model build reads (smp's
# `encoder_name`).
ENCODER="${ENCODER:-resnet34}"
# DINOv3 backbone whose cached weights the probe build reads (timm's name).
DINOV3_MODEL="${DINOV3_MODEL:-vit_large_patch16_dinov3.sat493m}"
# Cache the pretrained weights resolve from: the login-node fetch has to land
# where the compute node reads it, and `$HOME` is shared between the two.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# The recipe names the backbone it builds, so it also decides which cache entry
# the check below looks for and how a run directory is named by default.
case "$TRAIN_CONFIG" in
    *dinov3*)
        RUN_PREFIX=dinov3
        WEIGHTS_GLOB="$HF_HOME/hub/models--timm--$DINOV3_MODEL"
        WEIGHTS_HINT="python -m rsss.models.weights --dinov3 $DINOV3_MODEL"
        ;;
    *)
        RUN_PREFIX=unet
        WEIGHTS_GLOB="$HF_HOME"/hub/models--*--"$ENCODER".*
        WEIGHTS_HINT="python -m rsss.models.weights --encoder $ENCODER"
        ;;
esac
# Run artefacts: never the repository, which may sit on a small quota.
RUN_DIR="${RUN_DIR:-$HOME/rsss-runs/$RUN_PREFIX-${SLURM_JOB_ID:-local}}"

# The group selection lives here because every uv command needs it: the default
# groups are `dev` and `gcp`, whose wheels need glibc 2.28 and would fail to
# install (or import) on PANDO.
UV_GROUP_ARGS=(--no-default-groups --no-dev --group pando)

# Slurm opens --output/--error before this script runs, so logs/ must exist.
mkdir -p logs "$RUN_DIR"

if [ ! -d "$DATA_DIR/$DATASET_DIR/CSVs" ]; then
    echo "No $DATASET dataset under $DATA_DIR/$DATASET_DIR." >&2
    echo "Stage it on the login node first:" >&2
    echo "  SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \\" >&2
    echo "    uv run ${UV_GROUP_ARGS[*]} python -m rsss.data.download \\" >&2
    echo "    --dataset $DATASET --data-dir $DATA_DIR" >&2
    exit 1
fi

if ! ls -d $WEIGHTS_GLOB > /dev/null 2>&1; then
    echo "No cached weights for $TRAIN_CONFIG under $HF_HOME/hub." >&2
    echo "Stage them on the login node first:" >&2
    echo "  SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \\" >&2
    echo "    uv run ${UV_GROUP_ARGS[*]} $WEIGHTS_HINT" >&2
    exit 1
fi

EXTRA_ARGS=()
if [ -n "${LIMIT_TRAIN_BATCHES:-}" ]; then
    EXTRA_ARGS+=(--limit-train-batches "$LIMIT_TRAIN_BATCHES")
fi
if [ -n "$RESUME" ]; then
    EXTRA_ARGS+=(--resume "$RESUME")
fi
if [ -n "$EPOCHS" ]; then
    EXTRA_ARGS+=(--max-epochs "$EPOCHS")
fi

echo "Job ${SLURM_JOB_ID:-local} on $(hostname); config=$TRAIN_CONFIG dataset=$DATASET data=$DATA_DIR runs=$RUN_DIR epochs=${EPOCHS:-recipe} resume=${RESUME:-no}"
# Guard the optional-array expansion: PANDO's bash 4.2 treats an empty
# "${EXTRA_ARGS[@]}" as an unbound variable under `set -u` (fixed in 4.4).
uv run "${UV_GROUP_ARGS[@]}" --no-sync python -m rsss.train \
    --config "$TRAIN_CONFIG" \
    --data-config "$DATA_CONFIG" \
    --data-dir "$DATA_DIR" \
    --output-dir "$RUN_DIR" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    --seed 0
