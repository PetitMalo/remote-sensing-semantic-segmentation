#!/bin/bash
# Copy a finished run from the PANDO login node to the Scaleway bucket.
#
# The compute nodes have no outbound network, so a job cannot upload its own
# artefacts. Run this on the login node once the job has finished, then pull the
# result locally with the `rclone copy` in the README's "Inspect a run".
#
# rclone has to know the bucket: the remote defaults to `scw` and the bucket to
# `rsss-bucket`, and artefacts land under `runs/<run name>/`.
#
#   scripts/pando/upload_run.sh                                  # newest run
#   RUN_DIR=~/rsss-runs/unet-4022046 scripts/pando/upload_run.sh # a given one
#   RCLONE_REMOTE=scw RCLONE_BUCKET=rsss-bucket scripts/pando/upload_run.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Remote and bucket as configured on both PANDO and the local machine.
RCLONE_REMOTE="${RCLONE_REMOTE:-scw}"
RCLONE_BUCKET="${RCLONE_BUCKET:-rsss-bucket}"

if ! command -v rclone > /dev/null 2>&1; then
    echo "rclone is not on PATH; load it before uploading." >&2
    exit 1
fi

# Default to the most recently written run, so the common case needs no input.
RUN_DIR="${RUN_DIR:-}"
if [ -z "$RUN_DIR" ]; then
    RUN_DIR="$(ls -1dt "$HOME"/rsss-runs/*/ 2> /dev/null | head -n 1 || true)"
fi
if [ -z "$RUN_DIR" ]; then
    echo "No run directory under $HOME/rsss-runs; set RUN_DIR to upload one." >&2
    exit 1
fi
if [ ! -d "$RUN_DIR" ]; then
    echo "RUN_DIR is not a directory: $RUN_DIR" >&2
    exit 1
fi
RUN_NAME="$(basename "$RUN_DIR")"
DEST="$RCLONE_REMOTE:$RCLONE_BUCKET/runs/$RUN_NAME"

echo "Uploading $RUN_DIR -> $DEST"
rclone copy "$RUN_DIR" "$DEST" --progress

# The Slurm log lives in the repository, not the run directory; send the one
# named after the run's job id when the run was named `unet-<jobid>`.
JOB_ID="${RUN_NAME##*-}"
for suffix in out err; do
    LOG="$REPO_ROOT/logs/slurm-$JOB_ID.$suffix"
    if [ -f "$LOG" ]; then
        rclone copyto "$LOG" "$DEST/logs/slurm-$JOB_ID.$suffix" --progress
    fi
done

echo "Done. Pull it locally with:"
echo "  rclone copy $DEST/ outputs/pando/$RUN_NAME/ --progress"
