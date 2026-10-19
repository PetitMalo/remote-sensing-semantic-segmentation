#!/bin/bash
# Submit one training run to a spot L4 VM on GCP.
#
# The VM runs the committed HEAD, so commit first. The first run also stages
# the full dataset from IGN into GCS; later runs pull it from the bucket.
#
#   scripts/gcp/submit.sh                          # unet-<timestamp>, 30 epochs
#   RUN=unet-01 EPOCHS=30 scripts/gcp/submit.sh    # named run
#   scripts/gcp/submit.sh --run unet-01 --dry-run  # print the gcloud commands
#   RUN=unet-01 RESUME=1 scripts/gcp/submit.sh     # continue after preemption
#
# See configs/gcp.yaml for the project, zone, bucket, image and disk.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

exec uv run python -m rsss.gcp.submit "$@"
