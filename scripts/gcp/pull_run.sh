#!/bin/bash
# Pull a finished GCP run from the bucket, for local inspection.
#
#   scripts/gcp/pull_run.sh unet-20261019-120000
#
# The run lands in gitignored outputs/gcp/<run>/. Then predict and plot from it
# with the rsss.predict command the script prints.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${GCP_CONFIG:-configs/gcp.yaml}"
DATA_CONFIG="${DATA_CONFIG:-configs/data_full.yaml}"
RUN="${1:-${RUN:-}}"
if [ -z "$RUN" ]; then
    echo "usage: $0 RUN" >&2
    exit 1
fi
BUCKET="${BUCKET:-$(uv run python -c "from rsss.gcp.config import load_gcp_config; print(load_gcp_config('$CONFIG').bucket)")}"
DEST="outputs/gcp/$RUN"

mkdir -p "$DEST"
gcloud storage rsync --recursive "gs://$BUCKET/runs/$RUN" "$DEST"

cat <<EOF

Pulled gs://$BUCKET/runs/$RUN -> $DEST
Score and plot it locally with:
  uv run python -m rsss.predict \\
      --checkpoint "$DEST/checkpoints/last.ckpt" \\
      --metrics "$DEST/csv" \\
      --data-config "$DATA_CONFIG" \\
      --data-dir data \\
      --output-dir "$DEST/figures"
EOF
