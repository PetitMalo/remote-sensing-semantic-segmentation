# A remote sensing semantic segmentation project

Fine-tune a DINO vision backbone for land-cover segmentation over IGN's OCS GE
labels, and measure how much it improves over a classic U-Net/ResNet baseline.

## Setup

```bash
uv sync    # local and GCP: the latest compatible stack
```

`uv.lock` carries two mutually exclusive runtime stacks, selected by dependency
group. The default `gcp` group tracks the latest releases; the `pando` group is
pinned to the last wheels that load on the cluster's RHEL 7.8 (glibc 2.17),
CUDA 12.4 driver and Tesla V100 GPUs. A change can be checked against the stack
training actually uses, in an environment of its own:

```bash
UV_PROJECT_ENVIRONMENT=.venv-pando \
    uv run --no-default-groups --group dev --group pando pytest -q
```

Give the `pando` stack its own environment rather than switching the main one:
the two stacks ship different `nvidia-*` wheels that install into the same
directories, so switching back can leave `libcudnn.so.9` deleted, which uv does
not notice (it tracks packages by metadata, not by file). If the main
environment ends up in that state, `uv sync --reinstall-package
nvidia-cudnn-cu13` repairs it.

`tests/test_lock_platform.py` fails when a `pando` package starts needing a newer
glibc, which a modern host would otherwise only discover on the cluster.

## Checks

```bash
uv run ruff check . --fix
uv run ruff format .
uv run ty check
uv run pytest -q
```

## Data

`configs/data.yaml` names the toy dataset used by tests and smoke runs, and
`configs/data_full.yaml` the full FLAIR#1 dataset. The pipeline downloads,
extracts and verifies the selected dataset on first use:

```bash
# Toy (216 MiB)
uv run python -m rsss.data.download --dataset flair_1_toy --data-dir data

# Full FLAIR#1: four archives plus the three split CSVs, ~65 GiB extracted
uv run python -m rsss.data.download --dataset flair_1_full --data-dir data

# Check an already-staged tree without downloading
uv run python -m rsss.data.download --dataset flair_1_full --data-dir data \
    --verify-only
```

The full dataset needs roughly 120 GiB free at peak: the largest archive is
51 GiB and is extracted before it is deleted. Stage it on a scratch filesystem
on the cluster. An interrupted staging run resumes, skipping archives already
extracted; pass `--force` to rebuild a tree that fails verification.

## Train

`configs/train.yaml` holds the recipe (model, loss, optimizer, schedule, Trainer)
and `configs/data.yaml` the data source; the flags only override per-run choices.

```bash
uv run python -m rsss.train --help
uv run python -m rsss.train --fast-dev-run    # one batch; no artefacts written
uv run python -m rsss.train --max-epochs 1    # a real, short run

# Full FLAIR#1 with the official normalization
uv run python -m rsss.train --data-config configs/data_full.yaml \
    --data-dir /scratch/me/rsss-data --output-dir /scratch/me/runs/unet-full-01

# Continue an interrupted run to its total epoch count
uv run python -m rsss.train --data-config configs/data_full.yaml \
    --data-dir /scratch/me/rsss-data --output-dir /scratch/me/runs/unet-full-01 \
    --max-epochs 30 --resume
```

Logs and checkpoints land under `--output-dir` (default `outputs/`). A run keeps
only two checkpoints, `last.ckpt` and the one with the best validation mIoU, so
its directory does not grow with the epoch count. It ends by scoring the test
split, on the weights `last.ckpt` holds; add `--skip-test` to fit only.
`--resume` alone continues from `<output-dir>/checkpoints/last.ckpt`; pass a
path instead to pick another checkpoint. `--max-epochs` is always the total for
the run, not the number of extra epochs.

## Inspect a run

The compute nodes have no outbound network, so a finished run travels through
the Scaleway bucket: upload it from the login node, pull it locally, then draw
the predictions and the training curves. The bucket is the `rclone` remote
`scw` (run `rclone listremotes` to confirm the name), and the run directory on
PANDO is `~/rsss-runs/unet-<jobid>`:

```bash
# PANDO login node, in the repository clone: push the newest run (or set
# RUN_DIR to pick one).
scripts/pando/upload_run.sh

# Locally, in the repository clone: pull that run into gitignored outputs/.
RUN=unet-4022046
rclone copy "scw:rsss-bucket/runs/$RUN/" "outputs/pando/$RUN/" --progress

uv run python -m rsss.predict \
    --checkpoint "outputs/pando/$RUN/checkpoints/last.ckpt" \
    --metrics    "outputs/pando/$RUN/csv" \
    --data-dir   data \
    --output-dir "outputs/pando/$RUN/figures"
```

`predictions.png` shows a few test tiles as image / ground truth / prediction,
and, when `--metrics` is given, `results.png` draws the loss, mIoU and per-class
IoU curves. The plots need matplotlib, which lives in the `dev` group, so run
this step locally rather than on the cluster. `last.ckpt` holds the weights the
test split was scored with; the `<epoch>-<step>.ckpt` beside it is the best
validation mIoU, if you would rather predict from that one.

## Run on PANDO

PANDO's compute nodes have no outbound network, so stage the environment, the
dataset and the encoder weights from the login node, then submit. Every command
there selects the `pando` group, because the default stack does not install on
PANDO's glibc:

```bash
# Login node, in the repository clone.
uv sync --frozen --no-default-groups --no-dev --group pando

# Stage the dataset. Two trust stores are in play: uv carries its own CA set,
# and so installs happily, while the CPython it installs trusts only its own
# bundled certificates and does not see the institute CA that PANDO's HTTPS
# interception is signed with. SSL_CERT_FILE points Python at the system
# bundle; without it the download stops at "CERTIFICATE_VERIFY_FAILED:
# self-signed certificate in certificate chain". If that CA lives outside the
# bundle, point the variable at the .pem the IT service provides instead.
# Point DATA_DIR at a scratch filesystem: the full dataset is ~65 GiB extracted.
SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
    uv run --no-default-groups --group pando python -m rsss.data.download \
    --dataset flair_1_full --data-dir "$HOME/rsss-data"

# Stage the encoder weights the model build fetches from the Hugging Face Hub,
# into the same cache the compute node reads ($HOME/.cache/huggingface).
SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
    uv run --no-default-groups --group pando python -m rsss.models.weights \
    --encoder resnet34

# Then, from the repository root.
DATA_DIR=$HOME/rsss-data RUN_DIR=$HOME/rsss-runs/unet-full-01 \
    sbatch scripts/pando/job.sh
```

`scripts/pando/job.sh` runs the U-Net baseline for 30 epochs over the full
FLAIR#1 dataset by default, then evaluates on the test split. It allocates 24
hours; if the run hits the walltime, submit it again with the same `RUN_DIR` and
`RESUME=last` to continue to epoch 30. It fails early if either the dataset or
the encoder weights were not staged. The log lands in `logs/slurm-<jobid>.out`
and the checkpoints, CSV metrics and per-class IoU under `RUN_DIR`. Override
`DATA_DIR`, `RUN_DIR`, `EPOCHS`, `ENCODER` or `RESUME` to change that:

```bash
DATA_DIR=/scratch/me/rsss-data RUN_DIR=/scratch/me/runs/run-01 EPOCHS=2 \
    sbatch scripts/pando/job.sh

# Finish a run that reached the walltime.
DATA_DIR=/scratch/me/rsss-data RUN_DIR=/scratch/me/runs/run-01 RESUME=last \
    sbatch scripts/pando/job.sh
```

For a toy smoke test of the GPU path, select the toy config and cap the batches:

```bash
DATA_CONFIG=configs/data.yaml DATASET=flair_1_toy EPOCHS=1 \
    LIMIT_TRAIN_BATCHES=2 sbatch scripts/pando/job.sh
```

Staging the dataset is the only step that needs the network. If the download
cannot be made to work at all, copy the archives in by other means:
`ensure_dataset` accepts a complete toy archive at
`$DATA_DIR/flair_1_toy_dataset.zip` (226 168 911 bytes), and for the full
dataset it reuses any of the four archives whose size matches, so a manually
copied `flair_aerial_train.zip` (54 458 900 252 bytes) is picked up too.

Once a job has finished, `scripts/pando/upload_run.sh` (login node) copies its
run directory and Slurm log to the bucket, ready to pull locally: see
[Inspect a run](#inspect-a-run).

To run an interactive session:

```bash
srun --pty --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=01:00:00 /bin/bash
```
