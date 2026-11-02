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

`configs/data.yaml` names the toy dataset used by tests and smoke runs,
`configs/data_full.yaml` the full FLAIR#1 dataset, and `configs/data_subset.yaml`
a whole-domain subset of it for clusters that cannot hold the full tree. The
pipeline downloads, extracts and verifies the selected dataset on first use:

```bash
# Toy (216 MiB)
uv run python -m rsss.data.download --dataset flair_1_toy --data-dir data

# Full FLAIR#1: four archives plus the three split CSVs, 95 GiB extracted
uv run python -m rsss.data.download --dataset flair_1_full --data-dir data

# Subset: the alphabetically first 3 train, 2 validation and 2 test domains
# (5,275 / 3,550 / 2,250 tiles, 13.6 GiB), fetched tile by tile -- no archive
# is ever staged, so peak disk equals the final tree.
uv run python -m rsss.data.download --dataset flair_1_subset --data-dir data \
    --report-only   # size it first, without fetching any tile
uv run python -m rsss.data.download --dataset flair_1_subset --data-dir data
# Change the recipe (domains per split, alphabetically first). The staged tree is
# reconciled in place: dropped domains are pruned, kept tiles are reused, only
# the domains the new recipe adds are fetched. No --force needed.
uv run python -m rsss.data.download --dataset flair_1_subset --data-dir data \
    --num-domains 5,4,10

# Check an already-staged tree without downloading
uv run python -m rsss.data.download --dataset flair_1_subset --data-dir data \
    --verify-only
```

The full dataset needs roughly 146 GiB free at peak: the largest archive is
51 GiB and is extracted before it is deleted. Stage it on a scratch filesystem
on the cluster. An interrupted staging run resumes, skipping archives already
extracted; pass `--force` to rebuild a tree that fails verification. The subset
needs only its final 13.6 GiB, and records the domains it kept in
`<data-dir>/flair_1_subset_dataset/subset_manifest.json`; changing
`--num-domains` reconciles the tree to the new recipe in place (see above), and
`--force` is what throws it away and starts over.

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

`configs/train_dinov3.yaml` is the comparison arm. It swaps the U-Net for
timm's DINOv3 ViT-L/16 distilled on SAT-493M, frozen, with the paper's linear
probe as head: a 1x1 convolution on the last block's patch map, upsampled 16x.
Only the head (1.3 M parameters) and the patch-embedding stem train, so the run
measures what the pretrained features already contain; `train_stem: true` is
what lets the zero-initialized NIR/nDSM filters learn, at 1e-4 rather than the
head's 1e-3. The recipe also swaps in SAT-493M's RGB normalization (NIR and
nDSM keep FLAIR's), and asks for 10 epochs at batch 8. Select it with
`--config`:

```bash
uv run python -m rsss.train --config configs/train_dinov3.yaml \
    --data-config configs/data_subset.yaml --data-dir data \
    --output-dir outputs/dinov3-probe-01
```

Two fine-tuned arms sit between that probe and training the whole backbone,
differing only in which parameters move. `configs/train_dinov3_bitfit.yaml`
adds every bias, every LayerNorm affine pair and every LayerScale vector
(0.3 M parameters) but no token-mixing weight. `configs/train_dinov3_last_blocks.yaml`
instead trains the last four blocks whole — attention, MLP, norms and scales,
51.7 M parameters — plus the trunk's final LayerNorm, at 1e-4 (`blocks_lr`)
against the head's 1e-3, and asks for 20 epochs rather than 10, since that many
weights at a tenth of the rate need longer to settle. That longer budget is the
one knob it does not share with the probe, so its mIoU is indicative rather than
a like-for-like comparison; a control at 10 epochs is the same command with
`--max-epochs 10`:

```bash
uv run python -m rsss.train --config configs/train_dinov3_last_blocks.yaml \
    --data-config configs/data_subset.yaml --data-dir data \
    --output-dir outputs/dinov3-last-blocks-01
```

Logs and checkpoints land under `--output-dir` (default `outputs/`). A run keeps
only two checkpoints, `last.ckpt` and the one with the best validation mIoU, so
its directory does not grow with the epoch count. It ends by scoring the test
split on the best-validation checkpoint, not on the weights the last epoch
left; add `--skip-test` to fit only, or `--test-checkpoint <path>` to score an
existing checkpoint without training. `--resume` alone continues from
`<output-dir>/checkpoints/last.ckpt`; pass a path instead to pick another
checkpoint. `--max-epochs` is always the total for the run, not the number of
extra epochs.

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
    --checkpoint  "outputs/pando/$RUN/checkpoints/last.ckpt" \
    --metrics     "outputs/pando/$RUN/csv" \
    --data-config configs/data_subset.yaml \
    --data-dir    data \
    --output-dir  "outputs/pando/$RUN/figures"
```

`predictions.png` shows a few test tiles as image / ground truth / prediction,
and, when `--metrics` is given, `results.png` draws the loss, mIoU and per-class
IoU curves. The plots need matplotlib, which lives in the `dev` group, so run
this step locally rather than on the cluster. `last.ckpt` holds the final
epoch's weights; the `<epoch>-<step>.ckpt` beside it is the best validation
mIoU, which is what the test split was scored with.

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
# The subset is 13.6 GiB and is fetched tile by tile; --report-only only reads
# the split CSVs and the remote archive tables, so run it first to check the
# numbers against the quota. The uv wheel cache (~6 GiB here) competes for the
# same session: `uv cache clean` after the sync frees it until the next
# dependency change.
SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
    uv run --no-default-groups --group pando python -m rsss.data.download \
    --dataset flair_1_subset --data-dir "$HOME/rsss-data" --report-only
SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
    uv run --no-default-groups --group pando python -m rsss.data.download \
    --dataset flair_1_subset --data-dir "$HOME/rsss-data"

# Stage the pretrained weights the model build fetches from the Hugging Face
# Hub, into the same cache the compute node reads ($HOME/.cache/huggingface).
# The U-Net reads smp's ImageNet encoder; the DINOv3 recipe reads timm's
# SAT-493M ViT-L/16 instead (~1.2 GB). Stage whichever one you train.
SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
    uv run --no-default-groups --group pando python -m rsss.models.weights \
    --encoder resnet34
SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
    uv run --no-default-groups --group pando python -m rsss.models.weights \
    --dinov3

# Then, from the repository root.
DATA_DIR=$HOME/rsss-data RUN_DIR=$HOME/rsss-runs/unet-subset-01 \
    sbatch scripts/pando/job.sh

# The DINOv3 linear probe instead: TRAIN_CONFIG selects the recipe, and the
# job follows it for the cache check and the default run directory.
DATA_DIR=$HOME/rsss-data RUN_DIR=$HOME/rsss-runs/dinov3-probe-01 \
    TRAIN_CONFIG=configs/train_dinov3.yaml sbatch scripts/pando/job.sh

# The BitFit+ and last-blocks arms read the same staged SAT-493M weights.
DATA_DIR=$HOME/rsss-data RUN_DIR=$HOME/rsss-runs/dinov3-last-blocks-01 \
    TRAIN_CONFIG=configs/train_dinov3_last_blocks.yaml sbatch scripts/pando/job.sh
```

`scripts/pando/job.sh` runs the U-Net baseline for 30 epochs over the FLAIR#1
subset by default (whole domains: 5,275 train, 3,550 val, 2,250 test tiles,
13.6 GiB), then scores the best-validation checkpoint on the test split. The
full dataset stays reachable with
`DATA_CONFIG=configs/data_full.yaml DATASET=flair_1_full`, where the quota
allows its ~146 GiB peak. It allocates 24 hours; if the run hits the walltime,
submit it again with the same `RUN_DIR` and `RESUME=last` to continue to epoch
30. It fails early if either the dataset or
the pretrained weights were not staged. The log lands in `logs/slurm-<jobid>.out`
and the checkpoints, CSV metrics and per-class IoU under `RUN_DIR`. Override
`DATA_DIR`, `RUN_DIR`, `TRAIN_CONFIG`, `EPOCHS`, `ENCODER`, `DINOV3_MODEL`,
`RESUME` or `TEST_CHECKPOINT` to change that; `EPOCHS` defaults to the recipe's
own `max_epochs`:

```bash
DATA_DIR=/scratch/me/rsss-data RUN_DIR=/scratch/me/runs/run-01 EPOCHS=2 \
    sbatch scripts/pando/job.sh

# Finish a run that reached the walltime.
DATA_DIR=/scratch/me/rsss-data RUN_DIR=/scratch/me/runs/run-01 RESUME=last \
    sbatch scripts/pando/job.sh

# Score an existing checkpoint on the test split without training, e.g. the
# best one of an earlier run, to compare arms on their best epoch.
DATA_DIR=/scratch/me/rsss-data RUN_DIR=/scratch/me/runs/run-01-eval \
    TEST_CHECKPOINT=/scratch/me/runs/run-01/checkpoints/epoch=6-step=4613.ckpt \
    TRAIN_CONFIG=configs/train_dinov3_bitfit.yaml sbatch scripts/pando/job.sh
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

## Run on GCP

Training runs on a single spot GPU VM (one NVIDIA L4) that GCS feeds and
collects from: the dataset, the code archive and the run artefacts all live in
one bucket, created in the zone's region so transfers to the VM stay local.
`configs/gcp.yaml` holds the project, zone, bucket, image and disk.

### One-time setup

Create the bucket and the service account the VM runs as, and grant it access to
the bucket. No key file is involved: the account comes from the instance
metadata.

```bash
gcloud services enable compute.googleapis.com storage.googleapis.com \
    --project rsss-project

gcloud storage buckets create gs://rsss-data-bucket \
    --project rsss-project --location=europe-west4 \
    --uniform-bucket-level-access

gcloud iam service-accounts create rsss-vm --project rsss-project
gcloud storage buckets add-iam-policy-binding gs://rsss-data-bucket \
    --member=serviceAccount:rsss-vm@rsss-project.iam.gserviceaccount.com \
    --role=roles/storage.objectAdmin
```

Check the zone has L4 capacity and the project has GPU quota (`NVIDIA_L4_GPUS`,
and `PREEMPTIBLE_NVIDIA_L4_GPUS` for spot), then edit `configs/gcp.yaml` if any
of the values differ.

### Submit a run

The VM runs the committed HEAD, so commit first: the launcher refuses a dirty
worktree. The first run also stages the full dataset from IGN into GCS
(~95 GiB), which takes a while; every later run pulls it from the bucket
instead. The boot disk (200 GB in `configs/gcp.yaml`) holds the extracted tree
and, at peak, the archives it is extracted from.

```bash
RUN=unet-gcp-01 EPOCHS=30 scripts/gcp/submit.sh

# A smoke run on the toy dataset: one epoch over two batches.
scripts/gcp/submit.sh --run unet-smoke --epochs 1 --limit-train-batches 2 \
    --data-config configs/data.yaml

# Print the gcloud commands without uploading or creating anything.
scripts/gcp/submit.sh --run unet-gcp-01 --dry-run
```

The launcher archives the commit, uploads it to `gs://<bucket>/code/<sha>.tar.gz`
once per commit, renders the VM's startup script and creates the instance. It
prints the commands that follow it:

```bash
gcloud compute ssh rsss-unet-gcp-01 --project rsss-project \
    --zone europe-west4-a -- tail -f /var/log/rsss-startup.log
gcloud storage ls -r gs://rsss-data-bucket/runs/unet-gcp-01/
```

If the VM dies before the log exists, read the boot console instead:

```bash
gcloud compute instances get-serial-port-output rsss-unet-gcp-01 \
    --project rsss-project --zone europe-west4-a
```

A spot VM can be preempted at any time. The job uploads the run directory every
10 minutes, and a checkpoint only becomes `last.ckpt` after its upload has
completed, so a resubmission resumes cleanly:

```bash
RUN=unet-gcp-01 RESUME=last EPOCHS=30 scripts/gcp/submit.sh
```

The VM shuts itself down when the job ends, whatever the outcome; delete it to
stop paying for the boot disk. `configs/gcp.yaml` also caps the run with
`max_run_duration`, after which the instance is deleted outright.

### Inspect a run

```bash
scripts/gcp/pull_run.sh unet-gcp-01
uv run python -m rsss.predict \
    --checkpoint  outputs/gcp/unet-gcp-01/checkpoints/last.ckpt \
    --metrics     outputs/gcp/unet-gcp-01/csv \
    --data-config configs/data_full.yaml \
    --data-dir    data \
    --output-dir  outputs/gcp/unet-gcp-01/figures
```
