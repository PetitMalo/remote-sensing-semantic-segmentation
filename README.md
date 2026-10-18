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

`configs/data.yaml` names the dataset and where it is extracted. The pipeline
downloads, extracts and verifies it on first use; by hand:

```bash
uv run python -m rsss.data.download --data-dir data
```

## Train

`configs/train.yaml` holds the recipe (model, loss, optimizer, schedule, Trainer)
and `configs/data.yaml` the data source; the flags only override per-run choices.

```bash
uv run python -m rsss.train --help
uv run python -m rsss.train --fast-dev-run    # one batch; no artefacts written
uv run python -m rsss.train --max-epochs 1    # a real, short run
```

Logs and checkpoints land under `--output-dir` (default `outputs/`).

## Run on PANDO

PANDO's compute nodes have no outbound network, so stage the environment and the
dataset from the login node, then submit. Every command there selects the
`pando` group, because the default stack does not install on PANDO's glibc:

```bash
uv sync --frozen --no-default-groups --no-dev --group pando
uv run --no-default-groups --group pando python -m rsss.data.download \
    --data-dir "$HOME/rsss-data"
sbatch scripts/pando/job.sh
```

`scripts/pando/job.sh` runs one epoch of the U-Net baseline over the FLAIR#1 toy
split and writes its log to `logs/slurm-<jobid>.out` and its checkpoints to
`$HOME/rsss-runs/`. Override `DATA_DIR`, `RUN_DIR` or `EPOCHS` to change that.
