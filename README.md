# A remote sensing semantic segmentation project

Fine-tune a DINO vision backbone for land-cover segmentation over IGN's OCS GE
labels, and measure how much it improves over a classic U-Net/ResNet baseline.

## Setup

```bash
uv sync
```

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
dataset from the login node, then submit:

```bash
uv sync --frozen
uv run python -m rsss.data.download --data-dir "$HOME/rsss-data"
sbatch scripts/pando/job.sh
```

`scripts/pando/job.sh` runs one epoch of the U-Net baseline over the FLAIR#1 toy
split and writes its log to `logs/slurm-<jobid>.out` and its checkpoints to
`$HOME/rsss-runs/`. Override `DATA_DIR`, `RUN_DIR` or `EPOCHS` to change that.
