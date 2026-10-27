# AGENTS.md

## Project
Fine-tune a DINO vision backbone for semantic segmentation of land cover,
using IGN's OCS GE (Occupation du Sol à Grande Échelle) as labels over
orthophoto tiles. Goal: measure how much a DINO backbone improves over a
classic baseline (U-Net/ResNet) and how much fine-tuning is needed.
Current scope: land cover (couverture) only, one small area first.

## Stack
- Python, managed with uv (never use pip directly)
- Lint/format: ruff. Types: ty. Tests: pytest. Hooks: pre-commit
- Training/eval: GPU jobs on ISAE-SUPAERO's PANDO cluster via Slurm (now);
  Google Cloud Platform (GCS for data, GPU VM for runs) later, once the PANDO
  runs have proved the pipeline
- Two runtime stacks in `uv.lock`, selected by dependency group: `gcp` (a
  default, latest releases) and `pando` (RHEL 7.8 / glibc 2.17, CUDA 12.4
  driver, V100 GPUs). A package whose version differs between the two targets
  goes in both groups, never in `[project.dependencies]`, which applies to
  every fork. `tests/test_lock_platform.py` guards the PANDO side
- DINO backbone: DINOv3 ViT-L/16 distilled on SAT-493M, through timm (the
  reference repo needs torch >= 2.7.1, PANDO caps at 2.6). Frozen probe first,
  with a 1x1 convolutional head; `configs/train_dinov3.yaml` is the recipe

## Commands
- Install: `uv sync`
- Per-target install: `uv sync --frozen --no-default-groups --no-dev --group pando` (PANDO);
  local and GCP use the default `gcp` group
- Lint: `uv run ruff check . --fix` / Format: `uv run ruff format .`
- Types: `uv run ty check`
- Tests: `uv run pytest -q` (the `pando` stack too when dependencies change:
  `UV_PROJECT_ENVIRONMENT=.venv-pando uv run --no-default-groups --group dev
  --group pando pytest -q`, in an environment of its own, since the two stacks
  install different `nvidia-*` wheels into the same directories and switching
  the main one can delete files uv does not know it owns)
- All checks: `uv run pre-commit run --all-files`
Run lint, types and tests before declaring any task done.

## Compute
Development happens locally, training does not. PANDO (ISAE-SUPAERO) is the GPU
target now; GCP comes later, once these runs have proved the pipeline.
1. Log in to PANDO, clone the repo, and prepare the environment from the login
   node (compute nodes have no internet). Every command there selects the
   `pando` group, since the default stack needs glibc 2.28:
   `uv sync --frozen --no-default-groups --no-dev --group pando`, then stage the
   dataset and the pretrained weights with
   `uv run --no-default-groups --group pando python -m rsss.data.download --data-dir <root>`
   and `uv run --no-default-groups --group pando python -m rsss.models.weights`
   (`--encoder resnet34` for the U-Net, `--dinov3` for the probe).
2. Submit a run from the repository root: `sbatch scripts/pando/job.sh`. It
   takes the dataset root, run directory, recipe and epoch count from
   `DATA_DIR`, `RUN_DIR`, `TRAIN_CONFIG` and `EPOCHS`, and writes logs and
   checkpoints outside the repo.
3. Read the results in `logs/slurm-<jobid>.out` and the run directory.
See the `pando-gpu` skill for the batch-script conventions. Never launch jobs
yourself: give me the `sbatch` (or `gcloud`) command instead.

## Code conventions
- Small, pure, single-purpose functions; no hidden global state
- Type hints on all public functions; short docstrings (what, not how)
- Reuse before writing: check `src/` for an existing helper first
- No premature abstraction: three similar lines beat a speculative framework
- Config in YAML files, never hard-coded paths or hyperparameters
- Every new function gets a pytest test in `tests/` (mirror the `src/rsss/` layout)

## Git workflow
- Never commit to `main`. One feature branch per task: `feat/...`, `fix/...`
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`
- Small, atomic commits. Do not bypass hooks (`--no-verify` is forbidden)
- Small Pull Requests. Above 300 lines of code is too much for a single PR to be manually reviewed by human.

## Working agreement
- Work in small tasks. For each: state a brief plan, implement, add tests,
  run the checks, then summarise what changed
- Ask before: adding a dependency, changing the project layout, or touching
  anything outside the task's scope
- If requirements are ambiguous, ask one question rather than guessing

## Boundaries (never do)
- Never commit data, model weights, credentials or `.env` files
- Never launch GCP or PANDO jobs or create cloud resources; give me the command instead
- Never download full datasets; use small samples for tests and smoke runs
- Never fabricate metrics or results; report only what was actually run

## Data notes
- Labels: OCS GE vector (open licence, IGN). Rasterise onto orthophoto tiles
- Classes are heavily imbalanced: report per-class IoU, not only mIoU
- TODO: tile size, area of interest, train/val/test split strategy

## Pull Request Workflow
When instructed to create a Pull Request (PR):
1. **Push Branch:** Push the current feature/fix branch to remote (`git push -u origin <branch-name>`).
2. **Run GitHub CLI:** Create the PR using `gh pr create` via terminal execution (`!`).
3. **PR Description Template:** Ensure the PR body contains:
   - **Summary:** High-level summary of the implementation.
   - **Key Changes:** Bulleted list of added/modified/deleted files or features.
   - **Verification:** Outline tests and pre-commit checks executed before pushing (`uv run pre-commit run --all-files`, `uv run pytest`).
   - **Reviewer Focus:** Specific design choices or logic needing human attention (e.g., DINO backbone choices, OCS GE class handling, raster transformations).
