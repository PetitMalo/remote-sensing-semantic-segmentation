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
- Training/eval on Google Cloud Platform (GCS for data, GPU VM for runs)
- TODO: DINO version (DINOv2 or DINOv3), to decide before the model code

## Commands
- Install: `uv sync`
- Lint: `uv run ruff check . --fix` / Format: `uv run ruff format .`
- Types: `uv run ty check`
- Tests: `uv run pytest -q`
- All checks: `uv run pre-commit run --all-files`
Run lint, types and tests before declaring any task done.

## Code conventions
- Small, pure, single-purpose functions; no hidden global state
- Type hints on all public functions; short docstrings (what, not how)
- Reuse before writing: check `src/` for an existing helper first
- No premature abstraction: three similar lines beat a speculative framework
- Config in YAML files, never hard-coded paths or hyperparameters
- Every new function gets a pytest test in `tests/` (mirror the `src/` layout)
- TODO: package name and layout (src/<name>/ with data, models, train, eval)

## Git workflow
- Never commit to `main`. One feature branch per task: `feat/...`, `fix/...`
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`
- Small, atomic commits. Do not bypass hooks (`--no-verify` is forbidden)

## Working agreement
- Work in small tasks. For each: state a brief plan, implement, add tests,
  run the checks, then summarise what changed
- Ask before: adding a dependency, changing the project layout, or touching
  anything outside the task's scope
- If requirements are ambiguous, ask one question rather than guessing

## Boundaries (never do)
- Never commit data, model weights, credentials or `.env` files
- Never launch GCP jobs or create cloud resources; give me the command instead
- Never download full datasets; use small samples for tests and smoke runs
- Never fabricate metrics or results; report only what was actually run

## Data notes
- Labels: OCS GE vector (open licence, IGN). Rasterise onto orthophoto tiles
- Classes are heavily imbalanced: report per-class IoU, not only mIoU
- TODO: tile size, area of interest, train/val/test split strategy
