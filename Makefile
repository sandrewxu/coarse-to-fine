.PHONY: install lint format test clean

# `uv run --no-sync` runs the command in the existing .venv WITHOUT re-resolving
# the dep graph. We do this for every target because the project's GPU deps
# (vllm, flash-attn, verl) require CUDA at build time — running `uv run` (which
# syncs first) on a CPU-only login node would try to rebuild them and fail.
# To refresh the lock file, run `uv lock` on a GPU node where CUDA_HOME is set.

UV ?= uv run --no-sync

# Install dev tooling into an existing venv (assumes GPU deps already present).
# For a fresh install, run on a GPU node so vllm/flash-attn can build.
install:
	uv pip install -e . --no-deps
	uv pip install ruff pytest pytest-cov pre-commit nbstripout
	$(UV) pre-commit install

# Check (don't fix) lint + format.
lint:
	$(UV) ruff check .
	$(UV) ruff format --check .

# Auto-fix lint + format.
format:
	$(UV) ruff format .
	$(UV) ruff check --fix .

# Run the unit test suite.
test:
	$(UV) pytest

# Remove build / cache artifacts.
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
