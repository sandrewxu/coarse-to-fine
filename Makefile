.PHONY: install lint format test clean

# Install dev tooling and set up pre-commit hooks.
install:
	uv sync --extra dev
	uv run pre-commit install

# Check (don't fix) lint + format.
lint:
	uv run ruff check .
	uv run ruff format --check .

# Auto-fix lint + format.
format:
	uv run ruff format .
	uv run ruff check --fix .

# Run the unit test suite.
test:
	uv run pytest

# Remove build / cache artifacts.
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
