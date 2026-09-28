.PHONY: setup lint test download audit baseline-activity baseline-tdi

setup:
	uv sync --dev

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest --cov=somerra_openadmet --cov-report=term-missing

download:
	uv run openadmet-cyp download --output data/raw

audit:
	uv run openadmet-cyp audit --data-dir data/raw

baseline-activity:
	uv run openadmet-cyp baseline activity --data-dir data/raw --output-dir runs/activity-lgbm

baseline-tdi:
	uv run openadmet-cyp baseline tdi --data-dir data/raw --output-dir runs/tdi-lgbm
