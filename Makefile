.PHONY: setup setup-conda lint test download verify-data audit baseline-activity baseline-tdi

setup:
	uv sync --dev

setup-conda:
	conda env create -f environment.yml
	conda run -n somerra-openadmet-cyp python -m pip install --no-deps -e .

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest --cov=somerra_openadmet --cov-report=term-missing

download:
	uv run openadmet-cyp download --output data/raw

verify-data:
	uv run openadmet-cyp verify-data --data-dir data/raw

audit:
	uv run openadmet-cyp audit --data-dir data/raw

baseline-activity:
	uv run openadmet-cyp baseline activity --data-dir data/raw --output-dir runs/activity-lgbm

baseline-tdi:
	uv run openadmet-cyp baseline tdi --data-dir data/raw --output-dir runs/tdi-lgbm
