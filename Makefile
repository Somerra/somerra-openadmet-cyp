.PHONY: setup lint test download verify-data audit baseline-activity baseline-tdi

CONDA_ENV ?= somerra-openadmet-cyp
CONDA_RUN = conda run --no-capture-output -n $(CONDA_ENV)

setup:
	conda env update --file environment.yml --prune
	$(CONDA_RUN) python -m pip install --no-build-isolation --no-deps -e .

lint:
	$(CONDA_RUN) ruff check .
	$(CONDA_RUN) ruff format --check .

test:
	$(CONDA_RUN) pytest --cov=somerra_openadmet --cov-report=term-missing

download:
	$(CONDA_RUN) openadmet-cyp download --output data/raw

verify-data:
	$(CONDA_RUN) openadmet-cyp verify-data --data-dir data/raw

audit:
	$(CONDA_RUN) openadmet-cyp audit --data-dir data/raw

baseline-activity:
	$(CONDA_RUN) openadmet-cyp baseline activity --data-dir data/raw --output-dir runs/activity-lgbm

baseline-tdi:
	$(CONDA_RUN) openadmet-cyp baseline tdi --data-dir data/raw --output-dir runs/tdi-lgbm
