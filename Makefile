.PHONY: help install test run run-date clean format lint

START_DATE ?= 2024-10-01
END_DATE ?= 2024-10-21

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make run        - Run the pipeline"
	@echo "  make clean      - Clean generated files"
	@echo "  make format     - Format code with black"
	@echo "  make lint       - Lint code"

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src --cov-report=html

run:
	python -m pipeline run --start-date $(START_DATE) --end-date $(END_DATE)

run-date:
	python -m pipeline run --start-date $(DATE) --end-date $(DATE)

clean:
	python -c "import shutil, pathlib; paths=[pathlib.Path('outputs'), pathlib.Path('reports/qa'), pathlib.Path('reports/metrics'), pathlib.Path('reports/validation')]; [shutil.rmtree(p, ignore_errors=True) for p in paths]; [p.mkdir(parents=True, exist_ok=True) for p in paths]"
	python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"

format:
	black src/ tests/

lint:
	flake8 src/ tests/
	mypy src/

setup-dirs:
	mkdir -p data/raw data/clean data/features
	mkdir -p outputs/preds_baseline outputs/preds_model outputs/signals
	mkdir -p models/trained
	mkdir -p reports/qa reports/metrics reports/validation reports/trading reports/commentary reports/llm_logs reports/figures
	mkdir -p report/figures
