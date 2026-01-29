.PHONY: help install test run clean format lint

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
	python -m pipeline.cli run

run-date:
	python -m pipeline.cli run --date $(DATE)

clean:
	rm -rf outputs/* reports/qa/* reports/metrics/* reports/validation/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

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
