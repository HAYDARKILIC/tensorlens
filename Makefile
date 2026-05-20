.PHONY: help install lint format typecheck test coverage all clean

help:
	@echo "TensorLens — development targets"
	@echo "  install    Install package with dev extras"
	@echo "  lint       Run ruff + black --check"
	@echo "  format     Apply black + ruff --fix"
	@echo "  typecheck  Run mypy --strict on src/"
	@echo "  test       Run pytest with coverage"
	@echo "  all        lint + typecheck + test"
	@echo "  clean      Remove caches and build artifacts"

install:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	ruff check --fix src/ tests/
	black src/ tests/

typecheck:
	mypy --strict src/

test:
	pytest --cov=src --cov-report=term-missing

all: lint typecheck test

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
