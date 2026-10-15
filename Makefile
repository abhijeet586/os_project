PY := .venv/bin/python
# 12 seeds is what docs/RESULTS.md reports; `make experiment` therefore
# reproduces the published numbers by default. Five is the reportable minimum.
SEEDS ?= 1,2,3,4,5,6,7,8,9,10,11,12
DURATION ?= 3600
POLICY ?= lease
SEED ?= 42

.PHONY: help venv test lint format run headless experiment sweep figures clean

help:
	@echo "Autonomous Warehouse Fleet Allocator"
	@echo ""
	@echo "  make venv        create .venv and install dependencies"
	@echo "  make test        run the full pytest suite"
	@echo "  make lint        ruff + black --check + mypy --strict"
	@echo "  make format      apply black and ruff --fix"
	@echo "  make run         interactive simulation (POLICY=, SEED=)"
	@echo "  make headless    one headless run, prints derived statistics"
	@echo "  make experiment  all policies over SEEDS, writes CSVs"
	@echo "  make sweep       parameter sweeps, writes sweep.csv"
	@echo "  make figures     regenerate all five figures"
	@echo "  make clean       remove caches and generated results"

venv:
	uv venv --python 3.12
	uv pip install -r requirements.txt

test:
	$(PY) -m pytest tests/ -v

lint:
	$(PY) -m ruff check warehouse_allocator/ tests/
	$(PY) -m black --check warehouse_allocator/ tests/
	$(PY) -m mypy --strict warehouse_allocator/

format:
	$(PY) -m black warehouse_allocator/ tests/
	$(PY) -m ruff check --fix warehouse_allocator/ tests/

run:
	$(PY) -m warehouse_allocator --policy $(POLICY) --seed $(SEED) --duration 600

headless:
	$(PY) -m warehouse_allocator --policy $(POLICY) --seed $(SEED) \
		--duration $(DURATION) --headless

experiment:
	$(PY) -m warehouse_allocator.experiments.runner \
		--seeds $(SEEDS) --duration $(DURATION)

sweep:
	$(PY) -m warehouse_allocator.experiments.sweep \
		--seeds $(SEEDS) --duration $(DURATION)

figures:
	$(PY) -m warehouse_allocator.experiments.runner \
		--seeds $(SEEDS) --duration $(DURATION) --figures

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f results/*.csv results/*.png results/*.pdf results/*.txt
