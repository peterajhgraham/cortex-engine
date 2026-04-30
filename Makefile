.PHONY: help install dev-install lint format typecheck test test-fast test-gpu \
        train-xs train-s train-m bench-kernels bench-serving \
        serve docker-build docker-up docker-down clean

help:
	@echo "Cortex-Engine Makefile targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    install        Install package in production mode"
	@echo "    dev-install    Install with all dev dependencies + pre-commit"
	@echo ""
	@echo "  Quality:"
	@echo "    lint           Run ruff"
	@echo "    format         Run black + isort"
	@echo "    typecheck      Run mypy --strict"
	@echo "    test           Full test suite"
	@echo "    test-fast      Skip slow + gpu tests"
	@echo "    test-gpu       GPU tests only"
	@echo ""
	@echo "  Training:"
	@echo "    train-xs       Train Cortex-XS"
	@echo "    train-s        Train Cortex-S"
	@echo "    train-m        Train Cortex-M"
	@echo ""
	@echo "  Benchmarks:"
	@echo "    bench-kernels  Run all kernel benchmarks"
	@echo "    bench-serving  Run k6 load tests"
	@echo ""
	@echo "  Serving:"
	@echo "    serve          Run inference server locally"
	@echo "    docker-build   Build production Docker image"
	@echo "    docker-up      Bring up full stack (engine + Prometheus + Grafana)"
	@echo "    docker-down    Tear down stack"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev,baselines,notebooks,load-test]"
	pre-commit install

lint:
	ruff check cortex tests

format:
	black cortex tests
	isort cortex tests
	ruff check --fix cortex tests

typecheck:
	mypy cortex

test:
	pytest tests -v

test-fast:
	pytest tests -v -m "not slow and not gpu"

test-gpu:
	pytest tests -v -m gpu

train-xs:
	python -m cortex.training.train --config-name=cortex_xs

train-s:
	python scripts/train_benchmark.py --max-steps 2000 --device auto

train-m:
	python -m cortex.training.train --config-name=cortex_m

bench-kernels:
	python -m cortex.benchmarks.kernels --output benchmarks/kernels/results.json

bench-serving:
	k6 run ops/k6/load_test.js

serve:
	uvicorn cortex.serve.app:app --host 0.0.0.0 --port 8080 --reload

docker-build:
	docker build -t cortex-engine:latest -f ops/docker/Dockerfile .

docker-up:
	docker compose -f ops/docker/docker-compose.yml up -d

docker-down:
	docker compose -f ops/docker/docker-compose.yml down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/
