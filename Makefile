PWD              := $(shell pwd)
PYTHON           := poetry run python
PYTEST           := poetry run pytest
PYSEN            := poetry run pysen
MODULE           := chomsky_neural


.PHONY: all lint format test clean clean-pyc clean-build setup

all: format lint test

lint:
	$(PYSEN) run lint

format:
	$(PYSEN) run format

test:
	PYTHONPATH=$(PWD) $(PYTEST)

clean: clean-pyc clean-build

clean-pyc:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf $(MODULE).egg-info/
	rm -rf pip-wheel-metadata/

setup:
	poetry install
