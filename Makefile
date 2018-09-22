.PHONY: help
help:
	@echo "clean - remove all artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "clean-logs - remove all logs"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "lint - check style with flake8"
	@echo "install-dev - install package in development mode"
	@echo "install-prod - install package in production mode"
	@echo "qa - run linters and test coverage"
	@echo "qa-all - run QA plus tox, docs, and packaging"
	@echo "release - package and upload a release"
	@echo "sdist - package"
	@echo "test-local - run local tests quickly with the default Python"
	@echo "tox - run tox"

VENV_NAME=venv
PYTHON=$(VENV_NAME)/bin/python

venv_activate:
	test -d $(VENV_NAME) || virtualenv -p python3 $(VENV_NAME)

.PHONY: setup-venv
setup-venv: venv_activate

.PHONY: clean
clean: clean-build clean-logs clean-pyc clean-venv clean-test

.PHONY: clean-build
clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

.PHONY: clean-pyc
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-logs
clean-logs:
	find . -name '*.log*' -exec rm -f {} +

.PHONY: clean-venv
clean-venv:
	rm -rf .venv/

.PHONY: clean-test
clean-test:
	rm -rf .tox/
	rm -f .coverage
	rm -rf htmlcov/

.PHONY: docs
docs:
	$(MAKE) -C docs clean
	$(PYTHON) -m tox -c docs.ini --skip-missing-interpreter 
	$(MAKE) -C docs html
	open docs/build/html/index.html

.PHONY: install-dev
install-dev: clean-build clean-pyc setup-venv
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r pip_requirements.txt
	$(PYTHON) -m pip install -e .

.PHONY: install-prod
install-prod: clean-build clean-pyc
	$(PYTHON) -m pip install -r pip_requirements.txt
	$(PYTHON) setup.py install

.PHONY: lint
lint:
	$(PYTHON) -m flake8 loadbalanceRL

.PHONY: qa
qa: lint test-local

.PHONY: qa-all
qa-all: lint test-prod docs

.PHONY: release
release: clean
	$(PYTHON) setup.py sdist

.PHONY: test-local
test-local: install-dev
	$(PYTHON) -m pytest -s --cov=loadbalanceRL -v -m "not production" --ignore=tests/functional_tests/

.PHONY: tox
tox: install-dev
	$(PYTHON) -m tox -c tox.ini --skip-missing-interpreters

.PHONY: start-cellular-simulator
start-cellular-simulator:
	$(PYTHON) loadbalanceRL/lib/environment/cellular/dev/server.py
