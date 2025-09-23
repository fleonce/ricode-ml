
VIRTUALENV := $(shell which virtualenv 2> /dev/null)
ifndef VIRTUALENV
	VIRTUALENV := "python3 -m venv"
endif

PYTHON_MAJOR := 3
PYTHON_MINOR := 11
PYTHON := .venv/bin/python$(PYTHON_MAJOR).$(PYTHON_MINOR)
PIP := .venv/bin/pip$(PYTHON_MAJOR)
PRE_COMMIT := .venv/bin/pre-commit
PRE_COMMIT_GIT_HOOK := .git/hooks/pre-commit
FIND := $(shell which find)

setup: $(PYTHON) $(PRE_COMMIT_GIT_HOOK);

$(PRE_COMMIT_GIT_HOOK): $(PRE_COMMIT)
	$(PRE_COMMIT) install

# setup for the virtualenv, we use python 3.11
$(PYTHON):
	$(VIRTUALENV) --prompt ricode-ml .venv -p python$(PYTHON_MAJOR).$(PYTHON_MINOR)

$(PIP): $(PYTHON);

$(PRE_COMMIT): $(PIP)
	$(PIP) install --upgrade pip setuptools
	$(PIP) install --upgrade pre-commit

install: setup;
	$(PIP) install --upgrade pip setuptools pre-commit
	$(PIP) install .
