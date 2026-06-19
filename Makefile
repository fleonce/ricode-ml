PROJECT_NAME := $(lastword $(subst /, ,$(shell pwd)))
PYTHON_MAJOR ?= 3
PYTHON_MINOR ?= 11
PYTHON := .venv/bin/python
GLOBAL_PYTHON := python$(PYTHON_MAJOR).$(PYTHON_MINOR)
UV := $(shell which uv)
PIP := $(UV) pip
PRE_COMMIT := .venv/bin/pre-commit
PRE_COMMIT_GIT_HOOK := .git/hooks/pre-commit
FIND := $(shell which find)
LN := $(shell which ln)
VENV_SETUP ?= local
HOME_DIR ?= $(HOME)
MKDIR := $(shell which /usr/bin/mkdir)

ifndef UV
	UV := /bin/echo "uv is not installed, install via `pip install uv`; Command was: "
endif

# default setup
setup: $(PYTHON) $(PRE_COMMIT_GIT_HOOK);

# setup the virtual environment in the home directory
home_setup:
	VENV_SETUP=home $(MAKE) setup;

# setup for the virtualenv, we use python 3.11
$(PYTHON):
ifeq ($(VENV_SETUP),local)
	$(UV) venv --prompt $(PROJECT_NAME) .venv
else
	$(MKDIR) -p $(HOME_DIR)/.environments/$(PROJECT_NAME)
	$(UV) venv --prompt $(PROJECT_NAME) --directory $(HOME_DIR)/.environments/$(PROJECT_NAME)
	$(LN) -s $(HOME_DIR)/.environments/$(PROJECT_NAME)/.venv .venv
endif

$(PIP): $(PYTHON);

$(PRE_COMMIT_GIT_HOOK): $(PRE_COMMIT)
	$(PRE_COMMIT) install

$(PRE_COMMIT): $(PIP)
	$(PIP) install --upgrade pip setuptools
	$(PIP) install --upgrade pre-commit

install: setup;
	$(PIP) install --upgrade pip setuptools pre-commit
	$(PIP) install -e .
