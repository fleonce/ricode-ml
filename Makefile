PROJECT_NAME := $(lastword $(subst /, ,$(shell pwd)))
PYTHON_MAJOR ?= 3
PYTHON_MINOR ?= 12
PYTHON := .venv/bin/python
GLOBAL_PYTHON := python$(PYTHON_MAJOR).$(PYTHON_MINOR)
UV := $(shell which uv)
PIP := $(UV) pip
PRE_COMMIT := .venv/bin/pre-commit
PRE_COMMIT_GIT_HOOK := .git/hooks/pre-commit
FIND := $(shell which find)
LN := $(shell which ln)
GIT := $(shell which git)
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
	$(UV) venv -p $(GLOBAL_PYTHON) --prompt $(PROJECT_NAME) .venv
else
	$(MKDIR) -p $(HOME_DIR)/.environments/$(PROJECT_NAME)
	$(UV) venv -p $(GLOBAL_PYTHON) --prompt $(PROJECT_NAME) --directory $(HOME_DIR)/.environments/$(PROJECT_NAME)
	$(LN) -s $(HOME_DIR)/.environments/$(PROJECT_NAME)/.venv .venv
endif

$(PIP): $(PYTHON);

$(PRE_COMMIT_GIT_HOOK): $(PRE_COMMIT)
	$(PRE_COMMIT) install

$(PRE_COMMIT): $(PYTHON)
	$(UV) add pre-commit

install: setup _common_install;

home_install: home_setup _common_install;

_common_install:
	$(PIP) install -e .
