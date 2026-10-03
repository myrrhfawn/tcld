# TCLD Makefile. Every target works both on the host (with deps installed) and
# inside the Docker container. Use `make docker-run CMD="make <target>"` to run
# a target inside the container.

SHELL := /bin/bash
PROFILE ?= laptop
MODEL   ?= dfine_s
CMD     ?= bash
PYTHON  ?= python3

# Host: read .env (volume paths, UID/GID). Inside the container the paths are
# already /data, /runs, /ckpt — do not let the host .env override them.
ifndef TCLD_IN_DOCKER
-include .env
endif
export

# Make `tcld` and D-FINE's `src` importable without pip install (host or container).
export PYTHONPATH := $(CURDIR):$(CURDIR)/external/D-FINE:$(PYTHONPATH)

.PHONY: help setup docker-build docker-shell docker-run smoke test lint \
        download-ckpt download-coco-subset aggregate

help:
	@echo "Targets:"
	@echo "  setup                 pip install deps + editable package (host, non-Docker)"
	@echo "  docker-build          build tcld:latest with your UID/GID"
	@echo "  docker-shell          interactive shell in the container"
	@echo "  docker-run CMD=...    run a command in the container"
	@echo "  download-ckpt         download D-FINE checkpoints into \$$TCLD_CKPT"
	@echo "  download-coco-subset  COCO val2017 annotations + 50-image subset into \$$TCLD_DATA/coco"
	@echo "  smoke                 inference sanity check (PROFILE=laptop|server, MODEL=dfine_s|dfine_l)"
	@echo "  test                  pytest (CPU)"
	@echo "  lint                  ruff"
	@echo "  aggregate             rebuild results tables from results/experiments.csv"

setup:
	pip install -r requirements.txt
	pip install --no-deps -e .

# ---------------------------------------------------------------- docker
docker-build:
	TCLD_UID=$$(id -u) TCLD_GID=$$(id -g) docker compose build

docker-shell:
	docker compose run --rm tcld bash

docker-run:
	docker compose run --rm tcld $(CMD)

# ---------------------------------------------------------------- data / ckpt
download-ckpt:
	bash scripts/download_dfine_ckpt.sh $(MODELS)

download-coco-subset:
	$(PYTHON) scripts/download_coco_subset.py

# ---------------------------------------------------------------- checks
smoke:
	$(PYTHON) scripts/smoke.py --profile $(PROFILE) --model $(MODEL)

test:
	$(PYTHON) -m pytest tests

lint:
	ruff check tcld scripts tests

aggregate:
	$(PYTHON) scripts/aggregate.py
