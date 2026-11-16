SHELL := /bin/bash
PYTHON ?= python
export PYTHONPATH := src

.PHONY: setup bench data train sft dpo eval export status test chaos

setup:
	bash scripts/bootstrap.sh

bench:
	$(PYTHON) -m t4llm bench --config configs/cpt.yaml

data:
	$(PYTHON) -m t4llm data --config configs/cpt.yaml

train:
	$(PYTHON) -m t4llm supervise --config configs/$(or $(PHASE),cpt).yaml

sft:
	$(PYTHON) -m t4llm sft --config configs/sft.yaml $(if $(TRAIN_FILE),--train-file $(TRAIN_FILE),)

dpo:
	$(PYTHON) -m t4llm dpo --config configs/dpo.yaml $(if $(TRAIN_FILE),--train-file $(TRAIN_FILE),)

eval:
	$(PYTHON) -m t4llm eval --config configs/$(or $(PHASE),cpt).yaml

export:
	$(PYTHON) -m t4llm export --config configs/$(or $(PHASE),cpt).yaml

status:
	$(PYTHON) -m t4llm status --config configs/$(or $(PHASE),cpt).yaml

test:
	$(PYTHON) -m pytest tests -m "not gpu and not slow"

chaos:
	bash scripts/chaos_test.sh
