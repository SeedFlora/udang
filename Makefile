# ShrimpVision-MTL
#
# Full pipeline, in dependency order. Each target refuses to run on unsound
# inputs: `audit` exits non-zero when a leakage-critical check fails, and
# `splits` raises if any group straddles train/val/test.

PY ?= .venv/Scripts/python.exe
PROFILE ?= standard
SEED ?= 42
NBOOT ?= 2000

.PHONY: help setup data manifest audit splits preflight test smoke \
        e1 e2 e3 e4 e5 e6 e7 e8 e9 baselines evaluate figures errors export \
        report all clean-outputs

help:
	@echo "make setup      - create venv and install pinned dependencies"
	@echo "make data       - download + verify the three datasets (SHA-256 checked)"
	@echo "make manifest   - build the unified manifest"
	@echo "make audit      - run the mandatory audit (BLOCKS on leakage-critical failure)"
	@echo "make splits     - build and verify all leakage-safe splits"
	@echo "make preflight  - tests + overfit + smoke + leakage checks"
	@echo "make e1..e9     - run individual experiments"
	@echo "make all        - E0 through E9 plus evaluation, figures and export"

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
	$(PY) -m pip install -r requirements-lock.txt

data:
	$(PY) scripts/download_data.py

manifest:
	$(PY) scripts/build_manifests.py --workers 10

audit:
	$(PY) scripts/audit_data.py

splits:
	$(PY) scripts/build_splits.py --seed $(SEED)

test:
	$(PY) -m pytest tests/ -q

preflight: test
	$(PY) scripts/preflight.py --profile $(PROFILE)

smoke:
	$(PY) scripts/train_disease.py --profile smoke
	$(PY) scripts/train_biomass.py --profile smoke

baselines:
	$(PY) scripts/run_biomass_baselines.py

e1:
	$(PY) scripts/train_disease.py --profile $(PROFILE) --only E1
e2:
	$(PY) scripts/train_disease.py --profile $(PROFILE) --only E2
e3:
	$(PY) scripts/train_disease.py --profile $(PROFILE) --only E3
e4:
	$(PY) scripts/train_biomass.py --profile $(PROFILE) --only E4
e5:
	$(PY) scripts/train_biomass.py --profile $(PROFILE) --only E5
e6:
	$(PY) scripts/train_multitask.py --profile $(PROFILE) --only E6
e7:
	$(PY) scripts/train_multitask.py --profile $(PROFILE) --only E7
e8:
	$(PY) scripts/run_ablations.py --profile $(PROFILE) --only E8
e9:
	$(PY) scripts/run_e9_seeds.py --profile $(PROFILE)

evaluate:
	$(PY) scripts/evaluate.py --n-boot $(NBOOT)

figures:
	$(PY) scripts/make_figures.py

errors:
	$(PY) scripts/error_analysis.py

export:
	$(PY) scripts/export_best_lightweight.py

report:
	$(PY) scripts/write_report.py

# E0 must pass before any training starts.
all: manifest audit splits preflight baselines e1 e2 e3 e4 e5 e6 e7 e8 e9 \
     evaluate figures errors export report

clean-outputs:
	rm -rf outputs/checkpoints/* outputs/predictions/* outputs/metrics/* \
	       outputs/figures/* outputs/error_analysis/* outputs/logs/*
