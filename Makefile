# Recompute everything that can be recomputed from the deposited results.
# The analysis stages in pipeline/ need the source imaging data; see
# docs/DATA_ACCESS.md.

PY ?= python3
STAGES = 00_preprocessing 01_operator_divergence 02_ec_uncertainty 03_linearization 04_nonlinear_control 05_clinical

.PHONY: all install test values tables clean

all: test values tables

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

values:
	@status=0; for s in $(STAGES); do $(PY) pipeline/$$s/reported_values.py --results results || status=1; done; exit $$status

tables:
	$(PY) tables/check_tables.py --results results --out tables/out --check

clean:
	rm -rf tables/out
