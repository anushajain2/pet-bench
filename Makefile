PYTHON ?= .venv/bin/python
SCOUT ?= .venv/bin/pet-scout
BENCH ?= .venv/bin/pet-bench

.PHONY: prepare embed evaluate report all test app benchmark
prepare:
	$(SCOUT) prepare
embed:
	$(SCOUT) embed
evaluate:
	$(SCOUT) evaluate
report:
	$(SCOUT) report
all:
	$(SCOUT) all
benchmark:
	$(BENCH) all
campaign:
	$(PYTHON) -m pet_scout.campaign
validate-study:
	$(PYTHON) -m pet_scout.validation
test:
	$(PYTHON) -m pytest -q
app:
	$(PYTHON) -m streamlit run app.py --server.address 127.0.0.1
