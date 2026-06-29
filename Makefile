.PHONY: setup discover-data download-data audit-data check-hardware test run-small run-full run-real-small run-real-full run-real-small-resume run-real-full-resume run-real-realtime run-real-realtime-resume revision-inventory revision-reproduce revision-phase-b revision-supervisor-comparison revision-fpr-ablation revision-compute-cost revision-safety-adversarial revision-closing-experiments revision-seed-baselines revision-cpu-extensions revision-agentic-subset revision-bundle serve-llm serve-transformers-llm serve-diffusiongemma-llm paper clean-results docker-build docker-run docker-run-small real-paper-pipeline

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
export MPLCONFIGDIR ?= $(CURDIR)/.cache/matplotlib

setup:
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -r requirements.txt

discover-data:
	$(PYTHON) scripts/discover_datasets.py

download-data:
	$(PYTHON) scripts/download_datasets.py

audit-data:
	$(PYTHON) scripts/audit_datasets.py

check-hardware:
	$(PYTHON) scripts/check_hardware.py

test:
	$(PYTHON) -m pytest -q

run-small:
	$(PYTHON) scripts/run_experiments.py --size small --data-mode auto

run-full:
	$(PYTHON) scripts/run_experiments.py --size full --data-mode auto

run-real-small:
	$(PYTHON) scripts/run_experiments.py --size small --data-mode real

run-real-full:
	$(PYTHON) scripts/run_experiments.py --size full --data-mode real

run-real-small-resume:
	$(PYTHON) scripts/run_experiments.py --size small --data-mode real --resume

run-real-full-resume:
	$(PYTHON) scripts/run_experiments.py --size full --data-mode real --resume

run-real-realtime:
	$(PYTHON) scripts/run_experiments.py --size realtime --data-mode real

run-real-realtime-resume:
	$(PYTHON) scripts/run_experiments.py --size realtime --data-mode real --resume

revision-inventory:
	$(PYTHON) scripts/revision_inventory.py

revision-reproduce:
	$(PYTHON) scripts/revision_reproduce.py

revision-phase-b:
	$(PYTHON) scripts/revision_phase_b.py

revision-supervisor-comparison:
	$(PYTHON) scripts/revision_supervisor_comparison.py

revision-fpr-ablation:
	$(PYTHON) scripts/revision_fpr_ablation.py

revision-compute-cost:
	$(PYTHON) scripts/revision_compute_cost.py

revision-safety-adversarial:
	$(PYTHON) scripts/revision_safety_adversarial.py

revision-closing-experiments:
	$(PYTHON) scripts/revision_closing_experiments.py

revision-seed-baselines:
	$(PYTHON) scripts/revision_seed_baselines.py --data-mode real --resume

revision-cpu-extensions:
	$(PYTHON) scripts/revision_run_extensions.py --mode cpu

revision-agentic-subset:
	$(PYTHON) scripts/revision_run_extensions.py --mode agentic-subset

revision-bundle:
	$(PYTHON) scripts/revision_bundle.py

serve-llm:
	bash scripts/serve_llm.sh

serve-transformers-llm:
	$(PYTHON) scripts/serve_transformers_openai.py --model $${LLM_MODEL:-google/gemma-4-12B-it}

serve-diffusiongemma-llm:
	$(PYTHON) scripts/serve_transformers_openai.py --model $${LLM_MODEL:-google/diffusiongemma-26B-A4B-it}

paper:
	$(PYTHON) scripts/generate_report.py

clean-results:
	find results -type f ! -name .gitkeep -delete

docker-build:
	docker build -t agentic-lightweight-consensus:latest .

docker-run:
	docker compose run --rm app

docker-run-small:
	docker compose run --rm app make discover-data audit-data test run-small paper

real-paper-pipeline:
	$(PYTHON) scripts/discover_datasets.py
	$(PYTHON) scripts/download_datasets.py --allow
	$(PYTHON) scripts/audit_datasets.py
	$(PYTHON) -m pytest -q
	$(PYTHON) scripts/run_experiments.py --size full --data-mode real
	$(PYTHON) scripts/generate_report.py
