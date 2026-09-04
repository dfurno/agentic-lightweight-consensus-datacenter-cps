# Agentic Lightweight Consensus for Smart-City Data-Center CPS

Reproducible research framework for **Agentic Lightweight Consensus for Resilient Monitoring and Control of Modern Data Centers in Smart Cities** (manuscript `smartcities-4475705`, Round 1 revision).

This repository is the public reproducibility bundle for the Smart Cities manuscript. It includes source code, configurations, prompt/schema documentation, verifier policy, scenario-generation scripts, aggregate result files, and scripts for recomputing reported quantities. The Round 1 release supersedes the paper-facing artifacts in the initial commit `a935bb0`, whose ASOC label reflected an earlier packaging target; Git history is retained for provenance.

Start here:

- `DATA_AVAILABILITY.md`: dataset sources, roles, licenses/terms and redistribution policy.
- `REPRODUCIBILITY.md`: commands for reproducing reported tables and checks.
- `RELEASE_MANIFEST.md`: what is included and intentionally excluded.
- `docs/llm_prompt_schema.md`: LLM supervisor prompt/schema and action-only compatibility.
- `docs/verifier_policy.md`: tested process-local actuation-verifier policy and limits.
- `docs/execution_trace_schema.md`: trace schema and metadata structure.
- `paper/smartcities-round1/`: revised MDPI manuscript source and compiled clean PDF.
- `results/round1_evidence/`: authoritative curated Round 1 evidence packages.
- `docs/round1/`: runtime contract, formal ordering note, and frozen thermal protocol.

Raw third-party datasets are not redistributed. Place locally obtained datasets under `data/raw/` after accepting the providers' terms.

The repository implements a hierarchical IoT/MAS simulation for data-center CPS control:

- sensing agents emit noisy and attacked thermal readings;
- edge moderating agents run baselines and FPR-informed OWA consensus;
- an orchestrating ReAct-style planner proposes structured actions;
- a deterministic actuation verifier is the tested process-local command gate.

## Smart Cities Round 1 update

- Corrected bias-scenario parsing and selectively recomposed beta-dependent results. The displayed 10-seed/6,300-scenario robustness values, rankings, and conclusions are unchanged at reported precision.
- Added stable reporter-ID ordering, a formal ordering proposition, an exact-tie audit, and corrected dependent paper artifacts.
- Added a persistent CRP implementation and an aligned temporal ablation on 45 frozen trajectories and 10,800 ticks.
- Added a shared software control runtime and a 24-case controlled-time fault-injection matrix; all 24 cases passed.
- Added 135 action-dependent closed-loop condition-runs and an 18-run pre-specified hot-start boundary test, retaining adverse and non-discriminating outcomes.
- Added true persistent-CRP cost profiles for `n=5..100`, separating one-round convergence from a forced three-round stress profile.
- Renamed historical `token_usage.csv` to `legacy_call_counts.csv`: it records planner/verifier call counts, not model-token consumption.

The 7,560 safety denominator is `1,890 indexed scenarios × 4 separately executed variants`: these are scenario-variant safety episodes, not 7,560 decisions per variant and not four selected ticks within a scenario.

No quantitative paper result is fabricated. The Results section is generated only from measured files under `results/`.

## Dataset Discovery And Audit

Dataset names in the project prompt are treated as hypotheses, not facts. Run:

```bash
make discover-data
make download-data
make audit-data
```

`configs/datasets.yaml` records primary datasets and substitutes:

- ENACT, Kaggle Hot Corridor, and CuCD-ID are marked `manual_required` until exact source, version, license, and schema are supplied.
- Google Cluster Data is supported as a workload/power substitute.
- Building Data Genome Project 2 / ASHRAE GEPIII is supported as an energy/weather/cooling-meter substitute, not a direct hot-corridor dataset.
- CICIoT2023 is supported only for pattern-level IoT attack timing, not as thermal telemetry.

Outputs:

- `results/dataset_discovery.md`
- `results/dataset_discovery.json`
- `results/data_audit.md`
- `results/data_audit.json`

## Real Dataset Pipeline

For a paper-oriented run that must fail instead of silently using synthetic fallback, place or keep the three local archives under `data/raw/`:

```text
data/raw/archive.zip
data/raw/18920397.zip
data/raw/CubeSat Cybersecurity Dataset for Intrusion Detect.zip
```

Then extract them if needed:

```bash
unzip -o data/raw/archive.zip -d data/raw/kaggle_hot_corridor
unzip -o data/raw/18920397.zip -d data/raw/enact
unzip -o "data/raw/CubeSat Cybersecurity Dataset for Intrusion Detect.zip" -d data/raw/cucd_id
```

Run:

```bash
export ALLOW_DATASET_DOWNLOADS=true
make real-paper-pipeline
```

The full experiment now evaluates:

- consensus methods: mean, median, trimmed mean, Hampel, direct OWA, FPR-OWA;
- attack ratios, attack types, sensor noise levels, sensor group sizes, and OWA alpha/beta settings;
- agentic control variants: deterministic policy, ReAct without verifier, ReAct with verifier, and ReAct with verifier plus self-refinement;
- verifier outcomes: unsafe proposed, unsafe blocked, unsafe executed, safe accepted, refinement cycles, planner/verifier calls.

This performs:

```bash
make discover-data
python3 scripts/download_datasets.py --allow
make audit-data
make test
make run-real-full
make paper
```

The primary real-data adapter uses:

- Data Centre Warm Channel Temperature Prediction from Kaggle as thermal source (`TLHC`, `T_MEAS-*`, `P_cu-*`);
- ENACT Edge-Cloud telemetry from Zenodo record `18920397` as workload/energy source (`CPU (%)`, `MEM`, `Energy (watts)`);
- CuCD-ID from Mendeley Data as attack timing source (`Label != 3`).

The older substitute adapter can still use **Building Data Genome Project 2** when its Git LFS CSVs are materialized under:

```text
data/raw/kaggle_hot_corridor/building_data_genome_2/
```

BDG2 is not direct hot-corridor data-center telemetry. The experiment therefore treats it as a documented substitute: electricity meter data drives the workload/power proxy, weather air temperature drives ambient temperature, and the thermal state is generated by the transparent model in `configs/simulation.yaml`. The generated paper section records this limitation.

If `run-real-full` fails with a Git LFS pointer message, run:

```bash
cd data/raw/kaggle_hot_corridor/building_data_genome_2
git lfs install
git lfs pull
cd -
make run-real-full
make paper
```

To confirm that a run used the three real datasets:

```bash
grep -R '"source"' results/traces/*.metadata.json | head
```

You should see:

```text
kaggle_hot_corridor_enact_cucd
```

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
make setup
make check-hardware
make discover-data
make audit-data
make test
make run-small
make paper
```

## Docker Setup

Docker is recommended for reproducible CPU experiments and report generation:

```bash
make docker-build
make docker-run-small
```

GPU checks use the Compose GPU profile:

```bash
docker compose --profile gpu run --rm app-gpu
```

Before GPU Docker runs, verify host access:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

If Docker cannot access the GPU, configure NVIDIA Container Toolkit:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## LLM Serving

Agentic experiments use a LangGraph planner backed by a real OpenAI-compatible LLM endpoint. Start vLLM before running `run-real-*` targets:

```bash
make serve-llm
make run-real-small
```

Initial target model:

```text
google/gemma-4-12B-it
```

Do not silently switch models. Any fallback must be recorded in `results/llm_report.json` and the paper.

If vLLM does not yet support the installed Gemma checkpoint/Transformers combination, keep Gemma and run the Transformers-based OpenAI-compatible server:

```bash
python -m pip install -r requirements-gpu.txt
python -m pip install --upgrade "git+https://github.com/huggingface/transformers.git"
export LLM_MODEL=google/gemma-4-12B-it
make serve-transformers-llm PYTHON=.venv/bin/python
```

The LangGraph planner still calls `http://127.0.0.1:8000/v1/chat/completions`; only the serving backend changes.

## Outputs

The experiment runner writes:

- `results/metrics.csv`
- `results/metrics_by_scenario.csv`
- `results/safety_metrics.csv`
- `results/latency_metrics.csv`
- `results/legacy_call_counts.csv` (historical planner/verifier call counts; no token measurement)
- `results/traces/`
- `results/figures/`
- `results/config_snapshot.yaml`
- `results/realtime_metrics.csv`
- `results/realtime_safety_metrics.csv`
- `results/realtime_config_snapshot.yaml`

## Realtime Evaluation

The `realtime` suite keeps the time-critical path lightweight:

- FPR-OWA consensus and deterministic verifier run on every tick;
- the LLM planner is only an event-triggered supervisory layer;
- a cooldown prevents repeated LLM calls across adjacent ticks;
- if the LLM returns invalid JSON or times out, the deterministic loop continues and the failure is recorded.

Run:

```bash
export LLM_MODEL=google/gemma-4-12B-it
make run-real-realtime-resume
make paper
```

The current realtime policy is intentionally strict:

- low-confidence trigger only below `0.35`;
- anomaly trigger only at `>= 3`;
- thermal margin trigger only within `0.02` of the SLA threshold;
- minimum spacing between LLM invocations: `20` ticks.

This is deliberate: the realtime study is intended to show that the MAS remains lightweight in the primary control loop and only escalates to the agentic planner under sharper uncertainty.

The report generator writes:

- `paper/sections/04_experimental_setup.tex`
- `paper/sections/05_results.tex`
- `paper/sections/06_discussion.tex`
- `paper/appendices/appendix_reproducibility.tex`
- `paper/appendices/appendix_code_snippets.tex`
- `paper/report.md`

## Limitations

This is a reproducible CPS/MAS simulation study, not a production deployment. FPR-informed OWA is a lightweight robust aggregation method, not a formal Byzantine consensus guarantee. Cyber datasets are not mapped to thermal measurements unless their schema directly supports that role. The Round 1 evidence does not establish physical hardware performance, production PKI/RBAC or operating-system isolation, electrical-energy savings, model-token consumption, live asynchronous LLM behavior, or a causal advantage of LLM supervision. The verifier result is limited to the tested process-local runtime.
