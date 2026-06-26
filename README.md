# QuantumQUBO Agent

### Automating Quadratic Unconstrained Binary Optimization (QUBO) Formulation Generation from Natural Language

[![ICML 2026 Workshop](https://img.shields.io/badge/ICML%202026-AI%20for%20Math%2C%20CS%20%26%20ML-blue?style=flat-square&logo=academia)](https://openreview.net/forum?id=9YTedapat4)
[![Paper](https://img.shields.io/badge/OpenReview-9YTedapat4-red?style=flat-square&logo=openreview)](https://openreview.net/forum?id=9YTedapat4)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-green?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Project Website](https://img.shields.io/badge/Website-QuantumQUBO%20Agent-purple?style=flat-square&logo=github)](https://quitttcat.github.io/QuantumQUBOAgent/)

> **Accepted** at the ICML 2026 Workshop: *AI as a Tool for Mathematics, Computer Science, and Machine Learning*
> [📄 Paper](https://openreview.net/forum?id=9YTedapat4) · [🌐 Project Website](https://quitttcat.github.io/QuantumQUBOAgent/)

---

![Architecture Diagram](QuantumQUBO%20Agent/ICML-Page-5-1.png)


> **Note:** All commands below must be run from inside the `QuantumQUBO Agent/` directory.
> ```bash
> cd "QuantumQUBO Agent"
> ```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenRouter API key
cp .env.example .env
# Edit .env: OPENROUTER_API_KEY=<your_key>

# 3. Run a single benchmark
python scripts/run_single.py benchmarks/max_cut --input benchmarks/max_cut/sample_cases.txt

# 4. Run all benchmarks and save results to a named CSV
python scripts/run_benchmark_suite.py --seed 42 --name full_pipeline

# 5. Output: results/full_pipeline_42.csv  |  results/runs.jsonl  |  deliverables/  |  transcripts/
```

---




---

## Agents

| Agent | File | Stage | Role |
|---|---|---|---|
| Planner Agent | `agents/planner_agent.py` | 1 | NL → StructuredSpec |
| Judge Agent | `agents/judge_agent.py` | 1b, 2b | Correctness verification |
| Formulizer Agent | `agents/formulizer_agent.py` | 2 | StructuredSpec → QUBOFormulation |
| Debugger Agent | `agents/debugger_agent.py` | 2.5 | Plain-text → JSON test cases |
| Coder Agent | `agents/coder_agent.py` | 3 | QUBOFormulation → Python code |
| Writer Agent | `agents/writer_agent.py` | post | QUBOFormulation → LaTeX deliverable |

Each agent has a corresponding prompt in `qubo_auto/prompts/<agent_name>.txt`.

---

## Run Commands

### Single benchmark

```bash
# Run with plain-text test cases (Debugger builds JSON cases automatically)
python scripts/run_single.py benchmarks/max_cut --input benchmarks/max_cut/sample_cases.txt

# Custom seed
python scripts/run_single.py benchmarks/knapsack --input benchmarks/knapsack/sample_cases.txt --seed 123

# Custom config file
python scripts/run_single.py benchmarks/max_cut --input benchmarks/max_cut/sample_cases.txt --config config_no_retries.yaml
```

### Full benchmark suite

```bash
# Run all benchmarks — CSV saved to results/<name>_<seed>.csv
python scripts/run_benchmark_suite.py --seed 42 --name full_pipeline

# Run a subset of benchmarks
python scripts/run_benchmark_suite.py --seed 42 --name subset_run --benchmarks max_cut,knapsack,graph_coloring

# Skip specific benchmarks
python scripts/run_benchmark_suite.py --seed 42 --name full_pipeline --skip custom,travelling_salesman

# Preview list of benchmarks without running
python scripts/run_benchmark_suite.py --dry-run

# Use a different config (e.g. no-retries ablation)
python scripts/run_benchmark_suite.py --seed 42 --name no_retries_full_pipeline --config config_no_retries.yaml
```

---

## Ablation Studies

Four ablation configurations are provided under `scripts/ablation_scripts/`.

### A1 — No Judge (retries kept)

Removes both Judge verification gates (spec check + QUBO check). Planner, Formulizer, Debugger, Coder, and Test Runner are intact.

```bash
# All benchmarks
python "scripts/ablation_scripts/No Judge ablation study/run_no_judge.py" --seed 42

# Single benchmark
python "scripts/ablation_scripts/No Judge ablation study/run_no_judge.py" --seed 42 --benchmarks max_cut

# Multiple benchmarks
python "scripts/ablation_scripts/No Judge ablation study/run_no_judge.py" --seed 42 --benchmarks max_cut,knapsack

# Custom CSV name
python "scripts/ablation_scripts/No Judge ablation study/run_no_judge.py" --seed 42 --name ablation_no_judge

# No retries + No Judge combined
python "scripts/ablation_scripts/No Judge ablation study/run_no_judge.py" --seed 42 --config config_no_retries.yaml --name no_judge_no_retries

# Preview
python "scripts/ablation_scripts/No Judge ablation study/run_no_judge.py" --dry-run
```

Results: `scripts/ablation_scripts/No Judge ablation study/results/<name>_<seed>.csv`

---

### A2 — No Retries (Judge kept)

Full pipeline with all retry budgets set to zero. One shot per stage — Judge gates are active but no repair budget.

```bash
# All benchmarks
python scripts/run_benchmark_suite.py --seed 42 --name no_retries_full_pipeline --config config_no_retries.yaml

# Single benchmark
python scripts/run_single.py benchmarks/max_cut --input benchmarks/max_cut/sample_cases.txt --config config_no_retries.yaml
```

Results: `results/no_retries_full_pipeline_42.csv`

---

### A3 — No Planner (NL direct to Formulizer)

Removes the Planner agent and both Judge gates. The raw NL problem goes directly to the Formulizer. Debugger receives NL + formulation output. Coder and Test Runner are intact.

```bash
# All benchmarks
python "scripts/ablation_scripts/No Planner ablation study/run_no_planner.py" --seed 42

# Single benchmark
python "scripts/ablation_scripts/No Planner ablation study/run_no_planner.py" --seed 42 --benchmarks max_cut

# Multiple benchmarks
python "scripts/ablation_scripts/No Planner ablation study/run_no_planner.py" --seed 42 --benchmarks max_cut,knapsack

# Custom CSV name
python "scripts/ablation_scripts/No Planner ablation study/run_no_planner.py" --seed 42 --name ablation_no_planner

# Preview
python "scripts/ablation_scripts/No Planner ablation study/run_no_planner.py" --dry-run
```

Results: `scripts/ablation_scripts/No Planner ablation study/results/<name>_<seed>.csv`

---

### A4 - Direct Method

## Caution : It reuses the json parsed test case generated by full architecture run. So that should be run first.

Single LLM call directly from the benchmark problem statement and example instance structures to `build_qubo(instance)`. No Planner, Formulizer, Judge, Debugger, or Coder pipeline stages are used.

```bash
# All benchmarks with the default configured model
python "scripts/ablation_scripts/Direct Method/direct_method.py" --seed 42

# All benchmarks with qwen/qwen3.6-35b-a3b
python "scripts/ablation_scripts/Direct Method/direct_method.py" --seed 42 --model qwen/qwen3.6-35b-a3b

# Single benchmark
python "scripts/ablation_scripts/Direct Method/direct_method.py" --seed 42 --benchmarks max_cut --model qwen/qwen3.6-35b-a3b

# Multiple benchmarks
python "scripts/ablation_scripts/Direct Method/direct_method.py" --seed 42 --benchmarks max_cut,knapsack --model qwen/qwen3.6-35b-a3b

# Custom CSV name
python "scripts/ablation_scripts/Direct Method/direct_method.py" --seed 42 --name ablation_direct_method --model qwen/qwen3.6-35b-a3b

# Preview
python "scripts/ablation_scripts/Direct Method/direct_method.py" --dry-run --model qwen/qwen3.6-35b-a3b
```

Results: `scripts/ablation_scripts/Direct Method/results/<name>_<model>_<seed>.csv`

---

### Ablation Summary

| Configuration | Planner | Judge | Retries | Success Rate |
|---|---|---|---|---|
| Full pipeline | ✓ | ✓ | ✓ | 68% |
| A1 — No Judge | ✓ | ✗ | ✓ | — |
| A2 — No Retries | ✓ | ✓ | ✗ | 35% |
| A1+A2 — No Judge, No Retries | ✓ | ✗ | ✗ | 36% |
| A3 — No Planner | ✗ | ✗ | ✓ | — |
| A4 - Direct Method | no | no | no | - |

---

## Configuration

### `config.yaml` (full pipeline)

```yaml
models:
  planner_agent:    qwen/qwen3.6-35b-a3b
  formulizer_agent: qwen/qwen3.6-35b-a3b
  judge_agent:      qwen/qwen3.6-35b-a3b
  coder_agent:      qwen/qwen3-coder-next
  debugger_agent:   qwen/qwen3.6-35b-a3b

temperatures:
  planner_agent:    0.0
  formulizer_agent: 0.0
  judge_agent:      0.0
  coder_agent:      0.0
  debugger_agent:   0.0

retries:
  restructure: 2    # Planner Agent max retries
  formulate:   2    # Formulizer Agent max retries
  code:        2    # Coder Agent max retries

verification:
  n_test_instances: 6   # test cases generated by Debugger
  max_n_vars:       12  # brute-force cap (cases above this are skipped)
  pass_threshold:   1.0 # fraction of test cases that must pass

limits:
  max_tokens_per_run: 200000
  use_cache: false
```

### `config_no_retries.yaml` (A2 ablation)

Identical to `config.yaml` but all retry budgets set to zero:

```yaml
retries:
  restructure: 0
  formulate:   0
  code:        0
```

Any OpenRouter model ID can be substituted. The pipeline is model-agnostic.

---

## Benchmark Folder Format

```
benchmarks/<name>/
    prompt.txt          ← natural-language problem statement
    sample_cases.txt    ← human-readable test cases (any format)
```

Test case JSON files are generated at runtime and written to `transcripts/<run_id>/cases/` — benchmark folders are never modified.

### Test case JSON schema

```json
{
  "name": "n4_example",
  "description": "...",
  "n_variables": 16,
  "instance": { "n_nodes": 4, "edges": [[0,1,1], [0,2,1]] },
  "ground_truth": {
    "optimal_value": -4.0,
    "optimal_bitstrings": [[0,0,1,1], [0,1,0,1]]
  }
}
```

`optimal_value` follows QUBO convention: negated for maximisation problems, raw for minimisation.

### Adding your own benchmark

1. Create `benchmarks/<name>/prompt.txt` with the problem statement in plain English.
2. Write `benchmarks/<name>/sample_cases.txt` with one or more test instances in any human-readable format.
3. Run:
   ```bash
   python scripts/run_single.py benchmarks/<name> --input benchmarks/<name>/sample_cases.txt
   ```

No code changes needed.

---

## Project Structure

```
qubo_auto/
├── qubo_auto/
│   ├── pipeline.py                   # Orchestrator — all stages + retry logic
│   ├── schemas.py                    # Pydantic models: StructuredSpec, QUBOFormulation, RunResult
│   ├── llm_client.py                 # OpenRouter wrapper, token + cost tracking, transcript logging
│   ├── agents/
│   │   ├── planner_agent.py          # Stage 1    : NL → StructuredSpec
│   │   ├── judge_agent.py            # Stage 1b, 2b : spec + QUBO correctness checks
│   │   ├── formulizer_agent.py       # Stage 2    : StructuredSpec → QUBOFormulation
│   │   ├── debugger_agent.py         # Stage 2.5  : plain-text → JSON test cases
│   │   ├── coder_agent.py            # Stage 3    : QUBOFormulation → Python build_qubo
│   │   └── writer_agent.py           # Post-success: LaTeX/HTML/PDF deliverable
│   ├── verification/
│   │   ├── brute_force.py            # Exhaustive 2^n search + Q matrix validation
│   │   ├── test_runner.py            # Run build_qubo vs ground-truth test cases
│   │   └── test_cases.py             # Load cases JSON into TestCase objects
│   └── prompts/
│       ├── planner_agent.txt
│       ├── judge_agent_spec.txt       # Judge prompt for spec verification
│       ├── judge_agent.txt            # Judge prompt for QUBO verification
│       ├── formulizer_agent.txt
│       ├── debugger_agent.txt
│       ├── coder_agent.txt
│       └── writer_agent.txt
├── benchmarks/
│   └── <name>/
│       ├── prompt.txt
│       └── sample_cases.txt
├── scripts/
│   ├── run_single.py                 # Run one benchmark
│   ├── run_benchmark_suite.py        # Run all benchmarks, write named CSV
│   └── ablation_scripts/
│       ├── No Judge ablation study/
│       │   ├── run_no_judge.py       # A1: pipeline without Judge gates
│       │   ├── results/              # CSV + logs
│       │   └── transcripts/
│       └── No Planner ablation study/
│           ├── run_no_planner.py     # A3: NL direct to Formulizer, no Planner/Judge
│           ├── results/
│           └── transcripts/
├── zero shot benchmark/
│   ├── QUBOBenchModified/            # Problem folders: prompt.txt + tests/case_N/
│   └── scripts/
│       └── direct_method_single.py       # Direct LLM → QUBO solve, no pipeline
├── transcripts/                      # Per-run LLM call logs (JSON, one file per step)
│   └── <run_id>/
│       ├── restructure_<ts>.json
│       ├── verify_spec_<ts>.json
│       ├── formulate_<ts>.json
│       ├── verify_formulation_<ts>.json
│       ├── parse_test_cases_<ts>.json
│       ├── code_<ts>.json
│       └── cases/                    # Generated test case JSON files
├── results/
│   ├── runs.jsonl                    # One JSON line per pipeline run
│   └── <name>_<seed>.csv            # Aggregated CSV from run_benchmark_suite.py
├── deliverables/
│   └── <benchmark_name>/
│       ├── <name>_<seed>.tex
│       ├── <name>_<seed>.html        # MathJax viewer (always generated)
│       └── <name>_<seed>.pdf         # Compiled PDF (requires pdflatex)
├── config.yaml                       # Full pipeline config
├── config_no_retries.yaml            # A2 ablation — zero retry budgets
└── .env                              # OPENROUTER_API_KEY
```

---

## Results Format

Each run appends one JSON line to `results/runs.jsonl`:

```json
{
  "run_id": "max_cut_42_a1b2c3d4",
  "benchmark": "max_cut",
  "seed": 42,
  "status": "success",
  "n_iterations": 2,
  "wall_time_s": 37.4,
  "models_used": {
    "planner_agent": "qwen/qwen3.6-35b-a3b",
    "formulizer_agent": "qwen/qwen3.6-35b-a3b",
    "coder_agent": "qwen/qwen3-coder-next"
  },
  "tokens_used": 9861,
  "cost_used": 0.0182,
  "verification": { "passed": 3, "total": 3 },
  "failure_modes": []
}
```

CSV columns written by `run_benchmark_suite.py`:

| Column | Description |
|---|---|
| `benchmark` | Benchmark folder name |
| `seed` | Run seed |
| `status` | `success` / `failed` / `script_failed` |
| `tests_passed` | Test cases passed |
| `total_tests` | Total test cases |
| `wall_time_s` | Wall clock time (seconds) |
| `tokens_used` | Total tokens consumed |
| `total_llm_cost` | Total LLM cost in USD |
| `planner_retries` | Stage 1 retry count |
| `formulizer_retries` | Stage 2 retry count |
| `coder_retries` | Stage 3 retry count |
| `debugger_retries` | Stage 2.5 retry count |

---

## Failure Modes

| Mode | Stage | Description |
|---|---|---|
| `spec_verification_failed` | 1b | Planner output missing variables or wrong objective |
| `formulation_verification_failed` | 2b | QUBO missing penalty terms or wrong objective direction |
| `coding_error` | 3 | Generated `build_qubo` raises exception or wrong Q shape |
| `formulation_error` | 3b | Q valid but argmin disagrees with ground truth |
| `token_budget_exceeded` | any | Run exceeded `max_tokens_per_run` |

---

## Running Tests (no API key needed)

```bash
pytest tests/
```

Covers: brute-force solver, code extraction, schema validation. No LLM calls made.

---

## Citation

If you use this code or the benchmarks in your research, please cite:

```bibtex
@inproceedings{mondal2026quantumqubo,
  title     = {Quantum{QUBO} Agent: Automating Quadratic Unconstrained Binary Optimization ({QUBO}) Formulation Generation from Natural Language},
  author    = {Niloy Kumar Mondal and Md Rizwan Parvez},
  booktitle = {ICML 2026 Workshop: AI as a Tool for Mathematics, Computer Science, and Machine Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=9YTedapat4},
}
```

If you use the benchmark problems, also cite [QUBOBench](https://quitttcat.github.io/QUBOBench/).

---

## License

This project is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE).

© 2026 Niloy Kumar Mondal, Md Rizwan Parvez
