from __future__ import annotations
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable

import yaml

from .agents.coder_agent import compile_build_qubo, generate_code
from .agents.formulizer_agent import formulate
from .agents.planner_agent import restructure
from .agents.debugger_agent import build_test_cases
from .agents.judge_agent import verify_formulation, verify_spec
from .llm_client import LLMClient, TokenBudgetExceeded
from .schemas import QUBOFormulation, RunResult, StructuredSpec
from .verification.test_cases import load_test_cases
from .verification.test_runner import run_tests


def load_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _log(msg: str) -> None:
    elapsed = time.strftime("%H:%M:%S")
    print(f"[{elapsed}] {msg}", flush=True)


class Pipeline:
    def __init__(self, config: dict, transcript_root: Path, results_path: Path):
        self.config = config
        self.transcript_root = transcript_root
        self.results_path = results_path
        self.results_path.parent.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        nl_problem: str,
        benchmark_name: str,
        seed: int,
        test_cases: list,
        benchmark_dir: Path | None = None,
        raw_test_input: str | None = None,
    ) -> RunResult:
        run_id = f"{benchmark_name}_{seed}_{uuid.uuid4().hex[:8]}"
        start = time.time()
        cfg = self.config
        models = cfg["models"]
        temps = cfg["temperatures"]
        retries = cfg["retries"]
        verif_cfg = cfg["verification"]
        limits = cfg.get("limits", {})
        max_tokens = limits.get("max_tokens_per_run", 50_000)
        use_cache = limits.get("use_cache", True)
        failure_modes: list[str] = []

        _log(f"=== Pipeline START | run_id={run_id} ===")
        _log(f"Benchmark : {benchmark_name}  |  seed={seed}")
        _log(f"Models    : planner_agent={models['planner_agent']}")
        _log(f"            formulizer_agent={models['formulizer_agent']}")
        _log(f"            judge_agent={models['judge_agent']}")
        _log(f"            coder_agent={models['coder_agent']}")
        _log(f"Cache     : {'enabled' if use_cache else 'disabled'}")

        client = LLMClient(
            transcript_dir=self.transcript_root,
            run_id=run_id,
            max_tokens=max_tokens,
            use_cache=use_cache,
        )

        result = RunResult(
            run_id=run_id,
            benchmark=benchmark_name,
            seed=seed,
            status="failed",
            models_used=dict(models),
        )

        try:
            # ----------------------------------------------------------------
            # Stage 1: Restructure NL → StructuredSpec
            # ----------------------------------------------------------------
            spec: StructuredSpec | None = None
            spec_issues: list[str] = []
            for attempt in range(retries["restructure"] + 1):
                _log(f"[Stage 1] Planner — attempt {attempt + 1}/{retries['restructure'] + 1}"
                     + (f" | feedback from: Judge (spec check)" if spec_issues else ""))
                try:
                    spec = restructure(nl_problem, client, models["planner_agent"], temps["planner_agent"],
                                       previous_issues=spec_issues or None)
                    _log(f"[Stage 1] Planner OK — {len(spec.variables)} vars, "
                         f"{len(spec.constraints)} constraints")
                except Exception as e:
                    _log(f"[Stage 1] Planner EXCEPTION: {e}")
                    spec_issues = [str(e)]
                    if attempt == retries["restructure"]:
                        result.error_trace = traceback.format_exc()
                        return self._finalize(result, client, start)
                    continue

                # Stage 1b: Verify spec
                _log(f"[Stage 1b] Judge — checking spec matches NL ...")
                spec_check = verify_spec(nl_problem, spec, client, models["judge_agent"], temps["judge_agent"])
                if spec_check.passed:
                    _log(f"[Stage 1b] Spec verification PASSED")
                    break
                spec_issues = spec_check.issues
                _log(f"[Stage 1b] Spec verification FAILED — issues: {spec_issues}")
                if attempt < retries["restructure"]:
                    failure_modes.append("spec_verification_failed")
            else:
                _log(f"[Stage 1] Planner exhausted all retries — aborting")
                result.error_trace = "Planner failed after max retries"
                result.failure_modes = failure_modes
                return self._finalize(result, client, start)

            result.n_iterations += 1

            # ----------------------------------------------------------------
            # Stage 2: QUBO Formulation
            # ----------------------------------------------------------------
            formulation: QUBOFormulation | None = None
            previous_issues: list[str] = []
            for attempt in range(retries["formulate"] + 1):
                _log(f"[Stage 2] Formulizer — attempt {attempt + 1}/{retries['formulate'] + 1}")
                try:
                    formulation = formulate(spec, client, models["formulizer_agent"], temps["formulizer_agent"],
                                            previous_issues or None, feedback_source="judge_agent")
                    _log(f"[Stage 2] Formulation OK — n_vars={formulation.n_variables}, "
                         f"{len(formulation.objective_terms)} obj terms, "
                         f"{len(formulation.penalty_terms)} penalty groups")
                except Exception as e:
                    _log(f"[Stage 2] Formulizer EXCEPTION: {e}")
                    previous_issues = [str(e)]
                    if attempt == retries["formulate"]:
                        result.error_trace = traceback.format_exc()
                        return self._finalize(result, client, start)
                    continue

                # Stage 2b: Verify formulation
                _log(f"[Stage 2b] Judge — checking QUBO formulation ...")
                qubo_check = verify_formulation(spec, formulation, client, models["judge_agent"], temps["judge_agent"])
                if qubo_check.passed:
                    _log(f"[Stage 2b] Formulation verification PASSED")
                    break
                _log(f"[Stage 2b] Formulation verification FAILED — issues: {qubo_check.issues}")
                previous_issues = qubo_check.issues
                if attempt < retries["formulate"]:
                    failure_modes.append("formulation_verification_failed")
            else:
                _log(f"[Stage 2] Formulizer exhausted all retries — aborting")
                result.error_trace = "Formulizer failed after max retries"
                result.failure_modes = failure_modes
                return self._finalize(result, client, start)

            result.n_iterations += 1

            # ----------------------------------------------------------------
            # Stage 2.5: Build test cases from raw input (if provided)
            # ----------------------------------------------------------------
            if raw_test_input and benchmark_dir:
                _log(f"[Stage 2.5] Building test cases from user input ...")
                try:
                    transcript_cases_dir = self.transcript_root / run_id / "cases"
                    written = build_test_cases(
                        benchmark_dir=benchmark_dir,
                        raw_input=raw_test_input,
                        client=client,
                        model=models.get("debugger_agent", models["planner_agent"]),
                        spec=spec,
                        formulation=formulation,
                        temperature=temps.get("debugger_agent", 0.1),
                        cases_dir=transcript_cases_dir,
                    )
                    test_cases = load_test_cases(benchmark_dir, cases_dir=transcript_cases_dir)
                    _log(f"[Stage 2.5] Wrote {len(written)} case(s) to transcript, loaded {len(test_cases)} total")
                except Exception as e:
                    _log(f"[Stage 2.5] Test case building FAILED: {e}")
                    result.error_trace = traceback.format_exc()
                    return self._finalize(result, client, start)

            # ----------------------------------------------------------------
            # Stage 3: Code generation + test loop
            # ----------------------------------------------------------------
            # Use the first test case's instance as a sample so the coder sees exact dict keys
            sample_instance = test_cases[0].instance if test_cases else {}
            n_cases = len(test_cases)

            previous_error: str | None = None
            for attempt in range(retries["code"] + 1):
                _log(f"[Stage 3] Coder — attempt {attempt + 1}/{retries['code'] + 1}")
                try:
                    code = generate_code(spec, formulation, client, models["coder_agent"], temps["coder_agent"],
                                         previous_error, sample_instance=sample_instance)
                    build_qubo_fn = compile_build_qubo(code)
                    _log(f"[Stage 3] Code generated and compiled OK")
                except Exception as e:
                    _log(f"[Stage 3] Coder EXCEPTION: {e}")
                    previous_error = traceback.format_exc()
                    failure_modes.append("coding_error")
                    if attempt == retries["code"]:
                        result.error_trace = previous_error
                        return self._finalize(result, client, start)
                    continue

                _log(f"[Stage 3b] Test Runner — running {n_cases} pre-computed test cases ...")
                test_results, failure_class = run_tests(
                    build_qubo=build_qubo_fn,
                    test_cases=test_cases,
                )

                n_pass = sum(1 for r in test_results if r.match)
                result.verification = {"passed": n_pass, "total": n_cases}
                pass_rate = n_pass / n_cases if n_cases else 0
                _log(f"[Stage 3b] Tests: {n_pass}/{n_cases} passed "
                     f"({pass_rate*100:.0f}%)  failure_class={failure_class}")
                for r in test_results:
                    if r.error_message:
                        _log(f"[Stage 3b] Instance {r.instance_id} error: {r.error_message[:200]}")

                if pass_rate >= verif_cfg["pass_threshold"]:
                    _log(f"=== Pipeline SUCCESS ===")
                    result.status = "success"
                    result.failure_modes = failure_modes
                    result.tokens_used = client.tokens_used
                    result.cost_used = client.cost_used
                    return self._finalize(result, client, start)

                failure_modes.append(failure_class)
                errors = [r.error_message for r in test_results if r.error_message]
                previous_error = "\n".join(errors[:3])

                if failure_class == "formulation_error" and attempt < retries["code"]:
                    _log(f"[Stage 3b] Structural error — looping back to Formulizer ...")
                    try:
                        formulation = formulate(spec, client, models["formulizer_agent"],
                                                temps["formulizer_agent"], errors[:3],
                                                feedback_source="test_runner")
                        _log(f"[Stage 3b] Re-formulation done, re-verifying ...")
                        qubo_check = verify_formulation(spec, formulation, client,
                                                        models["judge_agent"], temps["judge_agent"])
                        if not qubo_check.passed:
                            _log(f"[Stage 3b] Re-formulation still failed verification")
                            failure_modes.append("formulation_verification_failed")
                        else:
                            _log(f"[Stage 3b] Re-formulation passed verification")
                    except Exception as e:
                        _log(f"[Stage 3b] Re-formulation EXCEPTION: {e}")

                result.n_iterations += 1

            _log(f"=== Pipeline FAILED — code exhausted all retries ===")
            result.error_trace = f"Code failed after max retries. Last failure: {failure_class}"
            result.failure_modes = failure_modes
            return self._finalize(result, client, start)

        except TokenBudgetExceeded as e:
            _log(f"=== Pipeline ABORTED — token budget exceeded: {e} ===")
            result.error_trace = str(e)
            result.failure_modes = failure_modes + ["token_budget_exceeded"]
            return self._finalize(result, client, start)
        except Exception:
            _log(f"=== Pipeline EXCEPTION ===")
            result.error_trace = traceback.format_exc()
            result.failure_modes = failure_modes
            return self._finalize(result, client, start)

    def _finalize(self, result: RunResult, client: LLMClient, start: float) -> RunResult:
        result.wall_time_s = time.time() - start
        result.tokens_used = client.tokens_used
        result.cost_used = client.cost_used
        _log(f"Finalizing | status={result.status} | tokens={result.tokens_used} | "
             f"cost=${result.cost_used:.4f} | wall={result.wall_time_s:.1f}s")
        with self.results_path.open("a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")
        return result
