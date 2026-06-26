"""Convert human-written test case descriptions into benchmark JSON files.

The user provides instance data (connections, weights, energies, optimal bitstrings)
in any plain-text format. Called after formulation so the LLM has full QUBO context
to correctly map instance keys and sign the optimal_value.

Supports multiple test cases in a single input string.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

from ..llm_client import LLMClient
from ..schemas import QUBOFormulation, StructuredSpec

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "debugger_agent.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")


def _fill(template: str, **kwargs: str) -> str:
    result = template
    for k, v in kwargs.items():
        result = result.replace("{" + k + "}", v)
    return result


def _formulation_context(spec: StructuredSpec, formulation: QUBOFormulation) -> str:
    return (
        f"objective direction : {spec.objective.direction}\n"
        f"n_variables         : {formulation.n_variables}\n"
        f"variable_mapping    : {json.dumps(formulation.variable_mapping)}\n"
        f"instance_parameters : {spec.instance_parameters}"
    )


def _extract_json(raw: str) -> list[dict]:
    cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON.\nError: {e}\nRaw:\n{raw[:600]}")
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array of test cases, got: {type(parsed)}")
    return parsed


def _validate_case(case: dict, index: int) -> list[str]:
    errors = []
    for field in ("n_variables", "instance", "ground_truth"):
        if field not in case:
            errors.append(f"case[{index}] missing '{field}'")
    if "ground_truth" in case:
        gt = case["ground_truth"]
        if "optimal_value" not in gt:
            errors.append(f"case[{index}] ground_truth missing 'optimal_value'")
        if "optimal_bitstrings" not in gt:
            errors.append(f"case[{index}] ground_truth missing 'optimal_bitstrings'")
        else:
            n = case.get("n_variables")
            for bs in gt["optimal_bitstrings"]:
                if n and len(bs) != n:
                    errors.append(
                        f"case[{index}] bitstring length {len(bs)} != n_variables {n}"
                    )
    return errors


def build_test_cases(
    benchmark_dir: Path,
    raw_input: str,
    client: LLMClient,
    model: str,
    spec: StructuredSpec,
    formulation: QUBOFormulation,
    temperature: float = 0.1,
    max_retries: int = 3,
    cases_dir: Path | None = None,
) -> list[Path]:
    """Parse raw_input and write one JSON file per test case into benchmark_dir/cases/.

    Args:
        benchmark_dir: path to benchmarks/<name>/ (must contain prompt.txt)
        raw_input:     plain-text description(s) of one or more test cases
        client:        LLMClient instance
        model:         model identifier string
        spec:          StructuredSpec from the planner_agent (provides objective direction)
        formulation:   QUBOFormulation from the math expert (provides variable mapping)
        temperature:   LLM sampling temperature
        max_retries:   retry attempts if LLM output is invalid

    Returns:
        list of paths to the written JSON files
    """
    problem_prompt = (benchmark_dir / "prompt.txt").read_text(encoding="utf-8").strip()
    cases_dir = cases_dir if cases_dir is not None else benchmark_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    context = _formulation_context(spec, formulation)
    previous_feedback = ""

    for attempt in range(max_retries):
        prompt = _fill(
            _PROMPT_TEMPLATE,
            problem_prompt=problem_prompt,
            formulation_context=context,
            raw_input=raw_input,
            previous_feedback=previous_feedback,
        )
        raw = client.call(
            model=model,
            prompt=prompt,
            temperature=temperature,
            step_name="parse_test_cases",
        )
        try:
            cases = _extract_json(raw)
        except ValueError as e:
            previous_feedback = f"\nPREVIOUS ATTEMPT FAILED:\n{e}\nFix and return valid JSON array.\n"
            continue

        all_errors: list[str] = []
        for i, case in enumerate(cases):
            all_errors.extend(_validate_case(case, i))

        if all_errors:
            previous_feedback = (
                "\nPREVIOUS ATTEMPT HAD VALIDATION ERRORS — fix all before responding:\n"
                + "\n".join(f"  - {e}" for e in all_errors)
                + "\n"
            )
            continue

        written: list[Path] = []
        for case in cases:
            name = case.get("name") or f"case_{len(list(cases_dir.glob('*.json'))) + len(written) + 1}"
            out = {
                "name": name,
                "description": case.get("description", ""),
                "n_variables": case["n_variables"],
                "instance": case["instance"],
                "ground_truth": {
                    "optimal_value": float(case["ground_truth"]["optimal_value"]),
                    "optimal_bitstrings": [list(bs) for bs in case["ground_truth"]["optimal_bitstrings"]],
                },
            }
            path = cases_dir / f"{name}.json"
            path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            written.append(path)

        return written

    raise RuntimeError(
        f"Failed to parse test cases after {max_retries} attempts.\n"
        f"Last feedback: {previous_feedback}"
    )
