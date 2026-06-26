from __future__ import annotations
import json
import re
import traceback
from pathlib import Path
from typing import Callable

from ..llm_client import LLMClient
from ..schemas import QUBOFormulation, StructuredSpec


_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "coder_agent.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")


def _fill(template: str, **kwargs: str) -> str:
    result = template
    for k, v in kwargs.items():
        result = result.replace("{" + k + "}", v)
    return result


def generate_code(
    spec: StructuredSpec,
    formulation: QUBOFormulation,
    client: LLMClient,
    model: str,
    temperature: float,
    previous_error: str | None = None,
    sample_instance: dict | None = None,
) -> str:
    spec_json = spec.model_dump_json(indent=2)
    formulation_json = formulation.model_dump_json(indent=2)

    # Show the coder the exact instance dict it will receive at test time
    sample_block = ""
    if sample_instance:
        sample_block = (
            f"SAMPLE INSTANCE (exact dict your function will receive):\n"
            f"{json.dumps(sample_instance, indent=2)}\n"
            f"Use ONLY these keys when reading from `instance`."
        )

    feedback = ""
    if previous_error:
        feedback = (
            "PREVIOUS ATTEMPT FAILED — fix ALL issues before writing new code:\n"
            f"{previous_error.strip()}\n\n"
            "The test enumerates ALL 2^n binary strings and checks that the x minimizing "
            "x^T Q x + offset is the SAME string that minimizes the original problem. "
            "An 'Argmin mismatch' means your Q encodes the wrong objective or wrong coefficients.\n"
            "Common fixes:\n"
            "- Use exact keys from the SAMPLE INSTANCE above\n"
            "- Q shape must be (n, n) where n = len(instance data), not a hardcoded number\n"
            "- Q must be UPPER TRIANGULAR: off-diagonal [i,j] → write FULL coeff to Q[min(i,j), max(i,j)], leave Q[max,min]=0\n"
            "- Symbolic coefficients (e.g. 'a_i', 'A') must be computed from instance values\n"
            "- Symbolic offset (e.g. 'A**2') must be computed — do NOT leave it as 0.0"
        )

    prompt = _fill(
        _PROMPT_TEMPLATE,
        spec_json=spec_json,
        formulation_json=formulation_json,
        sample_instance=sample_block,
        previous_feedback=feedback,
    )

    raw = client.call(model=model, prompt=prompt, temperature=temperature, step_name="code")
    return _extract_code(raw)


def _extract_code(raw: str) -> str:
    match = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
    if "def build_qubo" in cleaned:
        return cleaned
    raise ValueError(f"Could not extract Python code from coder response:\n{raw[:500]}")


def compile_build_qubo(code: str) -> callable:
    namespace: dict = {}
    exec(compile(code, "<build_qubo>", "exec"), namespace)  # noqa: S102
    if "build_qubo" not in namespace:
        raise ValueError("Coder did not define `build_qubo` function")
    return namespace["build_qubo"]
