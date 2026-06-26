from __future__ import annotations
import traceback
from itertools import product
from typing import Callable

import numpy as np

from ..schemas import QUBOFormulation, StructuredSpec, TestInstanceResult
from .brute_force import validate_Q
from .test_cases import TestCase


def run_tests(
    build_qubo: Callable[[dict], tuple[np.ndarray, float]],
    test_cases: list[TestCase],
) -> tuple[list[TestInstanceResult], str]:
    """Run build_qubo on each pre-computed test case; check that the QUBO's
    argmin set overlaps with the case's optimal_bitstrings.

    Returns (results, failure_classification):
      - "success" if all cases pass
      - "coding_error" if Q construction fails or shape/triangularity wrong
      - "formulation_error" if Q is valid but its argmin disagrees with ground truth
    """
    results: list[TestInstanceResult] = []
    for i, case in enumerate(test_cases):
        results.append(_run_single(build_qubo, case, i))

    n_pass = sum(1 for r in results if r.match)
    if n_pass == len(test_cases):
        return results, "success"

    coding_errors = sum(1 for r in results if r.status == "coding_error")
    formulation_errors = sum(1 for r in results if r.status == "formulation_error")
    if coding_errors >= formulation_errors:
        return results, "coding_error"
    return results, "formulation_error"


def _run_single(
    build_qubo: Callable,
    case: TestCase,
    instance_id: int,
) -> TestInstanceResult:
    # Build Q
    try:
        Q, offset = build_qubo(case.instance)
        offset = float(offset) if isinstance(offset, (int, float)) else 0.0
    except Exception:
        return TestInstanceResult(
            instance_id=instance_id,
            status="coding_error",
            error_message=f"[{case.name}] build_qubo raised: {traceback.format_exc()[:600]}",
            match=False,
        )

    if Q.shape != (case.n_variables, case.n_variables):
        return TestInstanceResult(
            instance_id=instance_id,
            status="coding_error",
            error_message=(
                f"[{case.name}] Q shape {Q.shape} does not match expected "
                f"({case.n_variables},{case.n_variables})"
            ),
            match=False,
        )

    q_errors = validate_Q(Q, case.n_variables)
    if q_errors:
        return TestInstanceResult(
            instance_id=instance_id,
            status="coding_error",
            error_message=f"[{case.name}] " + "; ".join(q_errors),
            match=False,
        )

    n = case.n_variables

    # Enumerate all 2^n strings, compute QUBO energy
    qubo_energies: dict[tuple, float] = {}
    for bits in product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        qubo_energies[bits] = float(x @ Q @ x) + offset

    qubo_min = min(qubo_energies.values())
    tol = 1e-6
    qubo_optimal = {b for b, e in qubo_energies.items() if abs(e - qubo_min) <= tol}
    problem_optimal = set(case.optimal_bitstrings)

    overlap = qubo_optimal & problem_optimal
    if overlap:
        return TestInstanceResult(
            instance_id=instance_id,
            status="success",
            qubo_optimum=qubo_min,
            problem_optimum=case.optimal_value,
            match=True,
        )

    # Diagnose: report the disagreement without exposing ground-truth bitstrings or values
    error_msg = (
        f"[{case.name}] Formulation error: the QUBO minimum does not coincide with "
        f"the correct optimal solution. "
        f"QUBO found {len(qubo_optimal)} optimal bitstring(s); "
        f"none overlap with the {len(problem_optimal)} correct optimal bitstring(s). "
        f"Check penalty weights, objective sign, and variable indexing."
    )
    return TestInstanceResult(
        instance_id=instance_id,
        status="formulation_error",
        error_message=error_msg,
        qubo_optimum=qubo_min,
        problem_optimum=case.optimal_value,
        match=False,
    )
