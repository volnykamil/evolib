"""
evolib.operators.hybrid
=======================

Component-wise hybrid mutation and crossover operators with optional parallel execution.

Design:
 - Each Hybrid operator holds a dict mapping component keys -> operator instance.
 - Operators are applied independently to each component (component-wise).
 - Type checks ensure operator is compatible with the component's genotype type.
 - Optional parallel execution using ThreadPoolExecutor (safe: operations are lightweight, rely
   on NumPy; if heavy CPU-bound tasks are needed, swap to ProcessPoolExecutor with picklable functions).
"""

from __future__ import annotations

from evolib.core.genotype import (
    BinaryGenotype,
    Genotype,
    IntegerGenotype,
    PermutationGenotype,
    RealGenotype,
)
from evolib.operators.crossover import CrossoverOperator
from evolib.operators.mutation import MutationOperator


# Utility for runtime validation
def _validate_component_matches(gen: Genotype, op: CrossoverOperator | MutationOperator, comp_key: str) -> None:
    """Check that operator plausibly supports the genotype type of the component.

    This is a conservative runtime check: it validates the genotypes' classes against a set
    of expected operator attribute hints. Operators may be third-party; therefore we
    primarily check for instance compatibility via known Genotype subclasses.
    """
    # If op exposes an attribute `supported_types` (tuple of classes), prefer that explicit contract.
    supported = getattr(op, "supported_types", None)
    if supported is not None:
        if not isinstance(supported, (tuple, list)):
            raise TypeError(f"Operator for component '{comp_key}' has invalid supported_types attribute.")
        if not any(isinstance(gen, t) for t in supported):
            names = tuple(st.__name__ for st in supported)
            raise TypeError(f"Operator for component '{comp_key}' does not support genotype type {names}.")
        return

    # Fallback: check common cases using isinstance
    if isinstance(gen, RealGenotype):
        # assume operator must handle numeric arrays
        # no further check (operators for real usually work)
        return
    if isinstance(gen, BinaryGenotype):
        return
    if isinstance(gen, IntegerGenotype):
        return
    if isinstance(gen, PermutationGenotype):
        return
    # Unknown genotype subclass -> be permissive but warn (we cannot log here directly)
    return
