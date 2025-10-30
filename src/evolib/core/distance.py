"""Genotype distance utilities.

This module centralizes distance computations between genotypes so that
sharing / niching operators can rely on a consistent definition.

Two public helper functions are provided:

    genotype_distance(g1, g2) -> float
        Raw (unnormalized) distance. The scale depends on genotype type.

    normalized_genotype_distance(g1, g2) -> float
        Distance mapped to [0, 1]. The normalization rule depends on
        genotype meta-information (e.g., bounds, length). If a safe upper
        bound cannot be inferred, a fallback heuristic based on length is used.

Normalization rules (heuristics):
    - BinaryGenotype: Hamming distance / length.
    - RealGenotype: L2 distance divided by theoretical maximum given bounds
      (sqrt(length) * (high-low)).
    - IntegerGenotype: Similar to real; denominator = sqrt(length) * (high-low).
    - PermutationGenotype: We use (number of mismatched positions)/length.
      (Alternative metrics exist; this simple choice is length-invariant.)
    - HybridGenotype: Weighted average of component normalized distances, each
      weighted by its relative gene count.

Edge cases:
    - Different concrete genotype classes -> TypeError.
    - Different lengths -> ValueError (except Hybrid which checks per component).
    - Division by zero guarded with 1e-12 denominators.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import sqrt

import numpy as np

from .genotype import (
    BinaryGenotype,
    Genotype,
    HybridGenotype,
    IntegerGenotype,
    PermutationGenotype,
    RealGenotype,
)

__all__ = [
    "genotype_distance",
    "normalized_genotype_distance",
]


def _ensure_same_type(a: Genotype, b: Genotype) -> None:
    if a.__class__ is not b.__class__:
        raise TypeError(
            "Genotype distance requires operands of the same concrete type; "
            f"got {a.__class__.__name__} vs {b.__class__.__name__}."
        )


def genotype_distance(a: Genotype, b: Genotype) -> float:
    """Return a raw (unnormalized) distance between two genotypes.

    The metric depends on genotype type:
        - Binary / Integer / Real: L2 norm of element-wise difference.
        - Permutation: number of positions with different values.
        - Hybrid: L2 norm of concatenated components.
    """
    if isinstance(a, HybridGenotype) and isinstance(b, HybridGenotype):
        if a.components.keys() != b.components.keys():  # pragma: no cover - defensive
            raise ValueError("HybridGenotype keys must match for distance.")
        # Recursively compute concatenated difference
        diffs: list[np.ndarray] = []
        for k in a.components:  # iterate over keys
            comp_a = a.components[k]
            comp_b = b.components[k]
            diff = comp_a - comp_b  # type: ignore[operator]
            diffs.append(np.asarray(diff).ravel())
        concatenated = np.concatenate(diffs)
        return float(np.linalg.norm(concatenated))

    _ensure_same_type(a, b)

    if isinstance(a, BinaryGenotype):
        if len(a) != len(b):
            raise ValueError("BinaryGenotype lengths differ.")
        return float(np.count_nonzero(a.as_array() != b.as_array()))
    if isinstance(a, (RealGenotype, IntegerGenotype)):
        if len(a) != len(b):
            raise ValueError("Numeric genotype lengths differ.")
        diff = (a - b).astype(float)  # type: ignore[operator]
        return float(np.linalg.norm(diff))
    if isinstance(a, PermutationGenotype):
        if len(a) != len(b):
            raise ValueError("Permutation lengths differ.")
        return float(np.count_nonzero(a.as_array() != b.as_array()))

    raise TypeError(f"Unsupported genotype type {type(a).__name__} for distance computation.")


def normalized_genotype_distance(a: Genotype, b: Genotype) -> float:
    """Return a distance in [0, 1] between genotypes.

    See module docstring for normalization logic.
    """
    if isinstance(a, HybridGenotype) and isinstance(b, HybridGenotype):
        if a.components.keys() != b.components.keys():  # pragma: no cover - defensive
            raise ValueError("HybridGenotype keys must match for distance.")
        weighted: list[float] = []
        lengths: list[int] = []
        for k in a.components:
            comp_a = a.components[k]
            comp_b = b.components[k]
            d = normalized_genotype_distance(comp_a, comp_b)
            comp_len = len(comp_a)
            weighted.append(d * comp_len)
            lengths.append(comp_len)
        denom = sum(lengths) if lengths else 1
        return float(sum(weighted) / denom)

    _ensure_same_type(a, b)

    if isinstance(a, BinaryGenotype):
        raw = genotype_distance(a, b)
        return float(raw / max(1, len(a)))
    if isinstance(a, (RealGenotype, IntegerGenotype)):
        if len(a) != len(b):
            raise ValueError("Numeric genotype lengths differ.")
        low, high = getattr(a, "bounds", (0, 1))
        span = float(high - low)
        # Maximum possible L2 distance occurs when each gene differs by span
        max_l2 = float(sqrt(len(a)) * (span if span > 0 else 1.0))
        raw = genotype_distance(a, b)
        return float(min(1.0, raw / (max_l2 + 1e-12)))
    if isinstance(a, PermutationGenotype):
        raw = genotype_distance(a, b)
        return float(raw / max(1, len(a)))

    raise TypeError(f"Unsupported genotype type {type(a).__name__} for normalized distance.")


# Convenience: batch distance matrix (optionally normalized)
def distance_matrix(genotypes: Iterable[Genotype], normalized: bool = False) -> np.ndarray:
    """Return a symmetric distance matrix for provided genotypes."""
    genos = list(genotypes)
    n = len(genos)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if normalized:
                d = normalized_genotype_distance(genos[i], genos[j])
            else:
                d = genotype_distance(genos[i], genos[j])
            mat[i, j] = mat[j, i] = d
    return mat
