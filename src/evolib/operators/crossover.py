"""
evolib.operators.crossover
==========================

This module defines crossover (recombination) operators
for BinaryGenotype, RealGenotype, IntegerGenotype, and PermutationGenotype.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from evolib.core.genotype import (
    BinaryGenotype,
    Genotype,
    HybridGenotype,
    IntegerGenotype,
    PermutationGenotype,
    RealGenotype,
)


# =============================================================================
# Base class
# =============================================================================
class CrossoverOperator(ABC):
    """Abstract base class for crossover operators supporting RNG injection.

    Parameters
    ----------
    rng : numpy.random.Generator | None, default None
        Optional RNG for deterministic behavior. If ``None`` a new default
        generator is created.
    """

    supported_genotypes: tuple[type[Genotype], ...] = ()

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

    @abstractmethod
    def crossover(self, parent1: Genotype, parent2: Genotype) -> tuple[Genotype, Genotype]:
        """Return two offspring created from parent1 and parent2."""
        pass


# =============================================================================
# HybridGenotype crossovers
# =============================================================================
class HybridCrossover(CrossoverOperator):
    """Sequential component-wise crossover.

    Sub-operators may each have their own RNG; this wrapper's RNG is unused.
    """

    def __init__(self, operators: dict[str, CrossoverOperator], rng: np.random.Generator | None = None):
        super().__init__(rng=rng)
        self.operators = operators

    def crossover(self, parent1: HybridGenotype, parent2: HybridGenotype) -> tuple[HybridGenotype, HybridGenotype]:  # type: ignore[override]
        child1_parts, child2_parts = {}, {}
        for key in parent1.components:
            op = self.operators.get(key)
            if op:
                c1, c2 = op.crossover(parent1.components[key], parent2.components[key])
                child1_parts[key], child2_parts[key] = c1, c2
            else:
                child1_parts[key], child2_parts[key] = parent1.components[key], parent2.components[key]
        return HybridGenotype(child1_parts), HybridGenotype(child2_parts)


def _crossover_component(
    key: str, parent1_part: Genotype, parent2_part: Genotype, operator: CrossoverOperator | None
) -> tuple[str, Genotype, Genotype]:
    """
    Top-level helper function for parallel hybrid crossover.
    Must be defined at module scope for ProcessPoolExecutor pickling.
    Returns (key, child1_part, child2_part).
    """
    if operator is None:
        # Fallback — no crossover operator: clone parents
        child1_part = parent1_part.copy()
        child2_part = parent2_part.copy()
    else:
        result = operator.crossover(parent1_part, parent2_part)
        if isinstance(result, tuple) and len(result) == 2:
            child1_part, child2_part = result
        else:
            # Handle operators that return only one child
            child1_part = result
            child2_part = (
                operator.crossover(parent2_part, parent1_part)[0] if hasattr(operator, "crossover") else result
            )
    return key, child1_part, child2_part


class ParallelHybridCrossover:
    """
    Parallelized crossover for HybridGenotype using ProcessPoolExecutor.
    Each sub-genotype can have its own crossover operator.
    Returns two HybridGenotypes (children).
    """

    def __init__(
        self,
        operators: dict[str, CrossoverOperator],
        max_workers: int | None = None,
        persistent: bool = False,
        rng: np.random.Generator | None = None,
    ):
        """
        Args:
            operators: Mapping of genotype part keys to crossover operators.
            max_workers: Number of parallel workers (default: CPU count).
            persistent: If True, keeps a persistent pool open for reuse.
        """
        self.operators = operators
        self.max_workers = max_workers
        self.persistent = persistent
        self._executor = ProcessPoolExecutor(max_workers=max_workers) if persistent else None
        self.rng = rng if rng is not None else np.random.default_rng()

    def crossover(self, parent1: HybridGenotype, parent2: HybridGenotype) -> tuple[HybridGenotype, HybridGenotype]:
        """
        Perform parallel crossover between two hybrid parents.

        Returns:
            Tuple[HybridGenotype, HybridGenotype]: Two child genotypes.
        """
        if not isinstance(parent1, HybridGenotype) or not isinstance(parent2, HybridGenotype):
            raise TypeError("ParallelHybridCrossover expects HybridGenotype parents.")

        executor = self._executor or ProcessPoolExecutor(max_workers=self.max_workers)
        futures = []

        for key, _ in parent1.components.items():
            op = self.operators.get(key)
            part1 = parent1.components[key]
            part2 = parent2.components[key]
            futures.append(executor.submit(_crossover_component, key, part1, part2, op))

        child1_parts, child2_parts = {}, {}

        for f in as_completed(futures):
            key, part1, part2 = f.result()
            child1_parts[key] = part1
            child2_parts[key] = part2

        if not self.persistent:
            executor.shutdown()

        return HybridGenotype(child1_parts), HybridGenotype(child2_parts)

    def close(self):
        """Shutdown persistent process pool."""
        if self._executor:
            self._executor.shutdown()
            self._executor = None


# =============================================================================
# Binary / Real / Integer crossovers
# =============================================================================
class OnePointCrossover(CrossoverOperator):
    """
    Performs one-point crossover.

    This method creates two offspring by combining the genes of the parents at a single crossover point.
    """

    supported_genotypes: tuple[type[Genotype], ...] = (BinaryGenotype, RealGenotype, IntegerGenotype)

    def crossover(self, p1: Genotype, p2: Genotype):
        if type(p1) is not type(p2):
            raise TypeError("Parents must be of the same genotype type.")
        point = self.rng.integers(1, len(p1.genes))
        c1_genes = np.concatenate([p1.genes[:point], p2.genes[point:]])
        c2_genes = np.concatenate([p2.genes[:point], p1.genes[point:]])
        if isinstance(p1, RealGenotype):
            return RealGenotype(c1_genes.astype(p1.genes.dtype, copy=False), p1.bounds), RealGenotype(
                c2_genes.astype(p1.genes.dtype, copy=False), p1.bounds
            )
        if isinstance(p1, IntegerGenotype):
            return IntegerGenotype(c1_genes.astype(p1.genes.dtype, copy=False), p1.bounds), IntegerGenotype(
                c2_genes.astype(p1.genes.dtype, copy=False), p1.bounds
            )
        return p1.__class__(c1_genes), p1.__class__(c2_genes)


class TwoPointCrossover(CrossoverOperator):
    """
    Performs two-point crossover.

    This method creates two offspring by combining the genes of the parents between two crossover points.
    """

    supported_genotypes: tuple[type[Genotype], ...] = (BinaryGenotype, RealGenotype, IntegerGenotype)

    def crossover(self, p1: Genotype, p2: Genotype):
        if type(p1) is not type(p2):
            raise TypeError("Parents must be of the same genotype type.")
        a, b = sorted(self.rng.choice(range(len(p1.genes)), 2, replace=False))
        c1_genes = p1.genes.copy()
        c2_genes = p2.genes.copy()
        c1_genes[a:b], c2_genes[a:b] = p2.genes[a:b], p1.genes[a:b]
        if isinstance(p1, RealGenotype):
            return RealGenotype(c1_genes, p1.bounds), RealGenotype(c2_genes, p1.bounds)
        if isinstance(p1, IntegerGenotype):
            return IntegerGenotype(c1_genes, p1.bounds), IntegerGenotype(c2_genes, p1.bounds)
        return p1.__class__(c1_genes), p1.__class__(c2_genes)


class UniformCrossover(CrossoverOperator):
    """
    Performs uniform crossover with a mixing probability.

    This method creates two offspring by randomly selecting genes from each parent
    based on the specified probability.
    """

    supported_genotypes: tuple[type[Genotype], ...] = (BinaryGenotype, RealGenotype, IntegerGenotype)

    def __init__(self, probability: float = 0.5):
        if not (0.0 <= probability <= 1.0):
            raise ValueError("probability must be in [0,1]")
        super().__init__()
        self.probability = probability

    def crossover(self, p1: Genotype, p2: Genotype):
        if type(p1) is not type(p2):
            raise TypeError("Parents must be of the same genotype type.")
        mask = self.rng.random(len(p1.genes)) < self.probability
        c1_genes = np.where(mask, p1.genes, p2.genes)
        c2_genes = np.where(mask, p2.genes, p1.genes)
        if isinstance(p1, RealGenotype):
            return RealGenotype(c1_genes.astype(p1.genes.dtype, copy=False), p1.bounds), RealGenotype(
                c2_genes.astype(p1.genes.dtype, copy=False), p1.bounds
            )
        if isinstance(p1, IntegerGenotype):
            return IntegerGenotype(c1_genes.astype(p1.genes.dtype, copy=False), p1.bounds), IntegerGenotype(
                c2_genes.astype(p1.genes.dtype, copy=False), p1.bounds
            )
        return p1.__class__(c1_genes), p1.__class__(c2_genes)


# =============================================================================
# Real-valued specific crossovers
# =============================================================================
class ArithmeticCrossover(CrossoverOperator):
    """
    Performs arithmetic crossover: child = α*p1 + (1−α)*p2.

    This method creates two offspring by linearly combining the genes of the parents
    using a specified alpha parameter.
    """  # noqa: RUF002

    supported_genotypes: tuple[type[Genotype], ...] = (RealGenotype,)

    def __init__(self, alpha: float = 0.5):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1]")
        super().__init__()
        self.alpha = alpha

    def crossover(self, p1: RealGenotype, p2: RealGenotype):  # type: ignore[override]
        if not isinstance(p1, RealGenotype) or not isinstance(p2, RealGenotype):
            raise TypeError("ArithmeticCrossover is only applicable to RealGenotype.")
        c1_genes = self.alpha * p1.genes + (1 - self.alpha) * p2.genes
        c2_genes = self.alpha * p2.genes + (1 - self.alpha) * p1.genes
        low, high = p1.bounds
        c1_genes = np.clip(c1_genes, low, high)
        c2_genes = np.clip(c2_genes, low, high)
        return RealGenotype(c1_genes, p1.bounds), RealGenotype(c2_genes, p1.bounds)


class BlendCrossover(CrossoverOperator):
    """
    Implements BLX-α (blend) crossover.

    This method creates two offspring by sampling genes from an extended range
    around the parents' genes, controlled by the alpha parameter.
    """  # noqa: RUF002

    supported_genotypes: tuple[type[Genotype], ...] = (RealGenotype,)

    def __init__(self, alpha: float = 0.5):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1]")
        super().__init__()
        self.alpha = alpha

    def crossover(self, p1: RealGenotype, p2: RealGenotype):  # type: ignore[override]
        if not isinstance(p1, RealGenotype) or not isinstance(p2, RealGenotype):
            raise TypeError("BlendCrossover is only applicable to RealGenotype.")
        low, high = np.minimum(p1.genes, p2.genes), np.maximum(p1.genes, p2.genes)
        diff = high - low
        lower = low - self.alpha * diff
        upper = high + self.alpha * diff
        c1_genes = self.rng.uniform(lower, upper)
        c2_genes = self.rng.uniform(lower, upper)
        clip_low, clip_high = p1.bounds
        c1_genes = np.clip(c1_genes, clip_low, clip_high)
        c2_genes = np.clip(c2_genes, clip_low, clip_high)
        return RealGenotype(c1_genes, p1.bounds), RealGenotype(c2_genes, p1.bounds)


class SimulatedBinaryCrossover(CrossoverOperator):
    """
    Implements SBX (Simulated Binary Crossover).

    This method creates two offspring by simulating the behavior of single-point crossover
    on binary-encoded genotypes, adapted for real-valued genes.
    """

    supported_genotypes: tuple[type[Genotype], ...] = (RealGenotype,)

    def __init__(self, eta: float = 15.0, probability: float = 1.0):
        if eta <= 0:
            raise ValueError("eta must be > 0")
        if not (0.0 <= probability <= 1.0):
            raise ValueError("probability must be in [0,1]")
        super().__init__()
        self.eta = eta
        self.probability = probability

    def crossover(self, p1: RealGenotype, p2: RealGenotype):  # type: ignore[override]
        if not isinstance(p1, RealGenotype) or not isinstance(p2, RealGenotype):
            raise TypeError("SimulatedBinaryCrossover is only applicable to RealGenotype.")
        c1_genes = p1.genes.copy()
        c2_genes = p2.genes.copy()
        for i in range(len(p1.genes)):
            if self.rng.random() < self.probability:
                y1, y2 = min(p1.genes[i], p2.genes[i]), max(p1.genes[i], p2.genes[i])
                if y1 == y2:
                    continue
                low, high = p1.bounds
                rand = self.rng.random()
                beta = 1.0 + (2.0 * (y1 - low) / (y2 - y1))
                alpha = 2.0 - beta ** -(self.eta + 1)
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (self.eta + 1))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta + 1))
                c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                c1_genes[i], c2_genes[i] = np.clip([c1, c2], low, high)
        return RealGenotype(c1_genes, p1.bounds), RealGenotype(c2_genes, p1.bounds)


# =============================================================================
# Permutation crossovers
# =============================================================================
class OrderCrossover(CrossoverOperator):
    """
    Implements Order Crossover (OX).

    This method creates two offspring by preserving the relative order of genes from the parents.
    """

    supported_genotypes: tuple[type[Genotype], ...] = (PermutationGenotype,)

    def crossover(self, p1: PermutationGenotype, p2: PermutationGenotype):  # type: ignore[override]
        if not isinstance(p1, self.supported_genotypes) or not isinstance(p2, self.supported_genotypes):
            raise TypeError(f"OrderCrossover is only applicable to {self.supported_genotypes}.")
        n = len(p1.genes)
        a, b = sorted(self.rng.choice(range(n), 2, replace=False))
        c1 = [-1] * n
        c2 = [-1] * n
        c1[a:b], c2[a:b] = p1.genes[a:b], p2.genes[a:b]
        fill1 = [g for g in p2.genes if g not in c1]
        fill2 = [g for g in p1.genes if g not in c2]
        idxs = list(range(0, a)) + list(range(b, n))
        for i, g in zip(idxs, fill1, strict=True):
            c1[i] = g
        for i, g in zip(idxs, fill2, strict=True):
            c2[i] = g
        return PermutationGenotype(np.array(c1)), PermutationGenotype(np.array(c2))


class PartiallyMappedCrossover(CrossoverOperator):
    """
    Implements Partially Mapped Crossover (PMX) for permutation genotypes.

    This method creates two offspring by exchanging segments between parents
    and resolving conflicts through mapping.
    """

    supported_genotypes: tuple[type[Genotype], ...] = (PermutationGenotype,)

    def crossover(self, p1: PermutationGenotype, p2: PermutationGenotype):  # type: ignore[override]
        if not isinstance(p1, self.supported_genotypes) or not isinstance(p2, self.supported_genotypes):
            raise TypeError(f"PMX is only applicable to {self.supported_genotypes}.")
        n = len(p1.genes)
        a, b = sorted(self.rng.choice(range(n), 2, replace=False))

        # Initialize offspring
        c1, c2 = np.full(n, -1, dtype=int), np.full(n, -1, dtype=int)
        # Copy crossover segments
        c1[a:b] = p2.genes[a:b]
        c2[a:b] = p1.genes[a:b]

        # Build mapping dictionaries
        map1 = {p2.genes[i]: p1.genes[i] for i in range(a, b)}
        map2 = {p1.genes[i]: p2.genes[i] for i in range(a, b)}

        # Fill the rest of c1
        for i in list(range(0, a)) + list(range(b, n)):
            gene = p1.genes[i]
            # Resolve mapping conflicts
            while gene in c1[a:b]:
                gene = map1[gene]
            c1[i] = gene

        # Fill the rest of c2
        for i in list(range(0, a)) + list(range(b, n)):
            gene = p2.genes[i]
            while gene in c2[a:b]:
                gene = map2[gene]
            c2[i] = gene

        return PermutationGenotype(c1), PermutationGenotype(c2)


class CycleCrossover(CrossoverOperator):
    """
    Implements Cycle Crossover (CX).

    This method creates two offspring by preserving the position of genes from the parents
    through cycles.
    """

    supported_genotypes: tuple[type[Genotype], ...] = (PermutationGenotype,)

    def crossover(self, p1: PermutationGenotype, p2: PermutationGenotype):  # type: ignore[override]
        if not isinstance(p1, self.supported_genotypes) or not isinstance(p2, self.supported_genotypes):
            raise TypeError(f"CycleCrossover is only applicable to {self.supported_genotypes}.")
        n = len(p1.genes)
        c1, c2 = np.empty(n, dtype=int), np.empty(n, dtype=int)
        c1[:] = -1
        c2[:] = -1
        cycles = 0
        remaining = set(range(n))
        while remaining:
            start = next(iter(remaining))
            idx = start
            while True:
                c1[idx] = p1.genes[idx] if cycles % 2 == 0 else p2.genes[idx]
                c2[idx] = p2.genes[idx] if cycles % 2 == 0 else p1.genes[idx]
                remaining.discard(idx)
                idx = np.where(p1.genes == p2.genes[idx])[0][0]
                if idx == start:
                    break
            cycles += 1
        return PermutationGenotype(c1), PermutationGenotype(c2)


class EdgeRecombinationCrossover(CrossoverOperator):
    """
    Implements Edge Recombination Crossover (ERX).

    This method creates two offspring by combining the edges of the parents.
    """

    supported_genotypes: tuple[type[Genotype], ...] = (PermutationGenotype,)

    def crossover(self, p1: PermutationGenotype, p2: PermutationGenotype):  # type: ignore[override]
        if not isinstance(p1, self.supported_genotypes) or not isinstance(p2, self.supported_genotypes):
            raise TypeError(f"EdgeRecombinationCrossover is only applicable to {self.supported_genotypes}.")
        n = len(p1.genes)

        def build_edge_map(p1, p2):
            edges = {g: set() for g in p1.genes}
            for perm in [p1.genes, p2.genes]:
                for i in range(n):
                    left = perm[i - 1]
                    right = perm[(i + 1) % n]
                    edges[perm[i]].update([left, right])
            return edges

        def generate_child(edges):
            remaining = set(edges.keys())
            current = self.rng.choice(list(remaining))
            child = [current]
            remaining.remove(current)
            while remaining:
                for v in edges.values():
                    v.discard(current)
                if edges[current]:
                    next_gene = min(edges[current], key=lambda x: len(edges[x]))
                else:
                    next_gene = self.rng.choice(list(remaining))
                child.append(next_gene)
                remaining.remove(next_gene)
                current = next_gene
            return np.array(child)

        edges = build_edge_map(p1, p2)
        c1 = generate_child(edges)
        c2 = generate_child(edges)
        return PermutationGenotype(c1), PermutationGenotype(c2)
