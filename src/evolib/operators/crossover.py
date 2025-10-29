"""
evolib.operators.crossover
==========================

This module defines crossover (recombination) operators
for BinaryGenotype, RealGenotype, IntegerGenotype, and PermutationGenotype.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from evolib.core.genotype import (
    Genotype,
    IntegerGenotype,
    PermutationGenotype,
    RealGenotype,
)


# =============================================================================
# Base class
# =============================================================================
class CrossoverOperator(ABC):
    """Abstract base class for crossover operators."""

    @abstractmethod
    def crossover(self, parent1: Genotype, parent2: Genotype) -> tuple[Genotype, Genotype]:
        """Return two offspring created from parent1 and parent2."""
        pass


# =============================================================================
# Binary / Real / Integer crossovers
# =============================================================================
class OnePointCrossover(CrossoverOperator):
    """
    Performs one-point crossover.

    This method creates two offspring by combining the genes of the parents at a single crossover point.
    """

    def crossover(self, p1: Genotype, p2: Genotype):
        if type(p1) is not type(p2):
            raise TypeError("Parents must be of the same genotype type.")
        point = np.random.randint(1, len(p1.genes))
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

    def crossover(self, p1: Genotype, p2: Genotype):
        if type(p1) is not type(p2):
            raise TypeError("Parents must be of the same genotype type.")
        a, b = sorted(np.random.choice(range(len(p1.genes)), 2, replace=False))
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

    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def crossover(self, p1: Genotype, p2: Genotype):
        if type(p1) is not type(p2):
            raise TypeError("Parents must be of the same genotype type.")
        mask = np.random.rand(len(p1.genes)) < self.probability
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

    def __init__(self, alpha: float = 0.5):
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

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def crossover(self, p1: RealGenotype, p2: RealGenotype):  # type: ignore[override]
        if not isinstance(p1, RealGenotype) or not isinstance(p2, RealGenotype):
            raise TypeError("BlendCrossover is only applicable to RealGenotype.")
        low, high = np.minimum(p1.genes, p2.genes), np.maximum(p1.genes, p2.genes)
        diff = high - low
        lower = low - self.alpha * diff
        upper = high + self.alpha * diff
        c1_genes = np.random.uniform(lower, upper)
        c2_genes = np.random.uniform(lower, upper)
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

    def __init__(self, eta: float = 15.0, probability: float = 1.0):
        self.eta = eta
        self.probability = probability

    def crossover(self, p1: RealGenotype, p2: RealGenotype):  # type: ignore[override]
        if not isinstance(p1, RealGenotype) or not isinstance(p2, RealGenotype):
            raise TypeError("SimulatedBinaryCrossover is only applicable to RealGenotype.")
        c1_genes = p1.genes.copy()
        c2_genes = p2.genes.copy()
        for i in range(len(p1.genes)):
            if np.random.rand() < self.probability:
                y1, y2 = min(p1.genes[i], p2.genes[i]), max(p1.genes[i], p2.genes[i])
                if y1 == y2:
                    continue
                low, high = p1.bounds
                rand = np.random.rand()
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

    def crossover(self, p1: PermutationGenotype, p2: PermutationGenotype):  # type: ignore[override]
        if not isinstance(p1, PermutationGenotype) or not isinstance(p2, PermutationGenotype):
            raise TypeError("OrderCrossover is only applicable to PermutationGenotype.")
        n = len(p1.genes)
        a, b = sorted(np.random.choice(range(n), 2, replace=False))
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

    def crossover(self, p1: PermutationGenotype, p2: PermutationGenotype):  # type: ignore[override]
        if not isinstance(p1, PermutationGenotype) or not isinstance(p2, PermutationGenotype):
            raise TypeError("PMX is only applicable to PermutationGenotype.")
        n = len(p1.genes)
        a, b = sorted(np.random.choice(range(n), 2, replace=False))

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

    def crossover(self, p1: PermutationGenotype, p2: PermutationGenotype):  # type: ignore[override]
        if not isinstance(p1, PermutationGenotype) or not isinstance(p2, PermutationGenotype):
            raise TypeError("CycleCrossover is only applicable to PermutationGenotype.")
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

    def crossover(self, p1: PermutationGenotype, p2: PermutationGenotype):  # type: ignore[override]
        if not isinstance(p1, PermutationGenotype) or not isinstance(p2, PermutationGenotype):
            raise TypeError("EdgeRecombinationCrossover is only applicable to PermutationGenotype.")
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
            current = np.random.choice(list(remaining))
            child = [current]
            remaining.remove(current)
            while remaining:
                for v in edges.values():
                    v.discard(current)
                if edges[current]:
                    next_gene = min(edges[current], key=lambda x: len(edges[x]))
                else:
                    next_gene = np.random.choice(list(remaining))
                child.append(next_gene)
                remaining.remove(next_gene)
                current = next_gene
            return np.array(child)

        edges = build_edge_map(p1, p2)
        c1 = generate_child(edges)
        c2 = generate_child(edges)
        return PermutationGenotype(c1), PermutationGenotype(c2)
