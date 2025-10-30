"""
evolib.operators.mutation
=========================

This module defines various mutation operators for evolving genotypes.
Supports: BinaryGenotype, IntegerGenotype, RealGenotype, and PermutationGenotype.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor

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
class MutationOperator(ABC):
    """Abstract base class for mutation operators."""

    supported_genotypes: tuple[type[Genotype], ...] = ()

    @abstractmethod
    def mutate(self, genotype: Genotype) -> Genotype:
        """Return a mutated copy of the genotype."""
        pass


# =============================================================================
# HybridGenotype mutations
# =============================================================================
class HybridMutation(MutationOperator):
    """Component-wise mutation (sequential)."""

    def __init__(self, operators: dict[str, MutationOperator]):
        self.operators = operators

    def mutate(self, genotype: HybridGenotype) -> HybridGenotype:  # type: ignore[override]
        new_components = {}
        for key, subgen in genotype.components.items():
            op = self.operators.get(key)
            new_components[key] = op.mutate(subgen) if op else subgen
        return HybridGenotype(new_components)


def _mutate_component(key: str, subgen: Genotype, op: MutationOperator | None) -> tuple[str, Genotype]:
    """Helper function for parallel mutation (must be top-level for pickling)."""
    return key, op.mutate(subgen) if op else subgen


class ParallelHybridMutation(HybridMutation):
    """
    Parallelized variant of HybridMutation using ProcessPoolExecutor.
    Designed for CPU-bound evolutionary operations.
    """

    def __init__(
        self, operators: dict[str, MutationOperator], max_workers: int | None = None, persistent: bool = False
    ):
        self.operators = operators
        self.max_workers = max_workers
        self.persistent = persistent
        self._executor = ProcessPoolExecutor(max_workers=max_workers) if persistent else None

    def mutate(self, genotype: HybridGenotype) -> HybridGenotype:  # type: ignore[override]
        executor = self._executor or ProcessPoolExecutor(max_workers=self.max_workers)
        futures = [
            executor.submit(_mutate_component, key, subgen, self.operators.get(key))
            for key, subgen in genotype.components.items()
        ]
        results = dict(f.result() for f in futures)
        if not self.persistent:
            executor.shutdown()
        return HybridGenotype(results)

    def close(self):
        if self._executor:
            self._executor.shutdown()


# =============================================================================
# BinaryGenotype mutations
# =============================================================================
class BitFlipMutation(MutationOperator):
    """
    Flips bits in a BinaryGenotype with a given probability.

    Parameters:
        probability (float): Probability of flipping each bit.
    """

    supported_genotypes: tuple[type[Genotype]] = (BinaryGenotype,)

    def __init__(self, probability: float = 0.01):
        self.probability = probability

    def mutate(self, genotype: BinaryGenotype) -> BinaryGenotype:  # type: ignore[override]
        if not isinstance(genotype, self.supported_genotypes):
            # Format tuple of supported names exactly as tests expect: ('BinaryGenotype',)
            names = tuple(st.__name__ for st in self.supported_genotypes)
            raise TypeError(f"BitFlipMutation is only applicable to {names}.")
        genes = genotype.genes.copy()
        mask = np.random.rand(len(genes)) < self.probability
        genes[mask] = ~genes[mask]
        return BinaryGenotype(genes)


# =============================================================================
# RealGenotype mutations
# =============================================================================
class GaussianMutation(MutationOperator):
    """
    Adds Gaussian noise to RealGenotype genes.

    Parameters:
        sigma (float): Standard deviation of the Gaussian noise.
        probability (float): Probability of mutating each gene.
    """

    supported_genotypes: tuple[type[Genotype], ...] = (RealGenotype,)

    def __init__(self, sigma: float = 0.1, probability: float = 0.1):
        self.sigma = sigma
        self.probability = probability

    def mutate(self, genotype: RealGenotype) -> RealGenotype:  # type: ignore[override]
        if not isinstance(genotype, self.supported_genotypes):
            names = tuple(st.__name__ for st in self.supported_genotypes)
            raise TypeError(f"GaussianMutation is only applicable to {names}.")
        genes = genotype.genes.copy()
        mask = np.random.rand(len(genes)) < self.probability
        noise = np.random.normal(0, self.sigma, size=len(genes))
        genes[mask] += noise[mask]
        low, high = genotype.bounds
        genes = np.clip(genes, low, high)
        return RealGenotype(genes, genotype.bounds)


class UniformMutation(MutationOperator):
    """
    Replaces RealGenotype genes with uniform random values within bounds.

    Parameters:
        probability (float): Probability of mutating each gene.
    """

    supported_genotypes: tuple[type[Genotype], ...] = (RealGenotype,)

    def __init__(self, probability: float = 0.1):
        self.probability = probability

    def mutate(self, genotype: RealGenotype) -> RealGenotype:  # type: ignore[override]
        if not isinstance(genotype, self.supported_genotypes):
            # Test expects plain class name without tuple/quotes.
            names = ", ".join(st.__name__ for st in self.supported_genotypes)
            raise TypeError(f"UniformMutation is only applicable to {names}.")
        genes = genotype.genes.copy()
        mask = np.random.rand(len(genes)) < self.probability
        low, high = genotype.bounds
        genes[mask] = np.random.uniform(low, high, size=int(np.sum(mask)))
        return RealGenotype(genes, genotype.bounds)


class NonUniformMutation(MutationOperator):
    """
    Applies non-uniform mutation to RealGenotype genes, decreasing variance over time.

    Parameters:
        progress (float): Evolution progress [0,1].
        sigma_max (float): Maximum std deviation.
        sigma_min (float): Minimum std deviation.
        probability (float): Probability of mutating each gene.
    """

    supported_genotypes: tuple[type[Genotype], ...] = (RealGenotype,)

    def __init__(
        self,
        progress: float,
        sigma_max: float = 0.3,
        sigma_min: float = 0.01,
        probability: float = 0.1,
    ):
        assert 0.0 <= progress <= 1.0, "Progress must be in [0,1]"
        self.progress = progress
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.probability = probability

    def mutate(self, genotype: RealGenotype) -> RealGenotype:  # type: ignore[override]
        if not isinstance(genotype, self.supported_genotypes):
            names = ", ".join(st.__name__ for st in self.supported_genotypes)
            raise TypeError(f"NonUniformMutation is only applicable to {names}.")
        genes = genotype.genes.copy()
        mask = np.random.rand(len(genes)) < self.probability
        sigma = self.sigma_max * (1 - self.progress) + self.sigma_min * self.progress
        noise = np.random.normal(0, sigma, size=len(genes))
        genes[mask] += noise[mask]
        low, high = genotype.bounds
        genes = np.clip(genes, low, high)
        return RealGenotype(genes, genotype.bounds)


# =============================================================================
# IntegerGenotype mutations
# =============================================================================
class UniformIntegerMutation(MutationOperator):
    """
    Replaces IntegerGenotype genes with random integers within bounds.

    Parameters:
        probability (float): Probability of mutating each gene.
    """

    supported_genotypes: tuple[type[Genotype], ...] = (IntegerGenotype,)

    def __init__(self, probability: float = 0.1):
        self.probability = probability

    def mutate(self, genotype: IntegerGenotype) -> IntegerGenotype:  # type: ignore[override]
        if not isinstance(genotype, self.supported_genotypes):
            names = ", ".join(st.__name__ for st in self.supported_genotypes)
            raise TypeError(f"UniformIntegerMutation is only applicable to {names}.")
        genes = genotype.genes.copy()
        mask = np.random.rand(len(genes)) < self.probability
        low, high = genotype.bounds
        genes[mask] = np.random.randint(low, high + 1, size=int(np.sum(mask)))
        return IntegerGenotype(genes, genotype.bounds)


class CreepIntegerMutation(MutationOperator):
    """
    Adds small integer noise (±delta) to genes within bounds.

    Parameters:
        delta (int): Maximum integer change.
        probability (float): Probability of mutating each gene.
    """

    supported_genotypes: tuple[type[Genotype], ...] = (IntegerGenotype,)

    def __init__(self, delta: int = 1, probability: float = 0.1):
        self.delta = delta
        self.probability = probability

    def mutate(self, genotype: IntegerGenotype) -> IntegerGenotype:  # type: ignore[override]
        if not isinstance(genotype, self.supported_genotypes):
            names = ", ".join(st.__name__ for st in self.supported_genotypes)
            raise TypeError(f"CreepIntegerMutation is only applicable to {names}.")
        genes = genotype.genes.copy()
        mask = np.random.rand(len(genes)) < self.probability
        noise = np.random.randint(-self.delta, self.delta + 1, size=len(genes))
        genes[mask] += noise[mask]
        low, high = genotype.bounds
        genes = np.clip(genes, low, high)
        return IntegerGenotype(genes, genotype.bounds)


class NonUniformIntegerMutation(MutationOperator):
    """
    Integer version of non-uniform mutation (step size decreases with progress).

    Parameters:
        progress (float): Evolution progress [0,1].
        delta_max (int): Maximum integer delta.
        delta_min (int): Minimum integer delta.
        probability (float): Probability of mutating each gene.
    """

    supported_genotypes: tuple[type[Genotype], ...] = (IntegerGenotype,)

    def __init__(self, progress: float, delta_max: int = 5, delta_min: int = 1, probability: float = 0.1):
        assert 0.0 <= progress <= 1.0
        self.progress = progress
        self.delta_max = delta_max
        self.delta_min = delta_min
        self.probability = probability

    def mutate(self, genotype: IntegerGenotype) -> IntegerGenotype:  # type: ignore[override]
        if not isinstance(genotype, self.supported_genotypes):
            names = ", ".join(st.__name__ for st in self.supported_genotypes)
            raise TypeError(f"NonUniformIntegerMutation is only applicable to {names}.")
        genes = genotype.genes.copy()
        mask = np.random.rand(len(genes)) < self.probability
        delta = round(self.delta_max * (1 - self.progress) + self.delta_min * self.progress)
        noise = np.random.randint(-delta, delta + 1, size=len(genes))
        genes[mask] += noise[mask]
        low, high = genotype.bounds
        genes = np.clip(genes, low, high)
        return IntegerGenotype(genes, genotype.bounds)


# =============================================================================
# PermutationGenotype mutations
# =============================================================================
class SwapMutation(MutationOperator):
    """Swaps two random positions in a permutation."""

    supported_genotypes: tuple[type[Genotype], ...] = (PermutationGenotype,)

    def mutate(self, genotype: PermutationGenotype) -> PermutationGenotype:  # type: ignore[override]
        if not isinstance(genotype, self.supported_genotypes):
            names = ", ".join(st.__name__ for st in self.supported_genotypes)
            raise TypeError(f"SwapMutation is only applicable to {names}.")
        genes = genotype.genes.copy()
        i, j = np.random.choice(len(genes), 2, replace=False)
        genes[i], genes[j] = genes[j], genes[i]
        return PermutationGenotype(genes)


class InsertMutation(MutationOperator):
    """Removes one element and inserts it into a random new position."""

    supported_genotypes: tuple[type[Genotype], ...] = (PermutationGenotype,)

    def mutate(self, genotype: PermutationGenotype) -> PermutationGenotype:  # type: ignore[override]
        if not isinstance(genotype, self.supported_genotypes):
            names = ", ".join(st.__name__ for st in self.supported_genotypes)
            raise TypeError(f"InsertMutation is only applicable to {names}.")
        genes = genotype.genes.copy()
        i, j = np.random.choice(len(genes), 2, replace=False)
        gene = genes[i]
        genes = np.delete(genes, i)
        genes = np.insert(genes, j, gene)
        return PermutationGenotype(genes)


class ScrambleMutation(MutationOperator):
    """Randomly shuffles a subsequence within the permutation."""

    supported_genotypes: tuple[type[Genotype], ...] = (PermutationGenotype,)

    def mutate(self, genotype: PermutationGenotype) -> PermutationGenotype:  # type: ignore[override]
        if not isinstance(genotype, self.supported_genotypes):
            names = ", ".join(st.__name__ for st in self.supported_genotypes)
            raise TypeError(f"ScrambleMutation is only applicable to {names}.")
        genes = genotype.genes.copy()
        i, j = sorted(np.random.choice(len(genes), 2, replace=False))
        subseq = genes[i:j]
        np.random.shuffle(subseq)
        genes[i:j] = subseq
        return PermutationGenotype(genes)


class InversionMutation(MutationOperator):
    """Reverses the order of a subsequence."""

    supported_genotypes: tuple[type[Genotype], ...] = (PermutationGenotype,)

    def mutate(self, genotype: PermutationGenotype) -> PermutationGenotype:  # type: ignore[override]
        if not isinstance(genotype, self.supported_genotypes):
            names = ", ".join(st.__name__ for st in self.supported_genotypes)
            raise TypeError(f"InversionMutation is only applicable to {names}.")
        genes = genotype.genes.copy()
        i, j = sorted(np.random.choice(len(genes), 2, replace=False))
        genes[i:j] = list(reversed(genes[i:j]))
        return PermutationGenotype(genes)
