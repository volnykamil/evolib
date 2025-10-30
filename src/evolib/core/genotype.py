"""
evolib.core.genotype
====================

This module defines various genotype representations used in evolutionary algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Genotype(ABC):
    """Abstract base class for genotypes."""

    def __init__(self, genes: np.ndarray):
        self.genes: np.ndarray = genes

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the length of the genotype."""
        pass

    @abstractmethod
    def __sub__(self, other: Genotype) -> np.ndarray:
        """Return element-wise subtraction (self - other) of gene arrays."""
        pass

    def __repr__(self) -> str:
        """Return a string representation of the genotype."""
        return f"{self.__class__.__name__}(shape={self.as_array().shape})"

    @abstractmethod
    def copy(self) -> Genotype:
        """Create a deep copy of the genotype."""
        pass

    def as_array(self) -> np.ndarray:
        """Return the genes as a numpy array."""
        return self.genes


class BinaryGenotype(Genotype):
    def __init__(self, genes: np.ndarray):
        super().__init__(genes)
        if genes.dtype != np.bool_:
            raise TypeError(f"BinaryGenotype genes must be boolean (np.bool_), got dtype={genes.dtype}.")

    @classmethod
    def random(cls, length: int, p: float = 0.5, rng: np.random.Generator | None = None) -> BinaryGenotype:
        """Create a random binary genotype.

        Parameters
        ----------
        length : int
            Number of bits.
        p : float, default 0.5
            Probability that a bit is True.
        rng : numpy.random.Generator | None, default None
            Optional RNG for reproducibility. Falls back to global ``np.random`` if None.
        """
        _rng = rng if rng is not None else np.random.default_rng()
        genes = _rng.random(length) < p
        return cls(genes.astype(np.bool_))

    def __eq__(self, other) -> bool:
        if not isinstance(other, BinaryGenotype):
            return False
        return np.array_equal(self.genes, other.genes)

    def __hash__(self):
        return hash(self.genes.tobytes())

    def __len__(self) -> int:
        return self.genes.size

    def __sub__(self, other: BinaryGenotype) -> np.ndarray:  # type: ignore[override]
        if not isinstance(other, BinaryGenotype):
            raise TypeError("Subtraction only supported between BinaryGenotype instances.")
        return self.genes.astype(np.int8) - other.genes.astype(np.int8)

    def copy(self) -> BinaryGenotype:
        return BinaryGenotype(np.copy(self.genes))


# =============================================================================
# RealGenotype
# =============================================================================
class RealGenotype(Genotype):
    """Genotype with real-valued genes."""

    def __init__(self, genes: np.ndarray, bounds: tuple[float, float] = (0.0, 1.0)):
        if genes.dtype not in (np.float32, np.float64):
            raise TypeError(f"RealGenotype genes must be float32/float64, got dtype={genes.dtype}.")
        if bounds[0] >= bounds[1]:
            raise ValueError(f"Invalid bounds {bounds}: low must be < high.")
        super().__init__(genes)
        self.bounds: tuple[float, float] = bounds

    @classmethod
    def random(
        cls, length: int, bounds: tuple[float, float] = (0.0, 1.0), rng: np.random.Generator | None = None
    ) -> RealGenotype:
        """Create a random real-valued genotype.

        Parameters
        ----------
        length : int
            Number of genes.
        bounds : tuple[float, float], default (0.0, 1.0)
            Inclusive lower and upper bounds for uniform sampling.
        rng : numpy.random.Generator | None, default None
            Optional RNG for reproducibility. Falls back to global ``np.random`` if None.
        """
        low, high = bounds
        _rng = rng if rng is not None else np.random.default_rng()
        genes = _rng.uniform(low, high, size=length).astype(np.float64)
        return cls(genes, bounds)

    def __eq__(self, other) -> bool:
        if not isinstance(other, RealGenotype):
            return False
        return np.array_equal(self.genes, other.genes) and self.bounds == other.bounds

    def __hash__(self):
        return hash((self.genes.tobytes(), self.bounds))

    def __len__(self) -> int:
        return self.genes.size

    def __sub__(self, other: RealGenotype) -> np.ndarray:  # type: ignore[override]
        if not isinstance(other, RealGenotype):
            raise TypeError("Subtraction only supported between RealGenotype instances.")
        return self.genes - other.genes

    def copy(self) -> RealGenotype:
        return RealGenotype(np.copy(self.genes), self.bounds)


# =============================================================================
# IntegerGenotype
# =============================================================================
class IntegerGenotype(Genotype):
    """Genotype with integer-valued genes."""

    def __init__(self, genes: np.ndarray, bounds: tuple[int, int] = (0, 10)):
        if not np.issubdtype(genes.dtype, np.integer):
            raise TypeError(f"IntegerGenotype genes must be integer dtype, got dtype={genes.dtype}.")
        if bounds[0] > bounds[1]:
            raise ValueError(f"Invalid bounds {bounds}: low must be <= high.")
        super().__init__(genes)
        self.bounds: tuple[int, int] = bounds

    @classmethod
    def random(
        cls, length: int, bounds: tuple[int, int] = (0, 10), rng: np.random.Generator | None = None
    ) -> IntegerGenotype:
        """Create a random integer genotype.

        Parameters
        ----------
        length : int
            Number of genes.
        bounds : tuple[int, int], default (0, 10)
            Inclusive lower and upper bounds for uniform integer sampling.
        rng : numpy.random.Generator | None, default None
            Optional RNG for reproducibility. Falls back to global ``np.random`` if None.
        """
        low, high = bounds
        _rng = rng if rng is not None else np.random.default_rng()
        genes = _rng.integers(low, high + 1, size=length, dtype=np.int32)
        return cls(genes, bounds)

    def __eq__(self, other) -> bool:
        if not isinstance(other, IntegerGenotype):
            return False
        return np.array_equal(self.genes, other.genes) and self.bounds == other.bounds

    def __hash__(self):
        return hash((self.genes.tobytes(), self.bounds))

    def __len__(self) -> int:
        return self.genes.size

    def __sub__(self, other: IntegerGenotype) -> np.ndarray:  # type: ignore[override]
        if not isinstance(other, IntegerGenotype):
            raise TypeError("Subtraction only supported between IntegerGenotype instances.")
        return self.genes - other.genes

    def copy(self) -> IntegerGenotype:
        return IntegerGenotype(np.copy(self.genes), self.bounds)


# =============================================================================
# PermutationGenotype
# =============================================================================
class PermutationGenotype(Genotype):
    """Genotype representing a permutation of integers."""

    def __init__(self, genes: np.ndarray):
        if not np.issubdtype(genes.dtype, np.integer):
            raise TypeError(f"PermutationGenotype genes must be integer dtype, got dtype={genes.dtype}.")
        super().__init__(genes)
        unique_genes = np.unique(genes)
        expected_genes = np.arange(len(genes))
        if not (len(unique_genes) == len(genes) and np.array_equal(np.sort(unique_genes), expected_genes)):
            raise ValueError("Genes must be a permutation of integers from 0 to len(genes)-1.")
        self.genes: np.ndarray = genes.astype(np.int32)

    @classmethod
    def random(cls, length: int, rng: np.random.Generator | None = None) -> PermutationGenotype:
        """Create a random permutation genotype.

        Parameters
        ----------
        length : int
            Size of the permutation (values 0..length-1).
        rng : numpy.random.Generator | None, default None
            Optional RNG for reproducibility. Falls back to global ``np.random`` if None.
        """
        _rng = rng if rng is not None else np.random.default_rng()
        genes = _rng.permutation(length).astype(np.int32)
        return cls(genes)

    def __eq__(self, other) -> bool:
        if not isinstance(other, PermutationGenotype):
            return False
        return np.array_equal(self.genes, other.genes)

    def __hash__(self):
        return hash(self.genes.tobytes())

    def __len__(self) -> int:
        return self.genes.size

    def __sub__(self, other: PermutationGenotype) -> np.ndarray:  # type: ignore[override]
        if not isinstance(other, PermutationGenotype):
            raise TypeError("Subtraction only supported between PermutationGenotype instances.")
        return self.genes - other.genes

    def copy(self) -> PermutationGenotype:
        return PermutationGenotype(np.copy(self.genes))


# =============================================================================
# CombinedGenotype
# =============================================================================
class HybridGenotype(Genotype):
    """Genotype composed of multiple sub-genotypes."""

    def __init__(self, components: dict[str, Genotype]):
        if not isinstance(components, dict) or not all(isinstance(v, Genotype) for v in components.values()):
            raise TypeError("Components must be a dictionary of Genotype instances.")
        self.components = components

    def __eq__(self, other) -> bool:
        if not isinstance(other, HybridGenotype):
            return False
        if self.components.keys() != other.components.keys():
            return False
        return all(self.components[k] == other.components[k] for k in self.components)

    def __hash__(self):
        return hash(tuple((k, self.components[k].__hash__()) for k in sorted(self.components.keys())))

    def __len__(self) -> int:
        return sum(len(v) for v in self.components.values())

    def __sub__(self, other: HybridGenotype) -> np.ndarray:  # type: ignore[override]
        if not isinstance(other, HybridGenotype):
            raise TypeError("Subtraction only supported between HybridGenotype instances.")
        if self.components.keys() != other.components.keys():
            raise ValueError("Both HybridGenotypes must have the same component keys for subtraction.")
        arrays = [self.components[k] - other.components[k] for k in self.components]
        return np.concatenate(arrays)

    def as_array(self) -> np.ndarray:
        """Vrací zřetězené hodnoty všech komponent."""
        arrays = [v.as_array().ravel() for v in self.components.values()]
        return np.concatenate(arrays)

    def __repr__(self):
        comp_info = ", ".join(f"{k}:{v.__class__.__name__}" for k, v in self.components.items())
        return f"HybridGenotype({comp_info})"

    def copy(self) -> HybridGenotype:
        return HybridGenotype({k: v.copy() for k, v in self.components.items()})
