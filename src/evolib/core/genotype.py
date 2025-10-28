"""
evolib.core.genotype
====================

This module defines various genotype representations used in evolutionary algorithms.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Genotype(ABC):
    """Abstract base class for genotypes."""
    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def copy(self) -> "Genotype":
        """Create a deep copy of the genotype."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the length of the genotype."""
        pass

    def __repr__(self) -> str:
        """Return a string representation of the genotype."""
        return f"{self.__class__.__name__}(shape={self.as_array().shape})"
    
    def as_array(self) -> np.ndarray:
        """Return the genes as a numpy array."""
        return self.genes


# =============================================================================
# BinaryGenotype (0/1)
# =============================================================================
class BinaryGenotype(Genotype):
    def __init__(self, genes: np.ndarray):
        if genes.dtype != np.bool_:
            raise TypeError(
                f"BinaryGenotype genes must be boolean (np.bool_), got dtype={genes.dtype}."
            )
        self.genes: np.ndarray = genes

    @classmethod
    def random(cls, length: int, p: float = 0.5) -> "BinaryGenotype":
        genes = np.random.rand(length) < p
        return cls(genes)

    def copy(self) -> "BinaryGenotype":
        return BinaryGenotype(np.copy(self.genes))
    
    def __len__(self) -> int:
        return self.genes.size
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, BinaryGenotype):
            return False
        return np.array_equal(self.genes, other.genes)
    

# =============================================================================
# RealGenotype
# =============================================================================
class RealGenotype(Genotype):
    """Genotype with real-valued genes."""
    def __init__(self, genes: np.ndarray, bounds: Tuple[float, float] = (0.0, 1.0)):
        if genes.dtype not in (np.float32, np.float64):
            raise TypeError(
                f"RealGenotype genes must be float32/float64, got dtype={genes.dtype}."
            )
        if bounds[0] >= bounds[1]:
            raise ValueError(f"Invalid bounds {bounds}: low must be < high.")
        self.genes: np.ndarray = genes
        self.bounds: Tuple[float, float] = bounds

    @classmethod
    def random(cls, length: int, bounds: Tuple[float, float] = (0.0, 1.0)) -> "RealGenotype":
        """Generate a random RealGenotype within the specified bounds."""
        low, high = bounds
        genes = np.random.uniform(low, high, size=length).astype(np.float64)
        return cls(genes, bounds)

    def copy(self) -> "RealGenotype":
        return RealGenotype(np.copy(self.genes), self.bounds)
    
    def __len__(self) -> int:
        return self.genes.size
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, RealGenotype):
            return False
        return np.array_equal(self.genes, other.genes) and self.bounds == other.bounds
    

# =============================================================================
# IntegerGenotype
# =============================================================================
class IntegerGenotype(Genotype):
    """Genotype with integer-valued genes."""
    def __init__(self, genes: np.ndarray, bounds: Tuple[int, int] = (0, 10)):
        if not np.issubdtype(genes.dtype, np.integer):
            raise TypeError(
                f"IntegerGenotype genes must be integer dtype, got dtype={genes.dtype}."
            )
        if bounds[0] > bounds[1]:
            raise ValueError(f"Invalid bounds {bounds}: low must be <= high.")
        self.genes: np.ndarray = genes
        self.bounds: Tuple[int, int] = bounds

    @classmethod
    def random(cls, length: int, bounds: Tuple[int, int] = (0, 10)) -> "IntegerGenotype":
        low, high = bounds
        genes = np.random.randint(low, high + 1, size=length, dtype=np.int32)
        return cls(genes, bounds)

    def copy(self) -> "IntegerGenotype":
        return IntegerGenotype(np.copy(self.genes), self.bounds)
    
    def __len__(self) -> int:
        return self.genes.size
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, IntegerGenotype):
            return False
        return np.array_equal(self.genes, other.genes) and self.bounds == other.bounds
    
    def __sub__(self, other: "IntegerGenotype"):
        if not isinstance(other, IntegerGenotype):
            raise TypeError("Subtraction only supported between IntegerGenotype instances.")
        return self.genes - other.genes
    

# =============================================================================
# PermutationGenotype
# =============================================================================
class PermutationGenotype(Genotype):
    """Genotype representing a permutation of integers."""
    def __init__(self, genes: np.ndarray):
        if not np.issubdtype(genes.dtype, np.integer):
            raise TypeError(
                f"PermutationGenotype genes must be integer dtype, got dtype={genes.dtype}."
            )
        unique_genes = np.unique(genes)
        expected_genes = np.arange(len(genes))
        if not (
            len(unique_genes) == len(genes)
            and np.array_equal(np.sort(unique_genes), expected_genes)
        ):
            raise ValueError(
                "Genes must be a permutation of integers from 0 to len(genes)-1."
            )
        self.genes: np.ndarray = genes.astype(np.int32)

    @classmethod
    def random(cls, length: int) -> "PermutationGenotype":
        genes = np.random.permutation(length).astype(np.int32)
        return cls(genes)

    def copy(self) -> "PermutationGenotype":
        return PermutationGenotype(np.copy(self.genes))
    
    def __len__(self) -> int:
        return self.genes.size
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, PermutationGenotype):
            return False
        return np.array_equal(self.genes, other.genes)
