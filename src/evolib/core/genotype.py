"""
evolib.core.genotype
====================

This module defines various genotype representations used in evolutionary algorithms.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Tuple
import numpy as np


class Genotype(ABC):
    """Abstract base class for genotypes."""

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


# =============================================================================
# BinaryGenotype (0/1)
# =============================================================================
class BinaryGenotype(Genotype):
    def __init__(self, genes: np.ndarray):
        assert genes.dtype == np.bool_, "Genes must be a numpy array of boolean type."
        self.genes = genes

    @classmethod
    def random(cls, length: int, p: float = 0.5) -> "BinaryGenotype":
        genes = np.random.rand(length) < p
        return cls(genes)

    def copy(self) -> "BinaryGenotype":
        return BinaryGenotype(np.copy(self.genes))
    
    def __len__(self) -> int:
        return self.genes.size
    
    def as_array(self) -> np.ndarray:
        return self.genes
    

# =============================================================================
# RealGenotype
# =============================================================================
class RealGenotype(Genotype):
    """Genotype with real-valued genes."""
    def __init__(self, genes: np.ndarray, bounds: Tuple[float, float] = (0.0, 1.0)):
        assert genes.dtype in (np.float32, np.float64), "Genes must be a numpy array of float type."
        self.genes = genes
        self.bounds = bounds

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
    
    def as_array(self) -> np.ndarray:
        return self.genes
    

# =============================================================================
# IntegerGenotype
# =============================================================================
class IntegerGenotype(Genotype):
    """Genotype with integer-valued genes."""
    def __init__(self, genes: np.ndarray, bounds: Tuple[int, int] = (0, 10)):
        assert np.issubdtype(genes.dtype, np.integer), "Genes must be a numpy array of integer type."
        self.genes = genes
        self.bounds = bounds

    @classmethod
    def random(cls, length: int, bounds: Tuple[int, int] = (0, 10)) -> "IntegerGenotype":
        low, high = bounds
        genes = np.random.randint(low, high + 1, size=length, dtype=np.int32)
        return cls(genes, bounds)

    def copy(self) -> "IntegerGenotype":
        return IntegerGenotype(np.copy(self.genes), self.bounds)
    
    def __len__(self) -> int:
        return self.genes.size
    
    def as_array(self) -> np.ndarray:
        return self.genes
    

# =============================================================================
# PermutationGenotype
# =============================================================================
class PermutationGenotype(Genotype):
    """Genotype representing a permutation of integers."""
    def __init__(self, genes: np.ndarray):
        assert np.issubdtype(genes.dtype, np.integer), "Genes must be a numpy array of integer type."
        unique_genes = np.unique(genes)
        expected_genes = np.arange(len(genes))
        assert len(unique_genes) == len(genes) and np.array_equal(np.sort(unique_genes), expected_genes), (
            "Genes must be a permutation of integers from 0 to len(genes)-1."
        )
        self.genes = genes.astype(np.int32)

    @classmethod
    def random(cls, length: int) -> "PermutationGenotype":
        genes = np.random.permutation(length).astype(np.int32)
        return cls(genes)

    def copy(self) -> "PermutationGenotype":
        return PermutationGenotype(np.copy(self.genes))
    
    def __len__(self) -> int:
        return self.genes.size
    
    def as_array(self) -> np.ndarray:
        return self.genes
