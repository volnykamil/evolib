"""Core individual abstraction and population factory utilities.

The :class:`Individual` couples a genotype with evolutionary metadata
(fitness, age, etc.).
"""

from collections.abc import Callable, Iterable
from typing import NewType

from evolib.core.genotype import Genotype

Population = NewType("Population", list["Individual"])


class Individual:
    """Represents a single candidate solution.

    Parameters
    ----------
    genotype : Genotype
        Underlying genetic representation.
    age : int, default 0
        Non-negative integer counting generations survived.
    fitness : float, default 0.0
        Raw fitness value. Higher usually means better (problem dependent).
    """

    __slots__ = ("age", "fitness", "genotype")

    def __init__(self, genotype: Genotype, age: int = 0, fitness: float = 0.0) -> None:
        if age < 0:
            raise ValueError("age must be non-negative")
        self.genotype: Genotype = genotype
        self.age: int = age
        self.fitness: float = float(fitness)

    # ------------------------------------------------------------------
    # Core protocol helpers
    # ------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, Individual):
            return False
        return self.genotype == other.genotype and self.age == other.age and self.fitness == other.fitness

    def __repr__(self) -> str:  # Helpful for debugging/logging
        return (
            f"Individual(genotype={self.genotype.__class__.__name__}(len={len(self.genotype)}), "
            f"age={self.age}, fitness={self.fitness:.4f})"
        )

    def copy(self) -> "Individual":
        """Create a deep copy preserving metadata."""
        return Individual(
            genotype=self.genotype.copy(),
            age=self.age,
            fitness=self.fitness,
        )

    def __hash__(self):
        return hash((self.genotype, self.age, self.fitness))

    # ------------------------------------------------------------------
    # Population utilities
    # ------------------------------------------------------------------
    @staticmethod
    def create_population(genotype_factory: Callable[[], Genotype], size: int) -> Population:
        """Create a new population of individuals.

        Parameters
        ----------
        genotype_factory : Callable[[], Genotype]
            Factory returning a freshly randomized genotype instance.
        size : int
            Number of individuals to create (must be > 0).

        Returns
        -------
        Population
            List-based population wrapper.
        """
        if size <= 0:
            raise ValueError("size must be > 0")
        return Population([Individual(genotype_factory()) for _ in range(size)])

    @staticmethod
    def from_genotypes(genotypes: Iterable[Genotype]) -> Population:
        """Create a population directly from an iterable of genotypes."""
        return Population([Individual(g) for g in genotypes])
