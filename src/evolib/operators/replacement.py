"""
evolib.operators.replacement
============================

This module defines various population replacement strategies for evolutionary algorithms.

Replacement strategies determine how the next generation of individuals is formed
from the current population and the newly created offspring.

Available strategies:
- GenerationalReplacement
- SteadyStateReplacement
- MuLambdaReplacement
- MuPlusLambdaReplacement
- ElitismReplacement
- AgeBasedReplacement
- FitnessSharingReplacement

Each strategy implements the `replace` method:
    replace(parents, offspring, population_size) -> new_population
"""

from evolib.core.individual import Population, Individual
from abc import ABC, abstractmethod
import numpy as np
import warnings


class ReplacementStrategy(ABC):
    """Abstract base class for replacement strategies."""

    @abstractmethod
    def replace(self, parents: Population, offspring: Population, population_size: int) -> Population:
        """
        Replace individuals in the population.

        Args:
            parents (Population): The current population of parents.
            offspring (Population): The newly created offspring.
            population_size (int): The desired size of the new population.

        Returns:
            Population: The new population after replacement.
        """
        pass


class GenerationalReplacement(ReplacementStrategy):
    """
    Replaces the entire parent population with the offspring population.

    - Common in simple genetic algorithms.
    - Maximizes exploration but may lose good solutions if elitism is not used.
    """

    def replace(self, parents: Population, offspring: Population, population_size: int) -> Population:
        return Population(offspring[:population_size])
    

class SteadyStateReplacement(ReplacementStrategy):
    """
    Replaces only a few individuals per generation (usually the worst ones).

    - Maintains most of the old population.
    - Useful for continuous optimization where gradual improvement is desired.
    """

    def __init__(self, num_replacements: int = 2):
        """
        Initialize the steady-state replacement strategy.

        Args:
            num_replacements (int, optional): The number of individuals to replace. Defaults to 2.
        """
        self.num_replacements = num_replacements

    def replace(self, parents: Population, offspring: Population, population_size: int) -> Population:
        parents = sorted(parents, key=lambda ind: ind.fitness, reverse=True)
        offspring = sorted(offspring, key=lambda ind: ind.fitness, reverse=True)
        # Replace worst parents with best offspring
        new_population = parents[:-self.num_replacements] + offspring[:self.num_replacements]
        return new_population[:population_size]


class MuLambdaReplacement(ReplacementStrategy):
    """
    (μ, λ)-Strategy (Evolution Strategies)

    In this strategy, the new generation is created **only** from offspring.
    Parents do not survive directly; only the best μ individuals among λ offspring
    form the next generation.

    Typical parameters: λ ≥ μ.

    Attributes
    ----------
    mu : int
        Number of individuals (μ) to keep in the next generation.
    lambda_ : int
        Number of offspring (λ) generated from the parent population.

    Notes
    -----
    - Ensures strong selection pressure and promotes exploration.
    - Since parents are discarded, diversity remains higher but convergence can be slower.
    """

    def __init__(self, mu: int, lambda_: int):
        assert mu > 0, "mu must be > 0"
        assert lambda_ >= mu, "lambda should be >= mu"
        self.mu = mu
        self.lambda_ = lambda_

    def replace(self, parents: Population, offspring: Population, population_size: int) -> Population:
        if population_size <= 0:
            raise ValueError("population_size must be > 0")
        if len(offspring) != self.lambda_:
            warnings.warn(
                f"MuLambdaReplacement: Expected {self.lambda_} offspring, got {len(offspring)}.",
                stacklevel=2,
            )
        sorted_offspring = sorted(offspring, key=lambda ind: ind.fitness, reverse=True)
        selected = sorted_offspring[: self.mu]
        return Population(selected[:population_size])


class MuPlusLambdaReplacement(ReplacementStrategy):
    """
    (μ + λ)-Strategy (Evolution Strategies)

    Both parents and offspring compete for survival.
    The next generation is formed by selecting the best μ individuals
    from the combined pool of parents (μ) and offspring (λ).

    Attributes
    ----------
    mu : int
        Number of individuals to keep for the next generation.
    lambda_ : int
        Number of offspring generated in each iteration.

    Notes
    -----
    - Promotes **elitism** (best individuals always survive).
    - Provides more stable convergence compared to (μ, λ)-strategy.
    """

    def __init__(self, mu: int, lambda_: int):
        assert mu > 0, "mu must be > 0"
        assert lambda_ >= 0, "lambda must be >= 0"
        self.mu = mu
        self.lambda_ = lambda_

    def replace(self, parents: Population, offspring: Population, population_size: int) -> Population:
        if population_size <= 0:
            raise ValueError("population_size must be > 0")
        if len(offspring) != self.lambda_:
            warnings.warn(
                f"MuPlusLambdaReplacement: Expected {self.lambda_} offspring, got {len(offspring)}.",
                stacklevel=2,
            )
        combined = list(parents) + list(offspring)
        sorted_combined = sorted(combined, key=lambda ind: ind.fitness, reverse=True)
        selected = sorted_combined[: self.mu]
        return Population(selected[:population_size])


class ElitismReplacement(ReplacementStrategy):
    """
    Keeps the best individuals (elite) from the previous generation.

    - Commonly used as a complement to generational replacement.
    - Ensures that the best found solution is never lost.
    """

    def __init__(self, elite_size: int = 1):
        self.elite_size = elite_size

    def replace(self, parents: Population, offspring: Population, population_size: int) -> Population:
        if population_size <= 0:
            raise ValueError("population_size must be > 0")
        parents_sorted = sorted(parents, key=lambda ind: ind.fitness, reverse=True)
        offspring_sorted = sorted(offspring, key=lambda ind: ind.fitness, reverse=True)
        elites = parents_sorted[: self.elite_size]
        remaining_slots = max(0, population_size - len(elites))
        new_population = elites + offspring_sorted[:remaining_slots]
        return Population(new_population)


class AgeBasedReplacement(ReplacementStrategy):
    """
    Replaces individuals based on their age.

    - Each individual should have an 'age' attribute.
    - Oldest individuals are removed first.
    - Helps maintain evolutionary diversity.
    """

    def replace(self, parents: Population, offspring: Population, population_size: int) -> Population:
        if population_size <= 0:
            raise ValueError("population_size must be > 0")
        combined = parents + offspring
        combined = sorted(combined, key=lambda ind: ind.age)
        return Population(combined[:population_size])
    

class FitnessSharingReplacement(ReplacementStrategy):
    """
    Uses fitness sharing or crowding to maintain population diversity.

    - Penalizes individuals that are too similar to others.
    - Encourages exploration of multiple niches in the search space.
    """

    def __init__(self, sigma_share: float = 0.5, alpha: float = 1.0):
        self.sigma_share = sigma_share
        self.alpha = alpha

    def _sharing_function(self, distance):
        if distance < self.sigma_share:
            return 1 - (distance / self.sigma_share) ** self.alpha
        return 0

    def replace(self, parents: Population, offspring: Population, population_size: int) -> Population:
        if population_size <= 0:
            raise ValueError("population_size must be > 0")
        combined: list[Individual] = list(parents) + list(offspring)
        n = len(combined)
        if n == 0:
            return Population([])

        # Precompute genotype arrays (fallback to fitness distance if genotype missing)
        arrays: list[np.ndarray | None] = []
        for ind in combined:
            try:
                arrays.append(ind.genotype.as_array())
            except Exception:
                arrays.append(None)

        niche_counts = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if arrays[i] is not None and arrays[j] is not None:
                    # Use L2 distance between genotypes of same length; otherwise fallback
                    if arrays[i].shape == arrays[j].shape:
                        distance = float(np.linalg.norm(arrays[i] - arrays[j]))
                    else:
                        distance = abs(combined[i].fitness - combined[j].fitness)
                else:
                    distance = abs(combined[i].fitness - combined[j].fitness)
                niche_counts[i] += self._sharing_function(distance)
            # Ensure at least self-count contributes (avoid divide-by-zero)
            niche_counts[i] = max(niche_counts[i], 1.0)

        shared_fitness = np.array([ind.fitness for ind in combined]) / niche_counts
        order = np.argsort(shared_fitness)[::-1]
        selected = [combined[idx] for idx in order[:population_size]]
        return Population(selected)
