"""
evolib.operators.selection
==========================

This module defines various parent selection strategies used in evolutionary algorithms.
Each selection operator determines how individuals are chosen from the current population
to produce offspring for the next generation.

All selection strategies implement the same interface:

    select(self, population, n_parents) -> list

Where:
    - population: list of individuals (any genotype type)
    - n_parents: number of parents to select

Author: EvoLab project
"""

from collections.abc import Sequence

import numpy as np

from evolib.core.individual import Individual, Population


class SelectionStrategy:
    """Base class for all selection strategies."""

    def select(self, population: Population, n_parents: int) -> Population:  # pragma: no cover (interface)
        raise NotImplementedError("SelectionStrategy must implement select().")

    # Common input validation helper
    @staticmethod
    def _validate(population: Sequence[Individual], n_parents: int) -> None:
        if len(population) == 0:
            raise ValueError("population must not be empty")
        if n_parents <= 0:
            raise ValueError("n_parents must be > 0")


class RouletteWheelSelection(SelectionStrategy):
    """
    Roulette Wheel (Fitness-Proportionate) Selection.
    Each individual's probability of being selected is proportional to its fitness.
    """

    def select(self, population: Population, n_parents: int) -> Population:
        self._validate(population, n_parents)
        fitness = np.array([ind.fitness for ind in population], dtype=float)
        total_fitness = float(np.sum(fitness))
        probs = np.full(len(population), 1.0 / len(population)) if total_fitness <= 0.0 else fitness / total_fitness
        selected_indices = np.random.choice(len(population), size=n_parents, replace=True, p=probs)
        return Population([population[i] for i in selected_indices])


class StochasticUniversalSampling(SelectionStrategy):
    """
    Stochastic Universal Sampling (SUS).
    Provides more even selection pressure than standard roulette wheel.
    """

    def select(self, population: Population, n_parents: int) -> Population:
        self._validate(population, n_parents)
        fitness = np.array([ind.fitness for ind in population], dtype=float)
        total_fitness = float(np.sum(fitness))
        probs = np.full(len(population), 1.0 / len(population)) if total_fitness <= 0.0 else fitness / total_fitness
        cumulative = np.cumsum(probs)
        step = 1.0 / n_parents
        start = np.random.uniform(0.0, step)
        pointers = start + step * np.arange(n_parents)
        selected: list[Individual] = []
        i = 0
        for p in pointers:
            while p > cumulative[i]:
                i += 1
            selected.append(population[i])
        return Population(selected)


class RankSelection(SelectionStrategy):
    """
    Rank-Based Selection.
    Selection probability depends on sorted order (rank), not raw fitness.
    """

    def select(self, population: Population, n_parents: int) -> Population:
        self._validate(population, n_parents)
        fitness = np.array([ind.fitness for ind in population], dtype=float)
        ranks = np.argsort(np.argsort(fitness)) + 1  # rank starts at 1
        probs = ranks / np.sum(ranks)
        selected_indices = np.random.choice(len(population), size=n_parents, replace=True, p=probs)
        return Population([population[i] for i in selected_indices])


class TournamentSelection(SelectionStrategy):
    """
    Tournament Selection.
    Randomly select k individuals and pick the one with highest fitness.
    """

    def __init__(self, k: int = 3):
        self.k = k

    def select(self, population: Population, n_parents: int) -> Population:
        self._validate(population, n_parents)
        if self.k <= 0:
            raise ValueError("k must be > 0")
        if self.k > len(population):
            raise ValueError("k must be <= population size")
        selected: list[Individual] = []
        for _ in range(n_parents):
            contender_indices = np.random.choice(len(population), size=self.k, replace=False)
            contenders = [population[i] for i in contender_indices]
            winner = max(contenders, key=lambda ind: ind.fitness)
            selected.append(winner)
        return Population(selected)


class TruncationSelection(SelectionStrategy):
    """
    Truncation Selection.
    Select only from the top fraction of the population.
    """

    def __init__(self, fraction: float = 0.5):
        self.fraction = fraction

    def select(self, population: Population, n_parents: int) -> Population:
        self._validate(population, n_parents)
        if not (0 < self.fraction <= 1):
            raise ValueError("fraction must be in (0, 1]")
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        cutoff = int(len(sorted_pop) * self.fraction)
        top = sorted_pop[: max(1, cutoff)]
        chosen_indices = np.random.choice(len(top), size=n_parents, replace=True)
        chosen = [top[i] for i in chosen_indices]
        return Population(chosen)


class BoltzmannSelection(SelectionStrategy):
    """
    Boltzmann Selection.
    Selection probability follows Boltzmann distribution based on "temperature".
    Higher temperature = more exploration.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def select(self, population: Population, n_parents: int) -> Population:
        self._validate(population, n_parents)
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        fitness = np.array([ind.fitness for ind in population], dtype=float)
        # Numerical stabilization
        shifted = fitness - fitness.max()
        scaled = np.exp(shifted / (self.temperature + 1e-8))
        probs = scaled / np.sum(scaled)
        selected_indices = np.random.choice(len(population), size=n_parents, replace=True, p=probs)
        return Population([population[i] for i in selected_indices])


class FitnessSharingSelection(SelectionStrategy):
    """
    Fitness Sharing / Niching.
    Promotes diversity by penalizing individuals similar to others.
    """

    def __init__(self, sigma_share: float = 1.0):
        self.sigma_share = sigma_share

    def select(self, population: Population, n_parents: int) -> Population:
        self._validate(population, n_parents)
        if self.sigma_share <= 0:
            raise ValueError("sigma_share must be > 0")
        fitness = np.array([ind.fitness for ind in population], dtype=float)
        # Compute niche counts (using fitness distance
        # could extend to genotype distance if available)
        niche_counts = np.zeros(len(population))
        for i in range(len(population)):
            for j in range(len(population)):
                distance = abs(fitness[i] - fitness[j])
                if distance < self.sigma_share:
                    niche_counts[i] += 1 - (distance / self.sigma_share)
            niche_counts[i] = max(niche_counts[i], 1.0)
        shared_fitness = fitness / niche_counts
        shared_fitness = np.maximum(shared_fitness, 1e-12)
        probs = shared_fitness / np.sum(shared_fitness)
        selected_indices = np.random.choice(len(population), size=n_parents, replace=True, p=probs)
        return Population([population[i] for i in selected_indices])


class RandomSelection(SelectionStrategy):
    """
    Random Selection.
    Baseline random selection (no dependence on fitness).
    """

    def select(self, population: Population, n_parents: int) -> Population:
        self._validate(population, n_parents)
        chosen_indices = np.random.choice(len(population), size=n_parents, replace=True)
        chosen = [population[i] for i in chosen_indices]
        return Population(chosen)


class Elitism(SelectionStrategy):
    """
    Elitism.
    Always preserves the top `elite_size` individuals.
    """

    def __init__(self, elite_size: int = 1):
        self.elite_size = elite_size

    def select(self, population: Population, n_parents: int | None = None) -> Population:
        if len(population) == 0:
            raise ValueError("population must not be empty")
        if self.elite_size <= 0:
            raise ValueError("elite_size must be > 0")
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        elites = sorted_pop[: self.elite_size]
        return Population(elites)
