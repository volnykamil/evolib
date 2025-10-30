import time
from abc import ABC, abstractmethod

from evolib.core.individual import Population


class TerminationCondition(ABC):
    """Abstract base class for termination conditions in evolutionary algorithms."""

    @abstractmethod
    def should_terminate(self, generation: int, population: Population, best_fitness: float) -> bool:
        """Determine whether the evolutionary process should terminate.

        Args:
            generation (int): The current generation number.
            population (Population): The current population of individuals.
            best_fitness (float): The best fitness value in the current generation.

        Returns:
            bool: True if the process should terminate, False otherwise.
        """
        pass


class MaxGenerationsTermination(TerminationCondition):
    """Terminate after a maximum number of generations."""

    def __init__(self, max_generations: int):
        self.max_generations = max_generations

    def should_terminate(self, generation: int, population: Population, best_fitness: float) -> bool:
        return generation >= self.max_generations


class FitnessThresholdTermination(TerminationCondition):
    """Terminate when a fitness threshold is reached."""

    def __init__(self, fitness_threshold: float):
        self.fitness_threshold = fitness_threshold

    def should_terminate(self, generation: int, population: Population, best_fitness: float) -> bool:
        return best_fitness >= self.fitness_threshold
    

class StagnationTermination(TerminationCondition):
    """Terminate if there is no improvement in best fitness for a number of generations."""

    def __init__(self, max_stagnant_generations: int):
        self.max_stagnant_generations = max_stagnant_generations
        self.best_fitness_history: list[float] = []

    def should_terminate(self, generation: int, population: Population, best_fitness: float) -> bool:
        self.best_fitness_history.append(best_fitness)
        if len(self.best_fitness_history) > self.max_stagnant_generations:
            self.best_fitness_history.pop(0)
            if all(f == self.best_fitness_history[0] for f in self.best_fitness_history):
                return True
        return False


class TimeLimitTermination(TerminationCondition):
    """Terminate after a specified time limit (in seconds)."""

    def __init__(self, time_limit_seconds: float):
        self.time_limit_seconds = time_limit_seconds
        self.start_time = time.time()

    def should_terminate(self, generation: int, population: Population, best_fitness: float) -> bool:
        elapsed_time = time.time() - self.start_time
        return elapsed_time >= self.time_limit_seconds


class DiverseUnderMinimumTermination(TerminationCondition):
    """Terminate if population diversity falls below a minimum threshold."""

    def __init__(self, min_diversity: float):
        self.min_diversity = min_diversity

    def should_terminate(self, generation: int, population: Population, best_fitness: float) -> bool:
        try:
            # Assume population is iterable of individuals with a genotype having as_array()
            genotypes = [ind.genotype for ind in population]
            arrays = [g.as_array() for g in genotypes if hasattr(g, "as_array")]
            if arrays:
                unique = {tuple(a.tolist()) for a in arrays}
                diversity = len(unique) / len(arrays)
            else:
                diversity = 0.0
        except Exception:
            diversity = 0.0
        return diversity < self.min_diversity


class HybridTermination(TerminationCondition):
    """Combine multiple termination conditions; terminate if any condition is met."""

    def __init__(self, conditions: list[TerminationCondition]):
        self.conditions = conditions

    def should_terminate(self, generation: int, population: Population, best_fitness: float) -> bool:
        return any(
            condition.should_terminate(generation, population, best_fitness)
            for condition in self.conditions
        )
