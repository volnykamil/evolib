from evolib.core.genotype import Genotype
from typing import NewType, Callable
import pdb


Population = NewType("Population", list["Individual"])


class Individual:
    """Represents an individual in the population."""
    def __init__(self, genotype: Genotype, age: int = 0, fitness_score: float = 0.0):
        """
        Initialize an individual.

        Args:
            genotype (Genotype): The genotype of the individual.
            age (int, optional): The age of the individual. Defaults to 0.
            fitness_score (float, optional): The fitness score of the individual. Defaults to 0.0.
        """
        self.genotype: Genotype = genotype
        self.age: int = age
        self.fitnessScore: float = fitness_score

    def __eq__(self, value):
        if not isinstance(value, Individual):
            return False
        return (self.genotype == value.genotype and
                self.age == value.age and
                self.fitnessScore == value.fitnessScore
            )
    
    def copy(self) -> "Individual":
        return Individual(
            genotype=self.genotype.copy(),
            age=self.age,
            fitness_score=self.fitnessScore
        )
        
    @staticmethod
    def create_population(genotype_factory: Callable[[], Genotype], size: int) -> Population:
        """
        Create a population of individuals.

        Args:
            genotype_factory (Callable[[], Genotype]): A function that creates a new genotype.
            size (int): The number of individuals in the population.

        Returns:
            Population: A population of individuals.
        """
        return Population([Individual(genotype_factory()) for _ in range(size)])
